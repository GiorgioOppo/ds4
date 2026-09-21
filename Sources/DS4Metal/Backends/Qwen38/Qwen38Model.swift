import Foundation
import Metal
import DS4Core
import Darwin

/// Native Swift orchestration of the Qwen3.8 Flash Next text trunk. Calls must
/// be serialized by the owning backend, like the other Swift model decoders.
/// Original BF16 n-grams and routed experts are paged from the GGUF on demand.
public final class Qwen38Model: SwiftModelDecoder {
    public let contextCapacity: Int
    public var vocabularySize: Int { c.vocabulary }
    public private(set) var position = 0
    public let configuration: Qwen38Configuration
    let model: GGUFModel
    let weights: Qwen38Weights
    let rt: MetalRuntime
    let fd: Int32
    let batchCapacity: Int
    var c: Qwen38Configuration { configuration }
    var valid = true
    var ngrams = Qwen38NgramState()
    var mapped = [String: GPUTensor]()
    var scratch = [String: GPUTensor]()
    var states = [[String: GPUTensor]]()
    var specializedPipelines = [String: MTLComputePipelineState]()

    public init(model: GGUFModel, contextSize: Int) throws {
        let config = try Qwen38Configuration(model: model)
        let validated = try Qwen38Weights(model: model, configuration: config)
        guard contextSize > 0, contextSize <= config.nativeContext else {
            throw SwiftModelDecoderError.invalidInput("Qwen3.8 context must be in 1...\(config.nativeContext)")
        }
        let descriptor = open(model.path, O_RDONLY)
        guard descriptor >= 0 else { throw Qwen38Error.io(String(cString: strerror(errno))) }
        // Avoid duplicating active expert and n-gram reads in the file cache.
        _ = fcntl(descriptor, F_NOCACHE, 1)
        do {
            self.rt = try MetalRuntime(additionalSources: [Qwen38KernelSource.source])
        } catch { close(descriptor); throw error }
        self.fd = descriptor; self.model = model; self.weights = validated
        self.configuration = config; self.contextCapacity = contextSize
        let largestExpert = (0..<config.layers).map { layer in
            ["gate", "up", "down"].reduce(0) { $0 + Int(validated["blk.\(layer).ffn_\($1)_exps.weight"].bytes) / config.experts }
        }.max()!
        // Bound the three routed weight slabs together to 1 GiB, including
        // unusual F32 fixtures/exports. The production Q2 and Q4 recipes both
        // retain 32-token prefill. Selected expert IDs are unique per token.
        self.batchCapacity = min(32, (1 << 30) / largestExpert / config.expertsUsed)
        guard batchCapacity > 0 else { throw Qwen38Error.invalidModel("An active expert set exceeds the staging budget") }
        try allocate()
    }

    deinit { close(fd) }

    public func reset() throws {
        // evaluate waits for every submitted command, including failure paths.
        // Cache rows beyond position are never read, so only recurrent state
        // and convolution history require clearing.
        for state in states {
            state["conv"]?.zero(); state["gdn"]?.zero()
        }
        scratch["pleHistory"]?.zero()
        position = 0; ngrams = Qwen38NgramState(eos: c.pleEOS); valid = true
    }

    public func evaluate(tokens: [Int], cancelled: @Sendable () -> Bool) throws -> [Float] {
        guard valid else { throw SwiftModelDecoderError.invalidState }
        guard !tokens.isEmpty, tokens.allSatisfy({ (0..<vocabularySize).contains($0) }) else {
            throw SwiftModelDecoderError.invalidInput("Qwen3.8 requires nonempty, valid token IDs")
        }
        guard tokens.count <= contextCapacity - position else {
            throw SwiftModelDecoderError.contextOverflow(requested: position + tokens.count, capacity: contextCapacity)
        }
        if cancelled() { throw SwiftModelDecoderError.cancelled }
        do {
            var offset = 0
            while offset < tokens.count {
                if cancelled() { throw SwiftModelDecoderError.cancelled }
                let count = min(batchCapacity, tokens.count - offset)
                try forward(Array(tokens[offset..<(offset + count)]), cancelled: cancelled)
                position += count; offset += count
            }
            return s("logits").floatArray(vocabularySize)
        } catch {
            // Recurrent GDN/PLE histories cannot be rolled back after a partial
            // layer. Explicit reset is mandatory before this instance is reused.
            valid = false
            throw error
        }
    }

    func s(_ name: String) -> GPUTensor { scratch[name]! }
    func tensor(_ name: String) -> GGUFModel.Tensor { weights[name] }

    func weight(_ name: String) throws -> GPUTensor {
        if let found = mapped[name] { return found }
        let t = tensor(name)
        // Experts and n-grams must use pread, including on high-memory devices.
        guard !name.contains("_exps."), name != "per_layer_token_embd.weight",
              t.bytes <= UInt64(rt.device.maxBufferLength - Int(getpagesize())) else {
            throw Qwen38Error.invalidModel("Tensor \(name) exceeds the bounded dense mapping limit")
        }
        let b = try GPUTensor.mappedNoCopy(rt, ptr: model.mapBase + Int(t.absOffset), byteLength: Int(t.bytes), elementCount: Int(t.elements))
        mapped[name] = b
        return b
    }

    func allocate() throws {
        let T = batchCapacity, E = c.embedding, HC = c.hcDimension, Q = c.queryDimension
        let counts: [(String, Int)] = [
            ("R", T * HC), ("xn", T * HC), ("hcUp", T * HC), ("lo", T * c.hcRank), ("loAct", T * c.hcRank),
            ("mixed", T * E), ("blk", T * E), ("inj", T * c.hcCount * 8 * c.hcCount),
            ("pleEmbedding", T * E), ("pleKey", T * HC), ("pleValue", T * E), ("pleGated", T * HC), ("pleNormed", T * HC), ("pleHistory", 9 * HC),
            ("qkv", T * c.linearQKVDimension), ("z", T * c.linearValueDimension), ("alpha", T * c.linearVHeads), ("beta", T * c.linearVHeads), ("linearOut", T * c.linearValueDimension),
            ("qg", T * 2 * Q), ("kp", T * c.kvHeads * c.headDimension), ("vp", T * c.kvHeads * c.headDimension),
            ("iq", T * c.indexHeads * c.indexDimension), ("ik", T * c.indexDimension), ("iqn", T * c.indexHeads * c.indexDimension),
            ("q", T * Q), ("gate", T * Q), ("attentionOut", T * Q),
            ("scores", T * max(1, contextCapacity / 4)), ("selectedBlocks", T * (c.indexTopK / 4)),
            ("selectedTokens", T * (c.indexTopK + 4)), ("selectedCount", T),
            ("attentionPart", 2 * c.heads * 64 * (c.headDimension + 2)),
            ("router", T * c.experts), ("selected", T * c.expertsUsed), ("routeWeights", T * c.expertsUsed), ("sharedGate", T),
            ("mid", T * c.expertsUsed * c.expertWidth), ("part", T * c.expertsUsed * E),
            ("sharedA", T * c.expertWidth), ("sharedB", T * c.expertWidth), ("sharedMid", T * c.expertWidth), ("sharedOut", T * E),
            ("lists", c.experts * T), ("listCounts", c.experts), ("logits", c.vocabulary), ("dummy", 1)
        ]
        for (name, count) in counts { scratch[name] = try GPUTensor.zeros(rt, floatCount: count) }
        scratch["positions"] = try GPUTensor.uninitializedBytes(rt, byteLength: contextCapacity * 16, elementCount: contextCapacity * 4)
        for i in 0..<c.layers {
            if c.isLinear(layer: i) {
                states.append([
                    "conv": try GPUTensor.zeros(rt, floatCount: 3 * c.linearQKVDimension),
                    "gdn": try GPUTensor.zeros(rt, floatCount: c.linearVHeads * c.linearDimension * c.linearDimension)
                ])
            } else {
                states.append([
                    "k": try GPUTensor.uninitializedBytes(rt, byteLength: contextCapacity * c.kvHeads * c.headDimension * 2, elementCount: contextCapacity * c.kvHeads * c.headDimension),
                    "v": try GPUTensor.uninitializedBytes(rt, byteLength: contextCapacity * c.kvHeads * c.headDimension * 2, elementCount: contextCapacity * c.kvHeads * c.headDimension),
                    "ik": try GPUTensor.lazyZeros(rt, floatCount: contextCapacity * c.indexDimension),
                    "blocks": try GPUTensor.uninitializedBytes(rt, byteLength: max(1, contextCapacity / 4) * c.indexDimension * 2, elementCount: max(1, contextCapacity / 4) * c.indexDimension)
                ])
            }
        }
    }

    func read(_ target: UnsafeMutableRawPointer, bytes: Int, offset: UInt64) throws {
        guard offset <= model.size, UInt64(bytes) <= model.size - offset else { throw Qwen38Error.io("File range is out of bounds") }
        var done = 0
        while done < bytes {
            let n = pread(fd, target + done, bytes - done, off_t(offset) + off_t(done))
            if n < 0 && errno == EINTR { continue }
            guard n > 0 else { throw Qwen38Error.io(n == 0 ? "Unexpected EOF" : String(cString: strerror(errno))) }
            done += n
        }
    }

    func stage(_ tokens: [Int]) throws {
        let emb = tensor("token_embd.weight"), ng = tensor("per_layer_token_embd.weight")
        let bytes = try Qwen38Weights.rowBytes(type: emb.type, width: c.embedding)
        let row = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 16)
        let ngRow = UnsafeMutableRawPointer.allocate(byteCount: c.pleHeadDimension * 2, alignment: 16)
        defer { row.deallocate(); ngRow.deallocate() }
        let r = s("R").buffer.contents().assumingMemoryBound(to: Float.self)
        let ple = s("pleEmbedding").buffer.contents().assumingMemoryBound(to: Float.self)
        let positions = s("positions").buffer.contents().assumingMemoryBound(to: UInt32.self)
        for (t, token) in tokens.enumerated() {
            try read(row, bytes: bytes, offset: emb.absOffset + UInt64(token * bytes))
            let values = try Self.embeddingRow(UnsafeRawPointer(row), width: c.embedding, type: emb.type)
            for stream in 0..<c.hcCount { _ = values.withUnsafeBytes { memcpy(r + t * c.hcDimension + stream * c.embedding, $0.baseAddress!, c.embedding * 4) } }
            let p = (position + t) * 4
            positions[p] = UInt32(position + t); positions[p + 1] = positions[p]; positions[p + 2] = positions[p]; positions[p + 3] = 0
            let rows = ngrams.rows(token: token, multipliers: c.pleMultipliers, offsets: c.pleOffsets, vocabularies: c.pleVocabulary, eos: c.pleEOS)
            for (h, id) in rows.enumerated() {
                try read(ngRow, bytes: c.pleHeadDimension * 2, offset: ng.absOffset + UInt64(id) * UInt64(c.pleHeadDimension * 2))
                for j in 0..<c.pleHeadDimension {
                    ple[t * c.embedding + h * c.pleHeadDimension + j] = Float(bitPattern: UInt32(ngRow.loadUnaligned(fromByteOffset: j * 2, as: UInt16.self)) << 16)
                }
            }
        }
    }

    static func embeddingRow(_ p: UnsafeRawPointer, width: Int, type: UInt32) throws -> [Float] {
        var out = [Float](repeating: 0, count: width)
        for i in 0..<width {
            switch type {
            case 0: out[i] = p.loadUnaligned(fromByteOffset: i * 4, as: Float.self)
            case 1: out[i] = Float(Float16(bitPattern: p.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)))
            case 30: out[i] = Float(bitPattern: UInt32(p.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)) << 16)
            case 8:
                let b = (i / 32) * 34
                out[i] = Float(Float16(bitPattern: p.loadUnaligned(fromByteOffset: b, as: UInt16.self))) * Float(p.load(fromByteOffset: b + 2 + i % 32, as: Int8.self))
            case 2:
                let b = (i / 32) * 18, j = i % 32
                let q = p.load(fromByteOffset: b + 2 + j % 16, as: UInt8.self)
                let v = j < 16 ? q & 15 : q >> 4
                out[i] = Float(Float16(bitPattern: p.loadUnaligned(fromByteOffset: b, as: UInt16.self))) * Float(Int(v) - 8)
            default: throw Qwen38Error.invalidModel("Unsupported embedding type \(type)")
            }
        }
        return out
    }
}
