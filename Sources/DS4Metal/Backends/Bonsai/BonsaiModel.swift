import Foundation
import Metal
import DS4Core

/// Native Swift driver for the complete 64-layer Ternary Bonsai 2 trunk.
/// Packed weights remain resident; KV, convolution history and GDN state are
/// private to this instance. The owning executor must serialize its methods.
public final class BonsaiModel: SwiftModelDecoder {
    public let contextCapacity: Int
    public var vocabularySize: Int { configuration.vocabularySize }
    public private(set) var position = 0
    public let configuration: BonsaiConfiguration

    private let model: GGUFModel // owns the mmap used by no-copy Metal weights
    private let gpu: BonsaiGPU
    private let reservation: BonsaiMemoryReservation
    private let weights: [String: BonsaiWeight]
    private let scratch: [Slot: BonsaiBuffer]
    private let states: [Int: BonsaiBuffer], histories: [Int: BonsaiBuffer]
    private let keys: [Int: BonsaiBuffer], values: [Int: BonsaiBuffer]
    private let batchCapacity: Int, attentionBatchCapacity: Int
    private var invalid = false

    private enum Slot: Int, CaseIterable {
        case x, norm, result, qkv, z, alpha, beta, conv, gdn, q, k, v, attention, gate, up, mid, rotated, logits, scores
    }

    public init(model: GGUFModel, contextSize: Int) throws {
        let configuration = try BonsaiConfiguration(model: model)
        guard contextSize > 0, contextSize <= configuration.contextLimit else {
            throw SwiftModelDecoderError.contextOverflow(requested: contextSize, capacity: configuration.contextLimit)
        }
        guard let device = MTLCreateSystemDefaultDevice() else { throw SwiftModelDecoderError.gpu("Metal non disponibile") }
        let batch = min(contextSize, 128)
        let attentionBatch = min(batch, 8, max(1, (8 * 1024 * 1024) / (contextSize * 24)))
        let sizes = [5120,5120,5120,12288,6144,48,48,10240,6144,6144,1024,1024,6144,17408,17408,17408,17408,248320,contextSize*24*attentionBatch]
        let page = UInt64(getpagesize())
        // Round mapped ranges conservatively, including both possible edge pages.
        let weightBytes = configuration.weights.values.reduce(UInt64(0)) { $0 + $1.tensor.bytes + 2 * page }
        let scratchBytes = sizes.enumerated().reduce(UInt64(0)) { $0 + UInt64($1.element * ($1.offset < Slot.logits.rawValue ? batch : 1)) * 4 }
        let recurrentBytes = UInt64(48 * (6144 * 128 + 10240 * 3)) * 4
        let cacheBytes = UInt64(16 * 2 * contextSize * 1024) * 4
        let reservation = try BonsaiMemoryReservation(bytes: weightBytes + scratchBytes + recurrentBytes + cacheBytes + 28672 * 4, device: device)
        let gpu = try BonsaiGPU(device: device)
        var signs: [Int: BonsaiBuffer] = [:]
        for (width, table) in configuration.signs {
            let buffer = try gpu.allocate(floats: width, label: "Bonsai Hadamard \(width)")
            table.withUnsafeBytes { source in buffer.buffer.contents().copyMemory(from: source.baseAddress!, byteCount: source.count) }
            signs[width] = buffer
        }
        var weights: [String: BonsaiWeight] = [:]
        for (name, spec) in configuration.weights {
            weights[name] = try gpu.mapWeight(spec, model: model, signs: spec.folded ? signs[spec.columns] : nil)
        }
        var scratch: [Slot: BonsaiBuffer] = [:]
        for slot in Slot.allCases {
            scratch[slot] = try gpu.allocate(floats: sizes[slot.rawValue] * (slot.rawValue < Slot.logits.rawValue ? batch : 1), label: "Bonsai \(slot)")
        }
        var states: [Int: BonsaiBuffer] = [:], histories: [Int: BonsaiBuffer] = [:]
        var keys: [Int: BonsaiBuffer] = [:], values: [Int: BonsaiBuffer] = [:]
        for i in 0..<64 {
            if (i + 1) % 4 == 0 {
                keys[i] = try gpu.allocate(floats: contextSize * 1024, label: "Bonsai K \(i)")
                values[i] = try gpu.allocate(floats: contextSize * 1024, label: "Bonsai V \(i)")
            } else {
                states[i] = try gpu.allocate(floats: 6144 * 128, label: "Bonsai state \(i)")
                histories[i] = try gpu.allocate(floats: 10240 * 3, label: "Bonsai convolution \(i)")
            }
        }
        self.model = model; self.configuration = configuration; self.contextCapacity = contextSize
        self.gpu = gpu; self.reservation = reservation; self.weights = weights; self.scratch = scratch
        self.states = states; self.histories = histories; self.keys = keys; self.values = values
        self.batchCapacity = batch; self.attentionBatchCapacity = attentionBatch
        try reset()
    }

    public func reset() throws {
        for view in Array(states.values) + Array(histories.values) { memset(view.buffer.contents(), 0, view.buffer.length) }
        // Old KV rows are unreachable after resetting the causal frontier.
        position = 0; invalid = false
    }

    public func evaluate(tokens: [Int], cancelled: @Sendable () -> Bool) throws -> [Float] {
        try evaluateRows(tokens: tokens, embeddings: nil, positions: nil, cancelled: cancelled)
    }

    /// Optional original-space embedding rows replace token lookup and its
    /// inverse Hadamard. Positions are [time,height,width] triples per row.
    /// The caller owns image-grid construction and supplies subsequent text
    /// positions including its MRoPE delta; causal KV indexing stays sequential.
    public func evaluateRows(tokens: [Int], embeddings: [[Float]?]?, positions: [Int32]?,
                             cancelled: @Sendable () -> Bool) throws -> [Float] {
        guard !invalid else { throw SwiftModelDecoderError.invalidState }
        guard !tokens.isEmpty, tokens.allSatisfy({ $0 >= 0 && $0 < vocabularySize }),
              embeddings == nil || embeddings!.count == tokens.count,
              positions == nil || positions!.count == tokens.count * 3 else {
            throw SwiftModelDecoderError.invalidInput("Bonsai: token, embedding o posizioni non validi")
        }
        if let embeddings {
            for row in embeddings.compactMap({ $0 }) {
                guard row.count == configuration.embeddingSize, row.allSatisfy(\.isFinite) else {
                    throw SwiftModelDecoderError.invalidInput("Bonsai: ogni embedding deve contenere 5120 valori finiti")
                }
            }
        }
        guard tokens.count <= contextCapacity - position else {
            throw SwiftModelDecoderError.contextOverflow(requested: position + tokens.count, capacity: contextCapacity)
        }
        do {
            var start = 0
            while start < tokens.count {
                if cancelled() { throw SwiftModelDecoderError.cancelled }
                var count = min(batchCapacity, tokens.count - start)
                if count > 32 { count = count / 32 * 32 }
                let end = start + count
                try chunk(tokens: Array(tokens[start..<end]),
                          embeddings: embeddings.map { Array($0[start..<end]) },
                          positions: positions.map { Array($0[start*3..<end*3]) },
                          publishLogits: end == tokens.count, cancelled: cancelled)
                start = end
            }
            let pointer = s(.logits).buffer.contents().assumingMemoryBound(to: Float.self)
            let logits = Array(UnsafeBufferPointer(start: pointer, count: vocabularySize))
            guard logits.allSatisfy(\.isFinite) else { throw SwiftModelDecoderError.gpu("Bonsai ha prodotto logits non finiti") }
            return logits
        } catch {
            invalid = true
            throw error
        }
    }

    private func s(_ slot: Slot) -> BonsaiBuffer { scratch[slot]! }
    private func w(_ name: String) -> BonsaiWeight { weights[name]! }
    private func w(_ layer: Int, _ name: String) -> BonsaiWeight { w("blk.\(layer)." + name) }
    private func dispatch(_ enc: MTLComputeCommandEncoder, _ name: String, _ args: BonsaiArgs,
                          _ buffers: [BonsaiBuffer], _ x: Int, _ y: Int = 1, _ z: Int = 1, threads: Int = 256) throws {
        try gpu.dispatch(enc, name, args, buffers, x, y, z, threads: threads)
    }
    private func vector(_ enc: MTLComputeCommandEncoder, _ name: String, _ args: BonsaiArgs, _ buffers: [BonsaiBuffer], _ n: Int) throws {
        try dispatch(enc, name, args, buffers, (n + 255) / 256)
    }
    private func norm(_ enc: MTLComputeCommandEncoder, _ weight: BonsaiWeight?, _ input: BonsaiBuffer,
                      _ output: BonsaiBuffer, heads: Int, width: Int, stride: Int) throws {
        let a = BonsaiArgs(cols: width, type: weight?.type ?? 0, width: stride, mode: weight == nil ? 1 : 0, eps: configuration.epsilon)
        try dispatch(enc, "norm", a, [input, weight?.view ?? input, output], heads)
    }
    private func element(_ enc: MTLComputeCommandEncoder, _ x: BonsaiBuffer, _ y: BonsaiBuffer,
                         _ out: BonsaiBuffer, count: Int, mode: UInt32, dim: Int = 0) throws {
        try vector(enc, "element", BonsaiArgs(n: count, dim: dim, mode: mode), [x,y,out], count)
    }
    private func transform(_ enc: MTLComputeCommandEncoder, _ weight: BonsaiWeight, _ input: BonsaiBuffer,
                           _ output: BonsaiBuffer, count: Int, inverse: Bool = false, grouped: Bool = false) throws {
        guard let signs = weight.signs else { throw SwiftModelDecoderError.gpu("Bonsai Hadamard signs assenti") }
        let args = BonsaiArgs(n: weight.columns, heads: 48, dim: 128, mode: inverse ? 1 : 0, groups: grouped ? 16 : 0)
        try dispatch(enc, "hadamard", args, [input,signs,output], weight.columns / 1024, count)
    }

    /// Preserves the upstream dispatch gates: tiled PQ2 for long prefill,
    /// independent PTQ sums for 4/8 rows, and optimized resident decode GEMV.
    private func projectPrepared(_ enc: MTLComputeCommandEncoder, _ weight: BonsaiWeight,
                                 _ input: BonsaiBuffer, _ output: BonsaiBuffer, count: Int) throws {
        let a = BonsaiArgs(n: count, rows: weight.rows, cols: weight.columns, type: weight.type, rowBytes: weight.spec.rowBytes)
        if count == 1 {
            if weight.type == 142 && weight.rows >= 4096 {
                try dispatch(enc, weight.rows % 16 == 0 ? "pq2_mv_full" : "pq2_mv", a, [weight.view,input,output], (weight.rows + 15) / 16, threads: 64)
            } else { try dispatch(enc, "mv", a, [weight.view,input,output], (weight.rows + 3) / 4, threads: 128) }
        } else if weight.type == 142 && weight.rows >= 4096 && count >= 16 {
            try dispatch(enc, "mm_pq2_tiled", a, [weight.view,input,output], (weight.rows + 63) / 64, (count + 31) / 32, threads: 128)
        } else if weight.type == 143 && weight.rows >= 4096 && count >= 4 {
            let tile = count >= 8 ? 8 : 4
            try dispatch(enc, tile == 8 ? "ptq_mm_8" : "ptq_mm", a, [weight.view,input,output], (weight.rows + 7) / 8, (count + tile - 1) / tile, threads: 128)
        } else if count < 4 {
            for row in 0..<count {
                try dispatch(enc, "mv", a, [weight.view,input.row(row,weight.columns),output.row(row,weight.rows)], (weight.rows + 3) / 4, threads: 128)
            }
        } else {
            try dispatch(enc, "mm", a, [weight.view,input,output], (weight.rows + 3) / 4, (count + 3) / 4, threads: 128)
        }
    }
    private func project(_ enc: MTLComputeCommandEncoder, _ weight: BonsaiWeight, _ input: BonsaiBuffer,
                         _ output: BonsaiBuffer, count: Int, grouped: Bool = false) throws {
        var input = input
        if weight.signs != nil {
            try transform(enc, weight, input, s(.rotated), count: count, grouped: grouped); input = s(.rotated)
        }
        try projectPrepared(enc, weight, input, output, count: count)
    }
    private func siblings(_ enc: MTLComputeCommandEncoder, _ projections: [(BonsaiWeight, Slot)],
                          _ input: BonsaiBuffer, count: Int) throws {
        var signBuffer: MTLBuffer?, columns = 0
        for (weight, slot) in projections {
            var prepared = input
            if let signs = weight.signs {
                if signBuffer !== signs.buffer || columns != weight.columns {
                    try transform(enc, weight, input, s(.rotated), count: count)
                    signBuffer = signs.buffer; columns = weight.columns
                }
                prepared = s(.rotated)
            }
            try projectPrepared(enc, weight, prepared, s(slot), count: count)
        }
    }
    private func gateUp(_ enc: MTLComputeCommandEncoder, _ layer: Int, count: Int) throws {
        let gate = w(layer,"ffn_gate.weight"), up = w(layer,"ffn_up.weight")
        let ptq = gate.type == 143
        let compatible = (ptq || gate.type == 142) && up.type == gate.type && gate.rows >= 4096 &&
            gate.columns == up.columns && gate.rows == up.rows && gate.signs?.buffer === up.signs?.buffer
        guard compatible, count == 1 || count >= (ptq ? 4 : 16) else {
            try siblings(enc, [(gate,.gate),(up,.up)], s(.norm), count: count)
            try element(enc, s(.gate), s(.up), s(.mid), count: count * gate.rows, mode: 1)
            return
        }
        var input = s(.norm)
        if gate.signs != nil { try transform(enc, gate, input, s(.rotated), count: count); input = s(.rotated) }
        let a = BonsaiArgs(n: count, rows: gate.rows, cols: gate.columns, type: gate.type, rowBytes: gate.spec.rowBytes)
        let buffers = [gate.view,up.view,input,s(.mid)]
        if count == 1 {
            let name = ptq ? "ptq_gate_up" : (gate.rows % 8 == 0 ? "pq2_gate_up_full" : "pq2_gate_up")
            try dispatch(enc, name, a, buffers, (gate.rows + 7) / 8, threads: ptq ? 128 : 64)
        } else if ptq {
            let tile = count >= 8 ? 8 : 4
            try dispatch(enc, tile == 8 ? "ptq_gate_up_batch_8" : "ptq_gate_up_batch", a, buffers, (gate.rows + 3) / 4, (count + tile - 1) / tile, threads: 128)
        } else {
            try dispatch(enc, "mm_pq2_gate_up_tiled", a, buffers, (gate.rows + 31) / 32, (count + 31) / 32, threads: 128)
        }
    }
    private func alphaBeta(_ enc: MTLComputeCommandEncoder, _ layer: Int, count: Int) throws {
        let a = w(layer,"ssm_alpha.weight"), b = w(layer,"ssm_beta.weight")
        if a.type == 30 && b.type == 30 && a.signs == nil && b.signs == nil && a.rows == b.rows && a.columns == b.columns && (count == 1 || count >= 4) {
            let args = BonsaiArgs(n: count, rows: a.rows, cols: a.columns, type: 30, rowBytes: a.spec.rowBytes)
            try dispatch(enc, count == 1 ? "bf16_pair" : "bf16_pair_batch", args,
                         [a.view,b.view,s(.norm),s(.alpha),s(.beta)], (a.rows + 3) / 4, (count + 3) / 4, threads: 128)
        } else { try siblings(enc, [(a,.alpha),(b,.beta)], s(.norm), count: count) }
    }

    private func gdn(_ enc: MTLComputeCommandEncoder, _ layer: Int, count: Int) throws {
        try siblings(enc, [(w(layer,"attn_qkv.weight"),.qkv),(w(layer,"attn_gate.weight"),.z)], s(.norm), count: count)
        try alphaBeta(enc, layer, count: count)
        let conv = w(layer,"ssm_conv1d.weight"), a = w(layer,"ssm_a"), dt = w(layer,"ssm_dt.bias")
        let history = histories[layer]!, state = states[layer]!
        let scan = BonsaiArgs(n: count, type: a.type, heads: 48, kvheads: 16, dim: 128, mode: dt.type)
        if count >= 16 {
            try vector(enc, "conv_batch", BonsaiArgs(n: count, cols: 10240, type: conv.type, width: 4), [s(.qkv),conv.view,history,s(.conv)], 10240)
            try dispatch(enc, "l2_batch", BonsaiArgs(n: count, cols: 128, heads: 32, width: 10240, eps: configuration.epsilon), [s(.conv)], 32, count)
            try dispatch(enc, "gdn_batch_128", scan, [s(.conv),s(.alpha),s(.beta),a.view,dt.view,state,s(.gdn)], 48, threads: 128)
        } else {
            for row in 0..<count {
                let c = s(.conv).row(row,10240)
                try vector(enc, "conv", BonsaiArgs(n: 10240, type: conv.type, width: 4), [s(.qkv).row(row,10240),conv.view,history,c], 10240)
                try norm(enc, nil, c, c, heads: 32, width: 128, stride: 128)
                try dispatch(enc, "gdn_128", scan, [c,s(.alpha).row(row,48),s(.beta).row(row,48),a.view,dt.view,state,s(.gdn).row(row,6144)], 48, threads: 128)
            }
        }
        try norm(enc, w(layer,"ssm_norm.weight"), s(.gdn), s(.attention), heads: count * 48, width: 128, stride: 128)
        try element(enc, s(.z), s(.attention), s(.attention), count: count * 6144, mode: 1)
        try project(enc, w(layer,"ssm_out.weight"), s(.attention), s(.result), count: count, grouped: configuration.groupedValueHeads)
    }

    private func rope(_ enc: MTLComputeCommandEncoder, _ args: BonsaiArgs, _ input: BonsaiBuffer, count: Int, positions: [Int32]?) throws {
        var args = args; args.n = UInt32(count)
        if let positions {
            positions.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 2) }
        }
        try dispatch(enc, positions == nil ? "rope" : "mrope", args, [input], (Int(args.heads * args.rot / 2) + 255) / 256, count)
    }
    private func attention(_ enc: MTLComputeCommandEncoder, _ layer: Int, count: Int, positions: [Int32]?) throws {
        try siblings(enc, [(w(layer,"attn_q.weight"),.qkv),(w(layer,"attn_k.weight"),.k),(w(layer,"attn_v.weight"),.v)], s(.norm), count: count)
        try norm(enc, w(layer,"attn_q_norm.weight"), s(.qkv), s(.q), heads: count * 24, width: 256, stride: 512)
        try norm(enc, w(layer,"attn_k_norm.weight"), s(.k), s(.k), heads: count * 4, width: 256, stride: 256)
        let a = BonsaiArgs(n: 1024, pos: position, heads: 24, kvheads: 4, dim: 256, rot: 64, width: contextCapacity, base: configuration.rotaryBase)
        try rope(enc, a, s(.q), count: count, positions: positions)
        var ka = a; ka.heads = 4
        try rope(enc, ka, s(.k), count: count, positions: positions)
        try dispatch(enc, "cache", a, [s(.k),s(.v),keys[layer]!,values[layer]!], 4, count)
        if count == 1 {
            try dispatch(enc, "scores", a, [s(.q),keys[layer]!,s(.scores)], (position + 256) / 256, 24)
            try dispatch(enc, "softmax", a, [s(.scores)], 24)
            try vector(enc, "attention", a, [s(.scores),values[layer]!,s(.attention)], 6144)
        } else {
            for row in stride(from: 0, to: count, by: attentionBatchCapacity) {
                var b = a; b.pos += UInt32(row); b.n = UInt32(min(count - row, attentionBatchCapacity))
                try dispatch(enc, "scores_batch", b, [s(.q).row(row,6144),keys[layer]!,s(.scores)], (Int(b.pos + b.n) + 255) / 256, 24, Int(b.n))
                try dispatch(enc, "softmax_batch", b, [s(.scores)], 24, Int(b.n))
                try dispatch(enc, "attention_batch", b, [s(.scores),values[layer]!,s(.attention).row(row,6144)], 24, Int(b.n))
            }
        }
        try element(enc, s(.attention), s(.qkv), s(.attention), count: count * 6144, mode: 2, dim: 256)
        try project(enc, w(layer,"attn_output.weight"), s(.attention), s(.result), count: count)
    }

    private func chunk(tokens: [Int], embeddings: [[Float]?]?, positions: [Int32]?, publishLogits: Bool,
                       cancelled: @Sendable () -> Bool) throws {
        guard let command = gpu.queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw SwiftModelDecoderError.gpu("Impossibile creare il command buffer Bonsai")
        }
        command.label = "Bonsai \(position)..<\(position + tokens.count)"
        var ended = false
        defer { if !ended { encoder.endEncoding() } }
        let count = tokens.count, embedding = w("token_embd.weight")
        for row in 0..<count {
            if let vector = embeddings?[row] {
                vector.withUnsafeBytes { source in
                    s(.x).buffer.contents().advanced(by: row * 5120 * 4).copyMemory(from: source.baseAddress!, byteCount: source.count)
                }
            } else {
                let args = BonsaiArgs(cols: 5120, type: embedding.type, rowBytes: embedding.spec.rowBytes, pos: tokens[row])
                try self.vector(encoder, "embed", args, [embedding.view,s(.norm).row(row,5120)], 5120)
                try transform(encoder, embedding, s(.norm).row(row,5120), s(.x).row(row,5120), count: 1, inverse: true)
            }
        }
        for layer in 0..<64 {
            if cancelled() { throw SwiftModelDecoderError.cancelled }
            try norm(encoder, w(layer,"attn_norm.weight"), s(.x), s(.norm), heads: count, width: 5120, stride: 5120)
            if (layer + 1) % 4 == 0 { try attention(encoder, layer, count: count, positions: positions) }
            else { try gdn(encoder, layer, count: count) }
            try element(encoder, s(.x), s(.result), s(.x), count: count * 5120, mode: 0)
            try norm(encoder, w(layer,"post_attention_norm.weight"), s(.x), s(.norm), heads: count, width: 5120, stride: 5120)
            try gateUp(encoder, layer, count: count)
            try project(encoder, w(layer,"ffn_down.weight"), s(.mid), s(.result), count: count)
            try element(encoder, s(.x), s(.result), s(.x), count: count * 5120, mode: 0)
        }
        if publishLogits {
            try norm(encoder, w("output_norm.weight"), s(.x).row(count - 1,5120), s(.norm), heads: 1, width: 5120, stride: 5120)
            try project(encoder, w("output.weight"), s(.norm), s(.logits), count: 1)
        }
        encoder.endEncoding(); ended = true
        if cancelled() { throw SwiftModelDecoderError.cancelled }
        command.commit(); command.waitUntilCompleted()
        guard command.status == .completed else { throw SwiftModelDecoderError.gpu(command.error?.localizedDescription ?? "Bonsai command buffer fallito") }
        // A submitted recurrent scan must finish before returning cancellation;
        // callers can then reset/replay without racing an in-flight GPU write.
        if cancelled() { throw SwiftModelDecoderError.cancelled }
        position += count
    }
}
