import DS4Core
import Foundation

// Per-expert streaming for the resident decode graph: the planner and the
// LRU slot cache serve one packed gate|up|down record per selected expert,
// sliced into the GLM52QuantizedExpert the sparse FFN stage consumes. The
// slot cache makes repeated selections byte-identical hits across tokens.

public enum GLM52StreamedExpertProviderError: Error, Sendable, Equatable,
    CustomStringConvertible {
    case unsupportedExpertType(layer: Int, type: UInt32)

    public var description: String {
        switch self {
        case .unsupportedExpertType(let layer, let type):
            return "blk.\(layer): routed expert type \(GGUF.typeName(type)) "
                + "has no validated GLM kernel yet"
        }
    }
}

public final class GLM52StreamedExpertProvider {
    /// Types the validated MoE kernels dispatch today — including IQ2_XXS,
    /// the routed type of the published GGUF. Anything else is refused at
    /// load: silently wrong output is worse than an error.
    public static let supportedTypes: Set<UInt32> = [
        GLM52TensorSchema.q8_0, GLM52TensorSchema.q2_K,
        GLM52TensorSchema.q4_K, GLM52TensorSchema.q5_K,
        GLM52TensorSchema.q6_K, GLM52TensorSchema.iq2_XXS,
    ]

    public let layer: Int
    private let planner: GLM52ExpertStreamPlanner
    private let reader: GLM52PayloadReader
    private let cache: GLM52ExpertSlotCache
    public let gateUpType: UInt32
    public let downType: UInt32

    public var stats: GLM52ExpertSlotCacheStats { cache.stats }

    /// When `bundleDirectory` holds a valid per-layer bundle, experts are
    /// served from it in ONE contiguous read; a bundle that exists but does
    /// not match this GGUF throws instead of being silently ignored.
    public convenience init(reader: GLM52PayloadReader,
                            weightMap: GLM52WeightMap,
                            layer: Int,
                            slotCount: Int = 16,
                            bundleDirectory: String? = nil) throws {
        let weights = try weightMap.routedExperts(layer: layer)
        let bundle = try bundleDirectory.flatMap {
            try GLM52ExpertBundle.open(
                directory: $0, layer: layer, weights: weights,
                expertCount: Int(weightMap.configuration.shape.nExpert),
                sourceFileSize: reader.fileSize)
        }
        try self.init(reader: reader, layer: layer, weights: weights,
                      slotCount: slotCount, bundle: bundle)
    }

    public init(reader: GLM52PayloadReader,
                layer: Int,
                weights: GLM52RoutedExpertWeights,
                slotCount: Int = 16,
                bundle: GLM52ExpertBundle? = nil) throws {
        guard Self.supportedTypes.contains(weights.gate.type),
              Self.supportedTypes.contains(weights.down.type),
              weights.gate.type == weights.up.type else {
            throw GLM52StreamedExpertProviderError.unsupportedExpertType(
                layer: layer,
                type: Self.supportedTypes.contains(weights.gate.type)
                    ? weights.down.type : weights.gate.type)
        }
        self.layer = layer
        gateUpType = weights.gate.type
        downType = weights.down.type
        self.bundle = bundle
        self.reader = reader
        planner = try GLM52ExpertStreamPlanner(layer: layer, weights: weights)
        let probe = try planner.plan(selectedExperts: [0])
        let layout = try reader.packedLayout(of: probe)
        cache = try GLM52ExpertSlotCache(
            reader: reader, slotCount: slotCount,
            slotBytes: layout.recordBytes)
    }

    /// Packed gate|up|down record size — the unit the staging arena is
    /// sized in.
    public var recordBytes: Int {
        Int(planner.gateLayout.expertBytes + planner.upLayout.expertBytes
            + planner.downLayout.expertBytes)
    }

    public var gateRecordBytes: Int { Int(planner.gateLayout.expertBytes) }
    public var upRecordBytes: Int { Int(planner.upLayout.expertBytes) }
    public var downRecordBytes: Int { Int(planner.downLayout.expertBytes) }

    /// Whether the packed record can be bound with Metal buffer offsets:
    /// every offset in the record (and the record stride) must stay 4-byte
    /// aligned. True for the schema's quantizations at GLM shapes; a foreign
    /// layout falls back to the copying per-expert path instead of binding
    /// misaligned.
    public var bindableRecordLayout: Bool {
        planner.gateLayout.expertBytes % 4 == 0
            && planner.upLayout.expertBytes % 4 == 0
            && planner.downLayout.expertBytes % 4 == 0
    }

    /// Per-layer contiguous bundle; nil serves from the GGUF's three
    /// scattered tensors through the LRU slot cache.
    private let bundle: GLM52ExpertBundle?

    private let lock = NSLock()

    /// One expert's packed gate|up|down record straight into `destination`
    /// — the arena's fill primitive, called CONCURRENTLY for the records of
    /// a selection. Bundle: one contiguous pread; GGUF: one single-expert
    /// plan with the reader's concurrent scattered reads. No lock on either
    /// path — both are stateless pread into disjoint destinations.
    /// `destination` must hold exactly `recordBytes`.
    public func readRecord(_ id: UInt32,
                           into destination: UnsafeMutableRawBufferPointer)
        throws {
        if let bundle {
            guard bundle.recordBytes == recordBytes else {
                throw GLM52PayloadReaderError.destinationSizeMismatch(
                    expected: UInt64(recordBytes),
                    got: bundle.recordBytes)
            }
            try bundle.read(id, into: destination)
            return
        }
        let plan = try planner.plan(selectedExperts: [id])
        _ = try reader.read(plan: plan, into: destination, concurrent: true)
    }

    /// Speculative warm-up of the slot cache off-thread — the DeepSeek
    /// expert-lookahead analog: consecutive tokens reselect experts often,
    /// so the previous token's selection is a cheap bet. Errors are
    /// swallowed on purpose (a failed prefetch just means a cold read later).
    ///
    /// Bundle path: a NO-OP on purpose. There the only retention is the OS
    /// page cache, and at ~20 GiB of streaming per token it is churned long
    /// before the layer comes around again — the speculative reads were pure
    /// extra SSD traffic competing with the real ones.
    public func prefetch(_ ids: [UInt32]) {
        guard bundle == nil else { return }
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self else { return }
            for id in ids { _ = try? self.expert(id) }
        }
    }

    /// One expert's quantized record, from the slot cache or freshly read.
    /// Serialized: the LRU cache is not concurrency-safe and the prefetch
    /// path races the decode thread by design.
    public func expert(_ id: UInt32) throws -> GLM52QuantizedExpert {
        // Bundle path: one contiguous bounded read, thread-safe by
        // construction (stateless pread) — no lock and no LRU needed; the
        // page cache does the reuse.
        if let bundle { return try bundle.expert(id) }
        lock.lock()
        defer { lock.unlock() }
        let plan = try planner.plan(selectedExperts: [id])
        return try cache.withRecords(plan: plan) { records, layout in
            let record = records[0]
            func slice(_ offset: Int, _ count: Int) -> [UInt8] {
                Array(record[offset..<offset + count])
            }
            return GLM52QuantizedExpert(
                gateUpType: gateUpType,
                downType: downType,
                gate: slice(layout.gateOffset, layout.gateBytes),
                up: slice(layout.upOffset, layout.upBytes),
                down: slice(layout.downOffset, layout.downBytes))
        }
    }
}

/// Pure greedy decoding glue, separated from the Metal engine so the loop is
/// testable without a device: argmax with the deterministic lowest-index tie
/// rule, and the generation loop over an injected forward step.
public enum GLM52GreedyDecoding {
    /// Highest logit; exact ties prefer the lower token id.
    public static func argmax(_ logits: [Float]) -> Int32? {
        guard !logits.isEmpty else { return nil }
        var best = 0
        for i in 1..<logits.count where logits[i] > logits[best] {
            best = i
        }
        return Int32(best)
    }

    /// Greedy loop: sample from `logitsAfterPrompt`, then feed each sampled
    /// token through `step` for the next logits. Stops at an end token or
    /// after `maxNewTokens`. Returns the sampled tokens (end token included).
    public static func generate(
        logitsAfterPrompt: [Float],
        maxNewTokens: Int,
        endTokens: Set<Int32>,
        step: (Int32) throws -> [Float]) throws -> [Int32] {
        var generated: [Int32] = []
        var logits = logitsAfterPrompt
        while generated.count < maxNewTokens {
            guard let token = argmax(logits) else { break }
            generated.append(token)
            if endTokens.contains(token) { break }
            if generated.count == maxNewTokens { break }
            logits = try step(token)
        }
        return generated
    }
}
