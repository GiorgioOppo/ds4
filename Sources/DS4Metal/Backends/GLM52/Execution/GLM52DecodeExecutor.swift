import DS4Core
import Foundation
import Metal

// GPU composition of one GLM 5.2 DECODE step over the validated primitives,
// judged against GLM52DecodeCPUReference. Same validation-level split as the
// first-token executor: every heavy stage — the Q8_0 matvecs, both RoPE
// kernels, the KV-LoRA norm, the indexer key store, indexer scoring, the
// multi-block top-k and the indexed attention with per-row tail rotation —
// dispatches a validated GPU kernel; the cheap glue (RMSNorm over one row,
// residual adds, the tiny F32 router and indexer-proj matvecs) stays on the
// CPU oracle implementations so every divergence is attributable to a GPU
// kernel. The attention and indexer kernels are fixed at the v5_2 head
// geometry; the embedding width, Q-LoRA rank, FFN widths and top-k stay free
// so tests can shrink the fixture without touching kernel code. Host arrays
// move through shared buffers per dispatch — the persistent-GPUTensor decode
// graph comes later; this is its correctness baseline.

/// Decode attention weights as the GGUF stores them: Q8_0 projections, F32
/// norms. Row layouts match `GLM52DecodeAttentionWeightsF32`.
public struct GLM52QuantizedDecodeAttention: Sendable {
    public let attnNorm: [Float]
    public let qA: [UInt8]
    public let qANorm: [Float]
    public let qB: [UInt8]
    public let kvA: [UInt8]
    public let kvANorm: [Float]
    public let keyB: [UInt8]
    public let valueB: [UInt8]
    public let attnOutput: [UInt8]

    public init(attnNorm: [Float], qA: [UInt8], qANorm: [Float], qB: [UInt8],
                kvA: [UInt8], kvANorm: [Float], keyB: [UInt8],
                valueB: [UInt8], attnOutput: [UInt8]) {
        self.attnNorm = attnNorm
        self.qA = qA
        self.qANorm = qANorm
        self.qB = qB
        self.kvA = kvA
        self.kvANorm = kvANorm
        self.keyB = keyB
        self.valueB = valueB
        self.attnOutput = attnOutput
    }
}

/// Full-indexer layer weights: Q8_0 key/query projections, F32 LayerNorm
/// affine, F32 proj rows (the GGUF stores indexer.proj in F32 and its 32-row
/// matvec stays on CPU beside the router).
public struct GLM52QuantizedDecodeIndexer: Sendable {
    public let key: [UInt8]
    public let keyNorm: [Float]
    public let keyNormBias: [Float]
    public let queryB: [UInt8]
    public let proj: [Float]

    public init(key: [UInt8], keyNorm: [Float], keyNormBias: [Float],
                queryB: [UInt8], proj: [Float]) {
        self.key = key
        self.keyNorm = keyNorm
        self.keyNormBias = keyNormBias
        self.queryB = queryB
        self.proj = proj
    }
}

/// Host-side F16 decode caches of ONE layer for the validation executor.
public struct GLM52DecodeCaches: Sendable, Equatable {
    /// Interleaved compact rows `[position][512 + 64]` exactly as the
    /// attention kernel reads them: normalized KV-LoRA prefix, RAW tail.
    public var compactBits: [UInt16]
    /// Indexer key rows `[position][128]` — full-indexer layers only.
    public var indexerKeyBits: [UInt16]

    public init() {
        compactBits = []
        indexerKeyBits = []
    }
}

extension MetalRuntime {
    /// One decode attention step at `position`: appends this token's cache
    /// rows (before selection and attention, upstream's fused-store order),
    /// resolves the selection — GPU score+top-k on full-indexer layers when
    /// the visible range exceeds top-k, the causal fill range otherwise,
    /// verbatim reuse on IndexShare layers — and returns the attn_output
    /// projection. On error the caches are unspecified.
    public func glm52DecodeAttention(
        geometry: GLM52DecodeGeometry,
        input: [Float],
        attention: GLM52QuantizedDecodeAttention,
        indexer: GLM52QuantizedDecodeIndexer?,
        reusedSelection: [UInt32]?,
        caches: inout GLM52DecodeCaches,
        position: Int) throws -> (output: [Float], selection: [UInt32]) {
        try glm52ValidateDecode(
            geometry: geometry, input: input, attention: attention,
            indexer: indexer, reusedSelection: reusedSelection,
            caches: caches, position: position)
        let g = geometry
        let layer = g.layer
        let q8 = GLM52TensorSchema.q8_0

        // 1. CPU pre-norm, GPU LoRA down-projections, CPU q_a_norm.
        let normed = try GLM52FFNCPUReference.rmsNorm(
            input, weight: attention.attnNorm)
        let qRank = try glm52MoEDown(
            mid: normed, downRows: attention.qA,
            weightType: q8, outputWidth: g.qLoraRank)
        let kvRaw = try glm52MoEDown(
            mid: normed, downRows: attention.kvA,
            weightType: q8, outputWidth: layer.kvRawWidth)
        let qRankNorm = try GLM52FFNCPUReference.rmsNorm(
            qRank, weight: attention.qANorm)

        // 2. Compact-cache write BEFORE selection/attention: the GPU KV-LoRA
        //    norm emits the cache-ready 576 row (normalized prefix, raw
        //    tail), stored as F16.
        let cacheReady = try glm52NormalizeKVLoRA(
            rawRows: kvRaw, weight: attention.kvANorm)
        caches.compactBits.append(contentsOf: cacheReady.map(Half.bits))

        // 3. GPU query up-projection and tail RoPE before qk_lowrank.
        var query = try glm52MoEDown(
            mid: qRankNorm, downRows: attention.qB,
            weightType: q8, outputWidth: g.queryWidth)
        query = try glm52RopeTail(
            values: query, headCount: layer.headCount,
            headDimension: g.qkDimension,
            rotationDimension: layer.ropeDimension, position: position)

        // 4. Indexer key store and selection.
        let visible = position + 1
        let selection: [UInt32]
        if let indexer {
            let rawKey = try glm52MoEDown(
                mid: input, downRows: indexer.key,
                weightType: q8, outputWidth: g.indexerHeadDimension)
            caches.indexerKeyBits = try glm52StoreIndexerKeys(
                rawKeys: rawKey, weight: indexer.keyNorm,
                bias: indexer.keyNormBias, pos0: position,
                cacheCapacity: visible,
                initialCacheBits: caches.indexerKeyBits + [UInt16](
                    repeating: 0, count: g.indexerHeadDimension))
            if visible <= g.indexerTopK {
                // Upstream fill_selected_range: the whole causal range,
                // current token included.
                selection = (0..<visible).map(UInt32.init)
            } else {
                var indexerQuery = try glm52MoEDown(
                    mid: qRankNorm, downRows: indexer.queryB,
                    weightType: q8, outputWidth: g.indexerQueryWidth)
                indexerQuery = try glm52RopePrefix(
                    values: indexerQuery, headCount: g.indexerHeadCount,
                    headDimension: g.indexerHeadDimension,
                    rotationDimension: g.indexerRotationDimension,
                    position: position)
                let headWeights = try GLM52FFNCPUReference.matvec(
                    rows: indexer.proj, input: input,
                    rowCount: g.indexerHeadCount)
                let scored = try glm52IndexerScores(
                    queries: indexerQuery, headWeights: headWeights,
                    keyCacheBits: caches.indexerKeyBits, pos0: position)
                selection = try glm52IndexerTopK(
                    scores: scored.scores, rowCount: visible,
                    tokenCount: 1, topK: g.indexerTopK)
            }
        } else {
            // Validated above: IndexShare layers carry the reused selection.
            selection = reusedSelection ?? []
        }

        // 5. Indexed attention over raw-tail cache rows (per-row rotation in
        //    the kernel), then the GPU output projection.
        let qLow = try glm52QKLowRankQ8(query: query, keyBQ8: attention.keyB)
        let attnLora = try glm52AttentionIndexed(
            qLow: qLow, query: query, cacheBits: caches.compactBits,
            selection: selection, rotateTailByRowPosition: true)
        let heads = try glm52ValueProjectQ8(
            attnLora: attnLora, valueBQ8: attention.valueB)
        let output = try glm52MoEDown(
            mid: heads, downRows: attention.attnOutput,
            weightType: q8, outputWidth: layer.embeddingWidth)
        return (output, selection)
    }

    /// One full decode layer on GPU: attention residual, then the shared
    /// residual FFN stage. Returns the routing on sparse layers and the
    /// selection later IndexShare layers reuse.
    public func glm52DecodeLayer(
        geometry: GLM52DecodeGeometry,
        input: [Float],
        attention: GLM52QuantizedDecodeAttention,
        indexer: GLM52QuantizedDecodeIndexer?,
        reusedSelection: [UInt32]?,
        ffnNorm: [Float],
        ffn: GLM52QuantizedLayerFFN,
        caches: inout GLM52DecodeCaches,
        position: Int) throws
        -> (output: [Float], routing: GLM52RouterOutput?,
            selection: [UInt32]) {
        let attn = try glm52DecodeAttention(
            geometry: geometry, input: input, attention: attention,
            indexer: indexer, reusedSelection: reusedSelection,
            caches: &caches, position: position)
        let afterAttn = (0..<input.count).map { input[$0] + attn.output[$0] }
        let ffnResult = try glm52LayerFFNStage(
            geometry: geometry.layer, afterAttention: afterAttn,
            ffnNorm: ffnNorm, ffn: ffn)
        return (ffnResult.output, ffnResult.routing, attn.selection)
    }

    // MARK: - Validation

    private func glm52ValidateDecode(
        geometry: GLM52DecodeGeometry,
        input: [Float],
        attention: GLM52QuantizedDecodeAttention,
        indexer: GLM52QuantizedDecodeIndexer?,
        reusedSelection: [UInt32]?,
        caches: GLM52DecodeCaches,
        position: Int) throws {
        try glm52ValidateDecodeWeights(
            geometry: geometry, attention: attention, indexer: indexer)
        let g = geometry
        let layer = g.layer
        guard position >= 0 else {
            throw MetalError.unsupported(
                "GLM 5.2 decode position must be non-negative")
        }
        try glm52RequireCount(input.count, layer.embeddingWidth, "input")

        let cacheRowWidth = layer.kvLoraRank + layer.ropeDimension
        try glm52RequireCount(caches.compactBits.count,
                              position * cacheRowWidth, "compactBits")
        if indexer != nil {
            guard reusedSelection == nil else {
                throw MetalError.unsupported(
                    "full-indexer decode layer must not receive a reused "
                    + "selection")
            }
            try glm52RequireCount(caches.indexerKeyBits.count,
                                  position * g.indexerHeadDimension,
                                  "indexerKeyBits")
        } else {
            guard reusedSelection != nil else {
                throw MetalError.unsupported(
                    "IndexShare decode layer requires the preceding "
                    + "full-indexer selection")
            }
            guard caches.indexerKeyBits.isEmpty else {
                throw MetalError.unsupported(
                    "IndexShare decode layer must not own indexer keys")
            }
        }
    }

    /// Geometry and weight-count contract shared by the per-dispatch decode
    /// executor and the resident decode graph.
    func glm52ValidateDecodeWeights(
        geometry: GLM52DecodeGeometry,
        attention: GLM52QuantizedDecodeAttention,
        indexer: GLM52QuantizedDecodeIndexer?) throws {
        let g = geometry
        let layer = g.layer
        let fixed = GLM52AttentionGeometry.v5_2
        guard g.isValid,
              layer.headCount == fixed.headCount,
              g.nopeDimension == fixed.nopeDimension,
              layer.ropeDimension == fixed.ropeDimension,
              layer.kvLoraRank == fixed.kvLoraRank,
              layer.valueDimension == fixed.valueDimension,
              g.indexerHeadCount == GLM52IndexerScoresReference.headCount,
              g.indexerHeadDimension
                  == GLM52IndexerScoresReference.headDimension,
              g.indexerRotationDimension
                  == GLM52IndexerKeyStoreReference.rotationDimension,
              g.indexerTopK <= Int(GLM52Shape.v5_2.nIndexerTopK),
              layer.embeddingWidth.isMultiple(of: 32),
              g.qLoraRank.isMultiple(of: 32) else {
            throw MetalError.unsupported(
                "GLM 5.2 decode executor requires the fixed v5_2 attention "
                + "and indexer geometry (embedding, Q-LoRA, FFN widths and "
                + "top-k stay free)")
        }

        let embedBytes = Self.glm52Q8RowBytes(layer.embeddingWidth)
        let qLoraBytes = Self.glm52Q8RowBytes(g.qLoraRank)
        let headsWidth = layer.headCount * layer.valueDimension
        try glm52RequireCount(attention.attnNorm.count, layer.embeddingWidth,
                              "attnNorm")
        try glm52RequireCount(attention.qA.count, g.qLoraRank * embedBytes,
                              "qA")
        try glm52RequireCount(attention.qANorm.count, g.qLoraRank, "qANorm")
        try glm52RequireCount(attention.qB.count, g.queryWidth * qLoraBytes,
                              "qB")
        try glm52RequireCount(attention.kvA.count,
                              layer.kvRawWidth * embedBytes, "kvA")
        try glm52RequireCount(attention.kvANorm.count, layer.kvLoraRank,
                              "kvANorm")
        try glm52RequireCount(
            attention.keyB.count,
            layer.headCount * layer.kvLoraRank
                * Self.glm52Q8RowBytes(g.nopeDimension),
            "keyB")
        try glm52RequireCount(
            attention.valueB.count,
            headsWidth * Self.glm52Q8RowBytes(layer.kvLoraRank), "valueB")
        try glm52RequireCount(attention.attnOutput.count,
                              layer.embeddingWidth
                                  * Self.glm52Q8RowBytes(headsWidth),
                              "attnOutput")
        if let indexer {
            try glm52RequireCount(indexer.key.count,
                                  g.indexerHeadDimension * embedBytes,
                                  "indexer.key")
            try glm52RequireCount(indexer.keyNorm.count,
                                  g.indexerHeadDimension, "indexer.keyNorm")
            try glm52RequireCount(indexer.keyNormBias.count,
                                  g.indexerHeadDimension,
                                  "indexer.keyNormBias")
            try glm52RequireCount(indexer.queryB.count,
                                  g.indexerQueryWidth * qLoraBytes,
                                  "indexer.queryB")
            try glm52RequireCount(indexer.proj.count,
                                  g.indexerHeadCount * layer.embeddingWidth,
                                  "indexer.proj")
        }
    }

    private func glm52RequireCount(_ got: Int, _ expected: Int,
                                   _ component: String) throws {
        guard got == expected else {
            throw MetalError.unsupported(
                "GLM 5.2 decode \(component) has \(got) elements, "
                + "expected \(expected)")
        }
    }
}
