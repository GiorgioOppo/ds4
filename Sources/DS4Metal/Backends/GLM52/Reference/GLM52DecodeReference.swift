import Foundation
import DS4Core

// CPU oracle of one GLM 5.2 DECODE step — the port of upstream's
// glm_graph_forward_token wiring on the indexed-attention path, composed from
// the already-validated primitive oracles. The binding order of operations:
//
//   1. attn_norm = rmsNorm(cur); q_rank and kv_raw are both projected from it.
//   2. The caches are written BEFORE selection and attention: the compact row
//      stores rmsNorm(kv_raw[0..<512])·w plus the RAW 64-wide K-RoPE tail (no
//      norm, no rotation), and — on full-indexer layers only — the indexer key
//      row stores centered LayerNorm (eps 1e-6, weight+bias) of the key
//      projection of the RAW residual, prefix-rotated with the token position.
//   3. The query tail RoPE (last 64 of each 256-wide head) is applied before
//      qk_lowrank; the indexer query rotates its PREFIX instead.
//   4. visible = position + 1 — the current token always participates. When
//      visible fits the top-k the selection is the plain causal range
//      (upstream fill_selected_range); otherwise indexer scores over all
//      visible rows feed the top-k. IndexShare layers reuse the most recent
//      full-indexer selection verbatim (absolute row indices, no offset).
//   5. Attention consumes the raw cached tails and rotates each selected row
//      with the ROW's own absolute position at attention time.
//
// Cache stores round through IEEE binary16 because the real caches are F16 —
// that rounding is upstream numerics, not a GPU artifact.

public struct GLM52DecodeGeometry: Sendable, Equatable {
    public let layer: GLM52LayerGeometry
    /// attn_q_a output width (Q-LoRA rank, 2048 for v5_2).
    public let qLoraRank: Int
    /// Per-head QK width WITHOUT the RoPE tail (192 for v5_2).
    public let nopeDimension: Int
    public let indexerHeadCount: Int
    public let indexerHeadDimension: Int
    /// Rotated PREFIX width of indexer queries and keys (64 for v5_2).
    public let indexerRotationDimension: Int
    public let indexerTopK: Int

    public var qkDimension: Int { nopeDimension + layer.ropeDimension }
    public var queryWidth: Int { layer.headCount * qkDimension }
    public var indexerQueryWidth: Int { indexerHeadCount * indexerHeadDimension }
    /// Upstream indexer score scale: 1/sqrt(headDim · headCount), 1/64 for v5_2.
    public var indexerScale: Float {
        1 / Float(indexerHeadCount * indexerHeadDimension).squareRoot()
    }
    public var attentionGeometry: GLM52AttentionGeometry {
        GLM52AttentionGeometry(
            headCount: layer.headCount, nopeDimension: nopeDimension,
            ropeDimension: layer.ropeDimension, kvLoraRank: layer.kvLoraRank,
            valueDimension: layer.valueDimension)
    }

    /// GLM 5.2: Q-LoRA 2048, nope 192, indexer 32 heads of 128 with a
    /// 64-wide rotated prefix, top-2048.
    public static let v5_2 = GLM52DecodeGeometry(
        layer: .v5_2, qLoraRank: 2_048, nopeDimension: 192,
        indexerHeadCount: 32, indexerHeadDimension: 128,
        indexerRotationDimension: 64, indexerTopK: 2_048)

    public init(layer: GLM52LayerGeometry, qLoraRank: Int, nopeDimension: Int,
                indexerHeadCount: Int, indexerHeadDimension: Int,
                indexerRotationDimension: Int, indexerTopK: Int) {
        self.layer = layer
        self.qLoraRank = qLoraRank
        self.nopeDimension = nopeDimension
        self.indexerHeadCount = indexerHeadCount
        self.indexerHeadDimension = indexerHeadDimension
        self.indexerRotationDimension = indexerRotationDimension
        self.indexerTopK = indexerTopK
    }

    var isValid: Bool {
        qLoraRank > 0 && nopeDimension > 0 && indexerTopK > 0
            && indexerHeadCount > 0 && indexerHeadDimension > 0
            && indexerRotationDimension > 0
            && indexerRotationDimension.isMultiple(of: 2)
            && indexerRotationDimension <= indexerHeadDimension
            && layer.ropeDimension.isMultiple(of: 2)
    }
}

/// Decode attention weights, dequantized, row-major matvec layouts: `qA` is
/// `[qLora][embedding]`, `qB` is `[head*qk][qLora]`, `kvA` is
/// `[kvRaw][embedding]`, `keyB` is `[head][kvLora][nope]` (attn_k_b access
/// order), `valueB` is `[head*value][kvLora]`, `attnOutput` is
/// `[embedding][head*value]`.
public struct GLM52DecodeAttentionWeightsF32: Sendable {
    public let attnNorm: [Float]
    public let qA: [Float]
    public let qANorm: [Float]
    public let qB: [Float]
    public let kvA: [Float]
    public let kvANorm: [Float]
    public let keyB: [Float]
    public let valueB: [Float]
    public let attnOutput: [Float]

    public init(attnNorm: [Float], qA: [Float], qANorm: [Float], qB: [Float],
                kvA: [Float], kvANorm: [Float], keyB: [Float],
                valueB: [Float], attnOutput: [Float]) {
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

/// Indexer weights of a FULL-indexer layer: `key` is `[headDim][embedding]`
/// (indexer.attn_k — its input is the RAW residual, not a normalized
/// activation), `queryB` is `[idxHead*idxDim][qLora]` (input q_a_norm),
/// `proj` is `[idxHead][embedding]` (input the RAW residual; the 32 head
/// weights are used raw, no softmax). The k_norm pair is the centered
/// LayerNorm affine over the 128-wide key.
public struct GLM52DecodeIndexerWeightsF32: Sendable {
    public let key: [Float]
    public let keyNorm: [Float]
    public let keyNormBias: [Float]
    public let queryB: [Float]
    public let proj: [Float]

    public init(key: [Float], keyNorm: [Float], keyNormBias: [Float],
                queryB: [Float], proj: [Float]) {
        self.key = key
        self.keyNorm = keyNorm
        self.keyNormBias = keyNormBias
        self.queryB = queryB
        self.proj = proj
    }
}

/// Per-layer decode caches of the F32 oracle. Stored values are already
/// rounded through F16.
public struct GLM52DecodeCacheStateF32: Sendable, Equatable {
    /// `[position][kvLora + rope]`: normalized KV-LoRA prefix, RAW RoPE tail.
    public var compactRows: [Float]
    /// `[position][idxDim]` LayerNorm+prefix-RoPE'd keys — full-indexer
    /// layers only; IndexShare layers never store indexer keys.
    public var indexerKeys: [Float]

    public init() {
        compactRows = []
        indexerKeys = []
    }
}

public enum GLM52DecodeReferenceError: Error, Sendable, Equatable,
    CustomStringConvertible {
    case invalidGeometry
    case invalidDimensions(component: String, expected: Int, got: Int)
    case cachePositionMismatch(component: String, expected: Int, got: Int)
    case missingSharedSelection
    case unexpectedSharedSelection
    case indexerCacheOnSharedLayer

    public var description: String {
        switch self {
        case .invalidGeometry:
            return "invalid GLM 5.2 decode geometry"
        case .invalidDimensions(let component, let expected, let got):
            return "GLM decode \(component) has \(got) elements, expected \(expected)"
        case .cachePositionMismatch(let component, let expected, let got):
            return "GLM decode \(component) holds \(got) values, expected \(expected)"
        case .missingSharedSelection:
            return "IndexShare decode layer requires the preceding full-indexer selection"
        case .unexpectedSharedSelection:
            return "full-indexer decode layer must not receive a reused selection"
        case .indexerCacheOnSharedLayer:
            return "IndexShare decode layer must not own indexer keys"
        }
    }
}

public enum GLM52DecodeCPUReference {
    /// One decode attention step at `position`. Appends this token's rows to
    /// the caches (before selection and attention, like upstream's fused
    /// store), resolves the selection — computed on full-indexer layers
    /// (`indexer` present), reused verbatim on IndexShare layers — and
    /// returns the attn_output projection plus the selection that later
    /// IndexShare layers must reuse. On error the caches are unspecified.
    public static func decodeAttention(
        geometry: GLM52DecodeGeometry,
        input: [Float],
        weights: GLM52DecodeAttentionWeightsF32,
        indexer: GLM52DecodeIndexerWeightsF32?,
        reusedSelection: [Int]?,
        cache: inout GLM52DecodeCacheStateF32,
        position: Int) throws -> (output: [Float], selection: [Int]) {
        try validate(geometry: geometry, input: input, weights: weights,
                     indexer: indexer, reusedSelection: reusedSelection,
                     cache: cache, position: position)
        let g = geometry
        let layer = g.layer

        // 1. Pre-norm and both LoRA down-projections.
        let attnNorm = try GLM52FFNCPUReference.rmsNorm(
            input, weight: weights.attnNorm)
        let qRank = try GLM52FFNCPUReference.matvec(
            rows: weights.qA, input: attnNorm, rowCount: g.qLoraRank)
        let kvRaw = try GLM52FFNCPUReference.matvec(
            rows: weights.kvA, input: attnNorm, rowCount: layer.kvRawWidth)
        let qRankNorm = try GLM52FFNCPUReference.rmsNorm(
            qRank, weight: weights.qANorm)

        // 2. Compact-cache write BEFORE selection/attention: normalized
        //    KV-LoRA prefix, RAW tail, both rounded to F16.
        let kvNorm = try GLM52FFNCPUReference.rmsNorm(
            Array(kvRaw[0..<layer.kvLoraRank]), weight: weights.kvANorm)
        for value in kvNorm {
            cache.compactRows.append(Half.float(Half.bits(value)))
        }
        for i in layer.kvLoraRank..<layer.kvRawWidth {
            cache.compactRows.append(Half.float(Half.bits(kvRaw[i])))
        }

        // 3. Query up-projection, then tail RoPE before qk_lowrank.
        var query = try GLM52FFNCPUReference.matvec(
            rows: weights.qB, input: qRankNorm, rowCount: g.queryWidth)
        query = try GLM52RopeTailReference.rotate(
            values: query, headCount: layer.headCount,
            headDimension: g.qkDimension,
            rotationDimension: layer.ropeDimension, position: position)

        // 4. Indexer key store and selection.
        let visible = position + 1
        let selection: [Int]
        if let indexer {
            let rawKey = try GLM52FFNCPUReference.matvec(
                rows: indexer.key, input: input,
                rowCount: g.indexerHeadDimension)
            cache.indexerKeys.append(contentsOf: try storedIndexerKey(
                rawKey, indexer: indexer, geometry: g, position: position))
            if visible <= g.indexerTopK {
                // Upstream fill_selected_range: the whole causal range,
                // current token included.
                selection = Array(0..<visible)
            } else {
                var indexerQuery = try GLM52FFNCPUReference.matvec(
                    rows: indexer.queryB, input: qRankNorm,
                    rowCount: g.indexerQueryWidth)
                indexerQuery = try GLM52RopeTailReference.rotatePrefix(
                    values: indexerQuery, headCount: g.indexerHeadCount,
                    headDimension: g.indexerHeadDimension,
                    rotationDimension: g.indexerRotationDimension,
                    position: position)
                let headWeights = try GLM52FFNCPUReference.matvec(
                    rows: indexer.proj, input: input,
                    rowCount: g.indexerHeadCount)
                let scores = indexerScores(
                    query: indexerQuery, headWeights: headWeights,
                    geometry: g, keys: cache.indexerKeys, visible: visible)
                selection = try GLM52IndexerCPUReference.causalTopK(
                    scores: scores, queryPosition: position,
                    topK: g.indexerTopK)
            }
        } else {
            guard let reusedSelection else {
                throw GLM52DecodeReferenceError.missingSharedSelection
            }
            selection = reusedSelection
        }

        // 5. Absorbed attention over raw-tail cache rows (per-row rotation),
        //    then the output projection.
        let heads = try GLM52AttentionCPUReference.absorbed(
            geometry: g.attentionGeometry, query: query, keyB: weights.keyB,
            valueB: weights.valueB, cache: cache.compactRows,
            selection: selection, rotateTailByRowPosition: true)
        let output = try GLM52FFNCPUReference.matvec(
            rows: weights.attnOutput, input: heads,
            rowCount: layer.embeddingWidth)
        return (output, selection)
    }

    /// One full decode layer: pre-norm attention residual, then the shared
    /// residual FFN stage. Returns the routing on sparse layers and the
    /// selection later IndexShare layers reuse.
    public static func decodeLayer(
        geometry: GLM52DecodeGeometry,
        input: [Float],
        attention: GLM52DecodeAttentionWeightsF32,
        indexer: GLM52DecodeIndexerWeightsF32?,
        reusedSelection: [Int]?,
        ffnNorm: [Float],
        ffn: GLM52LayerFFNWeightsF32,
        cache: inout GLM52DecodeCacheStateF32,
        position: Int) throws
        -> (output: [Float], routing: GLM52RouterOutput?, selection: [Int]) {
        let attn = try decodeAttention(
            geometry: geometry, input: input, weights: attention,
            indexer: indexer, reusedSelection: reusedSelection,
            cache: &cache, position: position)
        let afterAttn = (0..<input.count).map { input[$0] + attn.output[$0] }
        let ffnResult = try GLM52LayerCPUReference.ffnStage(
            geometry: geometry.layer, afterAttention: afterAttn,
            ffnNorm: ffnNorm, ffn: ffn)
        return (ffnResult.output, ffnResult.routing, attn.selection)
    }

    // MARK: - Indexer primitives (parameterized geometry)

    /// Upstream kernel_glm_store_indexer_k: centered LayerNorm (mean, eps
    /// 1e-6) with affine weight+bias over the whole key, PREFIX RoPE with the
    /// token position, F16 rounding at the cache boundary.
    private static func storedIndexerKey(
        _ raw: [Float],
        indexer: GLM52DecodeIndexerWeightsF32,
        geometry: GLM52DecodeGeometry,
        position: Int) throws -> [Float] {
        let n = geometry.indexerHeadDimension
        var sum: Float = 0
        for value in raw { sum += value }
        let mean = sum / Float(n)
        var sumSquares: Float = 0
        for value in raw {
            let centered = value - mean
            sumSquares += centered * centered
        }
        let inverseDeviation = 1 / (sumSquares / Float(n) + 1e-6).squareRoot()
        var normalized = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let centered = (raw[i] - mean) * inverseDeviation
            normalized[i] = centered * indexer.keyNorm[i]
                + indexer.keyNormBias[i]
        }
        let rotated = try GLM52RopeTailReference.rotatePrefix(
            values: normalized, headCount: 1, headDimension: n,
            rotationDimension: geometry.indexerRotationDimension,
            position: position)
        return rotated.map { Half.float(Half.bits($0)) }
    }

    /// Parameterized indexer relevance score over the visible key rows:
    /// `sum_h relu(dot(q[h], key[row]) * scale) * headWeight[h]` — the ReLU
    /// wraps the scaled dot product; the raw head weight multiplies after.
    private static func indexerScores(query: [Float],
                                      headWeights: [Float],
                                      geometry: GLM52DecodeGeometry,
                                      keys: [Float],
                                      visible: Int) -> [Float] {
        let dimension = geometry.indexerHeadDimension
        let scale = geometry.indexerScale
        var scores = [Float](repeating: 0, count: visible)
        for row in 0..<visible {
            let keyBase = row * dimension
            var score: Float = 0
            for head in 0..<geometry.indexerHeadCount {
                let queryBase = head * dimension
                var dot: Float = 0
                for i in 0..<dimension {
                    dot += query[queryBase + i] * keys[keyBase + i]
                }
                score += max(dot * scale, 0) * headWeights[head]
            }
            scores[row] = score
        }
        return scores
    }

    // MARK: - Validation

    private static func validate(
        geometry: GLM52DecodeGeometry,
        input: [Float],
        weights: GLM52DecodeAttentionWeightsF32,
        indexer: GLM52DecodeIndexerWeightsF32?,
        reusedSelection: [Int]?,
        cache: GLM52DecodeCacheStateF32,
        position: Int) throws {
        guard geometry.isValid, position >= 0 else {
            throw GLM52DecodeReferenceError.invalidGeometry
        }
        let g = geometry
        let layer = g.layer
        let headsWidth = layer.headCount * layer.valueDimension
        try requireCount(input, layer.embeddingWidth, "input")
        try requireCount(weights.attnNorm, layer.embeddingWidth, "attnNorm")
        try requireCount(weights.qA, g.qLoraRank * layer.embeddingWidth, "qA")
        try requireCount(weights.qANorm, g.qLoraRank, "qANorm")
        try requireCount(weights.qB, g.queryWidth * g.qLoraRank, "qB")
        try requireCount(weights.kvA,
                         layer.kvRawWidth * layer.embeddingWidth, "kvA")
        try requireCount(weights.kvANorm, layer.kvLoraRank, "kvANorm")
        try requireCount(weights.keyB,
                         layer.headCount * layer.kvLoraRank * g.nopeDimension,
                         "keyB")
        try requireCount(weights.valueB, headsWidth * layer.kvLoraRank,
                         "valueB")
        try requireCount(weights.attnOutput,
                         layer.embeddingWidth * headsWidth, "attnOutput")

        let cacheRowWidth = layer.kvLoraRank + layer.ropeDimension
        guard cache.compactRows.count == position * cacheRowWidth else {
            throw GLM52DecodeReferenceError.cachePositionMismatch(
                component: "compactRows",
                expected: position * cacheRowWidth,
                got: cache.compactRows.count)
        }
        if let indexer {
            guard reusedSelection == nil else {
                throw GLM52DecodeReferenceError.unexpectedSharedSelection
            }
            try requireCount(indexer.key,
                             g.indexerHeadDimension * layer.embeddingWidth,
                             "indexer.key")
            try requireCount(indexer.keyNorm, g.indexerHeadDimension,
                             "indexer.keyNorm")
            try requireCount(indexer.keyNormBias, g.indexerHeadDimension,
                             "indexer.keyNormBias")
            try requireCount(indexer.queryB,
                             g.indexerQueryWidth * g.qLoraRank,
                             "indexer.queryB")
            try requireCount(indexer.proj,
                             g.indexerHeadCount * layer.embeddingWidth,
                             "indexer.proj")
            guard cache.indexerKeys.count
                == position * g.indexerHeadDimension else {
                throw GLM52DecodeReferenceError.cachePositionMismatch(
                    component: "indexerKeys",
                    expected: position * g.indexerHeadDimension,
                    got: cache.indexerKeys.count)
            }
        } else {
            guard cache.indexerKeys.isEmpty else {
                throw GLM52DecodeReferenceError.indexerCacheOnSharedLayer
            }
        }
    }

    private static func requireCount(_ values: [Float], _ expected: Int,
                                     _ component: String) throws {
        guard values.count == expected else {
            throw GLM52DecodeReferenceError.invalidDimensions(
                component: component, expected: expected, got: values.count)
        }
    }
}
