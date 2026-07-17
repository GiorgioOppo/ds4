import Foundation

// CPU correctness oracle for the GLM 5.2 compact DSA attention core: the stage
// that consumes the indexer's selected rows and the compact KV-LoRA cache.
// Two deliberately different evaluation orders are kept:
//
// - `expanded` is the textbook order of upstream's F32 reference
//   (layer_glm_attention_prefill_f32_ref): materialize each selected row's
//   per-head key (k_b · kv_lora ++ shared RoPE tail) and value (v_b · kv_lora),
//   then softmax-attend in the per-head domain.
// - `absorbed` is the order the future Metal kernels use
//   (kernel_glm_qk_lowrank / kernel_glm_attention_indexed_decode): absorb the
//   query into k_b once, score directly against the 512-wide cache rows,
//   accumulate the softmax in the KV-LoRA domain and project through v_b once.
//
// The two must agree within float tolerance on every input — that equivalence
// is the fixture later GPU kernels are checked against.

public struct GLM52AttentionGeometry: Sendable, Equatable {
    /// Number of query heads sharing one compact cache row.
    public let headCount: Int
    /// Per-head QK dimension WITHOUT the RoPE tail (k_b output rows).
    public let nopeDimension: Int
    /// Shared K-RoPE tail width appended to every head's key.
    public let ropeDimension: Int
    /// Compact cache row prefix width (KV-LoRA rank).
    public let kvLoraRank: Int
    /// Per-head value dimension (v_b output rows).
    public let valueDimension: Int

    public var qkDimension: Int { nopeDimension + ropeDimension }
    /// Compact cache row width: KV-LoRA prefix plus raw RoPE tail.
    public var cacheRowWidth: Int { kvLoraRank + ropeDimension }
    /// Upstream scale: 1/sqrt(per-head QK dimension), 1/16 for GLM 5.2.
    public var scale: Float { 1.0 / (Float(qkDimension)).squareRoot() }

    /// GLM 5.2: 64 heads, per-head QK 192+64, KV-LoRA 512, value 256.
    public static let v5_2 = GLM52AttentionGeometry(
        headCount: 64,
        nopeDimension: 192,
        ropeDimension: 64,
        kvLoraRank: 512,
        valueDimension: 256
    )

    public init(headCount: Int,
                nopeDimension: Int,
                ropeDimension: Int,
                kvLoraRank: Int,
                valueDimension: Int) {
        self.headCount = headCount
        self.nopeDimension = nopeDimension
        self.ropeDimension = ropeDimension
        self.kvLoraRank = kvLoraRank
        self.valueDimension = valueDimension
    }

    var isValid: Bool {
        headCount > 0 && nopeDimension > 0 && ropeDimension > 0
            && kvLoraRank > 0 && valueDimension > 0
    }
}

public enum GLM52AttentionReferenceError: Error, Sendable, Equatable,
    CustomStringConvertible {
    case invalidGeometry(GLM52AttentionGeometry)
    case invalidDimensions(component: String, expected: Int, got: Int)
    case emptySelection
    case selectionOutOfRange(row: Int, rowCount: Int)
    case duplicateSelectedRow(Int)
    case nonFiniteValue(component: String, index: Int)

    public var description: String {
        switch self {
        case .invalidGeometry(let geometry):
            return "invalid GLM attention geometry \(geometry)"
        case .invalidDimensions(let component, let expected, let got):
            return "GLM attention \(component) has \(got) elements, expected \(expected)"
        case .emptySelection:
            return "GLM attention selection is empty"
        case .selectionOutOfRange(let row, let rowCount):
            return "GLM attention selected row \(row) is outside 0..<\(rowCount)"
        case .duplicateSelectedRow(let row):
            return "GLM attention selection repeats row \(row)"
        case .nonFiniteValue(let component, let index):
            return "GLM attention \(component)[\(index)] is not finite"
        }
    }
}

/// Scalar oracle of the indexed attention over the compact cache. All arrays
/// are row-major `[Float]`:
///
/// - `query`: `[head][nope + rope]`, RoPE already applied to the tail;
/// - `keyB`: `[head][kvLora][nope]` (upstream `attn_k_b` access order);
/// - `valueB`: `[head][value][kvLora]` (upstream `attn_v_b` access order);
/// - `cache`: `[row][kvLora + rope]` — the compact layout `GLM52CompactKV`
///   stores: normalized KV-LoRA prefix, then the raw-bit RoPE tail;
/// - `selection`: unique cache row indices, any order (the indexer's top-k).
public enum GLM52AttentionCPUReference {
    /// Textbook order: expand keys/values per head, then attend.
    /// Returns `[head][value]`.
    public static func expanded(geometry: GLM52AttentionGeometry,
                                query: [Float],
                                keyB: [Float],
                                valueB: [Float],
                                cache: [Float],
                                selection: [Int]) throws -> [Float] {
        try validate(geometry: geometry, query: query, keyB: keyB,
                     valueB: valueB, cache: cache, selection: selection)
        let g = geometry
        var output = [Float](repeating: 0, count: g.headCount * g.valueDimension)

        var expandedKey = [Float](repeating: 0, count: g.qkDimension)
        var scores = [Float](repeating: 0, count: selection.count)
        for head in 0..<g.headCount {
            let queryBase = head * g.qkDimension
            let keyBBase = head * g.kvLoraRank * g.nopeDimension
            var maxScore = -Float.infinity
            for (rank, row) in selection.enumerated() {
                let cacheBase = row * g.cacheRowWidth
                // k_nope[i] = sum_j kv_lora[j] * k_b[j][i]
                for i in 0..<g.nopeDimension { expandedKey[i] = 0 }
                for j in 0..<g.kvLoraRank {
                    let x = cache[cacheBase + j]
                    let rowBase = keyBBase + j * g.nopeDimension
                    for i in 0..<g.nopeDimension {
                        expandedKey[i] += x * keyB[rowBase + i]
                    }
                }
                for i in 0..<g.ropeDimension {
                    expandedKey[g.nopeDimension + i] = cache[cacheBase + g.kvLoraRank + i]
                }
                var dot: Float = 0
                for i in 0..<g.qkDimension {
                    dot += query[queryBase + i] * expandedKey[i]
                }
                scores[rank] = dot * g.scale
                if scores[rank] > maxScore { maxScore = scores[rank] }
            }

            var denominator: Float = 0
            for rank in 0..<selection.count {
                scores[rank] = exp(scores[rank] - maxScore)
                denominator += scores[rank]
            }
            if denominator < 1e-20 { denominator = 1e-20 }

            // v[row][d] = sum_j v_b[d][j] * kv_lora[row][j]
            let valueBBase = head * g.valueDimension * g.kvLoraRank
            let outputBase = head * g.valueDimension
            for d in 0..<g.valueDimension {
                let rowBase = valueBBase + d * g.kvLoraRank
                var accumulator: Float = 0
                for (rank, row) in selection.enumerated() {
                    let cacheBase = row * g.cacheRowWidth
                    var value: Float = 0
                    for j in 0..<g.kvLoraRank {
                        value += valueB[rowBase + j] * cache[cacheBase + j]
                    }
                    accumulator += scores[rank] * value
                }
                output[outputBase + d] = accumulator / denominator
            }
        }
        try requireFinite(output, component: "output")
        return output
    }

    /// Kernel order: absorb the query into k_b once, score against raw cache
    /// rows, accumulate softmax in the KV-LoRA domain, project through v_b
    /// once. Mathematically identical to `expanded` by linearity.
    /// Returns `[head][value]`.
    public static func absorbed(geometry: GLM52AttentionGeometry,
                                query: [Float],
                                keyB: [Float],
                                valueB: [Float],
                                cache: [Float],
                                selection: [Int]) throws -> [Float] {
        try validate(geometry: geometry, query: query, keyB: keyB,
                     valueB: valueB, cache: cache, selection: selection)
        let g = geometry
        var output = [Float](repeating: 0, count: g.headCount * g.valueDimension)

        var lowRankQuery = [Float](repeating: 0, count: g.kvLoraRank)
        var scores = [Float](repeating: 0, count: selection.count)
        var loraAccumulator = [Float](repeating: 0, count: g.kvLoraRank)
        for head in 0..<g.headCount {
            let queryBase = head * g.qkDimension
            let keyBBase = head * g.kvLoraRank * g.nopeDimension
            // q_low[j] = sum_i q_nope[i] * k_b[j][i]  (absorb once per head)
            for j in 0..<g.kvLoraRank {
                let rowBase = keyBBase + j * g.nopeDimension
                var dot: Float = 0
                for i in 0..<g.nopeDimension {
                    dot += query[queryBase + i] * keyB[rowBase + i]
                }
                lowRankQuery[j] = dot
            }

            var maxScore = -Float.infinity
            for (rank, row) in selection.enumerated() {
                let cacheBase = row * g.cacheRowWidth
                var dot: Float = 0
                for j in 0..<g.kvLoraRank {
                    dot += lowRankQuery[j] * cache[cacheBase + j]
                }
                for i in 0..<g.ropeDimension {
                    dot += query[queryBase + g.nopeDimension + i] *
                        cache[cacheBase + g.kvLoraRank + i]
                }
                scores[rank] = dot * g.scale
                if scores[rank] > maxScore { maxScore = scores[rank] }
            }

            var denominator: Float = 0
            for rank in 0..<selection.count {
                scores[rank] = exp(scores[rank] - maxScore)
                denominator += scores[rank]
            }
            if denominator < 1e-20 { denominator = 1e-20 }

            for j in 0..<g.kvLoraRank { loraAccumulator[j] = 0 }
            for (rank, row) in selection.enumerated() {
                let cacheBase = row * g.cacheRowWidth
                let weight = scores[rank]
                for j in 0..<g.kvLoraRank {
                    loraAccumulator[j] += weight * cache[cacheBase + j]
                }
            }

            // out[d] = (v_b[d] · lora_acc) / denom  (project ONCE per head)
            let valueBBase = head * g.valueDimension * g.kvLoraRank
            let outputBase = head * g.valueDimension
            for d in 0..<g.valueDimension {
                let rowBase = valueBBase + d * g.kvLoraRank
                var dot: Float = 0
                for j in 0..<g.kvLoraRank {
                    dot += valueB[rowBase + j] * loraAccumulator[j]
                }
                output[outputBase + d] = dot / denominator
            }
        }
        try requireFinite(output, component: "output")
        return output
    }

    // MARK: - Validation

    private static func validate(geometry: GLM52AttentionGeometry,
                                 query: [Float],
                                 keyB: [Float],
                                 valueB: [Float],
                                 cache: [Float],
                                 selection: [Int]) throws {
        guard geometry.isValid else {
            throw GLM52AttentionReferenceError.invalidGeometry(geometry)
        }
        let g = geometry
        try requireCount(query, g.headCount * g.qkDimension, "query")
        try requireCount(keyB, g.headCount * g.kvLoraRank * g.nopeDimension, "keyB")
        try requireCount(valueB, g.headCount * g.valueDimension * g.kvLoraRank, "valueB")
        guard cache.count.isMultiple(of: g.cacheRowWidth), !cache.isEmpty else {
            throw GLM52AttentionReferenceError.invalidDimensions(
                component: "cache", expected: g.cacheRowWidth, got: cache.count)
        }
        let rowCount = cache.count / g.cacheRowWidth
        guard !selection.isEmpty else {
            throw GLM52AttentionReferenceError.emptySelection
        }
        var seen = Set<Int>()
        seen.reserveCapacity(selection.count)
        for row in selection {
            guard row >= 0, row < rowCount else {
                throw GLM52AttentionReferenceError.selectionOutOfRange(
                    row: row, rowCount: rowCount)
            }
            guard seen.insert(row).inserted else {
                throw GLM52AttentionReferenceError.duplicateSelectedRow(row)
            }
        }
        try requireFinite(query, component: "query")
        try requireFinite(keyB, component: "keyB")
        try requireFinite(valueB, component: "valueB")
        try requireFinite(cache, component: "cache")
    }

    private static func requireCount(_ values: [Float],
                                     _ expected: Int,
                                     _ component: String) throws {
        guard values.count == expected else {
            throw GLM52AttentionReferenceError.invalidDimensions(
                component: component, expected: expected, got: values.count)
        }
    }

    private static func requireFinite(_ values: [Float],
                                      component: String) throws {
        for (index, value) in values.enumerated() where !value.isFinite {
            throw GLM52AttentionReferenceError.nonFiniteValue(
                component: component, index: index)
        }
    }
}
