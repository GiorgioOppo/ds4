import Foundation
import DS4Core

public enum GLM52IndexerReferenceError: Error, Sendable, Equatable, CustomStringConvertible {
    case invalidDimensions(query: Int, weights: Int, keys: Int)
    case nonFiniteValue(component: String, index: Int)
    case invalidSelection(queryPosition: Int, scoreCount: Int, topK: Int)
    case missingComputedSelection(layer: Int)
    case unexpectedComputedSelection(layer: Int)
    case unavailableSharedSelection(layer: Int, expectedSourceLayer: Int)

    public var description: String {
        switch self {
        case .invalidDimensions(let query, let weights, let keys):
            return "GLM indexer dimensions query=\(query) weights=\(weights) keys=\(keys) are invalid"
        case .nonFiniteValue(let component, let index):
            return "GLM indexer \(component)[\(index)] is not finite"
        case .invalidSelection(let position, let count, let topK):
            return "invalid GLM indexer selection position=\(position) scores=\(count) topK=\(topK)"
        case .missingComputedSelection(let layer):
            return "full-indexer layer \(layer) requires a newly computed selection"
        case .unexpectedComputedSelection(let layer):
            return "IndexShare layer \(layer) must reuse the preceding full-indexer selection"
        case .unavailableSharedSelection(let layer, let source):
            return "IndexShare layer \(layer) has no selection from required source layer \(source)"
        }
    }
}

/// CPU correctness oracle for the GLM 5.2 indexer score:
///
/// `sum_h ReLU(dot(q[h], key) / sqrt(32 * 128)) * headWeight[h]`.
///
/// The key is shared by all 32 query heads.  This deliberately implements the
/// simple scalar order, not a performance path or a bit-exact simulation of a
/// parallel Metal reduction.
public enum GLM52IndexerCPUReference {
    public static let headCount = 32
    public static let headDimension = 128
    public static let scale: Float = 1.0 / 64.0

    /// Score row-major indexer keys. `query` is `[32,128]`, `headWeights` is
    /// `[32]`, and `keys` is `[rows,128]`.
    public static func scores(query: [Float],
                              headWeights: [Float],
                              keys: [Float]) throws -> [Float] {
        let queryCount = headCount * headDimension
        guard query.count == queryCount,
              headWeights.count == headCount,
              keys.count % headDimension == 0 else {
            throw GLM52IndexerReferenceError.invalidDimensions(
                query: query.count, weights: headWeights.count, keys: keys.count)
        }
        try requireFinite(query, component: "query")
        try requireFinite(headWeights, component: "headWeights")
        try requireFinite(keys, component: "keys")

        let rowCount = keys.count / headDimension
        var result = [Float](repeating: 0, count: rowCount)
        for row in 0..<rowCount {
            let keyBase = row * headDimension
            var score: Float = 0
            for head in 0..<headCount {
                let queryBase = head * headDimension
                var dot: Float = 0
                for dimension in 0..<headDimension {
                    dot += query[queryBase + dimension] * keys[keyBase + dimension]
                }
                let activated = max(dot * scale, 0)
                score += activated * headWeights[head]
            }
            guard score.isFinite else {
                throw GLM52IndexerReferenceError.nonFiniteValue(
                    component: "score", index: row)
            }
            result[row] = score
        }
        return result
    }

    /// Deterministic causal top-k. Future cache rows are never candidates;
    /// equal scores prefer the lower row index, matching the scalar reference's
    /// ascending first-best scan. When all visible rows fit, the natural causal
    /// range is returned without scoring-dependent reordering.
    public static func causalTopK(scores: [Float],
                                  queryPosition: Int,
                                  topK: Int = Int(GLM52Shape.v5_2.nIndexerTopK)) throws
        -> [Int] {
        guard queryPosition >= 0,
              queryPosition < scores.count,
              !scores.isEmpty,
              topK > 0 else {
            throw GLM52IndexerReferenceError.invalidSelection(
                queryPosition: queryPosition,
                scoreCount: scores.count,
                topK: topK
            )
        }
        try requireFinite(scores, component: "scores")
        let visible = queryPosition + 1
        if visible <= topK { return Array(0..<visible) }

        var order = Array(0..<visible)
        order.sort { lhs, rhs in
            if scores[lhs] != scores[rhs] { return scores[lhs] > scores[rhs] }
            return lhs < rhs
        }
        return Array(order.prefix(topK))
    }

    public static func scoreAndSelect(query: [Float],
                                      headWeights: [Float],
                                      keys: [Float],
                                      queryPosition: Int,
                                      topK: Int = Int(GLM52Shape.v5_2.nIndexerTopK)) throws
        -> (scores: [Float], selected: [Int]) {
        let values = try scores(query: query, headWeights: headWeights, keys: keys)
        return (values, try causalTopK(
            scores: values, queryPosition: queryPosition, topK: topK))
    }

    private static func requireFinite(_ values: [Float], component: String) throws {
        for (index, value) in values.enumerated() where !value.isFinite {
            throw GLM52IndexerReferenceError.nonFiniteValue(
                component: component, index: index)
        }
    }
}

/// GLM IndexShare schedule. Layers 0, 1 and 2 compute independent selections;
/// from layer 6 onward every fourth layer recomputes, while intervening layers
/// reuse the most recent full-indexer result.
public enum GLM52IndexSharePolicy {
    public static func isFullIndexerLayer(_ layer: Int,
                                          shape: GLM52Shape = .v5_2) -> Bool {
        guard (0..<Int(shape.inferenceLayerCount)).contains(layer) else { return false }
        if layer < Int(shape.nLeadingDense) { return true }
        return layer >= 6 && (layer - 6) % 4 == 0
    }

    public static func fullIndexerLayers(shape: GLM52Shape = .v5_2) -> [Int] {
        (0..<Int(shape.inferenceLayerCount)).filter {
            isFullIndexerLayer($0, shape: shape)
        }
    }

    public static func selectionSourceLayer(for layer: Int,
                                            shape: GLM52Shape = .v5_2) -> Int? {
        guard (0..<Int(shape.inferenceLayerCount)).contains(layer) else { return nil }
        if isFullIndexerLayer(layer, shape: shape) { return layer }
        if layer < 6 { return Int(shape.nLeadingDense) - 1 }
        return 6 + ((layer - 6) / 4) * 4
    }
}

public struct GLM52SharedIndexSelection: Sendable, Equatable {
    public let sourceLayer: Int
    public let rows: [Int]
}

/// Small state machine that makes IndexShare reuse explicit. It is suitable for
/// CPU fixtures and for validating future graph wiring; it does not make the
/// GLM backend executable.
public struct GLM52IndexShareState: Sendable, Equatable {
    public private(set) var current: GLM52SharedIndexSelection?

    public init(current: GLM52SharedIndexSelection? = nil) {
        self.current = current
    }

    public mutating func resolve(layer: Int,
                                 computedRows: [Int]? = nil,
                                 shape: GLM52Shape = .v5_2) throws
        -> GLM52SharedIndexSelection {
        guard let source = GLM52IndexSharePolicy.selectionSourceLayer(
            for: layer, shape: shape) else {
            throw GLM52IndexerReferenceError.unavailableSharedSelection(
                layer: layer, expectedSourceLayer: -1)
        }

        if source == layer {
            guard let computedRows else {
                throw GLM52IndexerReferenceError.missingComputedSelection(layer: layer)
            }
            let selection = GLM52SharedIndexSelection(
                sourceLayer: layer, rows: computedRows)
            current = selection
            return selection
        }

        guard computedRows == nil else {
            throw GLM52IndexerReferenceError.unexpectedComputedSelection(layer: layer)
        }
        guard let current, current.sourceLayer == source else {
            throw GLM52IndexerReferenceError.unavailableSharedSelection(
                layer: layer, expectedSourceLayer: source)
        }
        return current
    }
}
