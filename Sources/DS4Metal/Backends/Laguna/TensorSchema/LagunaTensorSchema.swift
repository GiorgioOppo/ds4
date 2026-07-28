import Foundation
import DS4Core

/// Lightweight tensor-directory entry used by the schema validator and by
/// unit tests that should not mmap a 60+ GB model fixture.
public struct LagunaTensorRecord: Sendable, Equatable {
    public let name: String
    public let type: UInt32
    public let dimensions: [UInt64]

    public init(name: String, type: UInt32, dimensions: [UInt64]) {
        self.name = name
        self.type = type
        self.dimensions = dimensions
    }

    public init(_ tensor: GGUFModel.Tensor) {
        self.init(name: tensor.name, type: tensor.type, dimensions: tensor.dims)
    }
}

/// The two coherent quantization recipes Poolside published for Laguna S 2.1.
///
/// The current official file uses Q8_0 signal-path weights (embedding,
/// attention, dense FFN, shared experts, output head); the earlier recipe kept
/// F16 attention with Q4_K/Q6_K signal weights.  Upstream identifies the
/// layout from the embedding tensor type for full models, or from the first
/// bound layer's attention Q for layer-only/distributed weight views.
public enum LagunaQuantizationLayout: String, Sendable, Equatable {
    case signalQ8
    case legacy
}

public enum LagunaTensorSchemaError: Error, Sendable, Equatable, CustomStringConvertible {
    case invalidLayerRange(ClosedRange<Int>, modelLayerCount: Int)
    case duplicateTensor(String)
    case missing(String)
    case unidentifiableLayout
    case unsupportedLayoutMarker(type: UInt32)
    case partialOutputHead
    case layout(name: String, gotType: UInt32, gotDimensions: [UInt64],
                expectedTypes: Set<UInt32>, expectedDimensions: [UInt64])
    case unsupportedRoutedType(layer: Int, type: UInt32)
    case routedDownMismatch(layer: Int, downType: UInt32, routedType: UInt32)
    case unsupportedSharedDownType(layer: Int, type: UInt32)

    public var description: String {
        switch self {
        case .invalidLayerRange(let range, let count):
            return "invalid Laguna layer range \(range); model has \(count) blocks"
        case .duplicateTensor(let name):
            return "Laguna GGUF tensor directory contains duplicate entry: \(name)"
        case .missing(let name):
            return "required Laguna GGUF tensor is missing: \(name)"
        case .unidentifiableLayout:
            return "cannot identify the Laguna quantization layout"
        case .unsupportedLayoutMarker(let type):
            return "unsupported Laguna quantization layout marker \(GGUF.typeName(type)); "
                + "expected legacy Q4_K/F16 or Q8_0 signal weights"
        case .partialOutputHead:
            return "Laguna GGUF contains a partial output head"
        case .layout(let name, let type, let dimensions, let expectedTypes, let expectedDimensions):
            let expected = expectedTypes.sorted().map(GGUF.typeName).joined(separator: "/")
            return "\(name): type \(GGUF.typeName(type)) dimensions \(dimensions), expected \(expected) \(expectedDimensions)"
        case .unsupportedRoutedType(let layer, let type):
            return "blk.\(layer): routed experts have unsupported type \(GGUF.typeName(type))"
        case .routedDownMismatch(let layer, let down, let routed):
            return "blk.\(layer): routed down type \(GGUF.typeName(down)) is incompatible "
                + "with \(GGUF.typeName(routed)) gate/up experts"
        case .unsupportedSharedDownType(let layer, let type):
            return "blk.\(layer): shared down tensor has unsupported type \(GGUF.typeName(type))"
        }
    }
}

/// Exact GGUF tensor schema accepted for Laguna S 2.1, ported from
/// `weights_validate_laguna_layout` in the reference `laguna-s2.1` branch.
///
/// Mixed files may spend more bits on selected layers (the published
/// `RoutedQ2_K-Last27Q3_K` file uses Q2_K routed experts on the first MoE
/// layers and Q3_K on the last 27), but all three routed projections within
/// one layer must use a coherent layout; the legacy recipe additionally
/// accepts Q6_K routed/shared down projections next to Q4_K gate/up.
public enum LagunaTensorSchema {
    public static let f32: UInt32 = 0
    public static let f16: UInt32 = 1
    public static let q8_0: UInt32 = 8
    public static let q2_K: UInt32 = 10
    public static let q3_K: UInt32 = 11
    public static let q4_K: UInt32 = 12
    public static let q6_K: UInt32 = 14

    /// Routed gate/up/down expert types accepted by the reference validator.
    public static let routedTypes: Set<UInt32> = [q2_K, q3_K, q4_K]

    /// Identify which published recipe a tensor directory uses.
    /// `layerStart` selects the attention-Q fallback marker for layer-only
    /// views without an embedding tensor.
    public static func quantizationLayout(records: [LagunaTensorRecord],
                                          layerStart: Int = 0) throws
        -> LagunaQuantizationLayout {
        let embedding = records.first { $0.name == "token_embd.weight" }
        let marker = embedding
            ?? records.first { $0.name == "blk.\(layerStart).attn_q.weight" }
        guard let marker else {
            throw LagunaTensorSchemaError.unidentifiableLayout
        }
        if marker.type == q8_0 { return .signalQ8 }
        let legacy = (embedding != nil && marker.type == q4_K)
            || (embedding == nil && marker.type == f16)
        guard legacy else {
            throw LagunaTensorSchemaError.unsupportedLayoutMarker(type: marker.type)
        }
        return .legacy
    }

    /// Validate the tensor directory of a mapped GGUF without reading tensor
    /// payloads.  A layer range supports future sharded loaders, while the
    /// defaults validate a complete single-file model.
    public static func validate(model: GGUFModel,
                                shape: LagunaShape = .s2_1,
                                layerRange: ClosedRange<Int>? = nil,
                                requireTokenEmbedding: Bool = true,
                                requireOutput: Bool = true) throws {
        try validate(
            records: model.tensors.map(LagunaTensorRecord.init),
            shape: shape,
            layerRange: layerRange,
            requireTokenEmbedding: requireTokenEmbedding,
            requireOutput: requireOutput
        )
    }

    /// Record-based form used by model-inspection and deterministic unit tests.
    public static func validate(records: [LagunaTensorRecord],
                                shape: LagunaShape = .s2_1,
                                layerRange: ClosedRange<Int>? = nil,
                                requireTokenEmbedding: Bool = true,
                                requireOutput: Bool = true) throws {
        var directory: [String: LagunaTensorRecord] = [:]
        directory.reserveCapacity(records.count)
        for record in records {
            guard directory[record.name] == nil else {
                throw LagunaTensorSchemaError.duplicateTensor(record.name)
            }
            directory[record.name] = record
        }

        let layerCount = Int(shape.nLayer)
        let range = layerRange ?? 0...(layerCount - 1)
        guard range.lowerBound >= 0,
              range.upperBound < layerCount,
              range.lowerBound <= range.upperBound else {
            throw LagunaTensorSchemaError.invalidLayerRange(range, modelLayerCount: layerCount)
        }

        let layout = try quantizationLayout(records: records,
                                            layerStart: range.lowerBound)
        let signalQ8 = layout == .signalQ8

        let embeddingName = "token_embd.weight"
        if requireTokenEmbedding && directory[embeddingName] == nil {
            throw LagunaTensorSchemaError.missing(embeddingName)
        }
        if directory[embeddingName] != nil {
            try expect(embeddingName, [signalQ8 ? q8_0 : q4_K],
                       [shape.nEmbd, shape.nVocab], in: directory)
        }

        let hasOutputNorm = directory["output_norm.weight"] != nil
        let hasOutput = directory["output.weight"] != nil
        if hasOutputNorm != hasOutput {
            throw LagunaTensorSchemaError.partialOutputHead
        }
        if requireOutput && !hasOutput {
            throw LagunaTensorSchemaError.missing("output.weight")
        }
        if hasOutput {
            try expect("output_norm.weight", [f32], [shape.nEmbd], in: directory)
            try expect("output.weight", [signalQ8 ? q8_0 : q6_K],
                       [shape.nEmbd, shape.nVocab], in: directory)
        }

        let attentionType = signalQ8 ? q8_0 : f16
        for layer in range {
            let p = "blk.\(layer)."
            let heads = UInt64(shape.layerHeadCount(layer))
            let qWidth = heads * UInt64(shape.nHeadDim)
            let kvWidth = UInt64(shape.keyValueProjectionWidth)

            try expect(p + "attn_norm.weight", [f32], [shape.nEmbd], in: directory)
            try expect(p + "attn_q.weight", [attentionType],
                       [UInt64(shape.nEmbd), qWidth], in: directory)
            try expect(p + "attn_k.weight", [attentionType],
                       [UInt64(shape.nEmbd), kvWidth], in: directory)
            try expect(p + "attn_v.weight", [attentionType],
                       [UInt64(shape.nEmbd), kvWidth], in: directory)
            try expect(p + "attn_gate.weight", [attentionType],
                       [UInt64(shape.nEmbd), heads], in: directory)
            try expect(p + "attn_q_norm.weight", [f32], [shape.nHeadDim], in: directory)
            try expect(p + "attn_k_norm.weight", [f32], [shape.nHeadDim], in: directory)
            try expect(p + "attn_output.weight", [attentionType],
                       [qWidth, UInt64(shape.nEmbd)], in: directory)
            try expect(p + "ffn_norm.weight", [f32], [shape.nEmbd], in: directory)

            if layer < Int(shape.nLeadingDense) {
                try expect(p + "ffn_gate.weight", [signalQ8 ? q8_0 : q4_K],
                           [shape.nEmbd, shape.nFFDense], in: directory)
                try expect(p + "ffn_up.weight", [signalQ8 ? q8_0 : q4_K],
                           [shape.nEmbd, shape.nFFDense], in: directory)
                try expect(p + "ffn_down.weight", [signalQ8 ? q8_0 : q6_K],
                           [shape.nFFDense, shape.nEmbd], in: directory)
                continue
            }

            try expect(p + "ffn_gate_inp.weight", [f32],
                       [shape.nEmbd, shape.nExpert], in: directory)
            try expect(p + "exp_probs_b.bias", [f32], [shape.nExpert], in: directory)

            let gateName = p + "ffn_gate_exps.weight"
            guard let gate = directory[gateName] else {
                throw LagunaTensorSchemaError.missing(gateName)
            }
            let routedType = gate.type
            guard routedTypes.contains(routedType) else {
                throw LagunaTensorSchemaError.unsupportedRoutedType(
                    layer: layer, type: routedType)
            }
            try expect(gateName, [routedType],
                       [shape.nEmbd, shape.nFFExpert, shape.nExpert], in: directory)
            try expect(p + "ffn_up_exps.weight", [routedType],
                       [shape.nEmbd, shape.nFFExpert, shape.nExpert], in: directory)

            let downName = p + "ffn_down_exps.weight"
            guard let down = directory[downName] else {
                throw LagunaTensorSchemaError.missing(downName)
            }
            let downSupported = down.type == routedType
                || (routedType == q4_K && !signalQ8 && down.type == q6_K)
            guard downSupported else {
                throw LagunaTensorSchemaError.routedDownMismatch(
                    layer: layer, downType: down.type, routedType: routedType)
            }
            try expect(downName, [down.type],
                       [shape.nFFExpert, shape.nEmbd, shape.nExpert], in: directory)

            try expect(p + "ffn_gate_shexp.weight", [signalQ8 ? q8_0 : q4_K],
                       [shape.nEmbd, shape.nFFShared], in: directory)
            try expect(p + "ffn_up_shexp.weight", [signalQ8 ? q8_0 : q4_K],
                       [shape.nEmbd, shape.nFFShared], in: directory)

            let sharedDownName = p + "ffn_down_shexp.weight"
            guard let sharedDown = directory[sharedDownName] else {
                throw LagunaTensorSchemaError.missing(sharedDownName)
            }
            if signalQ8 {
                try expect(sharedDownName, [q8_0],
                           [shape.nFFShared, shape.nEmbd], in: directory)
            } else {
                guard sharedDown.type == q4_K || sharedDown.type == q6_K else {
                    throw LagunaTensorSchemaError.unsupportedSharedDownType(
                        layer: layer, type: sharedDown.type)
                }
                try expect(sharedDownName, [sharedDown.type],
                           [shape.nFFShared, shape.nEmbd], in: directory)
            }
        }
    }

    private static func expect(_ name: String, _ types: Set<UInt32>,
                               _ dimensions: [UInt32],
                               in directory: [String: LagunaTensorRecord]) throws {
        try expect(name, types, dimensions.map(UInt64.init), in: directory)
    }

    private static func expect(_ name: String, _ types: Set<UInt32>,
                               _ dimensions: [UInt64],
                               in directory: [String: LagunaTensorRecord]) throws {
        guard let tensor = directory[name] else {
            throw LagunaTensorSchemaError.missing(name)
        }
        guard types.contains(tensor.type),
              tensor.dimensions == dimensions else {
            throw LagunaTensorSchemaError.layout(
                name: name,
                gotType: tensor.type,
                gotDimensions: tensor.dimensions,
                expectedTypes: types,
                expectedDimensions: dimensions
            )
        }
    }
}
