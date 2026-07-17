import Foundation
import DS4Core

/// A single tensor contract in the GLM 5.2 GGUF layout.
public struct GLM52TensorRequirement: Sendable, Equatable {
    public let name: String
    public let acceptedTypes: Set<UInt32>
    public let dimensions: [UInt64]

    public init(name: String, acceptedTypes: Set<UInt32>, dimensions: [UInt64]) {
        self.name = name
        self.acceptedTypes = acceptedTypes
        self.dimensions = dimensions
    }

    public var acceptedTypeNames: [String] {
        acceptedTypes.sorted().map(GGUF.typeName)
    }
}

/// Lightweight tensor-directory entry used by the schema validator and by
/// unit tests that should not mmap a multi-hundred-gigabyte model fixture.
public struct GLM52TensorRecord: Sendable, Equatable {
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

public enum GLM52TensorSchemaError: Error, Sendable, Equatable, CustomStringConvertible {
    case invalidLayerRange(ClosedRange<Int>, modelLayerCount: Int)
    case duplicateTensor(String)
    case missing(String)
    case partialOutputHead
    case layout(name: String, gotType: UInt32, gotDimensions: [UInt64],
                expectedTypes: Set<UInt32>, expectedDimensions: [UInt64])
    case gateUpQuantizationMismatch(layer: Int, gateType: UInt32, upType: UInt32)

    public var description: String {
        switch self {
        case .invalidLayerRange(let range, let count):
            return "invalid GLM 5.2 layer range \(range); model has \(count) blocks"
        case .duplicateTensor(let name):
            return "GLM 5.2 GGUF tensor directory contains duplicate entry: \(name)"
        case .missing(let name):
            return "required GLM 5.2 GGUF tensor is missing: \(name)"
        case .partialOutputHead:
            return "GLM 5.2 GGUF contains a partial output head"
        case .layout(let name, let type, let dimensions, let expectedTypes, let expectedDimensions):
            let expected = expectedTypes.sorted().map(GGUF.typeName).joined(separator: "/")
            return "\(name): type \(GGUF.typeName(type)) dimensions \(dimensions), expected \(expected) \(expectedDimensions)"
        case .gateUpQuantizationMismatch(let layer, let gate, let up):
            return "blk.\(layer): routed gate/up quantizations differ (\(GGUF.typeName(gate)) vs \(GGUF.typeName(up)))"
        }
    }
}

/// Exact GGUF tensor schema consumed by the GLM 5.2 graph.
///
/// Dense/control tensors deliberately stay on the Q8_0/F32 path.  Routed
/// gate/up types match the executable graph in the reference branch; routed
/// down additionally accepts Q6_K.  Gate and up must use the same type.
public enum GLM52TensorSchema {
    public static let f32: UInt32 = 0
    public static let q8_0: UInt32 = 8
    public static let q2_K: UInt32 = 10
    public static let q4_K: UInt32 = 12
    public static let q5_K: UInt32 = 13
    public static let q6_K: UInt32 = 14
    public static let iq2_XXS: UInt32 = 16

    /// Types handled by the reference GLM graph's gate/up dispatch.
    public static let routedGateUpTypes: Set<UInt32> = [iq2_XXS, q2_K, q4_K, q5_K]

    /// Types handled by the reference GLM graph's down-projection dispatch.
    public static let routedDownTypes: Set<UInt32> = [iq2_XXS, q2_K, q4_K, q5_K, q6_K]

    public static func globalRequirements(shape: GLM52Shape = .v5_2) -> [GLM52TensorRequirement] {
        [
            exact("token_embd.weight", q8_0, [shape.nEmbd, shape.nVocab]),
            exact("output_norm.weight", f32, [shape.nEmbd]),
            exact("output.weight", q8_0, [shape.nEmbd, shape.nVocab]),
        ]
    }

    public static func layerRequirements(_ layer: Int,
                                         shape: GLM52Shape = .v5_2) throws
        -> [GLM52TensorRequirement] {
        let layerCount = Int(shape.nLayer)
        guard (0..<layerCount).contains(layer) else {
            throw GLM52TensorSchemaError.invalidLayerRange(
                layer...layer, modelLayerCount: layerCount)
        }

        let p = "blk.\(layer)."
        var result: [GLM52TensorRequirement] = [
            exact(p + "attn_norm.weight", f32, [shape.nEmbd]),
            exact(p + "attn_q_a.weight", q8_0, [shape.nEmbd, shape.nLoraQ]),
            exact(p + "attn_q_a_norm.weight", f32, [shape.nLoraQ]),
            exact(p + "attn_q_b.weight", q8_0,
                  [shape.nLoraQ, shape.queryProjectionWidth]),
            exact(p + "attn_kv_a_mqa.weight", q8_0, [shape.nEmbd, shape.nHeadDim]),
            exact(p + "attn_kv_a_norm.weight", f32, [shape.nKVLoRA]),
            exact(p + "attn_k_b.weight", q8_0,
                  [shape.queryNonRoPEWidth, shape.nKVLoRA, shape.nHead]),
            exact(p + "attn_v_b.weight", q8_0,
                  [shape.nKVLoRA, shape.nValueMLA, shape.nHead]),
            exact(p + "attn_output.weight", q8_0,
                  [shape.nHead * shape.nValueMLA, shape.nEmbd]),
            exact(p + "indexer.attn_q_b.weight", q8_0,
                  [shape.nLoraQ, shape.indexerQueryWidth]),
            exact(p + "indexer.attn_k.weight", q8_0,
                  [shape.nEmbd, shape.nIndexerHeadDim]),
            exact(p + "indexer.k_norm.weight", f32, [shape.nIndexerHeadDim]),
            exact(p + "indexer.k_norm.bias", f32, [shape.nIndexerHeadDim]),
            exact(p + "indexer.proj.weight", f32,
                  [shape.nEmbd, shape.nIndexerHead]),
            exact(p + "ffn_norm.weight", f32, [shape.nEmbd]),
        ]

        if layer < Int(shape.nLeadingDense) {
            result += [
                exact(p + "ffn_gate.weight", q8_0, [shape.nEmbd, shape.nFFDense]),
                exact(p + "ffn_up.weight", q8_0, [shape.nEmbd, shape.nFFDense]),
                exact(p + "ffn_down.weight", q8_0, [shape.nFFDense, shape.nEmbd]),
            ]
        } else {
            result += [
                exact(p + "ffn_gate_inp.weight", f32, [shape.nEmbd, shape.nExpert]),
                exact(p + "exp_probs_b.bias", f32, [shape.nExpert]),
                oneOf(p + "ffn_gate_exps.weight", routedGateUpTypes,
                      [shape.nEmbd, shape.nFFExpert, shape.nExpert]),
                oneOf(p + "ffn_up_exps.weight", routedGateUpTypes,
                      [shape.nEmbd, shape.nFFExpert, shape.nExpert]),
                oneOf(p + "ffn_down_exps.weight", routedDownTypes,
                      [shape.nFFExpert, shape.nEmbd, shape.nExpert]),
                exact(p + "ffn_gate_shexp.weight", q8_0,
                      [shape.nEmbd, shape.nFFExpert]),
                exact(p + "ffn_up_shexp.weight", q8_0,
                      [shape.nEmbd, shape.nFFExpert]),
                exact(p + "ffn_down_shexp.weight", q8_0,
                      [shape.nFFExpert, shape.nEmbd]),
            ]
        }

        if layer + Int(shape.nNextNPredict) >= layerCount {
            result += [
                exact(p + "nextn.eh_proj.weight", q8_0,
                      [2 * shape.nEmbd, shape.nEmbd]),
                exact(p + "nextn.enorm.weight", f32, [shape.nEmbd]),
                exact(p + "nextn.hnorm.weight", f32, [shape.nEmbd]),
                exact(p + "nextn.shared_head_norm.weight", f32, [shape.nEmbd]),
            ]
        }

        return result
    }

    /// Validate the tensor directory of a mapped GGUF without reading tensor
    /// payloads.  A layer range supports future distributed/sharded loaders,
    /// while the defaults validate a complete single-file model.
    public static func validate(model: GGUFModel,
                                shape: GLM52Shape = .v5_2,
                                layerRange: ClosedRange<Int>? = nil,
                                requireTokenEmbedding: Bool = true,
                                requireOutput: Bool = true) throws {
        try validate(
            records: model.tensors.map(GLM52TensorRecord.init),
            shape: shape,
            layerRange: layerRange,
            requireTokenEmbedding: requireTokenEmbedding,
            requireOutput: requireOutput
        )
    }

    /// Record-based form used by model-inspection and deterministic unit tests.
    public static func validate(records: [GLM52TensorRecord],
                                shape: GLM52Shape = .v5_2,
                                layerRange: ClosedRange<Int>? = nil,
                                requireTokenEmbedding: Bool = true,
                                requireOutput: Bool = true) throws {
        var directory: [String: GLM52TensorRecord] = [:]
        directory.reserveCapacity(records.count)
        for record in records {
            guard directory[record.name] == nil else {
                throw GLM52TensorSchemaError.duplicateTensor(record.name)
            }
            directory[record.name] = record
        }

        let layerCount = Int(shape.nLayer)
        let range = layerRange ?? 0...(layerCount - 1)
        guard range.lowerBound >= 0,
              range.upperBound < layerCount,
              range.lowerBound <= range.upperBound else {
            throw GLM52TensorSchemaError.invalidLayerRange(range, modelLayerCount: layerCount)
        }

        let globals = globalRequirements(shape: shape)
        if requireTokenEmbedding || directory[globals[0].name] != nil {
            try validate(globals[0], in: directory)
        }

        let hasOutputNorm = directory[globals[1].name] != nil
        let hasOutput = directory[globals[2].name] != nil
        if hasOutputNorm != hasOutput {
            throw GLM52TensorSchemaError.partialOutputHead
        }
        if requireOutput || hasOutputNorm {
            try validate(globals[1], in: directory)
            try validate(globals[2], in: directory)
        }

        for layer in range {
            for requirement in try layerRequirements(layer, shape: shape) {
                try validate(requirement, in: directory)
            }

            guard layer >= Int(shape.nLeadingDense) else { continue }
            let gateName = "blk.\(layer).ffn_gate_exps.weight"
            let upName = "blk.\(layer).ffn_up_exps.weight"
            guard let gate = directory[gateName], let up = directory[upName] else {
                // Missing entries were reported above; this is unreachable but
                // keeps the cross-tensor invariant explicit and total.
                continue
            }
            if gate.type != up.type {
                throw GLM52TensorSchemaError.gateUpQuantizationMismatch(
                    layer: layer, gateType: gate.type, upType: up.type)
            }
        }
    }

    private static func exact(_ name: String, _ type: UInt32,
                              _ dimensions: [UInt32]) -> GLM52TensorRequirement {
        oneOf(name, [type], dimensions)
    }

    private static func oneOf(_ name: String, _ types: Set<UInt32>,
                              _ dimensions: [UInt32]) -> GLM52TensorRequirement {
        GLM52TensorRequirement(
            name: name,
            acceptedTypes: types,
            dimensions: dimensions.map(UInt64.init)
        )
    }

    private static func validate(_ requirement: GLM52TensorRequirement,
                                 in directory: [String: GLM52TensorRecord]) throws {
        guard let tensor = directory[requirement.name] else {
            throw GLM52TensorSchemaError.missing(requirement.name)
        }
        guard requirement.acceptedTypes.contains(tensor.type),
              tensor.dimensions == requirement.dimensions else {
            throw GLM52TensorSchemaError.layout(
                name: requirement.name,
                gotType: tensor.type,
                gotDimensions: tensor.dimensions,
                expectedTypes: requirement.acceptedTypes,
                expectedDimensions: requirement.dimensions
            )
        }
    }
}
