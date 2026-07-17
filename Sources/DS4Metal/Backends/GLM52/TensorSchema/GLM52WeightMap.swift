import Foundation
import DS4Core

/// Typed names for weights that are not owned by a transformer block.
public enum GLM52GlobalTensorKey: String, CaseIterable, Sendable {
    case tokenEmbedding = "token_embd.weight"
    case outputNorm = "output_norm.weight"
    case output = "output.weight"
}

/// Typed suffixes for every tensor that can occur below `blk.<layer>.`.
///
/// A lookup can still fail when the key is structurally absent from that layer,
/// for example a routed-expert key in one of the three leading dense blocks.
public enum GLM52LayerTensorKey: String, CaseIterable, Sendable {
    case attentionNorm = "attn_norm.weight"
    case attentionQueryA = "attn_q_a.weight"
    case attentionQueryANorm = "attn_q_a_norm.weight"
    case attentionQueryB = "attn_q_b.weight"
    case attentionKVA = "attn_kv_a_mqa.weight"
    case attentionKVANorm = "attn_kv_a_norm.weight"
    case attentionKeyB = "attn_k_b.weight"
    case attentionValueB = "attn_v_b.weight"
    case attentionOutput = "attn_output.weight"
    case indexerQueryB = "indexer.attn_q_b.weight"
    case indexerKey = "indexer.attn_k.weight"
    case indexerKeyNorm = "indexer.k_norm.weight"
    case indexerKeyNormBias = "indexer.k_norm.bias"
    case indexerProjection = "indexer.proj.weight"
    case feedForwardNorm = "ffn_norm.weight"
    case denseGate = "ffn_gate.weight"
    case denseUp = "ffn_up.weight"
    case denseDown = "ffn_down.weight"
    case router = "ffn_gate_inp.weight"
    case routerBias = "exp_probs_b.bias"
    case routedGate = "ffn_gate_exps.weight"
    case routedUp = "ffn_up_exps.weight"
    case routedDown = "ffn_down_exps.weight"
    case sharedGate = "ffn_gate_shexp.weight"
    case sharedUp = "ffn_up_shexp.weight"
    case sharedDown = "ffn_down_shexp.weight"
    case nextNEmbeddingProjection = "nextn.eh_proj.weight"
    case nextNEmbeddingNorm = "nextn.enorm.weight"
    case nextNHiddenNorm = "nextn.hnorm.weight"
    case nextNSharedHeadNorm = "nextn.shared_head_norm.weight"

    public func name(layer: Int) -> String {
        "blk.\(layer).\(rawValue)"
    }
}

/// A payload-free view of one validated GGUF tensor-directory entry.
///
/// The descriptor intentionally owns no pointer or byte buffer.  Streaming code
/// can retain it without pinning or copying any tensor payload.
public struct GLM52WeightDescriptor: Sendable, Equatable {
    public let name: String
    public let type: UInt32
    public let dims: [UInt64]
    public let absOffset: UInt64
    public let bytes: UInt64

    public init(name: String,
                type: UInt32,
                dims: [UInt64],
                absOffset: UInt64,
                bytes: UInt64) {
        self.name = name
        self.type = type
        self.dims = dims
        self.absOffset = absOffset
        self.bytes = bytes
    }

    public init(_ tensor: GGUFModel.Tensor) {
        self.init(
            name: tensor.name,
            type: tensor.type,
            dims: tensor.dims,
            absOffset: tensor.absOffset,
            bytes: tensor.bytes
        )
    }
}

public struct GLM52RoutedExpertWeights: Sendable, Equatable {
    public let gate: GLM52WeightDescriptor
    public let up: GLM52WeightDescriptor
    public let down: GLM52WeightDescriptor
}

public enum GLM52WeightMapError: Error, Sendable, Equatable, CustomStringConvertible {
    case duplicateTensor(String)
    case missingValidatedTensor(String)
    case invalidLayer(Int, layerCount: Int)
    case tensorUnavailable(layer: Int, key: GLM52LayerTensorKey)

    public var description: String {
        switch self {
        case .duplicateTensor(let name):
            return "duplicate GLM 5.2 GGUF tensor: \(name)"
        case .missingValidatedTensor(let name):
            return "validated GLM 5.2 tensor is missing from the weight map: \(name)"
        case .invalidLayer(let layer, let count):
            return "GLM 5.2 layer \(layer) is outside 0..<\(count)"
        case .tensorUnavailable(let layer, let key):
            return "GLM 5.2 tensor \(key.name(layer: layer)) is not present in that layer"
        }
    }
}

/// Validated, payload-free GLM 5.2 GGUF weight directory.
///
/// Construction deliberately performs both metadata and tensor-schema
/// validation.  A valid-looking GLM variant therefore cannot reach a future
/// streaming path with the wrong geometry or quantization.
public struct GLM52WeightMap: Sendable {
    public let configuration: GLM52Configuration

    private let globals: [GLM52GlobalTensorKey: GLM52WeightDescriptor]
    private let layers: [[GLM52LayerTensorKey: GLM52WeightDescriptor]]

    public var layerCount: Int { layers.count }

    public init(model: GGUFModel, shape: GLM52Shape = .v5_2) throws {
        let configuration = try GLM52Configuration(model: model, shape: shape)
        try GLM52TensorSchema.validate(model: model, shape: shape)

        var directory: [String: GLM52WeightDescriptor] = [:]
        directory.reserveCapacity(model.tensors.count)
        for tensor in model.tensors {
            guard directory[tensor.name] == nil else {
                throw GLM52WeightMapError.duplicateTensor(tensor.name)
            }
            directory[tensor.name] = GLM52WeightDescriptor(tensor)
        }

        var globals: [GLM52GlobalTensorKey: GLM52WeightDescriptor] = [:]
        globals.reserveCapacity(GLM52GlobalTensorKey.allCases.count)
        for key in GLM52GlobalTensorKey.allCases {
            guard let descriptor = directory[key.rawValue] else {
                throw GLM52WeightMapError.missingValidatedTensor(key.rawValue)
            }
            globals[key] = descriptor
        }

        let layerCount = Int(shape.nLayer)
        var layers: [[GLM52LayerTensorKey: GLM52WeightDescriptor]] = []
        layers.reserveCapacity(layerCount)
        for layer in 0..<layerCount {
            var entries: [GLM52LayerTensorKey: GLM52WeightDescriptor] = [:]
            for key in GLM52LayerTensorKey.allCases {
                if let descriptor = directory[key.name(layer: layer)] {
                    entries[key] = descriptor
                }
            }
            layers.append(entries)
        }

        self.configuration = configuration
        self.globals = globals
        self.layers = layers
    }

    public func global(_ key: GLM52GlobalTensorKey) throws -> GLM52WeightDescriptor {
        guard let descriptor = globals[key] else {
            throw GLM52WeightMapError.missingValidatedTensor(key.rawValue)
        }
        return descriptor
    }

    public func layer(_ layer: Int,
                      _ key: GLM52LayerTensorKey) throws -> GLM52WeightDescriptor {
        guard layers.indices.contains(layer) else {
            throw GLM52WeightMapError.invalidLayer(layer, layerCount: layers.count)
        }
        guard let descriptor = layers[layer][key] else {
            throw GLM52WeightMapError.tensorUnavailable(layer: layer, key: key)
        }
        return descriptor
    }

    public func routedExperts(layer: Int) throws -> GLM52RoutedExpertWeights {
        GLM52RoutedExpertWeights(
            gate: try self.layer(layer, .routedGate),
            up: try self.layer(layer, .routedUp),
            down: try self.layer(layer, .routedDown)
        )
    }
}
