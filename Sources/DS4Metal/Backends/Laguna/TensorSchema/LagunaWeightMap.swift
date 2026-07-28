import Foundation
import DS4Core

/// Typed names for Laguna weights that are not owned by a transformer block.
public enum LagunaGlobalTensorKey: String, CaseIterable, Sendable {
    case tokenEmbedding = "token_embd.weight"
    case outputNorm = "output_norm.weight"
    case output = "output.weight"
}

/// Typed suffixes for every tensor that can occur below `blk.<layer>.`.
///
/// A lookup can still fail when the key is structurally absent from that
/// layer: the routed/shared keys in the leading dense block, or the dense FFN
/// keys in a MoE block.
public enum LagunaLayerTensorKey: String, CaseIterable, Sendable {
    case attentionNorm = "attn_norm.weight"
    case attentionQuery = "attn_q.weight"
    case attentionKey = "attn_k.weight"
    case attentionValue = "attn_v.weight"
    case attentionGate = "attn_gate.weight"
    case attentionQueryNorm = "attn_q_norm.weight"
    case attentionKeyNorm = "attn_k_norm.weight"
    case attentionOutput = "attn_output.weight"
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

    public func name(layer: Int) -> String {
        "blk.\(layer).\(rawValue)"
    }
}

/// A payload-free view of one validated GGUF tensor-directory entry.
///
/// The descriptor intentionally owns no pointer or byte buffer, exactly like
/// the GLM equivalent: future loading code can retain it without pinning or
/// copying any tensor payload.
public struct LagunaWeightDescriptor: Sendable, Equatable {
    public let name: String
    public let type: UInt32
    public let dims: [UInt64]
    public let absOffset: UInt64
    public let bytes: UInt64

    public init(name: String, type: UInt32, dims: [UInt64],
                absOffset: UInt64, bytes: UInt64) {
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

public struct LagunaRoutedExpertWeights: Sendable, Equatable {
    public let gate: LagunaWeightDescriptor
    public let up: LagunaWeightDescriptor
    public let down: LagunaWeightDescriptor
}

public enum LagunaWeightMapError: Error, Sendable, Equatable, CustomStringConvertible {
    case duplicateTensor(String)
    case missingValidatedTensor(String)
    case invalidLayer(Int, layerCount: Int)
    case tensorUnavailable(layer: Int, key: LagunaLayerTensorKey)

    public var description: String {
        switch self {
        case .duplicateTensor(let name):
            return "duplicate Laguna GGUF tensor: \(name)"
        case .missingValidatedTensor(let name):
            return "validated Laguna tensor is missing from the weight map: \(name)"
        case .invalidLayer(let layer, let count):
            return "Laguna layer \(layer) is outside 0..<\(count)"
        case .tensorUnavailable(let layer, let key):
            return "Laguna tensor \(key.name(layer: layer)) is not present in that layer"
        }
    }
}

/// Validated, payload-free Laguna S 2.1 GGUF weight directory.
///
/// Construction deliberately performs both metadata and tensor-schema
/// validation, so a valid-looking Laguna variant cannot reach a future
/// loading path with the wrong geometry or quantization recipe. The detected
/// quantization layout is retained because the engine dispatches per recipe
/// (Q8_0 signal path versus legacy Q4_K/F16).
public struct LagunaWeightMap: Sendable {
    public let configuration: LagunaConfiguration
    public let quantizationLayout: LagunaQuantizationLayout

    private let globals: [LagunaGlobalTensorKey: LagunaWeightDescriptor]
    private let layers: [[LagunaLayerTensorKey: LagunaWeightDescriptor]]

    public var layerCount: Int { layers.count }

    public init(model: GGUFModel, shape: LagunaShape = .s2_1) throws {
        let configuration = try LagunaConfiguration(model: model, shape: shape)
        let records = model.tensors.map(LagunaTensorRecord.init)
        try LagunaTensorSchema.validate(records: records, shape: shape)
        let layout = try LagunaTensorSchema.quantizationLayout(records: records)

        var directory: [String: LagunaWeightDescriptor] = [:]
        directory.reserveCapacity(model.tensors.count)
        for tensor in model.tensors {
            guard directory[tensor.name] == nil else {
                throw LagunaWeightMapError.duplicateTensor(tensor.name)
            }
            directory[tensor.name] = LagunaWeightDescriptor(tensor)
        }

        var globals: [LagunaGlobalTensorKey: LagunaWeightDescriptor] = [:]
        globals.reserveCapacity(LagunaGlobalTensorKey.allCases.count)
        for key in LagunaGlobalTensorKey.allCases {
            guard let descriptor = directory[key.rawValue] else {
                throw LagunaWeightMapError.missingValidatedTensor(key.rawValue)
            }
            globals[key] = descriptor
        }

        let layerCount = Int(shape.nLayer)
        var layers: [[LagunaLayerTensorKey: LagunaWeightDescriptor]] = []
        layers.reserveCapacity(layerCount)
        for layer in 0..<layerCount {
            var entries: [LagunaLayerTensorKey: LagunaWeightDescriptor] = [:]
            for key in LagunaLayerTensorKey.allCases {
                if let descriptor = directory[key.name(layer: layer)] {
                    entries[key] = descriptor
                }
            }
            layers.append(entries)
        }

        self.configuration = configuration
        self.quantizationLayout = layout
        self.globals = globals
        self.layers = layers
    }

    public func global(_ key: LagunaGlobalTensorKey) throws -> LagunaWeightDescriptor {
        guard let descriptor = globals[key] else {
            throw LagunaWeightMapError.missingValidatedTensor(key.rawValue)
        }
        return descriptor
    }

    public func layer(_ layer: Int,
                      _ key: LagunaLayerTensorKey) throws -> LagunaWeightDescriptor {
        guard layers.indices.contains(layer) else {
            throw LagunaWeightMapError.invalidLayer(layer, layerCount: layers.count)
        }
        guard let descriptor = layers[layer][key] else {
            throw LagunaWeightMapError.tensorUnavailable(layer: layer, key: key)
        }
        return descriptor
    }

    public func routedExperts(layer: Int) throws -> LagunaRoutedExpertWeights {
        LagunaRoutedExpertWeights(
            gate: try self.layer(layer, .routedGate),
            up: try self.layer(layer, .routedUp),
            down: try self.layer(layer, .routedDown)
        )
    }

    /// The descriptor whose payload ends farthest into the file. A reader can
    /// prove the whole validated directory fits the real file by checking this
    /// single end offset against the file size.
    public var farthestDescriptor: LagunaWeightDescriptor? {
        var farthest: LagunaWeightDescriptor?
        var farthestEnd: UInt64 = 0
        for descriptor in globals.values {
            let end = descriptor.absOffset + descriptor.bytes
            if end > farthestEnd { farthest = descriptor; farthestEnd = end }
        }
        for layer in layers {
            for descriptor in layer.values {
                let end = descriptor.absOffset + descriptor.bytes
                if end > farthestEnd { farthest = descriptor; farthestEnd = end }
            }
        }
        return farthest
    }
}
