import Foundation
import DS4Core

/// Reads portable GGUF metadata and validates Bonsai's Prism tensor descriptors
/// to distinguish it from ordinary Qwen3.5 checkpoints. It never constructs a
/// tokenizer or allocates Metal resources, so inspection precedes backend load.
public enum ModelInspector {
    public static func inspect(_ model: GGUFModel) throws -> RuntimeModelDescriptor {
        var detected = gateAwareAvailability(
            try ModelArchitectureDetector.detect(in: model))
        // qwen35 also identifies ordinary Qwen checkpoints. Only the complete
        // Prism schema identifies the folded Bonsai model this runtime admits.
        if detected.id == .bonsai2, (try? BonsaiConfiguration(model: model)) == nil {
            detected = .init(id: detected.id, family: detected.family,
                             backendAvailability: .recognizedButNotImplemented)
        }
        let prefix = detected.id.ggufMetadataNamespace
        let name = model.string("general.name").flatMap { $0.isEmpty ? nil : $0 }
            ?? fallbackName(for: detected)

        let descriptor = ModelDescriptor(
            architecture: detected,
            name: name,
            layerCount: integer(model, "\(prefix).trunk_block_count", "\(prefix).block_count", "\(prefix).num_hidden_layers"),
            embeddingLength: integer(model, "\(prefix).embedding_length", "\(prefix).hidden_size"),
            vocabularySize: integer(model, "\(prefix).vocab_size", "\(prefix).vocabulary_size")
                ?? model.array("tokenizer.ggml.tokens").flatMap { Int(exactly: $0.len) },
            capabilities: modelCapabilities(for: detected)
        )
        return RuntimeModelDescriptor(
            model: descriptor,
            capabilities: runtimeCapabilities(for: detected)
        )
    }

    /// Pure overload used by selector/UI tests without creating a synthetic GGUF.
    public static func inspect(generalArchitecture: String?, displayName: String? = nil,
                               hasDeepSeekV4Metadata: Bool = false,
                               layerCount: Int? = nil) throws
        -> RuntimeModelDescriptor {
        var detected = gateAwareAvailability(try ModelArchitectureDetector.detect(
            generalArchitecture: generalArchitecture,
            hasDeepSeekV4Metadata: hasDeepSeekV4Metadata
        ))
        if detected.id == .bonsai2 {
            detected = .init(id: detected.id, family: detected.family,
                             backendAvailability: .recognizedButNotImplemented)
        }
        let descriptor = ModelDescriptor(
            architecture: detected,
            name: displayName.flatMap { $0.isEmpty ? nil : $0 } ?? fallbackName(for: detected),
            layerCount: layerCount,
            capabilities: modelCapabilities(for: detected)
        )
        return RuntimeModelDescriptor(
            model: descriptor,
            capabilities: runtimeCapabilities(for: detected)
        )
    }

    private static func integer(_ model: GGUFModel, _ keys: String...) -> Int? {
        keys.lazy.compactMap { model.u64Compat($0).flatMap(Int.init(exactly:)) }.first
    }

    private static func modelCapabilities(for detected: DetectedModelArchitecture)
        -> ModelCapabilities {
        if [.bonsai2, .qwen38FlashNext, .deepSeekV41, .glm53Flash].contains(detected.id) {
            var result: ModelCapabilities = [.chat, .tools, .reasoning]
            if detected.id != .bonsai2 { result.insert(.mixtureOfExperts) }
            return result
        }
        switch detected.family {
        case .deepSeek:
            return DeepSeekV4BackendDefinition.modelCapabilities
        case .glm:
            return GLM52BackendDefinition.modelCapabilities
        case .laguna:
            return LagunaBackendDefinition.modelCapabilities
        case .kimi:
            return KimiK3BackendDefinition.modelCapabilities
        case .qwen:
            return QwenBackendDefinition.modelCapabilities(for: detected.id)
        case .unknown:
            return []
        }
    }

    private static func runtimeCapabilities(for detected: DetectedModelArchitecture)
        -> BackendCapabilities {
        if [.bonsai2, .qwen38FlashNext, .deepSeekV41, .glm53Flash].contains(detected.id) {
            return detected.backendAvailability == .implemented ? .nativeText : []
        }
        switch detected.backendAvailability {
        case .implemented where detected.id == .deepSeekV4:
            return DeepSeekV4BackendDefinition.runtimeCapabilities
        case .implemented where detected.family == .glm,
             .recognizedButNotImplemented where detected.family == .glm:
            return GLM52BackendDefinition.runtimeCapabilities
        case .implemented where detected.family == .laguna,
             .recognizedButNotImplemented where detected.family == .laguna:
            return LagunaBackendDefinition.runtimeCapabilities
        case .implemented where detected.family == .kimi,
             .recognizedButNotImplemented where detected.family == .kimi:
            return KimiK3BackendDefinition.runtimeCapabilities
        case .implemented, .recognizedButNotImplemented, .unknown:
            return QwenBackendDefinition.runtimeCapabilities
        }
    }

    /// DS4Core marks GLM and Laguna statically as recognized-but-not-
    /// implemented because it cannot see the runtime gates (they live above
    /// DS4Core). Overlay the gates here so descriptors reflect what THIS
    /// build actually runs: with `GLM52RuntimeGate.enabled` a `glm-dsa` file
    /// loads and generates, and the UI must not caption it as "backend not
    /// implemented". The Laguna gate follows the same contract once its
    /// decoder lands.
    private static func gateAwareAvailability(
        _ detected: DetectedModelArchitecture) -> DetectedModelArchitecture {
        let gated: Bool
        switch detected.family {
        case .glm:
            gated = detected.id == GLM52BackendDefinition.supportedArchitecture
                && GLM52BackendDefinition.runtimeEnabled
        case .laguna:
            gated = detected.id == LagunaBackendDefinition.supportedArchitecture
                && LagunaBackendDefinition.runtimeEnabled
        case .kimi:
            gated = detected.id == KimiK3BackendDefinition.supportedArchitecture
                && KimiK3BackendDefinition.runtimeEnabled
        case .deepSeek, .qwen, .unknown:
            gated = false
        }
        guard gated,
              detected.backendAvailability == .recognizedButNotImplemented
        else { return detected }
        return DetectedModelArchitecture(
            id: detected.id,
            family: detected.family,
            backendAvailability: .implemented)
    }

    private static func fallbackName(for detected: DetectedModelArchitecture) -> String {
        switch detected.id {
        case .bonsai2 where detected.backendAvailability == .implemented: return "Ternary Bonsai 2 27B"
        case .qwen38FlashNext: return "Qwen 3.8 Flash Next"
        case .deepSeekV41: return "DeepSeek V4.1 Flash"
        case .glm53Flash: return "GLM 5.3 Flash"
        default: break
        }
        switch detected.family {
        case .deepSeek: return "DeepSeek V4"
        case .glm: return "GLM 5.2"
        case .laguna: return "Laguna S 2.1"
        case .kimi: return "Kimi K3"
        case .qwen: return "Qwen (\(detected.id.rawValue))"
        case .unknown: return detected.id.rawValue.isEmpty ? "Modello GGUF" : detected.id.rawValue
        }
    }
}
