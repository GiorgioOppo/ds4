import Foundation
import DS4Core

/// Reads only portable GGUF metadata. It never constructs a tokenizer, validates
/// DeepSeek shapes or allocates Metal resources, so it is safe to call before a
/// backend has been selected.
public enum ModelInspector {
    public static func inspect(_ model: GGUFModel) throws -> RuntimeModelDescriptor {
        let detected = try ModelArchitectureDetector.detect(in: model)
        let prefix = detected.id.ggufMetadataNamespace
        let name = model.string("general.name").flatMap { $0.isEmpty ? nil : $0 }
            ?? fallbackName(for: detected)

        let descriptor = ModelDescriptor(
            architecture: detected,
            name: name,
            layerCount: model.u32("\(prefix).block_count").map(Int.init),
            embeddingLength: model.u32("\(prefix).embedding_length").map(Int.init),
            vocabularySize: model.u32("\(prefix).vocab_size").map(Int.init),
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
        let detected = try ModelArchitectureDetector.detect(
            generalArchitecture: generalArchitecture,
            hasDeepSeekV4Metadata: hasDeepSeekV4Metadata
        )
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

    private static func modelCapabilities(for detected: DetectedModelArchitecture)
        -> ModelCapabilities {
        switch detected.family {
        case .deepSeek:
            return DeepSeekV4BackendDefinition.modelCapabilities
        case .glm:
            return GLM52BackendDefinition.modelCapabilities
        case .qwen:
            return QwenBackendDefinition.modelCapabilities(for: detected.id)
        case .unknown:
            return []
        }
    }

    private static func runtimeCapabilities(for detected: DetectedModelArchitecture)
        -> BackendCapabilities {
        switch detected.backendAvailability {
        case .implemented where detected.id == .deepSeekV4:
            return DeepSeekV4BackendDefinition.runtimeCapabilities
        case .recognizedButNotImplemented where detected.family == .glm:
            return GLM52BackendDefinition.runtimeCapabilities
        case .implemented, .recognizedButNotImplemented, .unknown:
            return QwenBackendDefinition.runtimeCapabilities
        }
    }

    private static func fallbackName(for detected: DetectedModelArchitecture) -> String {
        switch detected.family {
        case .deepSeek: return "DeepSeek V4"
        case .glm: return "GLM 5.2"
        case .qwen: return "Qwen (\(detected.id.rawValue))"
        case .unknown: return detected.id.rawValue.isEmpty ? "Modello GGUF" : detected.id.rawValue
        }
    }
}
