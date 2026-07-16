import Foundation
import DS4Core

/// Concrete runtime implementations available in this build. This is a closed
/// enum on purpose: hot token loops keep their concrete decoder type.
public enum RuntimeBackendKind: String, Sendable, Equatable {
    case deepSeekV4
}

public enum BackendSelectionError: Error, Sendable, Equatable, CustomStringConvertible {
    case backendNotImplemented(ModelArchitectureID)
    case unsupportedProfile(ModelArchitectureID, layerCount: Int)
    case unsupportedArchitecture(ModelArchitectureID)

    public var description: String {
        switch self {
        case .backendNotImplemented(let architecture):
            return "backend \(architecture.rawValue) non ancora implementato"
        case .unsupportedProfile(let architecture, let layerCount):
            return "profilo \(architecture.rawValue) con \(layerCount) layer riconosciuto, ma il backend non è ancora implementato"
        case .unsupportedArchitecture(let architecture):
            return "architettura GGUF non supportata: \(architecture.rawValue)"
        }
    }
}

public enum BackendSelector {
    public static func select(_ descriptor: RuntimeModelDescriptor) throws
        -> RuntimeBackendKind {
        if descriptor.architecture == .deepSeekV4,
           descriptor.backendAvailability == .implemented {
            if let layerCount = descriptor.layerCount,
               !DeepSeekV4BackendDefinition.supportedLayerCounts.contains(layerCount) {
                throw BackendSelectionError.unsupportedProfile(
                    descriptor.architecture,
                    layerCount: layerCount
                )
            }
            return .deepSeekV4
        }
        if descriptor.family == .qwen {
            throw BackendSelectionError.backendNotImplemented(descriptor.architecture)
        }
        throw BackendSelectionError.unsupportedArchitecture(descriptor.architecture)
    }
}

public struct RuntimeBackendSelection: Sendable, Equatable {
    public let backend: RuntimeBackendKind
    public let descriptor: RuntimeModelDescriptor

    public init(backend: RuntimeBackendKind, descriptor: RuntimeModelDescriptor) {
        self.backend = backend
        self.descriptor = descriptor
    }
}

/// Single construction boundary shared by all DS4Engine consumers. It performs
/// architecture selection before any backend-specific configuration is parsed.
public enum RuntimeBackendFactory {
    public static func prepare(model: GGUFModel) throws -> RuntimeBackendSelection {
        let descriptor = try ModelInspector.inspect(model)
        let backend = try BackendSelector.select(descriptor)
        return RuntimeBackendSelection(backend: backend, descriptor: descriptor)
    }

    /// Metadata-only inspection for UI and diagnostics. This does not promise
    /// that a runtime backend is available.
    public static func inspect(modelPath: String) throws -> RuntimeModelDescriptor {
        let model = try GGUFModel(path: modelPath, metalMapping: false, prefetchCPU: false)
        return try ModelInspector.inspect(model)
    }
}
