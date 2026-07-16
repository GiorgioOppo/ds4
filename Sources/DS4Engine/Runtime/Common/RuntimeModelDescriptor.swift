import DS4Core

/// Engine-facing bridge around DS4Core's architecture-neutral descriptor.
///
/// `model` reports what the GGUF declares. `capabilities` reports which paths
/// this DS4Engine build can actually expose for that architecture.
public struct RuntimeModelDescriptor: Sendable, Equatable {
    public let model: ModelDescriptor
    public let capabilities: BackendCapabilities

    public init(model: ModelDescriptor, capabilities: BackendCapabilities) {
        self.model = model
        self.capabilities = capabilities
    }

    public var architecture: ModelArchitectureID { model.architecture.id }
    public var family: ModelFamily { model.architecture.family }
    public var backendAvailability: ModelBackendAvailability {
        model.architecture.backendAvailability
    }
    public var displayName: String { model.name }
    public var layerCount: Int? { model.layerCount }
    public var embeddingLength: Int? { model.embeddingLength }
    public var vocabularySize: Int? { model.vocabularySize }
}
