import Foundation

/// Conversation wire protocol paired with an architecture tokenizer.
public enum ConversationBackendID: String, Sendable, Codable, Equatable {
    case deepSeekDSML
    case glm52Native
}

public enum ConversationBackendSelectionError: Error, Sendable, Equatable,
                                                CustomStringConvertible {
    case missingArchitecture
    case conversationBackendNotImplemented(ModelArchitectureID, family: ModelFamily)
    case unsupportedArchitecture(ModelArchitectureID)

    public var description: String {
        switch self {
        case .missingArchitecture:
            return "GGUF model architecture is missing; conversation protocol selection is impossible"
        case .conversationBackendNotImplemented(let id, let family):
            return "conversation backend for architecture \(id) (family \(family.rawValue)) is not implemented"
        case .unsupportedArchitecture(let id):
            return "unsupported conversation architecture: \(id)"
        }
    }
}

/// Small frontend policy only; it neither instantiates nor advertises an
/// inference runtime.
public enum ConversationBackendPolicy {
    public static func backend(for detected: DetectedModelArchitecture) throws
        -> ConversationBackendID {
        if detected.id == .deepSeekV4 { return .deepSeekDSML }
        if detected.id == .glmDSA { return .glm52Native }

        switch detected.family {
        case .qwen:
            throw ConversationBackendSelectionError.conversationBackendNotImplemented(
                detected.id, family: detected.family
            )
        case .deepSeek, .glm, .unknown:
            throw ConversationBackendSelectionError.unsupportedArchitecture(detected.id)
        }
    }

    public static func backend(
        generalArchitecture: String?,
        hasDeepSeekV4Metadata: Bool = false
    ) throws -> ConversationBackendID {
        do {
            let detected = try ModelArchitectureDetector.detect(
                generalArchitecture: generalArchitecture,
                hasDeepSeekV4Metadata: hasDeepSeekV4Metadata
            )
            return try backend(for: detected)
        } catch let error as ConversationBackendSelectionError {
            throw error
        } catch let error as ModelArchitectureError {
            switch error {
            case .missingArchitecture:
                throw ConversationBackendSelectionError.missingArchitecture
            case .backendNotImplemented(let id, let family):
                throw ConversationBackendSelectionError.conversationBackendNotImplemented(
                    id, family: family
                )
            case .unsupportedArchitecture(let id):
                throw ConversationBackendSelectionError.unsupportedArchitecture(id)
            }
        }
    }
}
