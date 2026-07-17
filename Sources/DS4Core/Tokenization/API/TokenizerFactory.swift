import Foundation

/// Concrete tokenizer implementation selected from GGUF architecture metadata.
/// This is a frontend capability, not an inference-runtime availability claim.
public enum TokenizerBackendID: String, Sendable, Codable, Equatable {
    case deepSeekV4
    case glm52
}

public enum TokenizerFactoryError: Error, Sendable, Equatable, CustomStringConvertible {
    case missingArchitecture
    case tokenizerNotImplemented(ModelArchitectureID, family: ModelFamily)
    case unsupportedArchitecture(ModelArchitectureID)
    case invalidTokenizerMetadata(backend: TokenizerBackendID, reason: String)

    public var description: String {
        switch self {
        case .missingArchitecture:
            return "GGUF model architecture is missing; tokenizer selection is impossible"
        case .tokenizerNotImplemented(let id, let family):
            return "tokenizer for architecture \(id) (family \(family.rawValue)) is not implemented"
        case .unsupportedArchitecture(let id):
            return "unsupported tokenizer architecture: \(id)"
        case .invalidTokenizerMetadata(let backend, let reason):
            return "invalid \(backend.rawValue) tokenizer metadata: \(reason)"
        }
    }
}

/// Architecture-aware construction boundary for portable tokenization and
/// diagnostics. In particular, GLM tokenization is available while its runtime
/// remains `recognizedButNotImplemented`.
public enum TokenizerFactory {
    /// Pure selection policy, useful to test routing without mapping a GGUF.
    public static func backend(for detected: DetectedModelArchitecture) throws
        -> TokenizerBackendID {
        if detected.id == .deepSeekV4 { return .deepSeekV4 }
        if detected.id == .glmDSA { return .glm52 }

        switch detected.family {
        case .qwen:
            throw TokenizerFactoryError.tokenizerNotImplemented(
                detected.id, family: detected.family
            )
        case .deepSeek, .glm, .unknown:
            throw TokenizerFactoryError.unsupportedArchitecture(detected.id)
        }
    }

    /// Pure metadata overload. The DeepSeek legacy fallback is accepted only
    /// when architecture is absent; any explicit architecture is authoritative.
    public static func backend(
        generalArchitecture: String?,
        hasDeepSeekV4Metadata: Bool = false
    ) throws -> TokenizerBackendID {
        do {
            let detected = try ModelArchitectureDetector.detect(
                generalArchitecture: generalArchitecture,
                hasDeepSeekV4Metadata: hasDeepSeekV4Metadata
            )
            return try backend(for: detected)
        } catch let error as TokenizerFactoryError {
            throw error
        } catch let error as ModelArchitectureError {
            throw mapArchitectureError(error)
        }
    }

    /// Detect and construct exactly one tokenizer. There is deliberately no
    /// default branch that could reinterpret Qwen or an unknown model as
    /// DeepSeek merely because both expose GPT-style tokenizer tables.
    public static func make(for model: GGUFModel) throws -> any TokenizerProtocol {
        let detected: DetectedModelArchitecture
        do {
            detected = try ModelArchitectureDetector.detect(in: model)
        } catch let error as ModelArchitectureError {
            throw mapArchitectureError(error)
        }
        let selected = try backend(for: detected)
        do {
            switch selected {
            case .deepSeekV4: return try DeepSeekV4Tokenizer(model: model)
            case .glm52: return try GLM52Tokenizer(model: model)
            }
        } catch {
            throw TokenizerFactoryError.invalidTokenizerMetadata(
                backend: selected,
                reason: String(describing: error)
            )
        }
    }

    private static func mapArchitectureError(_ error: ModelArchitectureError)
        -> TokenizerFactoryError {
        switch error {
        case .missingArchitecture:
            return .missingArchitecture
        case .backendNotImplemented(let id, let family):
            return .tokenizerNotImplemented(id, family: family)
        case .unsupportedArchitecture(let id):
            return .unsupportedArchitecture(id)
        }
    }
}
