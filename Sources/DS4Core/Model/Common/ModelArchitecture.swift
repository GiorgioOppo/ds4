import Foundation

/// Canonical identifier read from the GGUF `general.architecture` metadata.
///
/// GGUF producers historically used small spelling variations in diagnostics
/// and filenames. The identifier normalizes case, surrounding whitespace and
/// separators so architecture selection has one deterministic representation.
public struct ModelArchitectureID: RawRepresentable, Hashable, Sendable, Codable,
                                   CustomStringConvertible {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = Self.normalize(rawValue)
    }

    public init(_ rawValue: String) {
        self.init(rawValue: rawValue)
    }

    public static let deepSeekV4 = ModelArchitectureID("deepseek4")
    public static let glmDSA = ModelArchitectureID("glm-dsa")

    public var description: String { rawValue }

    /// Namespace used by architecture-specific GGUF metadata keys.
    ///
    /// `ModelArchitectureID` deliberately removes separators so spelling
    /// variants compare deterministically. GGUF namespaces are not canonical
    /// identifiers, however: GLM 5.2 publishes keys under `glm-dsa.*`, not
    /// `glmdsa.*`. Keep that wire-format spelling explicit instead of deriving
    /// it from `rawValue`.
    public var ggufMetadataNamespace: String {
        switch self {
        case .deepSeekV4: return "deepseek4"
        case .glmDSA: return "glm-dsa"
        default: return rawValue
        }
    }

    /// Lowercase canonical form used by detection and registry keys.
    public static func normalize(_ value: String) -> String {
        let compact = value
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .filter { !$0.isWhitespace && $0 != "-" && $0 != "_" && $0 != "." }
        switch compact {
        case "deepseekv4": return "deepseek4"
        case "qwenv2": return "qwen2"
        case "qwenv3": return "qwen3"
        default: return compact
        }
    }
}

/// Broad model family. It is intentionally independent from backend support:
/// recognizing a Qwen GGUF must not imply that a Qwen decoder exists yet.
public enum ModelFamily: String, Sendable, Codable, Equatable {
    case deepSeek
    case glm
    case qwen
    case unknown
}

/// Whether this build can instantiate an inference backend for an architecture.
public enum ModelBackendAvailability: String, Sendable, Codable, Equatable {
    case implemented
    case recognizedButNotImplemented
    case unknown
}

/// Result of architecture detection, kept separate from the architecture-
/// specific shape/configuration parsed later by a backend.
public struct DetectedModelArchitecture: Sendable, Equatable {
    public let id: ModelArchitectureID
    public let family: ModelFamily
    public let backendAvailability: ModelBackendAvailability

    public init(id: ModelArchitectureID, family: ModelFamily,
                backendAvailability: ModelBackendAvailability) {
        self.id = id
        self.family = family
        self.backendAvailability = backendAvailability
    }
}

/// Minimal portable capabilities shared with UI/model-management layers. More
/// detailed architectural properties remain in each backend configuration.
public struct ModelCapabilities: OptionSet, Sendable, Equatable, Codable {
    public let rawValue: UInt32

    public init(rawValue: UInt32) { self.rawValue = rawValue }

    public static let chat = ModelCapabilities(rawValue: 1 << 0)
    public static let tools = ModelCapabilities(rawValue: 1 << 1)
    public static let reasoning = ModelCapabilities(rawValue: 1 << 2)
    public static let mixtureOfExperts = ModelCapabilities(rawValue: 1 << 3)
    public static let compressedAttention = ModelCapabilities(rawValue: 1 << 4)
}

/// Architecture-neutral information safe to expose without leaking DeepSeek-
/// specific HC/NSA/indexer fields into future model families.
public struct ModelDescriptor: Sendable, Equatable {
    public let architecture: DetectedModelArchitecture
    public let name: String
    public let layerCount: Int?
    public let embeddingLength: Int?
    public let vocabularySize: Int?
    public let capabilities: ModelCapabilities

    public init(architecture: DetectedModelArchitecture, name: String,
                layerCount: Int? = nil, embeddingLength: Int? = nil,
                vocabularySize: Int? = nil,
                capabilities: ModelCapabilities = []) {
        self.architecture = architecture
        self.name = name
        self.layerCount = layerCount
        self.embeddingLength = embeddingLength
        self.vocabularySize = vocabularySize
        self.capabilities = capabilities
    }
}

public enum ModelArchitectureError: Error, Sendable, Equatable, CustomStringConvertible {
    case missingArchitecture
    case backendNotImplemented(ModelArchitectureID, family: ModelFamily)
    case unsupportedArchitecture(ModelArchitectureID)

    public var description: String {
        switch self {
        case .missingArchitecture:
            return "GGUF model architecture is missing"
        case .backendNotImplemented(let id, let family):
            return "model architecture \(id) (family \(family.rawValue)) is recognized, but its backend is not implemented"
        case .unsupportedArchitecture(let id):
            return "unsupported model architecture: \(id)"
        }
    }
}

/// Pure architecture classification plus a GGUF convenience overload.
public enum ModelArchitectureDetector {
    /// Detect from already-extracted metadata. The DeepSeek fallback is used only
    /// when `general.architecture` is absent/blank: an explicit architecture is
    /// authoritative and is never silently reinterpreted from unrelated keys.
    public static func detect(generalArchitecture: String?,
                              hasDeepSeekV4Metadata: Bool = false) throws
        -> DetectedModelArchitecture {
        let explicit = ModelArchitectureID(generalArchitecture ?? "")
        let id: ModelArchitectureID
        if explicit.rawValue.isEmpty {
            guard hasDeepSeekV4Metadata else { throw ModelArchitectureError.missingArchitecture }
            id = .deepSeekV4
        } else {
            id = explicit
        }

        let family = family(for: id)
        return DetectedModelArchitecture(
            id: id,
            family: family,
            backendAvailability: backendAvailability(for: id, family: family)
        )
    }

    /// Read architecture metadata from a mapped GGUF. Older DeepSeek-V4 files
    /// without `general.architecture` are recognized by their namespaced shape
    /// metadata, preserving compatibility with the existing loader.
    public static func detect(in model: GGUFModel) throws -> DetectedModelArchitecture {
        let hasDeepSeek = model.u32("deepseek4.block_count") != nil
            || model.u32("deepseek4.embedding_length") != nil
            || model.u32("deepseek4.vocab_size") != nil
        return try detect(generalArchitecture: model.string("general.architecture"),
                          hasDeepSeekV4Metadata: hasDeepSeek)
    }

    public static func family(for id: ModelArchitectureID) -> ModelFamily {
        if id == .deepSeekV4 || id.rawValue.hasPrefix("deepseek4") { return .deepSeek }
        if id == .glmDSA { return .glm }
        if id.rawValue.hasPrefix("qwen") { return .qwen }
        return .unknown
    }

    public static func backendAvailability(for id: ModelArchitectureID,
                                           family: ModelFamily? = nil)
        -> ModelBackendAvailability {
        if id == .deepSeekV4 { return .implemented }
        switch family ?? self.family(for: id) {
        case .glm, .qwen: return .recognizedButNotImplemented
        case .deepSeek, .unknown: return .unknown
        }
    }

    /// Boundary for runtime factories: detection can describe unavailable
    /// families, while construction fails explicitly before parsing the wrong
    /// architecture-specific metadata or tensor schema.
    public static func requireImplemented(_ detected: DetectedModelArchitecture) throws {
        switch detected.backendAvailability {
        case .implemented:
            return
        case .recognizedButNotImplemented:
            throw ModelArchitectureError.backendNotImplemented(detected.id, family: detected.family)
        case .unknown:
            throw ModelArchitectureError.unsupportedArchitecture(detected.id)
        }
    }
}
