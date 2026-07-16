import Foundation
import DS4Core

/// Stable identifiers exposed to the GUI and persisted by callers.
public enum ModelCatalogID: String, CaseIterable, Identifiable, Sendable, Hashable {
    case flashQ2Imatrix = "q2-imatrix"
    case flashQ2Q4Imatrix = "q2-q4-imatrix"
    case flashQ4Imatrix = "q4-imatrix"
    case proQ2Imatrix = "pro-q2-imatrix"
    case proQ4Split = "pro-q4-split"

    public var id: String { rawValue }
}

public enum DeepSeekV4Edition: String, Sendable, Hashable {
    case flash
    case pro
}

/// Whether this build can load an entry after downloading it.
public enum ModelRuntimeAvailability: Sendable, Hashable {
    case runnable
    case downloadOnly(reason: String)

    public var isRunnable: Bool {
        if case .runnable = self { return true }
        return false
    }

    public var unavailableReason: String? {
        if case .downloadOnly(let reason) = self { return reason }
        return nil
    }
}

public enum ModelArtifactRole: String, Sendable, Hashable {
    case mainModel
    case distributedShard
    case optionalComponent
}

/// One remote file. Catalog entries may contain one complete model or several
/// files that form a non-runnable package (for example the two PRO Q4 shards).
public struct ModelTarget: Sendable, Identifiable, Hashable {
    public let id: String
    public let file: String
    public let approxGB: Int
    public let note: String
    public let sha256: String?
    public let expectedSizeBytes: Int64?
    public let role: ModelArtifactRole

    public init(id: String, file: String, approxGB: Int, note: String,
                sha256: String? = nil, expectedSizeBytes: Int64? = nil,
                role: ModelArtifactRole = .mainModel) {
        self.id = id
        self.file = file
        self.approxGB = approxGB
        self.note = note
        self.sha256 = sha256
        self.expectedSizeBytes = expectedSizeBytes
        self.role = role
    }
}

public struct ModelCatalogEntry: Sendable, Identifiable, Hashable {
    public let id: ModelCatalogID
    public let displayName: String
    public let edition: DeepSeekV4Edition
    public let summary: String
    public let artifacts: [ModelTarget]
    public let runtimeAvailability: ModelRuntimeAvailability

    public init(id: ModelCatalogID, displayName: String, edition: DeepSeekV4Edition,
                summary: String, artifacts: [ModelTarget],
                runtimeAvailability: ModelRuntimeAvailability) {
        self.id = id
        self.displayName = displayName
        self.edition = edition
        self.summary = summary
        self.artifacts = artifacts
        self.runtimeAvailability = runtimeAvailability
    }

    /// Only a complete, single-file model supported by the current runtime may
    /// become the active model in the GUI.
    public var isSelectable: Bool {
        runtimeAvailability.isRunnable
            && artifacts.count == 1
            && artifacts[0].role == .mainModel
    }

    public var primaryArtifact: ModelTarget? {
        guard isSelectable else { return nil }
        return artifacts[0]
    }

    public var approximateSizeGB: Int {
        artifacts.reduce(0) { $0 + $1.approxGB }
    }
}

/// The single source of truth for models shown by the GUI.
///
/// Runtime availability is derived from the backend declaration. Artifact
/// packaging remains an independent constraint: a supported profile in a split
/// package is still download-only until a multi-shard loader exists.
public enum DeepSeekV4ModelCatalog {
    public static let entries: [ModelCatalogEntry] = [
        .init(
            id: .flashQ2Imatrix,
            displayName: "DeepSeek V4 Flash · IQ2XXS",
            edition: .flash,
            summary: "Quantizzazione più compatta, consigliata per Mac con 96–128 GB di memoria.",
            artifacts: [
                .init(
                    id: ModelCatalogID.flashQ2Imatrix.rawValue,
                    file: "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf",
                    approxGB: 87,
                    note: "2-bit routed experts",
                    sha256: "efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668"
                ),
            ],
            runtimeAvailability: singleFileAvailability(for: .flash)
        ),
        .init(
            id: .flashQ2Q4Imatrix,
            displayName: "DeepSeek V4 Flash · IQ2XXS/Q4_K",
            edition: .flash,
            summary: "Quantizzazione mista: gli ultimi sei layer usano esperti Q4 per una qualità maggiore.",
            artifacts: [
                .init(
                    id: ModelCatalogID.flashQ2Q4Imatrix.rawValue,
                    file: "DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf",
                    approxGB: 98,
                    note: "mixed q2/q4 routed experts",
                    sha256: "edabc92af63ad8b139f00087fbfc10a4072f37b7597f4fd9ad1dfa6f83002396"
                ),
            ],
            runtimeAvailability: singleFileAvailability(for: .flash)
        ),
        .init(
            id: .flashQ4Imatrix,
            displayName: "DeepSeek V4 Flash · Q4_K",
            edition: .flash,
            summary: "Quantizzazione a qualità più alta, consigliata per Mac con almeno 256 GB di memoria.",
            artifacts: [
                .init(
                    id: ModelCatalogID.flashQ4Imatrix.rawValue,
                    file: "DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf",
                    approxGB: 165,
                    note: "4-bit routed experts",
                    sha256: "a2a3b31eca06344b93d32b2095511c4d36f92739a68a599b22047b4b2335d859"
                ),
            ],
            runtimeAvailability: singleFileAvailability(for: .flash)
        ),
        .init(
            id: .proQ2Imatrix,
            displayName: "DeepSeek V4 Pro · IQ2XXS",
            edition: .pro,
            summary: "Modello PRO completo in un singolo GGUF.",
            artifacts: [
                .init(
                    id: ModelCatalogID.proQ2Imatrix.rawValue,
                    file: "DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf",
                    approxGB: 465,
                    note: "PRO q2 single GGUF",
                    sha256: "a0314d9c0e16122cd60071079124a2d17185d317c55a8f95ecb3ed3506278a96"
                ),
            ],
            runtimeAvailability: singleFileAvailability(for: .pro)
        ),
        .init(
            id: .proQ4Split,
            displayName: "DeepSeek V4 Pro · Q4_K split",
            edition: .pro,
            summary: "Pacchetto upstream a due shard, disponibile solo per il download; il loader multi-shard non è ancora implementato.",
            artifacts: [
                .init(
                    id: "pro-q4-layers00-30",
                    file: "DeepSeek-V4-Pro-Q4K-Layers00-30.gguf",
                    approxGB: 458,
                    note: "PRO Q4 layers 0…30",
                    sha256: "3c4526735ce204a99174059b216db155846b729bf5014c6b86d573323daa3cfa",
                    role: .distributedShard
                ),
                .init(
                    id: "pro-q4-layers31-output",
                    file: "DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf",
                    approxGB: 442,
                    note: "PRO Q4 layers 31…output",
                    sha256: "41d14e4ccf9a9b777899887ac4d6115b11e5a5125f051e9fa5e727656ad5179b",
                    role: .distributedShard
                ),
            ],
            runtimeAvailability: .downloadOnly(
                reason: "Il package PRO Q4 è multi-shard e non può ancora essere assemblato dal runtime locale o distribuito."
            )
        ),
    ]

    public static let selectableEntries: [ModelCatalogEntry] = entries.filter(\.isSelectable)

    private static func singleFileAvailability(for variant: DeepSeekV4Variant)
        -> ModelRuntimeAvailability {
        guard DeepSeekV4BackendDefinition.supportsLocalRuntime(variant) else {
            return .downloadOnly(
                reason: "Il runtime locale per questo profilo non è disponibile in questa versione."
            )
        }
        return .runnable
    }

    public static func entry(_ id: ModelCatalogID) -> ModelCatalogEntry? {
        entries.first { $0.id == id }
    }

    public static func entry(_ id: String) -> ModelCatalogEntry? {
        ModelCatalogID(rawValue: id).flatMap(entry)
    }

    public static var allArtifacts: [ModelTarget] {
        entries.flatMap(\.artifacts)
    }
}

/// Optional companions are deliberately not part of the main-model catalog.
public enum DeepSeekV4AccessoryCatalog {
    public static let mtp = ModelTarget(
        id: "mtp",
        file: "DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf",
        approxGB: 4,
        note: "optional speculative-decoding component",
        role: .optionalComponent
    )
}
