import Foundation
import DS4Core

/// Stable identifiers exposed to the GUI and persisted by callers.
public enum ModelCatalogID: String, CaseIterable, Identifiable, Sendable, Hashable {
    case flashQ2Imatrix = "q2-imatrix"
    case flashQ2Q4Imatrix = "q2-q4-imatrix"
    case flashQ4Imatrix = "q4-imatrix"
    case proQ2Imatrix = "pro-q2-imatrix"
    case proQ4Split = "pro-q4-split"
    case glm52IQ2XXS = "glm-5.2-iq2-xxs"
    case glm52Q2K = "glm-5.2-q2-k"
    case glm52Q4K = "glm-5.2-q4-k"

    public var id: String { rawValue }
}

public enum DeepSeekV4Edition: String, Sendable, Hashable {
    case flash
    case pro
}

/// Architecture/profile identity independent from runtime availability.
///
/// A profile may be present here while still being download-only. This keeps
/// artifact acquisition separate from the backend capability matrix.
public enum ModelCatalogProfile: Sendable, Hashable {
    case deepSeekV4(DeepSeekV4Edition)
    case glm52

    public var familyDisplayName: String {
        switch self {
        case .deepSeekV4: "DeepSeek V4"
        case .glm52: "GLM 5.2"
        }
    }
}

/// Immutable Hugging Face source for a catalog artifact. A revision can be a
/// branch, tag or commit. New catalogs should prefer a commit so a pinned
/// checksum cannot silently start referring to a different remote object.
public struct HuggingFaceSource: Sendable, Hashable {
    public let repository: String
    public let revision: String

    public init(repository: String, revision: String = "main") {
        self.repository = repository
        self.revision = revision
    }

    public static let deepSeekV4 = Self(
        repository: "antirez/deepseek-v4-gguf"
    )

    public static let glm52 = Self(
        repository: "antirez/glm-5.2-gguf",
        revision: "2638b3b878f5c6cc3ae7334b8dbea1275025f52e"
    )
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
    public let source: HuggingFaceSource

    public init(id: String, file: String, approxGB: Int, note: String,
                sha256: String? = nil, expectedSizeBytes: Int64? = nil,
                role: ModelArtifactRole = .mainModel,
                source: HuggingFaceSource = .deepSeekV4) {
        self.id = id
        self.file = file
        self.approxGB = approxGB
        self.note = note
        self.sha256 = sha256
        self.expectedSizeBytes = expectedSizeBytes
        self.role = role
        self.source = source
    }
}

public struct ModelCatalogEntry: Sendable, Identifiable, Hashable {
    public let id: ModelCatalogID
    public let displayName: String
    public let profile: ModelCatalogProfile
    public let summary: String
    public let artifacts: [ModelTarget]
    public let runtimeAvailability: ModelRuntimeAvailability

    public init(id: ModelCatalogID, displayName: String, profile: ModelCatalogProfile,
                summary: String, artifacts: [ModelTarget],
                runtimeAvailability: ModelRuntimeAvailability) {
        self.id = id
        self.displayName = displayName
        self.profile = profile
        self.summary = summary
        self.artifacts = artifacts
        self.runtimeAvailability = runtimeAvailability
    }

    /// Compatibility view for callers written while the catalog contained
    /// only DeepSeek V4. New code should switch on `profile` instead.
    @available(*, deprecated, message: "Use profile; the catalog now contains multiple architectures.")
    public var edition: DeepSeekV4Edition? {
        guard case .deepSeekV4(let edition) = profile else { return nil }
        return edition
    }

    /// Source-compatible initializer for pre multi-architecture clients.
    @available(*, deprecated, message: "Use init(... profile: ...) instead.")
    public init(id: ModelCatalogID, displayName: String, edition: DeepSeekV4Edition,
                summary: String, artifacts: [ModelTarget],
                runtimeAvailability: ModelRuntimeAvailability) {
        self.init(
            id: id,
            displayName: displayName,
            profile: .deepSeekV4(edition),
            summary: summary,
            artifacts: artifacts,
            runtimeAvailability: runtimeAvailability
        )
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

    /// Exact package size when every artifact is pinned to a known byte count.
    public var expectedSizeBytes: Int64? {
        var total: Int64 = 0
        for artifact in artifacts {
            guard let size = artifact.expectedSizeBytes else { return nil }
            let (next, overflow) = total.addingReportingOverflow(size)
            guard !overflow else { return nil }
            total = next
        }
        return total
    }
}

/// DeepSeek V4 portion of the model catalog shown by the GUI.
///
/// Runtime availability is derived from the backend declaration. Artifact
/// packaging remains an independent constraint: a supported profile in a split
/// package is still download-only until a multi-shard loader exists.
public enum DeepSeekV4ModelCatalog {
    public static let entries: [ModelCatalogEntry] = [
        .init(
            id: .flashQ2Imatrix,
            displayName: "DeepSeek V4 Flash · IQ2XXS",
            profile: .deepSeekV4(.flash),
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
            profile: .deepSeekV4(.flash),
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
            profile: .deepSeekV4(.flash),
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
            profile: .deepSeekV4(.pro),
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
            profile: .deepSeekV4(.pro),
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

/// Download catalog for the GLM 5.2 GGUF publication by antirez.
///
/// These entries deliberately remain download-only. The current build has no
/// GLM tokenizer, tensor schema, DSA/MoE decoder or Metal backend, so making a
/// monolithic file selectable would misrepresent runtime support.
public enum GLM52ModelCatalog {
    private static let unavailable = ModelRuntimeAvailability.downloadOnly(
        reason: "Download disponibile; il runtime GLM 5.2 non è ancora implementato."
    )

    public static let entries: [ModelCatalogEntry] = [
        .init(
            id: .glm52IQ2XXS,
            displayName: "GLM 5.2 · IQ2_XXS RoutedIQ",
            profile: .glm52,
            summary: "Quantizzazione imatrix più compatta del repository; consigliata come primo artefatto per lo sviluppo del backend.",
            artifacts: [
                .init(
                    id: ModelCatalogID.glm52IQ2XXS.rawValue,
                    file: "GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf",
                    approxGB: 211,
                    note: "GLM 5.2 IQ2_XXS routed experts",
                    sha256: "a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0",
                    expectedSizeBytes: 211_075_856_448,
                    source: .glm52
                ),
            ],
            runtimeAvailability: unavailable
        ),
        .init(
            id: .glm52Q2K,
            displayName: "GLM 5.2 · Q2_K RoutedQ2K",
            profile: .glm52,
            summary: "Variante Q2_K monolitica a qualità e dimensione intermedie.",
            artifacts: [
                .init(
                    id: ModelCatalogID.glm52Q2K.rawValue,
                    file: "GLM-5.2-UD-Q2_K_RoutedQ2K.gguf",
                    approxGB: 262,
                    note: "GLM 5.2 Q2_K routed experts",
                    sha256: "b9fa49d010dad35b96418c45831c212a746715b0646c1121ccfc414455bd6fe5",
                    expectedSizeBytes: 262_036_650_048,
                    source: .glm52
                ),
            ],
            runtimeAvailability: unavailable
        ),
        .init(
            id: .glm52Q4K,
            displayName: "GLM 5.2 · Q4_K RoutedQ4K",
            profile: .glm52,
            summary: "Variante Q4_K monolitica; richiede oltre 434 GB di spazio più il margine di sicurezza.",
            artifacts: [
                .init(
                    id: ModelCatalogID.glm52Q4K.rawValue,
                    file: "GLM-5.2-UD-Q4_K_RoutedQ4K.gguf",
                    approxGB: 434,
                    note: "GLM 5.2 Q4_K routed experts",
                    sha256: "7160879c87756236eea16ec6bfeb19288d16fa94dcfcef3a5ed5f38b1383d3a5",
                    expectedSizeBytes: 434_170_886_208,
                    source: .glm52
                ),
            ],
            runtimeAvailability: unavailable
        ),
    ]
}

/// Cross-family source of truth rendered by the downloader UI.
public enum ModelCatalogRegistry {
    public static let entries: [ModelCatalogEntry] =
        DeepSeekV4ModelCatalog.entries + GLM52ModelCatalog.entries

    public static let selectableEntries: [ModelCatalogEntry] = entries.filter(\.isSelectable)
    public static let allArtifacts: [ModelTarget] = entries.flatMap(\.artifacts)

    public static func entry(_ id: ModelCatalogID) -> ModelCatalogEntry? {
        entries.first { $0.id == id }
    }

    public static func entry(_ id: String) -> ModelCatalogEntry? {
        ModelCatalogID(rawValue: id).flatMap(entry)
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
