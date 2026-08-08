import Foundation
import DS4Core
import DS4Metal

/// Stable identifiers exposed to the GUI and persisted by callers.
public enum ModelCatalogID: String, CaseIterable, Identifiable, Sendable, Hashable {
    case flashQ2Imatrix = "q2-imatrix"
    case flashQ2Imatrix0731 = "q2-imatrix-0731"
    case flashQ2Q4Imatrix = "q2-q4-imatrix"
    case flashQ2Q4Imatrix0731 = "q2-q4-imatrix-0731"
    case flashQ4Imatrix = "q4-imatrix"
    case flashQ4Imatrix0731 = "q4-imatrix-0731"
    case flashMXFP40731 = "mxfp4-0731"
    case proQ2Imatrix = "pro-q2-imatrix"
    case proQ4Split = "pro-q4-split"
    case glm52IQ2XXS = "glm-5.2-iq2-xxs"
    case glm52Q2K = "glm-5.2-q2-k"
    case glm52Q4K = "glm-5.2-q4-k"
    case lagunaQ4 = "laguna-q4"
    case lagunaQ2Q3 = "laguna-q2-q3"
    case kimiK3IQ2XXSQ2K = "kimi-k3-iq2-xxs-q2-k"
    case dsparkSupport = "dspark-support"
    case dsparkSupport0731 = "dspark-support-0731"

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
    case laguna
    case kimiK3

    public var familyDisplayName: String {
        switch self {
        case .deepSeekV4: "DeepSeek V4"
        case .glm52: "GLM 5.2"
        case .laguna: "Laguna S 2.1"
        case .kimiK3: "Kimi K3"
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

    /// Official Poolside publication; the revision is the commit pinned by the
    /// upstream `download_model.sh` (`LAGUNA_REVISION`). The repository may be
    /// gated on Hugging Face, so downloads can require a saved HF token.
    public static let lagunaPoolside = Self(
        repository: "poolside/Laguna-S-2.1-GGUF",
        revision: "706fa69799926b6afde1af9e24ca2a4923f110a1"
    )

    /// Requantized companions published by antirez (mixed Q2_K/Q3_K main model
    /// and the Q8_0 DFlash draft). Upstream does not pin a revision for this
    /// repository yet; pin one here as soon as the artifacts are frozen.
    public static let lagunaAntirez = Self(
        repository: "antirez/Laguna-S-2.1-GGUF"
    )

    /// Five consecutive byte ranges of one GGUF. The commit, fragment sizes
    /// and LFS digests are pinned from the Hugging Face API; changing any part
    /// therefore creates a new catalog revision instead of silently mixing
    /// bytes from different uploads.
    public static let kimiK3 = Self(
        repository: "antirez/kimi-k3-gguf",
        revision: "b06d5a51043f84bce980cc31cf63ac645efb3a76"
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
    /// Consecutive byte fragment of one logical GGUF, not an independent GGUF
    /// shard. Keep every fragment in catalog order for the future virtual file
    /// reader; concatenating today would temporarily duplicate ~859 GB.
    case splitFragment
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

    /// Raw consecutive pieces of one GGUF. They are deliberately kept
    /// separate on disk until the virtual multi-part reader exists.
    public var isSplitFragmentPackage: Bool {
        artifacts.count > 1
            && artifacts.allSatisfy { $0.role == .splitFragment }
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
            displayName: "DeepSeek V4 Flash 0730 · IQ2XXS",
            profile: .deepSeekV4(.flash),
            summary: "Checkpoint 0730. Quantizzazione più compatta, consigliata per Mac con 96–128 GB di memoria.",
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
            id: .flashQ2Imatrix0731,
            displayName: "DeepSeek V4 Flash 0731 · IQ2XXS",
            profile: .deepSeekV4(.flash),
            summary: "Checkpoint 0731 aggiornato, compatto e compatibile con il supporto DSpark 0731.",
            artifacts: [
                .init(
                    id: ModelCatalogID.flashQ2Imatrix0731.rawValue,
                    file: "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf",
                    approxGB: 87,
                    note: "0731 · 2-bit routed experts",
                    sha256: "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0",
                    expectedSizeBytes: 86_720_111_488
                ),
            ],
            runtimeAvailability: singleFileAvailability(for: .flash)
        ),
        .init(
            id: .flashQ2Q4Imatrix,
            displayName: "DeepSeek V4 Flash 0730 · IQ2XXS/Q4_K",
            profile: .deepSeekV4(.flash),
            summary: "Checkpoint 0730. Gli ultimi sei layer usano esperti Q4 per una qualità maggiore.",
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
            id: .flashQ2Q4Imatrix0731,
            displayName: "DeepSeek V4 Flash 0731 · IQ2XXS/Q4_K",
            profile: .deepSeekV4(.flash),
            summary: "Checkpoint 0731 misto: esperti Q4 negli ultimi sei layer e IQ2XXS/Q2_K negli altri.",
            artifacts: [
                .init(
                    id: ModelCatalogID.flashQ2Q4Imatrix0731.rawValue,
                    file: "DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf",
                    approxGB: 98,
                    note: "0731 · mixed q2/q4 routed experts",
                    sha256: "659e22fbd01c9e13ea37a57c8d9c41e0a8819dffa3473d3c5286ee44b2d3398f",
                    expectedSizeBytes: 97_591_747_456
                ),
            ],
            runtimeAvailability: singleFileAvailability(for: .flash)
        ),
        .init(
            id: .flashQ4Imatrix,
            displayName: "DeepSeek V4 Flash 0730 · Q4_K",
            profile: .deepSeekV4(.flash),
            summary: "Checkpoint 0730. Quantizzazione a qualità più alta, consigliata per Mac con almeno 256 GB di memoria.",
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
            id: .flashQ4Imatrix0731,
            displayName: "DeepSeek V4 Flash 0731 · Q4_K",
            profile: .deepSeekV4(.flash),
            summary: "Checkpoint 0731 con tutti gli esperti routed Q4_K; richiede molto spazio e memoria.",
            artifacts: [
                .init(
                    id: ModelCatalogID.flashQ4Imatrix0731.rawValue,
                    file: "DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf",
                    approxGB: 165,
                    note: "0731 · 4-bit routed experts",
                    sha256: "6bb77b5ddcbc2d974c687cfb63d644ecfb295581b4a53fa4c1d810aea538254a",
                    expectedSizeBytes: 164_633_502_592
                ),
            ],
            runtimeAvailability: singleFileAvailability(for: .flash)
        ),
        .init(
            id: .flashMXFP40731,
            displayName: "DeepSeek V4 Flash 0731 · MXFP4",
            profile: .deepSeekV4(.flash),
            summary: "Pesi esperti MXFP4 nativi del checkpoint 0731. Scaricabile per ds4 C; il kernel MXFP4 non è ancora disponibile nel runtime Swift/Metal.",
            artifacts: [
                .init(
                    id: ModelCatalogID.flashMXFP40731.rawValue,
                    file: "DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf",
                    approxGB: 156,
                    note: "0731 · native MXFP4 routed experts",
                    sha256: "0e3a161b670f686128ec5f92a601dfde616a37bf5e7e48999fa2d32471b57ec6",
                    expectedSizeBytes: 155_976_458_848
                ),
            ],
            runtimeAvailability: .downloadOnly(
                reason: "Il formato GGUF MXFP4 (type 39) non è ancora eseguibile dal backend Swift/Metal."
            )
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
/// Availability is keyed off `GLM52RuntimeGate.enabled`: with the gate on,
/// the entries are `runnable` and selectable through the GLM streaming
/// backend; with the gate off they fall back to download-only.
public enum GLM52ModelCatalog {
    // Keyed off the single enablement switch: selectable the moment the
    // real-GGUF logits parity gate flips GLM52RuntimeGate.enabled.
    private static let unavailable: ModelRuntimeAvailability =
        GLM52RuntimeGate.enabled
            ? .runnable
            : .downloadOnly(
                reason: "Download disponibile; il runtime GLM 5.2 non è ancora abilitato (gate di parità logits)."
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

/// Download catalog for the Laguna S 2.1 GGUF publications.
///
/// Availability is keyed off `LagunaRuntimeGate.enabled`: with the gate on,
/// the entries become `runnable` and selectable through the future Laguna
/// backend; with the gate off they are download-only, exactly like GLM 5.2
/// before its runtime was enabled. Byte counts and checksums are not pinned
/// yet because this environment cannot reach Hugging Face; sizes follow the
/// upstream README (63.56 GiB and 44.95 GiB).
public enum LagunaModelCatalog {
    private static let unavailable: ModelRuntimeAvailability =
        LagunaRuntimeGate.enabled
            ? .runnable
            : .downloadOnly(
                reason: "Download disponibile; il runtime Laguna S 2.1 non è ancora abilitato (gate di parità logits del decoder — vedi docs/PORTING-GAPS, Gap 4)."
            )

    public static let entries: [ModelCatalogEntry] = [
        .init(
            id: .lagunaQ4,
            displayName: "Laguna S 2.1 · Q4_K_M",
            profile: .laguna,
            summary: "GGUF ufficiale Poolside quantizzato con imatrix: esperti instradati Q4_K e pesi signal-path Q8_0. Pensato per residenza completa su macchine con almeno 96 GB di memoria unificata.",
            artifacts: [
                .init(
                    id: ModelCatalogID.lagunaQ4.rawValue,
                    file: "laguna-s-2.1-Q4_K_M.gguf",
                    approxGB: 68,
                    note: "official Poolside Q4_K_M, Q8_0 signal path",
                    source: .lagunaPoolside
                ),
            ],
            runtimeAvailability: unavailable
        ),
        .init(
            id: .lagunaQ2Q3,
            displayName: "Laguna S 2.1 · Q2_K/Q3_K misti",
            profile: .laguna,
            summary: "Quantizzazione mista per sistemi in classe 64 GB: pesi densi del file ufficiale, esperti instradati Q2_K nei layer 1–20 e Q3_K nei layer 21–47.",
            artifacts: [
                .init(
                    id: ModelCatalogID.lagunaQ2Q3.rawValue,
                    file: "laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf",
                    approxGB: 48,
                    note: "mixed Q2_K/Q3_K routed experts",
                    source: .lagunaAntirez
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
            + LagunaModelCatalog.entries + KimiK3ModelCatalog.entries

    public static let selectableEntries: [ModelCatalogEntry] = entries.filter(\.isSelectable)
    public static let allArtifacts: [ModelTarget] = entries.flatMap(\.artifacts)

    /// Entries rendered by the downloader. Optional support models belong
    /// here, but never enter `entries`/`selectableEntries`, so downloading a
    /// DSpark sidecar cannot accidentally replace the active main model.
    public static let downloadEntries: [ModelCatalogEntry] =
        entries + DeepSeekV4AccessoryCatalog.downloadEntries

    public static func entry(_ id: ModelCatalogID) -> ModelCatalogEntry? {
        entries.first { $0.id == id }
    }

    public static func entry(_ id: String) -> ModelCatalogEntry? {
        ModelCatalogID(rawValue: id).flatMap(entry)
    }

    public static func downloadEntry(_ id: ModelCatalogID) -> ModelCatalogEntry? {
        downloadEntries.first { $0.id == id }
    }

    public static func downloadEntry(_ id: String) -> ModelCatalogEntry? {
        ModelCatalogID(rawValue: id).flatMap(downloadEntry)
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

    /// DeepSeek V4 Flash 0730 checkpoint support model. The upstream artifact
    /// predates the suffixed 0731 publication and keeps the unsuffixed name.
    public static let dspark = ModelTarget(
        id: ModelCatalogID.dsparkSupport.rawValue,
        file: "DeepSeek-V4-Flash-DSpark-support.gguf",
        approxGB: 6,
        note: "DSpark support model for the Flash 0730 checkpoint",
        sha256: "8b3adf5942bec22ae2ea867cd7079cf13530ba83ffcffaf00f5de48664a1a34e",
        expectedSizeBytes: 5_989_114_272,
        role: .optionalComponent
    )

    /// Checkpoint-specific support model for the 0731 refresh. Digest and exact
    /// size are pinned from the official Hugging Face manifest.
    public static let dspark0731 = ModelTarget(
        id: ModelCatalogID.dsparkSupport0731.rawValue,
        file: "DeepSeek-V4-Flash-DSpark-support-0731.gguf",
        approxGB: 6,
        note: "DSpark support model for the Flash 0731 checkpoint",
        sha256: "7e319924541db3f7a163ed7e11d7532a70d48228ab59d36cb81e1d4511885360",
        expectedSizeBytes: 5_989_114_272,
        role: .optionalComponent
    )

    public static let downloadEntries: [ModelCatalogEntry] = [
        .init(
            id: .dsparkSupport,
            displayName: "DeepSeek V4 Flash 0730 · supporto DSpark",
            profile: .deepSeekV4(.flash),
            summary: "Modello ausiliario DSpark per il checkpoint Flash 0730. Propone fino a cinque token; non è un modello autonomo.",
            artifacts: [dspark],
            runtimeAvailability: .downloadOnly(
                reason: "Accessorio DSpark eseguibile dal runtime Swift/Metal; richiede il modello Flash 0730 abbinato."
            )
        ),
        .init(
            id: .dsparkSupport0731,
            displayName: "DeepSeek V4 Flash 0731 · supporto DSpark",
            profile: .deepSeekV4(.flash),
            summary: "Supporto DSpark specifico per il checkpoint Flash 0731; non va usato con il checkpoint 0730.",
            artifacts: [dspark0731],
            runtimeAvailability: .downloadOnly(
                reason: "Accessorio DSpark eseguibile dal runtime Swift/Metal; richiede il checkpoint Flash 0731 abbinato."
            )
        ),
    ]
}

/// Optional Laguna companions, excluded from the main-model catalog and from
/// GUI selection exactly like the DeepSeek MTP sidecar. The DFlash draft
/// model accelerates greedy Laguna decoding without changing the generated
/// tokens; it becomes useful only when the Laguna decoder and its DFlash path
/// are ported.
public enum LagunaAccessoryCatalog {
    public static let dflash = ModelTarget(
        id: "laguna-dflash",
        file: "laguna-s-2.1-DFlash-Q8_0.gguf",
        approxGB: 1,
        note: "optional DFlash speculative-decoding draft (Q8_0)",
        role: .optionalComponent,
        source: .lagunaAntirez
    )
}
