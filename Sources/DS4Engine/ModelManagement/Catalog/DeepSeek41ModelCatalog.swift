import Foundation

// Sources and immutable LFS manifests: docs/FOUR-MODEL-INTEGRATION-NOTES.md.
// Encoder sidecars live in the separate, download-only companion catalog.
extension HuggingFaceSource {
    public static let deepSeek41 = Self(repository: "antirez/deepseek-v4.1-flash-gguf",
        revision: "dd8a266f7145edc19e2334b46e19b6821f221dc7")
}

public enum DeepSeek41ModelCatalog {
    public static let q4Output = ModelTarget(id: ModelCatalogID.deepSeek41Q4.rawValue,
        file: "DeepSeek-V4.1-Flash-Q4.gguf", approxGB: 519,
        note: "GGUF assemblato localmente e verificato SHA-256",
        sha256: "a5e2e2c3ada4b2e98d9f9e4b50f6d9c2a12c2c96f5da165c07e13aff9264984e",
        expectedSizeBytes: 518_596_067_328, source: .deepSeek41)

    public static let entries: [ModelCatalogEntry] = [
        .init(id: .deepSeek41Q2, displayName: "DeepSeek V4.1 Flash · Q2", profile: .deepSeek41,
              summary: "365,7 GB su disco (circa 152 GiB di modello principale e 189 GiB Engram). Gli esperti e le lookup Engram usano SSD; nessuna garanzia di prestazioni su 32 GB. Supporto testuale, decoder V4.1 dedicato.",
              artifacts: [.init(id: ModelCatalogID.deepSeek41Q2.rawValue,
                  file: "DeepSeek-V4.1-Flash-Q2.gguf", approxGB: 366,
                  note: "Decoder Swift/Metal · solo testo",
                  sha256: "1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42", expectedSizeBytes: 365713686528,
                  source: .deepSeek41)],
              runtimeAvailability: .runnable),
        .init(id: .deepSeek41Q4, displayName: "DeepSeek V4.1 Flash · Q4 (2 parti)", profile: .deepSeek41,
              summary: "518,6 GB da scaricare in due frammenti. Assemblaggio locale in un solo GGUF con verifica SHA-256. Le parti restano sul disco: picco circa 1,04 TB più riserva. Supporto testuale.",
              artifacts: [
                .init(id: "deepseek-4.1-q4-part1", file: "DeepSeek-V4.1-Flash-Q4.gguf.part1", approxGB: 480,
                      note: "Frammento 1 di 2 · non caricabile separatamente",
                      sha256: "6442b1f9224079662c02003c0ef9ef6be6e2aff509510f681dab9e6cc41df246",
                      expectedSizeBytes: 480_000_000_000, role: .splitFragment, source: .deepSeek41),
                .init(id: "deepseek-4.1-q4-part2", file: "DeepSeek-V4.1-Flash-Q4.gguf.part2", approxGB: 39,
                      note: "Frammento 2 di 2 · non caricabile separatamente",
                      sha256: "7c3e10646c918eeaffbc39305a75ec96117450262c61454ff194cef00d7617f0",
                      expectedSizeBytes: 38_596_067_328, role: .splitFragment, source: .deepSeek41)
              ],
              runtimeAvailability: .runnable,
              assemblyOutput: q4Output),
    ]
}
