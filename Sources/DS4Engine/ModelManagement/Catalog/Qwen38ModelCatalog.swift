import Foundation

// Sources and immutable LFS manifests: docs/FOUR-MODEL-INTEGRATION-NOTES.md.
// Encoder sidecars live in the separate, download-only companion catalog.
extension HuggingFaceSource {
    public static let qwen38 = Self(repository: "antirez/qwen3.8-flash-next-gguf",
        revision: "d600fe1a43d2e1cdcadb85144ce3142f66f9eefe")
}

public enum Qwen38ModelCatalog {
    public static let entries: [ModelCatalogEntry] = [
        .init(id: .qwen38Q2, displayName: "Qwen 3.8 Flash Next · Q2", profile: .qwen38,
              summary: "147,2 GB su disco, inclusa la tabella n-gram BF16 letta da SSD; il solo modello principale è circa 41,7 GiB. Streaming degli esperti; prestazioni e memoria su 32 GB non ancora misurate. Supporto testuale.",
              artifacts: [.init(id: ModelCatalogID.qwen38Q2.rawValue,
                  file: "Qwen3.8-Flash-Next-Q2.gguf", approxGB: 147,
                  note: "Decoder Swift/Metal · solo testo",
                  sha256: "b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9", expectedSizeBytes: 147207127040,
                  source: .qwen38)],
              runtimeAvailability: .runnable),
        .init(id: .qwen38Q4, displayName: "Qwen 3.8 Flash Next · Q4", profile: .qwen38,
              summary: "177,3 GB su disco; circa 69,7 GiB di pesi principali più tabella n-gram BF16 su SSD. La dimensione su disco non stima la memoria residente. Supporto testuale.",
              artifacts: [.init(id: ModelCatalogID.qwen38Q4.rawValue,
                  file: "Qwen3.8-Flash-Next-Q4.gguf", approxGB: 177,
                  note: "Decoder Swift/Metal · solo testo",
                  sha256: "680944460a8cbe93ba8b6d7b6107213ffb7e22320bd913000e563ca0a0f25a8a", expectedSizeBytes: 177280286720,
                  source: .qwen38)],
              runtimeAvailability: .runnable),
    ]
}
