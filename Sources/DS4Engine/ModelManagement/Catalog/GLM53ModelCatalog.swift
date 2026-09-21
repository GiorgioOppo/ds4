import Foundation

// Sources and immutable LFS manifests: docs/FOUR-MODEL-INTEGRATION-NOTES.md.
// Encoder sidecars live in the separate, download-only companion catalog.
extension HuggingFaceSource {
    public static let glm53 = Self(repository: "antirez/glm-5.3-flash-gguf",
        revision: "b2fa29d7a6b410db11221c904973967b80b760f5")
}

public enum GLM53ModelCatalog {
    public static let entries: [ModelCatalogEntry] = [
        .init(id: .glm53Q2, displayName: "GLM 5.3 Flash · Q2", profile: .glm53,
              summary: "96,5 GB su disco. Decoder KDA + attenzione sparsa con esperti selezionati mappati da SSD. Memoria e velocità dipendono da contesto e cache del sistema; nessuna misura completa su 32 GB. Supporto testuale.",
              artifacts: [.init(id: ModelCatalogID.glm53Q2.rawValue,
                  file: "GLM-5.3-Flash-Q2.gguf", approxGB: 97,
                  note: "Decoder Swift/Metal · solo testo",
                  sha256: "e81fd6241c6e55a64e1e14e47a3eab61a173fa8d7e4b5c1d1848827119705b32", expectedSizeBytes: 96505816384,
                  source: .glm53)],
              runtimeAvailability: .runnable),
        .init(id: .glm53Q4, displayName: "GLM 5.3 Flash · Q4_K", profile: .glm53,
              summary: "190,9 GB su disco; esperti Q4_K letti in gruppi limitati. La dimensione del file non coincide con la memoria residente, che include anche stato KDA e contesto. Supporto testuale.",
              artifacts: [.init(id: ModelCatalogID.glm53Q4.rawValue,
                  file: "GLM-5.3-Flash-Q4_K.gguf", approxGB: 191,
                  note: "Decoder Swift/Metal · solo testo",
                  sha256: "c7a0d950363238dd7804782c88340d737775aba53a15f8d4fdcc34e984f25221", expectedSizeBytes: 190875526464,
                  source: .glm53)],
              runtimeAvailability: .runnable),
    ]
}
