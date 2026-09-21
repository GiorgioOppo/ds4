import Foundation

// Sources and immutable LFS manifests: docs/FOUR-MODEL-INTEGRATION-NOTES.md.
// Encoder sidecars live in the separate, download-only companion catalog.
extension HuggingFaceSource {
    public static let bonsai2 = Self(repository: "prism-ml/Ternary-Bonsai-2-27B-gguf",
        revision: "6ed5e12bf84b7a63069882c91dd9e9218647d17b")
}

public enum BonsaiModelCatalog {
    public static let entries: [ModelCatalogEntry] = [
        .init(id: .bonsai2PQ2, displayName: "Ternary Bonsai 2 27B · PQ2_0", profile: .bonsai2,
              summary: "7,2 GB su disco; pesi ternari folded e contesto aggiuntivo in memoria. È la variante più leggera tra le quattro nuove famiglie. Supporto testuale.",
              artifacts: [.init(id: ModelCatalogID.bonsai2PQ2.rawValue,
                  file: "Ternary-Bonsai-2-27B-PQ2_0.gguf", approxGB: 7,
                  note: "Decoder Swift/Metal · solo testo",
                  sha256: "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1", expectedSizeBytes: 7206168928,
                  source: .bonsai2)],
              runtimeAvailability: .runnable),
        .init(id: .bonsai2PTQ1, displayName: "Ternary Bonsai 2 27B · PTQ1_0", profile: .bonsai2,
              summary: "5,9 GB su disco; codifica ternaria compatta PTQ1_0. Contesto e stato ricorrente richiedono memoria aggiuntiva. Supporto testuale.",
              artifacts: [.init(id: ModelCatalogID.bonsai2PTQ1.rawValue,
                  file: "Ternary-Bonsai-2-27B-PTQ1_0.gguf", approxGB: 6,
                  note: "Decoder Swift/Metal · solo testo",
                  sha256: "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3", expectedSizeBytes: 5946648928,
                  source: .bonsai2)],
              runtimeAvailability: .runnable),
    ]
}
