import Foundation

/// Download catalog for the antirez Kimi K3 mixed IQ2_XXS/Q2_K publication.
///
/// The upstream object is one 858,760,729,248-byte GGUF split into five
/// consecutive fragments. These are not llama.cpp-style GGUF shards: part 1
/// owns the header and parts 2...5 continue its raw byte stream. The current
/// downloader keeps the fragments separate so acquisition needs only the
/// model's own disk footprint. A future Kimi reader can expose them as one
/// logical file without an equally large concatenated copy.
public enum KimiK3ModelCatalog {
    public static let reconstructedFile =
        "Kimi-K3-IQ2_XXS-Q2_K.gguf"
    public static let reconstructedSizeBytes: Int64 = 858_760_729_248
    public static let reconstructedSHA256 =
        "58d7624a421149e65b430c20f3fa388886f565ea1e5a3410adeb8ce2d62caa19"

    public static let entries: [ModelCatalogEntry] = [
        .init(
            id: .kimiK3IQ2XXSQ2K,
            displayName: "Kimi K3 · IQ2_XXS/Q2_K · 5 parti",
            profile: .kimiK3,
            summary: "GGUF Kimi K3 da 2,8T parametri diviso in cinque frammenti consecutivi. Il downloader riprende e verifica ogni parte; il runtime Kimi non è ancora implementato.",
            artifacts: fragments,
            runtimeAvailability: .downloadOnly(
                reason: "Download disponibile; tokenizer, lettore GGUF virtuale multi-parte e runtime Kimi K3 non sono ancora implementati."
            )
        ),
    ]

    public static let fragments: [ModelTarget] = [
        fragment(
            index: 1,
            bytes: 171_752_145_849,
            sha256: "2a9ce3dc3754c1766eaff124d1c39610f6870873b61c2fd276e398196cc91198"
        ),
        fragment(
            index: 2,
            bytes: 171_752_145_849,
            sha256: "fe087129a04e0674982d9fe19b86beee1bd9ffd00e139772ba7055251bafdd65"
        ),
        fragment(
            index: 3,
            bytes: 171_752_145_849,
            sha256: "956e997f5cf2756025eaf4839759b8062700bb6dd165a5aa9c3db29da1bca5f6"
        ),
        fragment(
            index: 4,
            bytes: 171_752_145_849,
            sha256: "5171d378ca3a09825dd754325fdd49fa9786b85dbd745fa11c2f61ce16e5da1a"
        ),
        fragment(
            index: 5,
            bytes: 171_752_145_852,
            sha256: "e67274e7d2e2c4536f05ca15e38f873e2ea3731606b27f1a33ccb35e34c2e7e7"
        ),
    ]

    private static func fragment(
        index: Int,
        bytes: Int64,
        sha256: String
    ) -> ModelTarget {
        let ordinal = String(format: "%02d", index)
        return ModelTarget(
            id: "kimi-k3-part-\(ordinal)-of-05",
            file: "Kimi-K3-IQ2_XXS-Q2_K.gguf.part-\(ordinal)-of-05",
            approxGB: Int((bytes + 999_999_999) / 1_000_000_000),
            note: "Kimi K3 raw GGUF fragment \(ordinal)/05",
            sha256: sha256,
            expectedSizeBytes: bytes,
            role: .splitFragment,
            source: .kimiK3
        )
    }
}
