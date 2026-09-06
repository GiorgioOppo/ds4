import DS4Core
import DS4Metal

/// DeepSeek V4 Flash Vision-Exp artifacts from the publisher's Hugging Face
/// manifest (`revision/f71f23d552d664e523b422157b2befbf74040380?blobs=true`).
/// The encoder and DSpark draft are companions, never language-model choices.
public enum DeepSeekV4VisionCatalog {
    public static let encoder = ModelTarget(
        id: ModelCatalogID.visionEncoder.rawValue,
        file: "DeepSeek-V4-Flash-Vision-Encoder.gguf",
        approxGB: 1,
        note: "Required image encoder for the Vision-Exp checkpoint",
        sha256: "00cd4d81a435364967400a95c42703343e11da6b6f18c5143fe76e1d94d5035f",
        expectedSizeBytes: 932_857_760,
        role: .optionalComponent,
        source: .deepSeekV4Vision
    )

    public static let dspark = ModelTarget(
        id: ModelCatalogID.visionDSparkSupport.rawValue,
        file: "DeepSeek-V4-Flash-Vision-Exp-DSpark-support.gguf",
        approxGB: 6,
        note: "Optional DSpark draft matching the Vision-Exp checkpoint",
        sha256: "0807a67fd9ce5874bfc60d8d2461f50e11657e3dd94913d3473f85aa679bc877",
        expectedSizeBytes: 5_989_114_528,
        role: .optionalComponent,
        source: .deepSeekV4Vision
    )

    private static let availability: ModelRuntimeAvailability =
        DeepSeekV4BackendDefinition.supportsLocalRuntime(.flash)
            ? .runnable
            : .downloadOnly(reason: "Il runtime DeepSeek V4 Flash non è disponibile in questa versione.")

    public static let entries: [ModelCatalogEntry] = [
        .init(
            id: .flashVisionQ2,
            displayName: "DeepSeek V4 Flash Vision · IQ2XXS",
            profile: .deepSeekV4(.flash),
            summary: "Checkpoint Vision-Exp per testo e immagini. La variante più compatta; per le immagini occorre anche l'encoder Vision da 933 MB.",
            artifacts: [.init(
                id: ModelCatalogID.flashVisionQ2.rawValue,
                file: "DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf",
                approxGB: 87,
                note: "Vision-Exp · 2-bit routed experts",
                sha256: "8f2d42c0071ccf8a98f391cc2b835fd123f12330690b3059dbb7707920e5ad9e",
                expectedSizeBytes: 86_720_111_776,
                source: .deepSeekV4Vision
            )],
            runtimeAvailability: availability
        ),
        .init(
            id: .flashVisionQ2Q4,
            displayName: "DeepSeek V4 Flash Vision · IQ2XXS/Q4_K",
            profile: .deepSeekV4(.flash),
            summary: "Checkpoint Vision-Exp con gli ultimi sei layer Q4 per maggiore qualità. Richiede l'encoder Vision separato per le immagini.",
            artifacts: [.init(
                id: ModelCatalogID.flashVisionQ2Q4.rawValue,
                file: "DeepSeek-V4-Flash-Vision-Exp-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf",
                approxGB: 98,
                note: "Vision-Exp · mixed IQ2XXS/Q4_K routed experts",
                sha256: "cded4517bb9d033e778e8bc4ccf1e79ba96d1c2d2b9f1c071c1d4a9037c51b02",
                expectedSizeBytes: 97_591_747_744,
                source: .deepSeekV4Vision
            )],
            runtimeAvailability: availability
        ),
        .init(
            id: .flashVisionMXFP4,
            displayName: "DeepSeek V4 Flash Vision · MXFP4",
            profile: .deepSeekV4(.flash),
            summary: "Checkpoint Vision-Exp con esperti MXFP4 nativi. Disponibile per il download in vista del supporto al formato nel runtime Swift/Metal.",
            artifacts: [.init(
                id: ModelCatalogID.flashVisionMXFP4.rawValue,
                file: "DeepSeek-V4-Flash-Vision-Exp-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out.gguf",
                approxGB: 156,
                note: "Vision-Exp · native MXFP4 routed experts",
                sha256: "fc1efb96fa26e654b3530ce5f4b926b189a936d41d94dc1903c832f1e18eb3e7",
                expectedSizeBytes: 155_976_459_136,
                source: .deepSeekV4Vision
            )],
            runtimeAvailability: .downloadOnly(
                reason: "Il formato GGUF MXFP4 (type 39) non è ancora eseguibile dal backend Swift/Metal."
            )
        ),
    ]

    public static let accessoryEntries: [ModelCatalogEntry] = [
        .init(
            id: .visionEncoder,
            displayName: "DeepSeek V4 Flash Vision · encoder immagini",
            profile: .deepSeekV4(.flash),
            summary: "Converte le immagini per i modelli Vision-Exp. Si scarica una sola volta e si abbina a qualsiasi quantizzazione Vision compatibile.",
            artifacts: [encoder],
            runtimeAvailability: .downloadOnly(
                reason: "Componente Vision: richiede un modello principale Flash Vision-Exp."
            )
        ),
        .init(
            id: .visionDSparkSupport,
            displayName: "DeepSeek V4 Flash Vision · supporto DSpark",
            profile: .deepSeekV4(.flash),
            summary: "Draft DSpark facoltativo specifico per Vision-Exp. Non usare i draft 0730 o 0731 con questo checkpoint.",
            artifacts: [dspark],
            runtimeAvailability: .downloadOnly(
                reason: "Solo download: la generazione speculativa DSpark non è ancora attiva per Vision-Exp."
            )
        ),
    ]
}
