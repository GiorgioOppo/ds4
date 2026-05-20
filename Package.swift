// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "DeepSeekV4Pro",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "DeepSeekKit", targets: ["DeepSeekKit"]),
        .library(name: "DeepSeekConverter", targets: ["DeepSeekConverter"]),
        .library(name: "DeepSeekTraining", targets: ["DeepSeekTraining"]),
        .library(name: "DeepSeekTools", targets: ["DeepSeekTools"]),
        .library(name: "DeepSeekIntegrations", targets: ["DeepSeekIntegrations"]),
        .library(name: "DeepSeekVocabPruner", targets: ["DeepSeekVocabPruner"]),
        .executable(name: "deepseek", targets: ["deepseek"]),
        .executable(name: "deepseek_gguf", targets: ["deepseek_gguf"]),
        .executable(name: "deepseek_calibrate", targets: ["deepseek_calibrate"]),
        .executable(name: "converter", targets: ["converter"]),
        .executable(name: "vocab_pruner", targets: ["vocab_pruner"]),
        .executable(name: "DeepSeekUI", targets: ["DeepSeekUI"]),
    ],
    targets: [
        .target(
            name: "DeepSeekKit",
            path: "Sources/DeepSeekKit",
            // Kernels is owned by MetalLibPlugin, which compiles the .metal
            // files into default.metallib and emits that as a resource.
            // Excluding the directory here prevents SwiftPM from also
            // copying the raw .metal sources into the bundle.
            exclude: ["Kernels"],
            plugins: [
                .plugin(name: "MetalLibPlugin"),
            ]
        ),
        .executableTarget(
            name: "deepseek",
            dependencies: ["DeepSeekKit"],
            path: "Sources/deepseek"
        ),
        // Minimal CLI for running a GGUF Llama-family checkpoint
        // (TODO §10.2 / T2). Kept separate from the V4-specific
        // `deepseek` CLI so the conversion-heavy / MLA-specific
        // machinery in main.swift doesn't bleed into the GGUF path.
        .executableTarget(
            name: "deepseek_gguf",
            dependencies: ["DeepSeekKit"],
            path: "Sources/deepseek_gguf"
        ),
        // Calibration runner for GPTQ / AWQ / SmoothQuant (TODO §1).
        // Loads a GGUF Llama, walks a calibration corpus, dumps
        // per-layer activation absmax/mean (always) and the
        // per-layer Hessian (with --collect-hessian) to disk. The
        // converter then consumes those when --quant-method is set.
        .executableTarget(
            name: "deepseek_calibrate",
            dependencies: ["DeepSeekKit"],
            path: "Sources/deepseek_calibrate"
        ),
        .target(
            name: "DeepSeekConverter",
            dependencies: ["DeepSeekKit"],
            path: "Sources/DeepSeekConverter"
        ),
        // Fine-tuning / retraining scaffold. Mirrors DeepSeekConverter
        // in shape (Spec + Progress + Runner + public entry-point
        // enum). The current runner is a stub — it validates the
        // spec and emits a plan, then throws `FineTuneNotImplemented`
        // because the Metal engine has no backward kernels yet. The
        // UI in DeepSeekUI binds to the same surface area so when a
        // real backend lands the swap is local to FineTuneRunner.
        .target(
            name: "DeepSeekTraining",
            path: "Sources/DeepSeekTraining"
        ),
        // Pure-Swift, model-agnostic toolbox: native code-agent tools
        // (read/write/edit/grep/glob/shell/apply_patch/...), permission
        // policy, plan/build agent modes, skill registry. Has no Metal
        // / DeepSeekKit dependency on purpose — the goal is that the
        // CLI, the UI, and a future headless server can all link the
        // same toolbox and observe identical semantics.
        .target(
            name: "DeepSeekTools",
            path: "Sources/DeepSeekTools"
        ),
        // Optional adapters that bridge DeepSeekTools (or the chat
        // engine) to external systems. Kept in a separate target so
        // the core build doesn't pull in their failure modes. Each
        // sub-folder is a scaffolded integration: see the integration
        // READMEs for the wiring status.
        .target(
            name: "DeepSeekIntegrations",
            dependencies: ["DeepSeekTools"],
            path: "Sources/DeepSeekIntegrations"
        ),
        .executableTarget(
            name: "converter",
            dependencies: ["DeepSeekKit", "DeepSeekConverter"],
            path: "Sources/converter"
        ),
        // Italiano-only (o multilingua latino) vocab pruner: legge un
        // checkpoint convertito + un corpus, calcola i token effettivamente
        // usati, e riscrive embed.weight + head.weight + tokenizer.json
        // + config.json con il vocab ridotto. Non tocca i pesi del
        // transformer e non richiede fine-tuning. Vedi
        // `docs/VOCAB-PRUNING.md`.
        .target(
            name: "DeepSeekVocabPruner",
            // DeepSeekConverter dep per riuso di `CancellationToken`
            // (ConversionProgress.swift) — stesso pattern condiviso
            // con i facade `Converter` / `FineTuner`.
            dependencies: ["DeepSeekKit", "DeepSeekConverter"],
            path: "Sources/DeepSeekVocabPruner"
        ),
        .executableTarget(
            name: "vocab_pruner",
            dependencies: ["DeepSeekKit", "DeepSeekConverter", "DeepSeekVocabPruner"],
            path: "Sources/vocab_pruner"
        ),
        .executableTarget(
            name: "DeepSeekUI",
            dependencies: [
                "DeepSeekKit",
                "DeepSeekConverter",
                "DeepSeekTraining",
                "DeepSeekTools",
                "DeepSeekIntegrations",
                "DeepSeekVocabPruner",
            ],
            path: "Sources/DeepSeekUI",
            // Info.plist is embedded into __TEXT/__info_plist via the
            // linker flags below — SwiftPM must not also try to copy
            // it as a bundle resource.
            //
            // Assets.xcassets is consumed only by the Xcode build
            // (`xcodegen` → DeepSeekV4Pro.xcodeproj → asset-catalog
            // compiler → Assets.car). SwiftPM has no asset-catalog
            // compiler, so excluding it keeps `swift build` clean and
            // avoids copying raw PNGs into the SPM bundle, where the
            // SwiftUI runtime can't find them anyway.
            // DeepSeekUI.entitlements is referenced by CODE_SIGN_ENTITLEMENTS
            // in project.yml and is meaningless to SPM.
            exclude: [
                "Resources/Info.plist",
                "Resources/Assets.xcassets",
                "Resources/DeepSeekUI.entitlements",
            ],
            // Embed Info.plist directly into the executable's __TEXT
            // section. This is the Apple-standard "non-bundle" trick:
            // macOS reads CFBundleIdentifier / CFBundleName etc. from
            // this section without needing a .app/Contents directory.
            // Silences the linkd.autoShortcut + Process Instance
            // Registry warnings at launch.
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "Sources/DeepSeekUI/Resources/Info.plist",
                ])
            ]
        ),
        .plugin(
            name: "MetalLibPlugin",
            capability: .buildTool(),
            path: "Plugins/MetalLibPlugin"
        ),
        .testTarget(
            name: "DeepSeekKitTests",
            dependencies: ["DeepSeekKit"],
            path: "Tests/DeepSeekKitTests"
        ),
        .testTarget(
            name: "DeepSeekToolsTests",
            dependencies: ["DeepSeekTools"],
            path: "Tests/DeepSeekToolsTests"
        ),
        .testTarget(
            name: "DeepSeekVocabPrunerTests",
            dependencies: ["DeepSeekVocabPruner", "DeepSeekKit"],
            path: "Tests/DeepSeekVocabPrunerTests"
        ),
        // Tests for DeepSeekUI's plain-Swift state types — the v2
        // chat storage Codable surface (`ChatManifest`,
        // `TurnSummary`, `StoredRound`) plus `ChatPersistence`'s
        // atomic-write helper and the round LRU. SwiftPM allows a
        // test target to `@testable import` an executable target
        // since 5.5, which is what lets us cover these without
        // promoting the types into DeepSeekKit.
        .testTarget(
            name: "DeepSeekUITests",
            dependencies: ["DeepSeekUI"],
            path: "Tests/DeepSeekUITests"
        ),
    ]
)
