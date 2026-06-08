// swift-tools-version: 6.0
import PackageDescription

// DS4-gui: a native Swift/SwiftUI front-end for DeepSeek V4 (DwarfStar). The
// engine is a pure-Swift reimplementation (DS4Core + DS4Metal): no C engine, no
// prebuilt static lib, no external links — so everything builds in a clean
// SwiftPM package and the standalone .xcodeproj. Run `swift build` or `make`.

let package = Package(
    name: "DS4Gui",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "DS4Core", targets: ["DS4Core"]),
        .library(name: "DS4Metal", targets: ["DS4Metal"]),
        .library(name: "DS4Engine", targets: ["DS4Engine"]),
        .executable(name: "DwarfStar", targets: ["DwarfStar"]),
        // Pure-Swift engine demo CLI — NO external links (no C engine).
        .executable(name: "DS4Demo", targets: ["DS4Demo"]),
    ],
    targets: [
        // Pure-Swift reimplementation of the ds4 engine, built up module by
        // module (see the C->Swift conversion phases).
        .target(
            name: "DS4Core"
        ),

        // Swift Metal runtime (Phase 8+): compiles the vendored metal/ kernels
        // at runtime and dispatches them. Links the Metal framework.
        .target(
            name: "DS4Metal",
            dependencies: ["DS4Core"],
            linkerSettings: [.linkedFramework("Metal")]
        ),

        // Unit tests for the pure-Swift engine modules (kernels, graph, GGUF,
        // tokenizer, sampler, format/serialization).
        .testTarget(
            name: "DS4CoreTests",
            dependencies: ["DS4Core", "DS4Metal", "DS4Engine"]
        ),

        // Swift-native inference service backing the GUI: pure-Swift engine
        // (DS4Core tokenizer/GGUF + DS4Metal StreamingDecoder) — NO external links.
        .target(
            name: "DS4Engine",
            dependencies: ["DS4Core", "DS4Metal"]
        ),

        // The SwiftUI GUI app — driven by the pure-Swift engine (DS4Engine).
        // No C engine, no prebuilt static lib: builds in the standalone .xcodeproj.
        .executableTarget(
            name: "DwarfStar",
            dependencies: ["DS4Engine"]
        ),

        // Pure-Swift engine demo CLI: drives DS4Core + DS4Metal directly (Metal
        // runtime self-test + optional GGUF streaming). NO C engine, NO external
        // links — this is the target the standalone .xcodeproj builds.
        .executableTarget(
            name: "DS4Demo",
            dependencies: ["DS4Core", "DS4Metal"]
        ),
    ]
)
