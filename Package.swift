// swift-tools-version: 6.0
import PackageDescription

// Shared link flags for any executable that drives the in-process engine: the
// prebuilt static library (built by the Makefile) plus its frameworks. Paths are
// relative to the package root (the build cwd).
let engineLinkerSettings: [LinkerSetting] = [
    .unsafeFlags(["-Lenginelib", "-lDS4Engine"]),
    .linkedLibrary("m"),
    .linkedFramework("Metal"),
    .linkedFramework("Foundation"),
]

// DS4-gui: a native Swift/SwiftUI front-end for the existing ds4 (DwarfStar)
// DeepSeek V4 inference engine. The C/Objective-C engine is left UNCHANGED and
// is consumed in-process through its narrow public boundary (ds4.h).
//
// The engine itself is compiled separately into enginelib/libDS4Engine.a by the
// Makefile in this folder (it reuses the upstream compiler flags so inference is
// byte-identical to ./ds4). Run `make` here, or `make engine` then `swift build`.

let package = Package(
    name: "DS4Gui",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "DS4Kit", targets: ["DS4Kit"]),
        .library(name: "DS4Core", targets: ["DS4Core"]),
        .library(name: "DS4Metal", targets: ["DS4Metal"]),
        .library(name: "DS4Engine", targets: ["DS4Engine"]),
        .executable(name: "DwarfStar", targets: ["DwarfStar"]),
        .executable(name: "ds4gui-smoke", targets: ["ds4gui-smoke"]),
        // Pure-Swift engine demo CLI — NO external links (no C engine).
        .executable(name: "DS4Demo", targets: ["DS4Demo"]),
    ],
    targets: [
        // C module exposing the engine's public header (ds4.h) plus a tiny
        // GUI-side shim. The implementation lives in the prebuilt static lib.
        .target(
            name: "CDS4",
            publicHeadersPath: "include"
        ),

        // Pure-Swift reimplementation of the ds4 engine, built up module by
        // module (see the C->Swift conversion phases). No dependency on CDS4 or
        // the C engine: this is the code that will eventually replace it.
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

        // Validates each converted module against its C original (via CDS4),
        // so the port stays faithful. Links the engine static lib for the C side.
        .testTarget(
            name: "DS4CoreTests",
            dependencies: ["DS4Core", "DS4Metal", "DS4Engine", "CDS4"],
            linkerSettings: engineLinkerSettings
        ),

        // Idiomatic Swift bridge over the C engine: actor-serialized access,
        // streaming generation, Swift value types for options and tokens.
        .target(
            name: "DS4Kit",
            dependencies: ["CDS4"]
        ),

        // Swift-native inference service backing the GUI: pure-Swift engine
        // (DS4Core tokenizer/GGUF + DS4Metal StreamingDecoder). Replaces the C
        // bridge (DS4Kit) as the app's inference path — NO external links.
        .target(
            name: "DS4Engine",
            dependencies: ["DS4Core", "DS4Metal"]
        ),

        // The SwiftUI GUI app — now driven by the pure-Swift engine (DS4Engine).
        // No C engine, no prebuilt static lib: builds in the standalone .xcodeproj.
        .executableTarget(
            name: "DwarfStar",
            dependencies: ["DS4Engine"]
        ),

        // Phase 0 smoke test: opens the engine and runs one short generation.
        .executableTarget(
            name: "ds4gui-smoke",
            dependencies: ["DS4Kit"],
            linkerSettings: engineLinkerSettings
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
