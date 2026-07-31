// swift-tools-version: 6.0
import PackageDescription
import Foundation

// DwarfStar: a native Swift/SwiftUI front-end for local LLM backends. DeepSeek
// V4 Flash and the single-file Pro Q2 profile are operational locally through
// one geometry-driven backend; split Pro Q4 loading is not implemented. Qwen
// and Kimi K3 are recognized but deliberately not executable until their own
// tokenizer, tensor mapping and decoder exist. The engine is pure Swift
// (DS4Core + DS4Metal): no
// C engine, prebuilt static library
// or external process, so everything builds in SwiftPM and the standalone
// .xcodeproj. Run `swift build` or `make`.

/// Documentation is intentionally colocated with every source/test directory.
/// Keep the manifest warning-free without duplicating that growing list here:
/// SwiftPM receives paths relative to the target root, while XcodeGen applies
/// the equivalent recursive `**/*.md` exclusions from project.yml.
func markdownFiles(in relativeRoot: String, also extra: [String] = []) -> [String] {
    let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    let root = packageRoot.appendingPathComponent(relativeRoot).standardizedFileURL
    let prefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
    var paths = extra
    if let files = FileManager.default.enumerator(
        at: root,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles]
    ) {
        for case let file as URL in files where file.pathExtension.lowercased() == "md" {
            let path = file.standardizedFileURL.path
            guard path.hasPrefix(prefix) else { continue }
            paths.append(String(path.dropFirst(prefix.count)))
        }
    }
    return Array(Set(paths)).sorted()
}

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
        // Portable formats, architecture inspection and backend-owned model
        // contracts. It has no Metal, UI or application dependencies.
        .target(
            name: "DS4Core",
            exclude: markdownFiles(in: "Sources/DS4Core")
        ),

        // Swift Metal runtime plus concrete architecture backends. It compiles
        // the vendored metal/ kernels at runtime and links Metal.framework.
        .target(
            name: "DS4Metal",
            dependencies: ["DS4Core"],
            exclude: markdownFiles(in: "Sources/DS4Metal"),
            linkerSettings: [.linkedFramework("Metal")]
        ),

        // Unit tests for the pure-Swift engine modules (kernels, graph, GGUF,
        // tokenizer, sampler, format/serialization).
        .testTarget(
            name: "DS4CoreTests",
            dependencies: ["DS4Core", "DS4Metal", "DS4Engine"],
            exclude: markdownFiles(in: "Tests/DS4CoreTests")
        ),

        // Swift-native inference service backing the GUI. It inspects the GGUF,
        // selects one concrete backend once, then keeps dispatch out of hot loops.
        .target(
            name: "DS4Engine",
            dependencies: ["DS4Core", "DS4Metal"],
            exclude: markdownFiles(in: "Sources/DS4Engine")
        ),

        // The SwiftUI GUI app — driven by the pure-Swift engine (DS4Engine).
        // No C engine, no prebuilt static lib: builds in the standalone .xcodeproj.
        .executableTarget(
            name: "DwarfStar",
            dependencies: ["DS4Engine", "DS4Core"],
            // Assets.xcassets is the .app icon catalog: consumed by the xcodegen
            // .xcodeproj build, but SwiftPM has no asset-catalog compiler — exclude
            // it here so `swift build`/`swift test` don't warn about unhandled files.
            exclude: markdownFiles(in: "Sources/DwarfStar", also: ["Assets.xcassets"])
        ),

        // Pure-Swift engine demo CLI: drives DS4Core + DS4Metal directly (Metal
        // runtime self-test + optional GGUF streaming). NO C engine, NO external
        // links — this is the target the standalone .xcodeproj builds.
        .executableTarget(
            name: "DS4Demo",
            dependencies: ["DS4Core", "DS4Metal"],
            exclude: markdownFiles(in: "Sources/DS4Demo")
        ),
    ]
)
