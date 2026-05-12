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
        .executable(name: "deepseek", targets: ["deepseek"]),
        .executable(name: "converter", targets: ["converter"]),
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
        .target(
            name: "DeepSeekConverter",
            dependencies: ["DeepSeekKit"],
            path: "Sources/DeepSeekConverter"
        ),
        .executableTarget(
            name: "converter",
            dependencies: ["DeepSeekKit", "DeepSeekConverter"],
            path: "Sources/converter"
        ),
        .executableTarget(
            name: "DeepSeekUI",
            dependencies: ["DeepSeekKit"],
            path: "Sources/DeepSeekUI",
            exclude: ["Resources/Info.plist"],
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
    ]
)
