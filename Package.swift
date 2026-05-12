// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "DeepSeekV4Pro",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "DeepSeekKit", targets: ["DeepSeekKit"]),
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
        .executableTarget(
            name: "converter",
            dependencies: ["DeepSeekKit"],
            path: "Sources/converter"
        ),
        .executableTarget(
            name: "DeepSeekUI",
            dependencies: ["DeepSeekKit"],
            path: "Sources/DeepSeekUI"
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
