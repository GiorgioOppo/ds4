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
    ],
    targets: [
        .target(
            name: "DeepSeekKit",
            path: "Sources/DeepSeekKit",
            resources: [
                .process("Kernels")
            ]
        ),
        .executableTarget(
            name: "deepseek",
            dependencies: ["DeepSeekKit"],
            path: "Sources/deepseek"
        ),
    ]
)
