import Foundation

/// `swift build` — Swift Package Manager build. Picks up Package.swift
/// from the current directory (or 'packagePath').
public struct SwiftBuildTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "swift_build",
            description:
                "Build a Swift package (Package.swift). " +
                "'packagePath' defaults to the agent root; 'configuration' is debug or release. " +
                "Use 'target' / 'product' to scope the build.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "packagePath": SchemaBuilder.string(description: "Directory containing Package.swift. Default agent root."),
                    "configuration": SchemaBuilder.string(
                        description: "Build configuration. Default debug.",
                        enumValues: ["debug", "release"]),
                    "target": SchemaBuilder.string(description: "Build a specific target."),
                    "product": SchemaBuilder.string(description: "Build a specific product."),
                    "timeoutSeconds": SchemaBuilder.integer(description: "Timeout. Default 600.", minimum: 1),
                ]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "swift build \(input["target"] as? String ?? input["product"] as? String ?? "all")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let pkg = try resolveInsideRoot(input.optionalString("packagePath") ?? ".",
                                        context: context)
        var args: [String] = ["build"]
        if let cfg = input.optionalString("configuration") {
            args.append("-c"); args.append(cfg)
        }
        if let target = input.optionalString("target") {
            args.append("--target"); args.append(target)
        }
        if let product = input.optionalString("product") {
            args.append("--product"); args.append(product)
        }
        let timeout = TimeInterval(input.optionalInteger("timeoutSeconds") ?? 600)
        return try await Xcrun.run(tool: "swift",
                                   arguments: args,
                                   context: context,
                                   cwd: pkg,
                                   timeout: timeout)
    }
}
