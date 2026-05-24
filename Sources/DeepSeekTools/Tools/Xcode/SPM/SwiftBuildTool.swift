import Foundation

/// `swift build` — Swift Package Manager build. Picks up Package.swift
/// from the current directory (or 'packagePath').
public struct SwiftBuildTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "swift_build",
            description:
                "Compila un pacchetto Swift Package Manager (la directory DEVE contenere un Package.swift). " +
                "Usalo per librerie SPM, CLI tool, Swift lato server e qualsiasi pacchetto che non abbia " +
                "un .xcodeproj/.xcworkspace. " +
                "Per un progetto Xcode (target di app iOS / macOS / visionOS) usa invece 'xcodebuild_build'. " +
                "'packagePath' è per default la root dell'agente; 'configuration' è debug o release; " +
                "'target' / 'product' restringe la build.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "packagePath": SchemaBuilder.string(description: "Directory contenente Package.swift. Default: root dell'agente."),
                    "configuration": SchemaBuilder.string(
                        description: "Configurazione di build. Default debug.",
                        enumValues: ["debug", "release"]),
                    "target": SchemaBuilder.string(description: "Compila uno specifico target."),
                    "product": SchemaBuilder.string(description: "Compila uno specifico product."),
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
