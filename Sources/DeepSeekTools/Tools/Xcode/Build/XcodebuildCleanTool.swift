import Foundation

/// `xcodebuild clean` — remove build artifacts for a scheme.
/// Lighter weight than `rm -rf DerivedData` because it only purges
/// the targets reachable from the requested scheme.
public struct XcodebuildCleanTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "xcodebuild_clean",
            description:
                "Remove build products and intermediates for a scheme. " +
                "Provide 'scheme' and 'workspace' or 'project'.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "scheme": SchemaBuilder.string(description: "Scheme name."),
                    "workspace": SchemaBuilder.string(description: ".xcworkspace, relative to agent root."),
                    "project": SchemaBuilder.string(description: ".xcodeproj, relative to agent root."),
                    "derivedDataPath": SchemaBuilder.string(description: "DerivedData dir, relative to agent root."),
                ],
                required: ["scheme"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "xcodebuild clean \(input["scheme"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let scheme = try input.string("scheme")
        var args: [String] = ["-scheme", scheme]
        if let ws = input.optionalString("workspace") {
            let url = try resolveInsideRoot(ws, context: context)
            args.append("-workspace"); args.append(url.path)
        } else if let proj = input.optionalString("project") {
            let url = try resolveInsideRoot(proj, context: context)
            args.append("-project"); args.append(url.path)
        }
        if let ddpRel = input.optionalString("derivedDataPath") {
            let ddp = try resolveInsideRoot(ddpRel, context: context)
            args.append("-derivedDataPath"); args.append(ddp.path)
        }
        args.append("clean")
        return try await Xcrun.run(tool: "xcodebuild",
                                   arguments: args,
                                   context: context,
                                   cwd: context.rootDirectory,
                                   timeout: 120)
    }
}
