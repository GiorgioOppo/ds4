import Foundation

/// Decompress a .gz file. Default replaces 'X.gz' with 'X'.
public struct GunzipTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "gunzip",
            description:
                "Decompress a .gz file. Default replaces 'X.gz' with 'X'; 'keep=true' keeps the original.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: ".gz file path, relative to agent root."),
                    "keep": SchemaBuilder.boolean(description: "Keep the .gz file. Default false.", defaultValue: false),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "gunzip \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let keep = input.optionalBool("keep") ?? false
        let url = try resolveInsideRoot(rel, context: context)
        var args: [String] = []
        if keep { args.append("-k") }
        args.append(url.path)
        return try await UnixBinary.runBinary(
            launchPath: "/usr/bin/gunzip",
            arguments: args,
            context: context)
    }
}
