import Foundation

/// Decompress a .gz file. Default replaces 'X.gz' with 'X'.
public struct GunzipTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "gunzip",
            description:
                "Decomprime un file .gz. Per default sostituisce 'X.gz' con 'X'; 'keep=true' mantiene l'originale.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Path del file .gz, relativo alla root dell'agente."),
                    "keep": SchemaBuilder.boolean(description: "Mantiene il file .gz. Default false.", defaultValue: false),
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
