import Foundation

/// Compress a file with gzip. Default in-place (replaces 'file' with
/// 'file.gz'); set 'keep=true' to retain the original.
public struct GzipTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "gzip",
            description:
                "Compress a file with gzip. Default replaces 'X' with 'X.gz'; " +
                "'keep=true' keeps the original alongside.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "File path, relative to agent root."),
                    "keep": SchemaBuilder.boolean(description: "Keep the original file. Default false.", defaultValue: false),
                    "level": SchemaBuilder.integer(description: "Compression level 1-9. Default 6.", minimum: 1),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "gzip \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let keep = input.optionalBool("keep") ?? false
        let level = input.optionalInteger("level") ?? 6
        if level < 1 || level > 9 {
            throw ToolError.invalidInput("level must be 1-9")
        }
        let url = try resolveInsideRoot(rel, context: context)
        var args: [String] = ["-\(level)"]
        if keep { args.append("-k") }
        args.append(url.path)
        return try await UnixBinary.runBinary(
            launchPath: "/usr/bin/gzip",
            arguments: args,
            context: context)
    }
}
