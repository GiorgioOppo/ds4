import Foundation

/// `git log` with a default short format so a single call stays inside
/// the 32 KB output cap. The model can override via 'n' (commit count)
/// and 'path' (limit to one path).
public struct GitLogTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "git_log",
            description:
                "Show commit history of the agent root. Default --oneline -n 20 keeps output bounded. " +
                "Pass 'path' to scope to one file/directory.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "n": SchemaBuilder.integer(description: "Number of commits. Default 20.", minimum: 1),
                    "path": SchemaBuilder.string(description: "Scope to one path, relative to agent root."),
                    "since": SchemaBuilder.string(description: "Only commits newer than this (e.g. '2 weeks ago')."),
                    "author": SchemaBuilder.string(description: "Match commit author."),
                ]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let n = input.optionalInteger("n") ?? 20
        var args: [String] = ["log", "--oneline", "-n", "\(n)"]
        if let since = input.optionalString("since") {
            args.append("--since=\(since)")
        }
        if let author = input.optionalString("author") {
            args.append("--author=\(author)")
        }
        if let pathRel = input.optionalString("path") {
            let url = try resolveInsideRoot(pathRel, context: context)
            args.append("--"); args.append(url.path)
        }
        return try await UnixBinary.runBinary(
            launchPath: "/usr/bin/git",
            arguments: args,
            context: context,
            cwd: context.rootDirectory)
    }
}
