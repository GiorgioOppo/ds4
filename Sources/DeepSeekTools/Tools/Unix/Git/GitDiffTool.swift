import Foundation

/// `git diff` against working tree, index, or a specific revision.
public struct GitDiffTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "git_diff",
            description:
                "Show a diff. mode='working' (default, vs index), 'staged' (index vs HEAD), " +
                "or 'commit' (one commit's diff, requires 'rev'). 'path' scopes to one file/directory.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "mode": SchemaBuilder.string(
                        description: "Diff source. Default 'working'.",
                        enumValues: ["working", "staged", "commit"]),
                    "rev": SchemaBuilder.string(description: "Revision (for mode='commit')."),
                    "path": SchemaBuilder.string(description: "Scope to one path, relative to agent root."),
                    "statOnly": SchemaBuilder.boolean(description: "Show --stat instead of the patch. Default false.", defaultValue: false),
                ]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let mode = input.optionalString("mode") ?? "working"
        let statOnly = input.optionalBool("statOnly") ?? false
        var args: [String] = ["diff"]
        switch mode {
        case "staged":
            args.append("--cached")
        case "commit":
            guard let rev = input.optionalString("rev") else {
                throw ToolError.invalidInput("mode='commit' requires 'rev'")
            }
            // Reject obvious shell metacharacters even though we don't
            // shell out — keeps the surface clean against weird inputs.
            if rev.contains(where: { "; |&`$".contains($0) }) {
                throw ToolError.invalidInput("invalid 'rev'")
            }
            args.append("\(rev)~1..\(rev)")
        case "working":
            break
        default:
            throw ToolError.invalidInput("unknown mode '\(mode)'")
        }
        if statOnly { args.append("--stat") }
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
