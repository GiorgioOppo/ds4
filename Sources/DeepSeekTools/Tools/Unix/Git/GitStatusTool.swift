import Foundation

/// `git status` against the agent root, with `--porcelain=v1` so the
/// output is stable and machine-friendly.
public struct GitStatusTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "git_status",
            description:
                "Show modified, staged, and untracked files in the agent root. " +
                "Uses --porcelain=v1 for stable parseable output.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "showIgnored": SchemaBuilder.boolean(description: "Include ignored files. Default false.", defaultValue: false),
                ]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let showIgnored = input.optionalBool("showIgnored") ?? false
        var args: [String] = ["status", "--porcelain=v1"]
        if showIgnored { args.append("--ignored") }
        return try await UnixBinary.runBinary(
            launchPath: "/usr/bin/git",
            arguments: args,
            context: context,
            cwd: context.rootDirectory)
    }
}
