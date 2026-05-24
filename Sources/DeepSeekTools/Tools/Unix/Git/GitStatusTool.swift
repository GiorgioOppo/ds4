import Foundation

/// `git status` against the agent root, with `--porcelain=v1` so the
/// output is stable and machine-friendly.
public struct GitStatusTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "git_status",
            description:
                "Mostra file modificati, in staged e non tracciati nella root dell'agente. " +
                "Usa --porcelain=v1 per un output stabile e parsabile.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "showIgnored": SchemaBuilder.boolean(description: "Include i file ignorati. Default false.", defaultValue: false),
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
