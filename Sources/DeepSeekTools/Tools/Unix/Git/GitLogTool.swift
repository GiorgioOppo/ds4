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
                "Mostra la storia dei commit della root dell'agente. Il default --oneline -n 20 mantiene l'output limitato. " +
                "Passa 'path' per restringere a un singolo file/directory.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "n": SchemaBuilder.integer(description: "Numero di commit. Default 20.", minimum: 1),
                    "path": SchemaBuilder.string(description: "Restringe a un singolo path, relativo alla root dell'agente."),
                    "since": SchemaBuilder.string(description: "Solo commit più recenti di questa data (es. '2 weeks ago')."),
                    "author": SchemaBuilder.string(description: "Filtra per autore del commit."),
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
