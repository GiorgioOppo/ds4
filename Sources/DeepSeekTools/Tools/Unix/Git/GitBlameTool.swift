import Foundation

/// `git blame` of a file, optionally restricted to a line range.
public struct GitBlameTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "git_blame",
            description:
                "Mostra l'autorialità riga per riga di un file. Opzionalmente 'startLine'+'endLine' " +
                "per restringere a un intervallo.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Path del file, relativo alla root dell'agente."),
                    "startLine": SchemaBuilder.integer(description: "Riga di partenza, a base 1.", minimum: 1),
                    "endLine": SchemaBuilder.integer(description: "Riga finale, a base 1.", minimum: 1),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "git_blame \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let url = try resolveInsideRoot(rel, context: context)
        var args: [String] = ["blame"]
        if let start = input.optionalInteger("startLine") {
            let end = input.optionalInteger("endLine") ?? start
            args.append("-L"); args.append("\(start),\(end)")
        }
        args.append(url.path)
        return try await UnixBinary.runBinary(
            launchPath: "/usr/bin/git",
            arguments: args,
            context: context,
            cwd: context.rootDirectory)
    }
}
