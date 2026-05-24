import Foundation

/// List open files. Wraps `/usr/sbin/lsof` with a narrow schema —
/// either filter by PID or by file path. Free-form lsof selectors
/// aren't exposed.
public struct LsofTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "lsof",
            description:
                "Elenca i file aperti. Fornisci 'pid' per ispezionare un singolo processo, o 'path' per trovare chi tiene aperto un file. " +
                "Senza argomenti, elenca ogni file aperto visibile all'utente corrente.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "pid": SchemaBuilder.integer(description: "PID del processo da ispezionare.", minimum: 1),
                    "path": SchemaBuilder.string(description: "Path del file da cercare (assoluto o relativo alla root dell'agente)."),
                ]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        var args: [String] = []
        if let pid = input.optionalInteger("pid") {
            args.append("-p"); args.append("\(pid)")
        }
        if let rel = input.optionalString("path") {
            let url = try resolveInsideRoot(rel, context: context)
            args.append(url.path)
        }
        return try await UnixBinary.runBinary(
            launchPath: "/usr/sbin/lsof",
            arguments: args,
            context: context,
            timeout: 15)
    }
}
