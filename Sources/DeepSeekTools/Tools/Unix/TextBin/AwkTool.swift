import Foundation

/// Awk wrapper. The model provides a program string and a file path;
/// we run `/usr/bin/awk -F<fs> <program> <file>`. No system(), no -i.
///
/// ⚠️ awk's DSL exposes `getline < "/path"` and similar, which can
/// read files outside the agent root. The host should consider that
/// when choosing whether to register this tool — for strict sandbox
/// guarantees, prefer the typed Swift tools (`cut`, `tr`, `grep`,
/// `wc`) or wrap this with `sandbox-exec`. For most workflows the
/// `.readOnly` category is the right trade-off: the file write side
/// of awk needs explicit redirection in the program, which the
/// permission gate would catch through `permissionSummary`.
public struct AwkTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "awk",
            description:
                "Esegue un programma awk su un file e restituisce stdout. Fornisci 'program' (sorgente awk) " +
                "e 'path' (file di input). Attenzione: i programmi awk possono leggere file fuori dalla root dell'agente tramite " +
                "'getline < \"...\"' — preferisci, quando possibile, i tool tipizzati (cut/tr/grep).",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "program": SchemaBuilder.string(description: "Sorgente awk (una o più regole)."),
                    "path": SchemaBuilder.string(description: "File di input, relativo alla root dell'agente."),
                    "fieldSeparator": SchemaBuilder.string(description: "Separatore di campo (-F). Default: spazi bianchi."),
                ],
                required: ["program", "path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        let prog = (input["program"] as? String ?? "")
            .prefix(40)
            .replacingOccurrences(of: "\n", with: " ")
        return "awk '\(prog)\((input["program"] as? String ?? "").count > 40 ? "…" : "")' \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let program = try input.string("program")
        let rel = try input.string("path")
        let url = try resolveInsideRoot(rel, context: context)
        var args: [String] = []
        if let fs = input.optionalString("fieldSeparator"), !fs.isEmpty {
            args.append("-F"); args.append(fs)
        }
        args.append(program)
        args.append(url.path)
        return try await UnixBinary.runBinary(
            launchPath: "/usr/bin/awk",
            arguments: args,
            context: context)
    }
}
