import Foundation

/// Drop adjacent duplicate lines (Unix `uniq` semantics — does NOT sort).
public struct UniqTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "uniq",
            description:
                "Rimuove le righe duplicate adiacenti (NON è una dedup globale — pre-ordina con 'sort' se ne hai bisogno). " +
                "Imposta 'count=true' per anteporre a ciascuna riga il numero di occorrenze.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Path del file, relativo alla root dell'agente."),
                    "count": SchemaBuilder.boolean(description: "Antepone a ciascuna riga la lunghezza della sua sequenza. Default false.", defaultValue: false),
                    "caseInsensitive": SchemaBuilder.boolean(description: "Confronto case-insensitive. Default false.", defaultValue: false),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "uniq \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let count = input.optionalBool("count") ?? false
        let caseInsens = input.optionalBool("caseInsensitive") ?? false
        let url = try resolveInsideRoot(rel, context: context)
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            throw ToolError.invalidInput("cannot read '\(rel)' as UTF-8")
        }
        var lines = text.components(separatedBy: "\n")
        if lines.last == "" { lines.removeLast() }

        var out: [String] = []
        var prev: String?
        var prevKey: String?
        var run = 0
        for line in lines {
            let key = caseInsens ? line.lowercased() : line
            if key == prevKey {
                run += 1
            } else {
                if let prev = prev {
                    out.append(count ? "\(run) \(prev)" : prev)
                }
                prev = line
                prevKey = key
                run = 1
            }
        }
        if let prev = prev {
            out.append(count ? "\(run) \(prev)" : prev)
        }
        return ToolOutput(output: out.joined(separator: "\n"))
    }
}
