import Foundation

/// Sort lines of a file. Pure Swift.
public struct SortTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "sort",
            description:
                "Ordina le righe di un file. Default: lessicografico crescente. " +
                "Imposta 'numeric=true' per ordinamento numerico, 'reverse=true' per decrescente, 'unique=true' per deduplicare.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Path del file, relativo alla root dell'agente."),
                    "numeric": SchemaBuilder.boolean(description: "Ordinamento numerico (vs lessicografico). Default false.", defaultValue: false),
                    "reverse": SchemaBuilder.boolean(description: "Inverte il risultato. Default false.", defaultValue: false),
                    "unique": SchemaBuilder.boolean(description: "Rimuove i duplicati adiacenti dopo l'ordinamento. Default false.", defaultValue: false),
                    "caseInsensitive": SchemaBuilder.boolean(description: "Confronto case-insensitive. Default false.", defaultValue: false),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "sort \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let numeric = input.optionalBool("numeric") ?? false
        let reverse = input.optionalBool("reverse") ?? false
        let unique = input.optionalBool("unique") ?? false
        let caseInsens = input.optionalBool("caseInsensitive") ?? false
        let url = try resolveInsideRoot(rel, context: context)
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            throw ToolError.invalidInput("cannot read '\(rel)' as UTF-8")
        }
        var lines = text.components(separatedBy: "\n")
        if lines.last == "" { lines.removeLast() }
        lines.sort { a, b in
            let lhs = caseInsens ? a.lowercased() : a
            let rhs = caseInsens ? b.lowercased() : b
            if numeric {
                let la = Double(lhs.trimmingCharacters(in: .whitespaces)) ?? .infinity
                let rb = Double(rhs.trimmingCharacters(in: .whitespaces)) ?? .infinity
                return reverse ? (la > rb) : (la < rb)
            }
            return reverse ? (lhs > rhs) : (lhs < rhs)
        }
        if unique {
            var dedup: [String] = []
            for l in lines where dedup.last != l { dedup.append(l) }
            lines = dedup
        }
        return ToolOutput(output: lines.joined(separator: "\n"),
                          metadata: ["lines": "\(lines.count)"])
    }
}
