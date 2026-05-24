import Foundation

/// Character-by-character translate / delete. Schema takes typed args
/// (from/to/delete) — no embedded character classes like `[:alpha:]`,
/// keeping the surface narrow and safe.
public struct TrTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "tr",
            description:
                "Traduce o cancella caratteri in una stringa o in un file. " +
                "Fornisci 'input' (stringa) o 'path' (file). Con 'from'/'to' (di pari lunghezza), " +
                "traduce ogni carattere di 'from' nel corrispondente di 'to'. " +
                "Con 'delete', rimuove ogni carattere che vi compare.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "input": SchemaBuilder.string(description: "Testo inline. Alternativa a 'path'."),
                    "path": SchemaBuilder.string(description: "Path del file, relativo alla root dell'agente."),
                    "from": SchemaBuilder.string(description: "Set di caratteri di origine (ogni carattere corrisponde a quello di pari indice in 'to')."),
                    "to": SchemaBuilder.string(description: "Set di caratteri di destinazione, stessa lunghezza di 'from'."),
                    "delete": SchemaBuilder.string(description: "Caratteri da rimuovere. Usato da solo (senza 'from'/'to')."),
                ]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let text: String
        if let inline = input.optionalString("input") {
            text = inline
        } else if let rel = input.optionalString("path") {
            let url = try resolveInsideRoot(rel, context: context)
            guard let s = try? String(contentsOf: url, encoding: .utf8) else {
                throw ToolError.invalidInput("cannot read '\(rel)' as UTF-8")
            }
            text = s
        } else {
            throw ToolError.invalidInput("provide 'input' or 'path'")
        }

        if let delete = input.optionalString("delete"), !delete.isEmpty {
            let set = Set(delete)
            let out = text.filter { !set.contains($0) }
            return ToolOutput(output: out)
        }
        guard let from = input.optionalString("from"),
              let to = input.optionalString("to") else {
            throw ToolError.invalidInput("provide 'from'+'to' or 'delete'")
        }
        if from.count != to.count {
            throw ToolError.invalidInput("'from' and 'to' must be the same length")
        }
        var map: [Character: Character] = [:]
        for (f, t) in zip(from, to) { map[f] = t }
        let out = String(text.map { map[$0] ?? $0 })
        return ToolOutput(output: out)
    }
}
