import Foundation

/// Extract fields or character ranges from each line of a file.
public struct CutTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "cut",
            description:
                "Estrae campi (separati da 'delimiter') o intervalli di caratteri da ogni riga. " +
                "Fornisci 'fields' (indici a base 1) con 'delimiter' opzionale (default tab), " +
                "oppure 'characters' (intervalli a base 1) senza 'delimiter'.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Path del file, relativo alla root dell'agente."),
                    "delimiter": SchemaBuilder.string(description: "Separatore di campo. Default '\\t'."),
                    "fields": SchemaBuilder.array(itemsType: "integer", description: "Indici di campo a base 1 da mantenere."),
                    "characters": SchemaBuilder.array(itemsType: "integer", description: "Posizioni di carattere a base 1 da mantenere."),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "cut \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let delim = input.optionalString("delimiter") ?? "\t"
        let fields = (input["fields"] as? [Int]) ?? (input["fields"] as? [NSNumber])?.map { $0.intValue } ?? []
        let chars = (input["characters"] as? [Int]) ?? (input["characters"] as? [NSNumber])?.map { $0.intValue } ?? []
        if fields.isEmpty && chars.isEmpty {
            throw ToolError.invalidInput("provide 'fields' or 'characters'")
        }
        let url = try resolveInsideRoot(rel, context: context)
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            throw ToolError.invalidInput("cannot read '\(rel)' as UTF-8")
        }
        var lines = text.components(separatedBy: "\n")
        if lines.last == "" { lines.removeLast() }

        let out: [String] = lines.map { line in
            if !fields.isEmpty {
                let parts = line.components(separatedBy: delim)
                return fields.compactMap { i in
                    (i >= 1 && i <= parts.count) ? parts[i - 1] : nil
                }.joined(separator: delim)
            } else {
                let str = Array(line)
                return chars.compactMap { i in
                    (i >= 1 && i <= str.count) ? String(str[i - 1]) : nil
                }.joined()
            }
        }
        return ToolOutput(output: out.joined(separator: "\n"))
    }
}
