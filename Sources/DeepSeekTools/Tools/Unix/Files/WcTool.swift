import Foundation

/// Count lines/words/bytes in one or more files. Pure Swift.
public struct WcTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "wc",
            description:
                "Conta righe, parole e byte nei file sotto la root dell'agente. " +
                "Restituisce una riga per input come 'lines words bytes path', più una riga 'total' in caso di più path.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "paths": SchemaBuilder.array(itemsType: "string", description: "Path dei file, relativi alla root dell'agente."),
                    "mode": SchemaBuilder.string(
                        description: "Restringe i contatori: 'lines', 'words', 'bytes', o 'all' (default).",
                        enumValues: ["lines", "words", "bytes", "all"]),
                ],
                required: ["paths"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        let paths = (input["paths"] as? [String]) ?? []
        return "wc \(paths.first ?? "?")\(paths.count > 1 ? " +\(paths.count - 1)" : "")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let paths = input.optionalStringArray("paths") ?? []
        guard !paths.isEmpty else {
            throw ToolError.invalidInput("'paths' must be a non-empty array")
        }
        let mode = input.optionalString("mode") ?? "all"

        var lines: [String] = []
        var totalL = 0, totalW = 0, totalB = 0
        for rel in paths {
            let url = try resolveInsideRoot(rel, context: context)
            guard let data = try? Data(contentsOf: url) else {
                throw ToolError.notFound("cannot read '\(rel)'")
            }
            let bytes = data.count
            let text = String(data: data, encoding: .utf8) ?? ""
            let lineCount = text.isEmpty ? 0 : text.components(separatedBy: "\n").count - (text.hasSuffix("\n") ? 1 : 0)
            let wordCount = text.split(whereSeparator: { $0.isWhitespace }).count
            totalL += lineCount; totalW += wordCount; totalB += bytes
            lines.append(format(lines: lineCount, words: wordCount, bytes: bytes, path: rel, mode: mode))
        }
        if paths.count > 1 {
            lines.append(format(lines: totalL, words: totalW, bytes: totalB, path: "total", mode: mode))
        }
        return ToolOutput(output: lines.joined(separator: "\n"))
    }

    private func format(lines l: Int, words w: Int, bytes b: Int, path: String, mode: String) -> String {
        switch mode {
        case "lines": return "\(l) \(path)"
        case "words": return "\(w) \(path)"
        case "bytes": return "\(b) \(path)"
        default: return "\(l) \(w) \(b) \(path)"
        }
    }
}
