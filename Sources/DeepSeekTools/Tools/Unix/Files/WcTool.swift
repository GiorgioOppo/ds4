import Foundation

/// Count lines/words/bytes in one or more files. Pure Swift.
public struct WcTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "wc",
            description:
                "Count lines, words, and bytes in files under the agent root. " +
                "Returns one line per input as 'lines words bytes path', plus a 'total' line if multiple paths.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "paths": SchemaBuilder.array(itemsType: "string", description: "File paths, relative to agent root."),
                    "mode": SchemaBuilder.string(
                        description: "Restrict counters: 'lines', 'words', 'bytes', or 'all' (default).",
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
