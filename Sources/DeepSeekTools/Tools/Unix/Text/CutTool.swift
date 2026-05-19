import Foundation

/// Extract fields or character ranges from each line of a file.
public struct CutTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "cut",
            description:
                "Extract fields (split by 'delimiter') or character ranges from each line. " +
                "Provide either 'fields' (1-based indices) with optional 'delimiter' (default tab), " +
                "or 'characters' (1-based ranges) without 'delimiter'.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "File path, relative to agent root."),
                    "delimiter": SchemaBuilder.string(description: "Field separator. Default '\\t'."),
                    "fields": SchemaBuilder.array(itemsType: "integer", description: "1-based field indices to keep."),
                    "characters": SchemaBuilder.array(itemsType: "integer", description: "1-based character positions to keep."),
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
