import Foundation

/// Merge corresponding lines of multiple files side-by-side.
public struct PasteTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "paste",
            description:
                "Merge lines of two or more files side by side, separated by 'delimiter' (default tab). " +
                "Shorter files are padded with empty strings to match the longest.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "paths": SchemaBuilder.array(itemsType: "string", description: "File paths, relative to agent root."),
                    "delimiter": SchemaBuilder.string(description: "Joiner between columns. Default '\\t'."),
                ],
                required: ["paths"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        let paths = (input["paths"] as? [String]) ?? []
        return "paste \(paths.joined(separator: " "))"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        guard let paths = input.optionalStringArray("paths"), paths.count >= 1 else {
            throw ToolError.invalidInput("'paths' must be a non-empty array")
        }
        let delim = input.optionalString("delimiter") ?? "\t"

        var columns: [[String]] = []
        for rel in paths {
            let url = try resolveInsideRoot(rel, context: context)
            guard let text = try? String(contentsOf: url, encoding: .utf8) else {
                throw ToolError.invalidInput("cannot read '\(rel)' as UTF-8")
            }
            var lines = text.components(separatedBy: "\n")
            if lines.last == "" { lines.removeLast() }
            columns.append(lines)
        }
        let maxLen = columns.map(\.count).max() ?? 0
        var out: [String] = []
        out.reserveCapacity(maxLen)
        for i in 0..<maxLen {
            let row = columns.map { i < $0.count ? $0[i] : "" }
            out.append(row.joined(separator: delim))
        }
        return ToolOutput(output: out.joined(separator: "\n"))
    }
}
