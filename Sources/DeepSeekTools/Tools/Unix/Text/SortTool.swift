import Foundation

/// Sort lines of a file. Pure Swift.
public struct SortTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "sort",
            description:
                "Sort the lines of a file. Default: lexicographic ascending. " +
                "Set 'numeric=true' for numeric sort, 'reverse=true' to descend, 'unique=true' to dedupe.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "File path, relative to agent root."),
                    "numeric": SchemaBuilder.boolean(description: "Numeric (vs lex) sort. Default false.", defaultValue: false),
                    "reverse": SchemaBuilder.boolean(description: "Reverse the result. Default false.", defaultValue: false),
                    "unique": SchemaBuilder.boolean(description: "Drop adjacent duplicates after sort. Default false.", defaultValue: false),
                    "caseInsensitive": SchemaBuilder.boolean(description: "Case-insensitive compare. Default false.", defaultValue: false),
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
