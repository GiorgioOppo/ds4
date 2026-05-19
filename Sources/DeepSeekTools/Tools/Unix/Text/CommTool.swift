import Foundation

/// Compare two sorted files line by line, emitting three columns:
/// lines only in A, lines only in B, lines in both.
public struct CommTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "comm",
            description:
                "Compare two sorted files line by line. Output columns: " +
                "1 = only in A, 2 = only in B, 3 = in both. " +
                "Suppress any column via 'suppress1'/'suppress2'/'suppress3'. " +
                "Files must already be sorted — use 'sort' first if needed.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "pathA": SchemaBuilder.string(description: "First sorted file."),
                    "pathB": SchemaBuilder.string(description: "Second sorted file."),
                    "suppress1": SchemaBuilder.boolean(description: "Hide column 1 (A-only).", defaultValue: false),
                    "suppress2": SchemaBuilder.boolean(description: "Hide column 2 (B-only).", defaultValue: false),
                    "suppress3": SchemaBuilder.boolean(description: "Hide column 3 (common).", defaultValue: false),
                ],
                required: ["pathA", "pathB"]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let relA = try input.string("pathA")
        let relB = try input.string("pathB")
        let s1 = input.optionalBool("suppress1") ?? false
        let s2 = input.optionalBool("suppress2") ?? false
        let s3 = input.optionalBool("suppress3") ?? false
        let a = try linesFrom(relA, context: context)
        let b = try linesFrom(relB, context: context)

        var out: [String] = []
        var i = 0, j = 0
        while i < a.count && j < b.count {
            if a[i] == b[j] {
                if !s3 { out.append("\t\t\(a[i])") }
                i += 1; j += 1
            } else if a[i] < b[j] {
                if !s1 { out.append(a[i]) }
                i += 1
            } else {
                if !s2 { out.append("\t\(b[j])") }
                j += 1
            }
        }
        while i < a.count { if !s1 { out.append(a[i]) }; i += 1 }
        while j < b.count { if !s2 { out.append("\t\(b[j])") }; j += 1 }
        return ToolOutput(output: out.joined(separator: "\n"))
    }

    private func linesFrom(_ rel: String, context: ToolContext) throws -> [String] {
        let url = try resolveInsideRoot(rel, context: context)
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            throw ToolError.invalidInput("cannot read '\(rel)' as UTF-8")
        }
        var lines = text.components(separatedBy: "\n")
        if lines.last == "" { lines.removeLast() }
        return lines
    }
}
