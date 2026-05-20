import Foundation

/// Print the last N lines (or bytes) of a file. Pure Swift; sandboxed.
public struct TailTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "tail",
            description:
                "Print the last N lines of a file (default 10). " +
                "Use this for the latest entries in a log, the last error trace, or the bottom of a long output. " +
                "For the whole file use 'read'; for the start use 'head'. " +
                "'bytes' counts bytes instead of lines.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "File path, relative to agent root."),
                    "lines": SchemaBuilder.integer(description: "Lines to print. Default 10.", minimum: 1),
                    "bytes": SchemaBuilder.integer(description: "If set, print this many bytes and ignore 'lines'.", minimum: 1),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "tail \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let url = try resolveInsideRoot(rel, context: context)
        guard let data = try? Data(contentsOf: url) else {
            throw ToolError.notFound("cannot read '\(rel)'")
        }

        if let nBytes = input.optionalInteger("bytes") {
            let count = min(nBytes, data.count)
            let suffix = data.suffix(count)
            let text = String(data: suffix, encoding: .utf8) ?? ""
            return ToolOutput(output: text, metadata: ["bytes": "\(suffix.count)"])
        }
        let nLines = input.optionalInteger("lines") ?? 10
        guard let text = String(data: data, encoding: .utf8) else {
            throw ToolError.invalidInput("'\(rel)' is not UTF-8")
        }
        let lines = text.components(separatedBy: "\n")
        let start = max(0, lines.count - nLines)
        let kept = lines[start..<lines.count].joined(separator: "\n")
        return ToolOutput(output: kept, metadata: ["lines": "\(lines.count - start)"])
    }
}
