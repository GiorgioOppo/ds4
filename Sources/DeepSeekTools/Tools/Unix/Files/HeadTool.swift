import Foundation

/// Print the first N lines (or bytes) of a file. Pure Swift; sandboxed.
public struct HeadTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "head",
            description:
                "Print the first N lines (default 10) of a file. " +
                "Use 'bytes' to count bytes instead. Sandboxed to agent root.",
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
        "head \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let url = try resolveInsideRoot(rel, context: context)
        guard let data = try? Data(contentsOf: url) else {
            throw ToolError.notFound("cannot read '\(rel)'")
        }

        if let nBytes = input.optionalInteger("bytes") {
            let prefix = data.prefix(nBytes)
            let text = String(data: prefix, encoding: .utf8) ?? ""
            return ToolOutput(output: text, metadata: ["bytes": "\(prefix.count)"])
        }
        let nLines = input.optionalInteger("lines") ?? 10
        guard let text = String(data: data, encoding: .utf8) else {
            throw ToolError.invalidInput("'\(rel)' is not UTF-8")
        }
        let lines = text.split(separator: "\n", maxSplits: nLines, omittingEmptySubsequences: false)
        let kept = lines.prefix(nLines).joined(separator: "\n")
        return ToolOutput(output: kept, metadata: ["lines": "\(min(nLines, lines.count))"])
    }
}
