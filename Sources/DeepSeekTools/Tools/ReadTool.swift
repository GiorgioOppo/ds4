import Foundation

/// Read a file's contents, optionally limited to a line range. The
/// output is cat-n formatted (line numbers as 1-based prefixes) so
/// the model can reference exact line numbers when proposing edits —
/// same convention as Claude Code's read tool.
public struct ReadTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "read",
            description:
                "Read a UTF-8 text file inside the agent's working directory. " +
                "Output is line-numbered. For very large files, supply 'offset' " +
                "and 'limit' (1-based) to read a window instead of the whole file.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Path relative to the agent root, or absolute."),
                    "offset": SchemaBuilder.integer(description: "1-based starting line.", minimum: 1),
                    "limit": SchemaBuilder.integer(description: "Maximum number of lines to read.", minimum: 1),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "read \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let path = try input.string("path")
        let url = try resolveInsideRoot(path, context: context)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ToolError.notFound(path)
        }
        let data = try Data(contentsOf: url)
        guard let text = String(data: data, encoding: .utf8) else {
            throw ToolError.invalidInput("file is not valid UTF-8")
        }
        let allLines = text.components(separatedBy: "\n")
        let offset = max(1, input.optionalInteger("offset") ?? 1)
        let limit = input.optionalInteger("limit") ?? 2000
        let start = offset - 1
        guard start < allLines.count else {
            return ToolOutput(output: "",
                              metadata: ["lines": "0", "total": "\(allLines.count)"])
        }
        let end = min(allLines.count, start + limit)
        let window = allLines[start..<end]
        let formatted = window.enumerated().map { idx, line in
            let n = start + idx + 1
            return "\(String(format: "%6d", n))\t\(line)"
        }.joined(separator: "\n")
        return ToolOutput(
            output: formatted,
            metadata: [
                "lines": "\(window.count)",
                "total": "\(allLines.count)",
                "path": url.path,
            ]
        )
    }
}
