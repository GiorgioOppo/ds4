import Foundation

/// Write a UTF-8 text file, creating intermediate directories. Use
/// only for *new* files or full rewrites — for surgical changes the
/// model should prefer `edit` (smaller token cost, diff-visible).
public struct WriteTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "write",
            description:
                "Create or completely overwrite a UTF-8 file. Creates parent " +
                "directories as needed. Prefer 'edit' for partial changes.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Target file path."),
                    "content": SchemaBuilder.string(description: "Full file content."),
                ],
                required: ["path", "content"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "write \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let path = try input.string("path")
        let content = try input.string("content")
        let url = try resolveInsideRoot(path, context: context)
        let dir = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dir,
                                                withIntermediateDirectories: true)
        guard let data = content.data(using: .utf8) else {
            throw ToolError.invalidInput("content is not valid UTF-8")
        }
        try data.write(to: url, options: .atomic)
        let lineCount = content.components(separatedBy: "\n").count
        return ToolOutput(
            output: "wrote \(data.count) bytes to \(url.lastPathComponent) (\(lineCount) lines)",
            metadata: ["path": url.path, "bytes": "\(data.count)"]
        )
    }
}
