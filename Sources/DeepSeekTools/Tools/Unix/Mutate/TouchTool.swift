import Foundation

/// Create an empty file if it doesn't exist, or update the mtime if
/// it does. Sandboxed to agent root.
public struct TouchTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "touch",
            description:
                "Create an empty file at 'path' if missing, or bump its mtime to now if present. " +
                "Sandboxed to agent root.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "File path, relative to agent root."),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "touch \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let url = try resolveInsideRoot(rel, context: context)
        let fm = FileManager.default
        if fm.fileExists(atPath: url.path) {
            try fm.setAttributes([.modificationDate: Date()], ofItemAtPath: url.path)
            return ToolOutput(output: "touched \(rel)", metadata: ["created": "false"])
        }
        let parent = url.deletingLastPathComponent()
        if !fm.fileExists(atPath: parent.path) {
            throw ToolError.notFound("parent directory missing: \(parent.path)")
        }
        guard fm.createFile(atPath: url.path, contents: Data()) else {
            throw ToolError.external("failed to create \(rel)")
        }
        return ToolOutput(output: "created \(rel)", metadata: ["created": "true"])
    }
}
