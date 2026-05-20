import Foundation

/// Create a directory. By default creates intermediate parents
/// (equivalent to `mkdir -p`).
public struct MkdirTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "mkdir",
            description:
                "Create a directory under the agent root. Creates intermediate parents by default. " +
                "Set 'parents=false' to require an existing parent.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Directory path, relative to agent root."),
                    "parents": SchemaBuilder.boolean(description: "Create intermediate directories. Default true.", defaultValue: true),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "mkdir \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let parents = input.optionalBool("parents") ?? true
        let url = try resolveInsideRoot(rel, context: context)
        let fm = FileManager.default
        if fm.fileExists(atPath: url.path) {
            var isDir: ObjCBool = false
            fm.fileExists(atPath: url.path, isDirectory: &isDir)
            if isDir.boolValue {
                return ToolOutput(output: "exists \(rel)", metadata: ["created": "false"])
            }
            throw ToolError.invalidInput("'\(rel)' exists and is not a directory")
        }
        try fm.createDirectory(at: url, withIntermediateDirectories: parents)
        return ToolOutput(output: "created \(rel)", metadata: ["created": "true"])
    }
}
