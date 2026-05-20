import Foundation

/// Detect file type via `/usr/bin/file`. Wrapper because the magic
/// number database is non-trivial to reimplement.
public struct FileTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "file",
            description: "Detect file type (text/binary/format) using libmagic via /usr/bin/file.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "File path, relative to agent root."),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "file \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let url = try resolveInsideRoot(rel, context: context)
        return try await UnixBinary.runBinary(
            launchPath: "/usr/bin/file",
            arguments: ["-b", url.path],
            context: context)
    }
}
