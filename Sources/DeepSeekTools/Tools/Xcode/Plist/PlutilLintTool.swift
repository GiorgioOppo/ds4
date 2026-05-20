import Foundation

/// `plutil -lint` — validate that a file is a well-formed plist.
public struct PlutilLintTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "plutil_lint",
            description: "Validate a plist file's syntax. Exit 0 + 'OK' on success.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Plist file, relative to agent root."),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "plutil -lint \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let url = try resolveInsideRoot(try input.string("path"), context: context)
        return try await UnixBinary.runBinary(
            launchPath: "/usr/bin/plutil",
            arguments: ["-lint", url.path],
            context: context,
            timeout: 30)
    }
}
