import Foundation

/// Strip the directory part of a path. Does not touch the filesystem.
public struct BasenameTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "basename",
            description:
                "Return the last path component. With 'suffix' set, strip it from the result if present. " +
                "Purely string-based; no filesystem access.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Any path string."),
                    "suffix": SchemaBuilder.string(description: "Optional suffix to strip (e.g. '.swift')."),
                ],
                required: ["path"]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let path = try input.string("path")
        var name = (path as NSString).lastPathComponent
        if let suffix = input.optionalString("suffix"), !suffix.isEmpty, name.hasSuffix(suffix) {
            name = String(name.dropLast(suffix.count))
        }
        return ToolOutput(output: name)
    }
}
