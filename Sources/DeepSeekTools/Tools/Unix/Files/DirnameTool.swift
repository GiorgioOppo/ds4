import Foundation

/// Strip the last component of a path. Pure string operation.
public struct DirnameTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "dirname",
            description:
                "Restituisce la parte directory di un path. Puramente basato su stringhe; nessun accesso al filesystem.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Una qualsiasi stringa di path."),
                ],
                required: ["path"]
            )
        )
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let path = try input.string("path")
        let dir = (path as NSString).deletingLastPathComponent
        return ToolOutput(output: dir.isEmpty ? "." : dir)
    }
}
