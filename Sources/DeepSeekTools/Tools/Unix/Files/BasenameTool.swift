import Foundation

/// Strip the directory part of a path. Does not touch the filesystem.
public struct BasenameTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "basename",
            description:
                "Restituisce l'ultimo componente del path. Con 'suffix' impostato, lo rimuove dal risultato se presente. " +
                "Puramente basato su stringhe; nessun accesso al filesystem.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Una qualsiasi stringa di path."),
                    "suffix": SchemaBuilder.string(description: "Suffisso opzionale da rimuovere (es. '.swift')."),
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
