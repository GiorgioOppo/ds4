import Foundation

/// Create an empty file if it doesn't exist, or update the mtime if
/// it does. Sandboxed to agent root.
public struct TouchTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "touch",
            description:
                "Crea un file vuoto in 'path' se mancante, o aggiorna il suo mtime ad adesso se presente. " +
                "Confinato alla root dell'agente.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Path del file, relativo alla root dell'agente."),
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
