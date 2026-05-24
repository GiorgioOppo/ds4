import Foundation

/// Create a directory. By default creates intermediate parents
/// (equivalent to `mkdir -p`).
public struct MkdirTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "mkdir",
            description:
                "Crea una directory sotto la root dell'agente. Per default crea anche i parent intermedi. " +
                "Imposta 'parents=false' per richiedere un parent già esistente.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Path della directory, relativo alla root dell'agente."),
                    "parents": SchemaBuilder.boolean(description: "Crea le directory intermedie. Default true.", defaultValue: true),
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
