import Foundation

/// Delete a file or directory. Multiple safety rails:
///  - Refuses to delete the agent root itself.
///  - Requires `confirm: true` when removing a directory or when
///    `recursive: true` is set (matches the spirit of `rm -rf`).
///  - The destructive intent is included in `permissionSummary`, so
///    the consent prompt shows what's about to disappear.
public struct RmTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "rm",
            description:
                "Cancella un file o (con recursive=true) un albero di directory dentro la root dell'agente. " +
                "Le cancellazioni ricorsive e quelle di directory RICHIEDONO 'confirm: true' — opt-in esplicito oltre " +
                "al normale prompt di permesso. Si rifiuta di cancellare la root dell'agente stessa.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Path da rimuovere, relativo alla root dell'agente."),
                    "recursive": SchemaBuilder.boolean(description: "Consente la rimozione di un albero di directory. Default false.", defaultValue: false),
                    "confirm": SchemaBuilder.boolean(description: "Richiesto per cancellazioni ricorsive o di directory.", defaultValue: false),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        let rec = (input["recursive"] as? Bool) == true ? " -r" : ""
        return "rm\(rec) \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let recursive = input.optionalBool("recursive") ?? false
        let confirm = input.optionalBool("confirm") ?? false
        let url = try resolveInsideRoot(rel, context: context)
        let fm = FileManager.default

        if url.standardizedFileURL.path == context.rootDirectory.standardizedFileURL.path {
            throw ToolError.permissionDenied("refusing to delete the agent root")
        }
        guard fm.fileExists(atPath: url.path) else {
            throw ToolError.notFound("'\(rel)' does not exist")
        }
        var isDir: ObjCBool = false
        fm.fileExists(atPath: url.path, isDirectory: &isDir)
        if isDir.boolValue {
            if !recursive {
                throw ToolError.invalidInput("'\(rel)' is a directory; set recursive=true (and confirm=true)")
            }
            if !confirm {
                throw ToolError.invalidInput("recursive delete of '\(rel)' requires confirm=true")
            }
        }
        if recursive && !confirm {
            throw ToolError.invalidInput("recursive=true requires confirm=true")
        }
        try fm.removeItem(at: url)
        return ToolOutput(output: "removed \(rel)\(isDir.boolValue ? "/" : "")")
    }
}
