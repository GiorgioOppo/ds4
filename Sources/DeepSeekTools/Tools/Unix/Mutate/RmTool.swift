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
                "Delete a file or (with recursive=true) a directory tree inside the agent root. " +
                "Recursive deletes and directory deletes REQUIRE 'confirm: true' — explicit opt-in beyond " +
                "the regular permission prompt. Refuses to delete the agent root itself.",
            category: .mutating,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Path to remove, relative to agent root."),
                    "recursive": SchemaBuilder.boolean(description: "Allow removing a directory tree. Default false.", defaultValue: false),
                    "confirm": SchemaBuilder.boolean(description: "Required for recursive or directory deletes.", defaultValue: false),
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
