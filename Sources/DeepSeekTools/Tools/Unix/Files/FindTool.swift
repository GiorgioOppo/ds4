import Foundation

/// Walk a directory tree filtering by exact name, type, size, and
/// modification time. **Does not** do glob matching — that's `glob`'s
/// job. Pure Swift via `UnixWalker`; symlinks not followed unless
/// `followSymlinks: true`.
///
/// Schema is intentionally orthogonal to `glob`: `name` is an *exact*
/// component match (no wildcards). For pattern matching the model
/// should pick `glob` instead.
public struct FindTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "find",
            description:
                "Walk a directory tree under the agent root filtering by exact name, type, size, and mtime. " +
                "For glob/wildcard matching use 'glob' instead. " +
                "Output is one matching path per line, capped at 'limit' (default 500).",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Root directory, relative to agent root. Default '.'."),
                    "name": SchemaBuilder.string(description: "Exact filename to match (no wildcards). Optional."),
                    "type": SchemaBuilder.string(
                        description: "Restrict to one kind: 'file', 'dir', 'symlink'.",
                        enumValues: ["file", "dir", "symlink"]),
                    "minSize": SchemaBuilder.integer(description: "Minimum size in bytes.", minimum: 0),
                    "maxSize": SchemaBuilder.integer(description: "Maximum size in bytes.", minimum: 0),
                    "mtimeNewerThanDays": SchemaBuilder.integer(description: "Only entries modified within the last N days.", minimum: 1),
                    "followSymlinks": SchemaBuilder.boolean(description: "Follow symlinks (with cycle detection). Default false.", defaultValue: false),
                    "limit": SchemaBuilder.integer(description: "Max results. Default 500.", minimum: 1),
                ]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "find \(input["path"] as? String ?? ".")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = input.optionalString("path") ?? "."
        let name = input.optionalString("name")
        let typeFilter = input.optionalString("type")
        let minSize = input.optionalInteger("minSize")
        let maxSize = input.optionalInteger("maxSize")
        let mtimeDays = input.optionalInteger("mtimeNewerThanDays")
        let follow = input.optionalBool("followSymlinks") ?? false
        let limit = input.optionalInteger("limit") ?? 500
        let root = try resolveInsideRoot(rel, context: context)

        var matches: [String] = []
        let cutoff: Date? = mtimeDays.map { Date(timeIntervalSinceNow: -Double($0 * 86_400)) }
        let opts = UnixWalker.Options(followSymlinks: follow)

        UnixWalker.walk(root: root, options: opts, isCancelled: context.isCancelled) { entry in
            if let typeFilter {
                switch typeFilter {
                case "file": if entry.isDirectory || entry.isSymlink { return true }
                case "dir": if !entry.isDirectory { return true }
                case "symlink": if !entry.isSymlink { return true }
                default: break
                }
            }
            if let name, entry.url.lastPathComponent != name { return true }
            if minSize != nil || maxSize != nil {
                let size = ((try? FileManager.default.attributesOfItem(atPath: entry.url.path)[.size] as? NSNumber)?.intValue) ?? 0
                if let minSize, size < minSize { return true }
                if let maxSize, size > maxSize { return true }
            }
            if let cutoff {
                let mtime = ((try? FileManager.default.attributesOfItem(atPath: entry.url.path)[.modificationDate] as? Date)) ?? Date(timeIntervalSince1970: 0)
                if mtime < cutoff { return true }
            }
            matches.append(entry.relativePath.isEmpty ? "." : entry.relativePath)
            return matches.count < limit
        }
        return ToolOutput(output: matches.joined(separator: "\n"),
                          metadata: ["matches": "\(matches.count)"])
    }
}
