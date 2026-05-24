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
                "Attraversa ricorsivamente un albero di directory sotto la root dell'agente e filtra le entry per " +
                "nome di file esatto, tipo (file/dir/symlink), intervallo di dimensione o data di modifica. " +
                "Usalo quando hai una query basata su attributi ('tutti i file .swift modificati nell'ultima settimana'). " +
                "Per il contenuto immediato di una singola directory usa 'ls'; per il matching per nome con wildcard " +
                "('Sources/**/*.swift') usa 'glob' — è più veloce e la traduzione in regex gestisce i glob in modo nativo. " +
                "L'output è un path corrispondente per riga, limitato da 'limit' (default 500).",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Directory di partenza, relativa alla root dell'agente. Default '.'."),
                    "name": SchemaBuilder.string(description: "Nome file esatto da trovare (senza wildcard). Opzionale."),
                    "type": SchemaBuilder.string(
                        description: "Restringe a un solo tipo: 'file', 'dir', 'symlink'.",
                        enumValues: ["file", "dir", "symlink"]),
                    "minSize": SchemaBuilder.integer(description: "Dimensione minima in byte.", minimum: 0),
                    "maxSize": SchemaBuilder.integer(description: "Dimensione massima in byte.", minimum: 0),
                    "mtimeNewerThanDays": SchemaBuilder.integer(description: "Solo entry modificate negli ultimi N giorni.", minimum: 1),
                    "followSymlinks": SchemaBuilder.boolean(description: "Segue i symlink (con rilevamento dei cicli). Default false.", defaultValue: false),
                    "limit": SchemaBuilder.integer(description: "Numero massimo di risultati. Default 500.", minimum: 1),
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
