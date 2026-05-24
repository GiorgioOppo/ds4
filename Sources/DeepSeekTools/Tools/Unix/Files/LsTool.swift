import Foundation

/// List directory contents. Pure Swift via `FileManager`; no shell-out.
/// Sandboxed to the agent root via `resolveInsideRoot`. Output is one
/// entry per line; with `long=true` each line carries permissions,
/// size, and mtime in a stable order so the model can parse it.
public struct LsTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "ls",
            description:
                "Elenca i figli immediati di una directory (non ricorsivo). " +
                "Usalo quando conosci la directory e vuoi vedere cosa contiene direttamente. " +
                "Per un walk ricorsivo filtrato per attributo usa 'find'; per il matching per nome con pattern " +
                "sull'intero albero usa 'glob'. " +
                "'long=true' aggiunge permessi/dimensione/mtime; 'all=true' include i dotfile; " +
                "'limit' limita il numero di entry (default 1000).",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Directory, relativa alla root dell'agente. Default '.'."),
                    "all": SchemaBuilder.boolean(description: "Include i dotfile. Default false.", defaultValue: false),
                    "long": SchemaBuilder.boolean(description: "Formato dettagliato. Default false.", defaultValue: false),
                    "limit": SchemaBuilder.integer(description: "Numero massimo di entry. Default 1000.", minimum: 1),
                ]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "ls \(input["path"] as? String ?? ".")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let relPath = input.optionalString("path") ?? "."
        let all = input.optionalBool("all") ?? false
        let long = input.optionalBool("long") ?? false
        let limit = input.optionalInteger("limit") ?? 1000
        let dir = try resolveInsideRoot(relPath, context: context)

        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: dir.path, isDirectory: &isDir) else {
            throw ToolError.notFound("'\(relPath)' does not exist")
        }
        guard isDir.boolValue else {
            throw ToolError.invalidInput("'\(relPath)' is not a directory")
        }

        let raw = try FileManager.default.contentsOfDirectory(atPath: dir.path).sorted()
        let contents = all ? raw : raw.filter { !$0.hasPrefix(".") }
        var lines: [String] = []
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd HH:mm"
        df.locale = Locale(identifier: "en_US_POSIX")

        for name in contents {
            if lines.count >= limit { break }
            if long {
                let child = dir.appendingPathComponent(name)
                let attrs = (try? FileManager.default.attributesOfItem(atPath: child.path)) ?? [:]
                let perms = (attrs[.posixPermissions] as? NSNumber)?.intValue ?? 0
                let size = (attrs[.size] as? NSNumber)?.int64Value ?? 0
                let mtime = (attrs[.modificationDate] as? Date) ?? Date(timeIntervalSince1970: 0)
                let type = (attrs[.type] as? FileAttributeType) ?? .typeUnknown
                let typeChar: String
                switch type {
                case .typeDirectory: typeChar = "d"
                case .typeSymbolicLink: typeChar = "l"
                case .typeRegular: typeChar = "-"
                default: typeChar = "?"
                }
                lines.append("\(typeChar)\(formatPermissions(perms)) \(formatSize(size)) \(df.string(from: mtime)) \(name)")
            } else {
                lines.append(name)
            }
        }

        let truncated = contents.count > lines.count
        var body = lines.joined(separator: "\n")
        if truncated {
            body += "\n[truncated: showing \(lines.count) of \(contents.count) entries]"
        }
        return ToolOutput(output: body,
                          metadata: ["count": "\(lines.count)", "total": "\(contents.count)"])
    }

    private func formatPermissions(_ perms: Int) -> String {
        let owner = triplet((perms >> 6) & 0o7)
        let group = triplet((perms >> 3) & 0o7)
        let other = triplet(perms & 0o7)
        return owner + group + other
    }

    private func triplet(_ bits: Int) -> String {
        let r = (bits & 0o4) != 0 ? "r" : "-"
        let w = (bits & 0o2) != 0 ? "w" : "-"
        let x = (bits & 0o1) != 0 ? "x" : "-"
        return r + w + x
    }

    private func formatSize(_ bytes: Int64) -> String {
        String(format: "%10lld", bytes)
    }
}
