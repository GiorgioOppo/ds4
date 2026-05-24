import Foundation

/// Inspect file metadata. Output is `key=value` lines in a stable
/// order regardless of platform — pure Swift via `FileManager` so
/// macOS vs Linux differences in the system `stat` binary don't matter.
public struct StatTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "stat",
            description:
                "Restituisce i metadati del file come righe key=value: type, size, perms, owner, group, mtime, ctime, atime. " +
                "Confinato alla root dell'agente.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Path del file, relativo alla root dell'agente."),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "stat \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let url = try resolveInsideRoot(rel, context: context)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ToolError.notFound("'\(rel)' does not exist")
        }
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        let type = (attrs[.type] as? FileAttributeType) ?? .typeUnknown
        let size = (attrs[.size] as? NSNumber)?.int64Value ?? 0
        let perms = (attrs[.posixPermissions] as? NSNumber)?.intValue ?? 0
        let owner = (attrs[.ownerAccountName] as? String) ?? ""
        let group = (attrs[.groupOwnerAccountName] as? String) ?? ""
        let mtime = (attrs[.modificationDate] as? Date) ?? Date(timeIntervalSince1970: 0)
        let ctime = (attrs[.creationDate] as? Date) ?? mtime
        let iso = ISO8601DateFormatter()

        let typeStr: String
        switch type {
        case .typeRegular: typeStr = "file"
        case .typeDirectory: typeStr = "directory"
        case .typeSymbolicLink: typeStr = "symlink"
        case .typeSocket: typeStr = "socket"
        case .typeBlockSpecial: typeStr = "block"
        case .typeCharacterSpecial: typeStr = "char"
        default: typeStr = "unknown"
        }

        var body = """
        path=\(rel)
        type=\(typeStr)
        size=\(size)
        perms=\(String(format: "%04o", perms))
        owner=\(owner)
        group=\(group)
        mtime=\(iso.string(from: mtime))
        ctime=\(iso.string(from: ctime))
        """
        if type == .typeSymbolicLink {
            let target = (try? FileManager.default.destinationOfSymbolicLink(atPath: url.path)) ?? ""
            body += "\nlink_target=\(target)"
        }
        return ToolOutput(output: body)
    }
}
