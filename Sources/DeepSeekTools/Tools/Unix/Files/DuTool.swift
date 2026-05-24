import Foundation

/// Disk usage summary. Pure Swift via `UnixWalker`; reports cumulative
/// size in bytes. Symlinks are NOT followed by default.
public struct DuTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "du",
            description:
                "Uso del disco per un albero di directory. Restituisce una riga 'size path' per ogni figlio diretto quando 'summarize=false', " +
                "o una singola riga per la root altrimenti. Le dimensioni sono in byte. I symlink non vengono seguiti.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Directory, relativa alla root dell'agente. Default '.'."),
                    "summarize": SchemaBuilder.boolean(description: "Se true, stampa solo il totale complessivo.", defaultValue: false),
                    "humanReadable": SchemaBuilder.boolean(description: "Formatta le dimensioni come KB/MB/GB. Default false.", defaultValue: false),
                ]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "du \(input["path"] as? String ?? ".")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = input.optionalString("path") ?? "."
        let summarize = input.optionalBool("summarize") ?? false
        let human = input.optionalBool("humanReadable") ?? false
        let root = try resolveInsideRoot(rel, context: context)

        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDir) else {
            throw ToolError.notFound("'\(rel)' does not exist")
        }
        if !isDir.boolValue {
            let size = sizeOf(root)
            return ToolOutput(output: "\(format(size, human: human)) \(rel)")
        }

        if summarize {
            let total = totalSize(root, isCancelled: context.isCancelled)
            return ToolOutput(output: "\(format(total, human: human)) \(rel)")
        }

        var lines: [String] = []
        var grand: Int64 = 0
        for child in (try? FileManager.default.contentsOfDirectory(atPath: root.path).sorted()) ?? [] {
            if context.isCancelled() { break }
            let url = root.appendingPathComponent(child)
            let size = totalSize(url, isCancelled: context.isCancelled)
            grand += size
            let childRel = rel == "." ? child : "\(rel)/\(child)"
            lines.append("\(format(size, human: human)) \(childRel)")
        }
        lines.append("\(format(grand, human: human)) \(rel)")
        return ToolOutput(output: lines.joined(separator: "\n"),
                          metadata: ["total": "\(grand)"])
    }

    private func totalSize(_ url: URL, isCancelled: @Sendable () -> Bool) -> Int64 {
        var total: Int64 = 0
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), !isDir.boolValue {
            return sizeOf(url)
        }
        UnixWalker.walk(root: url, isCancelled: isCancelled) { entry in
            if !entry.isDirectory && !entry.isSymlink {
                total += self.sizeOf(entry.url)
            }
            return true
        }
        return total
    }

    private func sizeOf(_ url: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value) ?? 0
    }

    private func format(_ bytes: Int64, human: Bool) -> String {
        if !human { return String(format: "%lld", bytes) }
        let units: [(Int64, String)] = [(1 << 30, "G"), (1 << 20, "M"), (1 << 10, "K")]
        for (mag, suffix) in units where bytes >= mag {
            return String(format: "%.1f%@", Double(bytes) / Double(mag), suffix)
        }
        return "\(bytes)B"
    }
}
