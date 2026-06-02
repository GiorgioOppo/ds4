import Foundation

/// Print the first N lines (or bytes) of a file. Pure Swift; sandboxed.
public struct HeadTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "head",
            description:
                "Stampa le prime N righe di un file (default 10). " +
                "Usalo quando ti serve un piccolo prefisso — es. le prime righe di un log, " +
                "lo shebang di uno script, l'header di un CSV. " +
                "Per l'intero file usa 'read'; per la coda usa 'tail'. " +
                "'bytes' conta i byte invece delle righe.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Path del file, relativo alla root dell'agente."),
                    "lines": SchemaBuilder.integer(description: "Righe da stampare. Default 10.", minimum: 1),
                    "bytes": SchemaBuilder.integer(description: "Se impostato, stampa questo numero di byte e ignora 'lines'.", minimum: 1),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "head \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let url = try resolveInsideRoot(rel, context: context)
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            // Sandbox-blocked symlink target → push the parent into
            // the project's pending list and explain how to grant
            // access instead of returning the opaque
            // "cannot read" string.
            if let parent = sandboxBlockedSymlinkTarget(
                from: error, accessedFrom: url)
            {
                context.reportSymlinkTargetNeeded?(parent)
                let resolved = URL(fileURLWithPath:
                    (url.path as NSString).resolvingSymlinksInPath)
                throw ToolError.permissionDenied(
                    symlinkPermissionDeniedMessage(
                        relative: rel,
                        resolved: resolved,
                        grantParent: parent))
            }
            throw ToolError.notFound("cannot read '\(rel)'")
        }

        if let nBytes = input.optionalInteger("bytes") {
            let prefix = data.prefix(nBytes)
            let text = String(data: prefix, encoding: .utf8) ?? ""
            return ToolOutput(output: text, metadata: ["bytes": "\(prefix.count)"])
        }
        let nLines = input.optionalInteger("lines") ?? 10
        guard let text = String(data: data, encoding: .utf8) else {
            throw ToolError.invalidInput("'\(rel)' is not UTF-8")
        }
        let lines = text.split(separator: "\n", maxSplits: nLines, omittingEmptySubsequences: false)
        let kept = lines.prefix(nLines).joined(separator: "\n")
        return ToolOutput(output: kept, metadata: ["lines": "\(min(nLines, lines.count))"])
    }
}
