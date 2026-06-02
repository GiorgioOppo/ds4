import Foundation

/// Print the last N lines (or bytes) of a file. Pure Swift; sandboxed.
public struct TailTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "tail",
            description:
                "Stampa le ultime N righe di un file (default 10). " +
                "Usalo per le entry più recenti in un log, l'ultima traccia di errore, o il fondo di un output lungo. " +
                "Per l'intero file usa 'read'; per l'inizio usa 'head'. " +
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
        "tail \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let rel = try input.string("path")
        let url = try resolveInsideRoot(rel, context: context)
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
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
            let count = min(nBytes, data.count)
            let suffix = data.suffix(count)
            let text = String(data: suffix, encoding: .utf8) ?? ""
            return ToolOutput(output: text, metadata: ["bytes": "\(suffix.count)"])
        }
        let nLines = input.optionalInteger("lines") ?? 10
        guard let text = String(data: data, encoding: .utf8) else {
            throw ToolError.invalidInput("'\(rel)' is not UTF-8")
        }
        let lines = text.components(separatedBy: "\n")
        let start = max(0, lines.count - nLines)
        let kept = lines[start..<lines.count].joined(separator: "\n")
        return ToolOutput(output: kept, metadata: ["lines": "\(lines.count - start)"])
    }
}
