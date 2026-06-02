import Foundation

/// Read a file's contents, optionally limited to a line range. The
/// output is cat-n formatted (line numbers as 1-based prefixes) so
/// the model can reference exact line numbers when proposing edits —
/// same convention as Claude Code's read tool.
public struct ReadTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "read",
            description:
                "Legge un file di testo UTF-8 all'interno della working directory dell'agente. " +
                "L'output è numerato per riga. Per file molto grandi, fornisci 'offset' " +
                "e 'limit' (a base 1) per leggere una finestra invece dell'intero file.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "path": SchemaBuilder.string(description: "Path relativo alla root dell'agente, oppure assoluto."),
                    "offset": SchemaBuilder.integer(description: "Riga di partenza, a base 1.", minimum: 1),
                    "limit": SchemaBuilder.integer(description: "Numero massimo di righe da leggere.", minimum: 1),
                ],
                required: ["path"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "leggi \(input["path"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let path = try input.string("path")
        let url = try resolveInsideRoot(path, context: context)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ToolError.notFound(path)
        }
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            // The macOS sandbox surfaces a "follow-symlink-to-
            // un-bookmarked-target" failure as EPERM
            // (NSFileReadNoPermissionError). Detect it, report the
            // target's parent to the host so the project's pending
            // list grows in real time, and rewrite the error into
            // something actionable.
            if let parent = sandboxBlockedSymlinkTarget(
                from: error, accessedFrom: url)
            {
                context.reportSymlinkTargetNeeded?(parent)
                let resolved = URL(fileURLWithPath:
                    (url.path as NSString).resolvingSymlinksInPath)
                throw ToolError.permissionDenied(
                    symlinkPermissionDeniedMessage(
                        relative: path,
                        resolved: resolved,
                        grantParent: parent))
            }
            throw error
        }
        guard let text = String(data: data, encoding: .utf8) else {
            throw ToolError.invalidInput("file is not valid UTF-8")
        }
        let allLines = text.components(separatedBy: "\n")
        let offset = max(1, input.optionalInteger("offset") ?? 1)
        let limit = input.optionalInteger("limit") ?? 2000
        let start = offset - 1
        guard start < allLines.count else {
            return ToolOutput(output: "",
                              metadata: ["lines": "0", "total": "\(allLines.count)"])
        }
        let end = min(allLines.count, start + limit)
        let window = allLines[start..<end]
        let formatted = window.enumerated().map { idx, line in
            let n = start + idx + 1
            return "\(String(format: "%6d", n))\t\(line)"
        }.joined(separator: "\n")
        return ToolOutput(
            output: formatted,
            metadata: [
                "lines": "\(window.count)",
                "total": "\(allLines.count)",
                "path": url.path,
            ]
        )
    }
}
