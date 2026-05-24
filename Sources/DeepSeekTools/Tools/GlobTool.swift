import Foundation

/// Find files by glob pattern inside the agent's root. Pattern syntax
/// mirrors `fnmatch`: `*` matches any run except `/`, `?` matches one
/// char, `**` matches across directories. Results are sorted by
/// modification time (most recent first) so the model surfaces
/// currently-relevant files.
public struct GlobTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "glob",
            description:
                "Trova i file all'interno della working directory tramite pattern glob " +
                "(es. 'Sources/**/*.swift'). Restituisce fino a 'limit' path " +
                "ordinati per modifica più recente.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "pattern": SchemaBuilder.string(description: "Pattern glob."),
                    "limit": SchemaBuilder.integer(description: "Numero massimo di risultati. Default 200.", minimum: 1),
                ],
                required: ["pattern"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "glob \(input["pattern"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let pattern = try input.string("pattern")
        let limit = input.optionalInteger("limit") ?? 200
        let regex = try globToRegex(pattern)
        let rootPath = context.rootDirectory.standardizedFileURL.path
        let walker = FileManager.default.enumerator(
            at: context.rootDirectory,
            includingPropertiesForKeys: [.contentModificationDateKey, .isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        )
        var hits: [(URL, Date)] = []
        while let next = walker?.nextObject() as? URL {
            if context.isCancelled() { break }
            let resolved = next.standardizedFileURL
            guard resolved.path.hasPrefix(rootPath) else { continue }
            let isFile = (try? resolved.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) ?? false
            guard isFile else { continue }
            let rel = String(resolved.path.dropFirst(rootPath.count).drop(while: { $0 == "/" }))
            let nsRel = rel as NSString
            let nsRange = NSRange(location: 0, length: nsRel.length)
            guard regex.firstMatch(in: rel, range: nsRange) != nil else { continue }
            let mtime = (try? resolved.resourceValues(forKeys: [.contentModificationDateKey])
                .contentModificationDate) ?? .distantPast
            hits.append((resolved, mtime))
            if hits.count > limit * 4 { break } // cheap early stop
        }
        hits.sort { $0.1 > $1.1 }
        let trimmed = hits.prefix(limit)
        let output = trimmed.map { tuple in
            String(tuple.0.path.dropFirst(rootPath.count).drop(while: { $0 == "/" }))
        }.joined(separator: "\n")
        return ToolOutput(
            output: output,
            metadata: ["matches": "\(trimmed.count)"]
        )
    }

    /// Translate a fnmatch-style glob to an `NSRegularExpression`.
    /// Supports `*`, `?`, `**`, and literal escaping for regex
    /// metachars. Not exhaustive — character classes are not
    /// supported. Good enough for the patterns models actually emit.
    private func globToRegex(_ pattern: String) throws -> NSRegularExpression {
        var regex = "^"
        var i = pattern.startIndex
        while i < pattern.endIndex {
            let c = pattern[i]
            switch c {
            case "*":
                let next = pattern.index(after: i)
                if next < pattern.endIndex && pattern[next] == "*" {
                    regex += ".*"
                    i = pattern.index(after: next)
                    continue
                } else {
                    regex += "[^/]*"
                }
            case "?":
                regex += "[^/]"
            case ".", "(", ")", "+", "|", "^", "$", "{", "}", "[", "]", "\\":
                regex += "\\\(c)"
            default:
                regex.append(c)
            }
            i = pattern.index(after: i)
        }
        regex += "$"
        return try NSRegularExpression(pattern: regex)
    }
}
