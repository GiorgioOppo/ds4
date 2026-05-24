import Foundation

/// Regex search across the agent's root. Pure-Swift implementation
/// using `NSRegularExpression` plus a manual walker — keeps the tool
/// dependency-free (no ripgrep shell-out) which matters because we
/// want this to work even when `shell` is denied.
///
/// Not as fast as ripgrep on huge trees; future versions can shell
/// out when `which rg` succeeds.
public struct GrepTool: Tool {
    public init() {}

    public var schema: ToolSchema {
        ToolSchema(
            name: "grep",
            description:
                "Cerca una regex all'interno dei file sotto la root dell'agente. " +
                "Restituisce righe 'path:line:match', limitate da 'limit'. " +
                "Usa 'glob' per restringere i file da scansionare.",
            category: .readOnly,
            inputSchema: SchemaBuilder.object(
                properties: [
                    "pattern": SchemaBuilder.string(description: "Regex (dialetto NSRegularExpression)."),
                    "glob": SchemaBuilder.string(description: "Glob opzionale per filtrare i file. Default '**/*'."),
                    "caseInsensitive": SchemaBuilder.boolean(description: "Default false.", defaultValue: false),
                    "limit": SchemaBuilder.integer(description: "Numero massimo di righe di match. Default 200.", minimum: 1),
                ],
                required: ["pattern"]
            )
        )
    }

    public func permissionSummary(input: [String: Any]) -> String {
        "grep \(input["pattern"] as? String ?? "?")"
    }

    public func run(input: [String: Any], context: ToolContext) async throws -> ToolOutput {
        let pattern = try input.string("pattern")
        let limit = input.optionalInteger("limit") ?? 200
        let caseInsensitive = input.optionalBool("caseInsensitive") ?? false
        let globPattern = input.optionalString("glob") ?? "**/*"
        var options: NSRegularExpression.Options = []
        if caseInsensitive { options.insert(.caseInsensitive) }
        let regex: NSRegularExpression
        do {
            regex = try NSRegularExpression(pattern: pattern, options: options)
        } catch {
            throw ToolError.invalidInput("invalid regex: \(error.localizedDescription)")
        }

        let glob = try globToRegex(globPattern)
        let rootPath = context.rootDirectory.standardizedFileURL.path
        let walker = FileManager.default.enumerator(
            at: context.rootDirectory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        )
        var matches: [String] = []
        outer: while let next = walker?.nextObject() as? URL {
            if context.isCancelled() { break }
            let isFile = (try? next.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) ?? false
            guard isFile else { continue }
            let resolved = next.standardizedFileURL
            let rel = String(resolved.path.dropFirst(rootPath.count).drop(while: { $0 == "/" }))
            let nsRel = rel as NSString
            guard glob.firstMatch(in: rel, range: NSRange(location: 0, length: nsRel.length)) != nil else { continue }
            guard let data = try? Data(contentsOf: resolved),
                  let text = String(data: data, encoding: .utf8) else { continue }
            let lines = text.components(separatedBy: "\n")
            for (i, line) in lines.enumerated() {
                let nsLine = line as NSString
                if regex.firstMatch(in: line, range: NSRange(location: 0, length: nsLine.length)) != nil {
                    matches.append("\(rel):\(i + 1):\(line)")
                    if matches.count >= limit { break outer }
                }
            }
        }
        return ToolOutput(
            output: matches.joined(separator: "\n"),
            metadata: ["matches": "\(matches.count)"]
        )
    }

    /// Same minimal glob→regex translation as `GlobTool` — duplicated
    /// because the two tools are independent and we don't want them
    /// to import each other's implementation details.
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
