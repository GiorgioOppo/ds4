import Foundation
import DS4Core

extension ToolRegistry {
    static let projectSearch = BuiltinTool(
        spec: ToolSpec(name: "project_search",
                       description: "Search project contents (case-insensitive) and return file:line matches. Use 'queries' to batch up to 8 independent texts in one call; legacy 'query' still accepts one text. Provide at least one of them. Optional 'path' restricts every search to a relative subfolder or file.",
                       parametersJSON: #"{"type":"object","properties":{"queries":{"type":"array","maxItems":8,"items":{"type":"string"},"description":"texts to search together (preferred; max 8)"},"query":{"type":"string","description":"legacy single text; use queries for multiple texts"},"path":{"type":"string","description":"optional subfolder or file applied to every search (relative)"}}}"#),
        run: ProjectSearchRequest.run)
}

/// Parses the compatibility single-query shape and the batch shape in one
/// place. Tool output is capped because it is fed back into the local model and
/// therefore directly affects the next prefill cost.
private enum ProjectSearchRequest {
    private static let maxQueries = 8
    private static let maxOutputCharacters = 32_000

    static func run(_ argumentsJSON: String) -> String {
        guard let data = argumentsJSON.data(using: .utf8),
              let arguments = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return "Invalid JSON arguments."
        }

        let legacyQuery = (arguments["query"] as? String).map { [$0] } ?? []
        let batchQueries = (arguments["queries"] as? [Any])?.compactMap { $0 as? String } ?? []
        let path = arguments["path"] as? String

        var seen = Set<String>()
        let requested = (legacyQuery + batchQueries).compactMap { raw -> String? in
            let query = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !query.isEmpty, seen.insert(query.lowercased()).inserted else { return nil }
            return query
        }

        guard !requested.isEmpty else {
            return "Missing 'queries' (or legacy 'query') argument."
        }

        // Preserve the original unsectioned response for existing callers.
        if batchQueries.isEmpty, requested.count == 1 {
            return ProjectCache.shared.searchTool(query: requested[0], pathPrefix: path)
        }

        let queries = Array(requested.prefix(maxQueries))
        let results = ProjectCache.shared.searchTools(queries: queries, pathPrefix: path)
        var sections = queries.indices.map { index in
            let query = queries[index]
            let title = query.replacingOccurrences(of: "\n", with: " ").prefix(160)
            return "## Search \(index + 1)/\(queries.count) · \(title)\n\(results[index])"
        }
        if requested.count > maxQueries {
            sections.append("## Limit\n\(requested.count - maxQueries) search(es) omitted; project_search accepts at most \(maxQueries) per call.")
        }

        let output = sections.joined(separator: "\n\n")
        guard output.count > maxOutputCharacters else { return output }
        let marker = "\n\n... [project_search truncated at \(maxOutputCharacters) characters; narrow the path or queries]"
        return String(output.prefix(max(0, maxOutputCharacters - marker.count))) + marker
    }
}
