import Foundation
import DS4Core

/// High-density, read-only project inspection. Local models pay a full
/// prefill/decode round for every tool result, so independent discovery and
/// evidence reads belong in one bounded request instead of one call per file.
extension ToolRegistry {
    static let projectInspect = BuiltinTool(
        spec: ToolSpec(
            name: "project_inspect",
            description: "Batch READ-ONLY project inspection in one call: optional Git changes, tree, folder listings, path finds, content searches, and multiple file ranges. 'search' accepts one query or an array; 'read' accepts one path or an array. Objects remain available for per-search paths and line ranges. Prefer this over multiple project_* invokes. At most 24 operations and 48,000 output characters.",
            parametersJSON: #"""
            {
              "type": "object",
              "properties": {
                "changes": {
                  "type": "string",
                  "enum": ["summary", "diff"],
                  "description": "Read-only working-tree scope: summary is status/stat; diff includes the tracked patch. Also refreshes the project index."
                },
                "tree_depth": {
                  "type": "integer",
                  "minimum": 1,
                  "maximum": 6,
                  "description": "Include the compact project tree to this depth."
                },
                "lists": {
                  "type": "array",
                  "maxItems": 4,
                  "items": {"type": "string"},
                  "description": "Relative folders to list. Use an empty string for the root."
                },
                "find": {
                  "type": "array",
                  "maxItems": 6,
                  "items": {"type": "string"},
                  "description": "File-name/path patterns; '*' is supported."
                },
                "search": {
                  "type": ["array", "string", "object"],
                  "maxItems": 16,
                  "items": {
                    "oneOf": [
                      {"type": "string"},
                      {
                        "type": "object",
                        "properties": {
                          "query": {"type": "string"},
                          "pattern": {"type": "string", "description": "Safe pattern alias: '|' means alternatives and '.*' an ordered gap."},
                          "path": {"type": "string", "description": "Optional relative folder or file."}
                        },
                        "anyOf": [
                          {"required": ["query"]},
                          {"required": ["pattern"]}
                        ]
                      }
                    ]
                  },
                  "description": "One query string/object, or an array mixing both (max 16). Objects allow a per-query path and accept query or safe pattern."
                },
                "read": {
                  "type": ["array", "string", "object"],
                  "maxItems": 12,
                  "items": {
                    "oneOf": [
                      {"type": "string"},
                      {
                        "type": "object",
                        "properties": {
                          "path": {"type": "string"},
                          "from_line": {"type": "integer", "minimum": 1},
                          "lines": {"type": "integer", "minimum": 1, "maximum": 240}
                        },
                        "required": ["path"]
                      }
                    ]
                  },
                  "description": "One path/range object, or an array mixing both (max 12). Plain paths use the default 160 lines; objects select from_line/lines, hard-capped at 240."
                },
                "max_chars": {
                  "type": "integer",
                  "minimum": 8000,
                  "maximum": 48000,
                  "description": "Global response budget; default 32000 characters."
                }
              }
            }
            """#
        ),
        run: { ProjectInspection.run(argumentsJSON: $0) }
    )
}

private enum ProjectInspection {
    private static let maxOperations = 24
    private static let defaultOutputCharacters = 32_000
    private static let maxOutputCharacters = 48_000
    private static let defaultReadLines = 160
    private static let maxReadLines = 240

    static func run(argumentsJSON: String) -> String {
        guard let data = argumentsJSON.data(using: .utf8),
              let args = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return "Invalid JSON arguments."
        }

        var sections: [String] = []
        var operations = 0
        var omittedOperations = 0

        func add(_ title: String, _ body: @autoclosure () -> String) {
            guard operations < maxOperations else {
                omittedOperations += 1
                return
            }
            operations += 1
            sections.append("## \(title)\n\(body())")
        }

        // A change-oriented review must see the current disk contents rather
        // than a potentially stale lazy cache. Reload changes no project files.
        if let changes = args["changes"] as? String {
            _ = ProjectCache.shared.reload()
            switch changes {
            case "summary":
                add("Git changes · summary", gitSummary())
            case "diff":
                add("Git changes · diff", gitDiff())
            default:
                add("Git changes", "Invalid 'changes' value; use 'summary' or 'diff'.")
            }
        }

        // Put requested source evidence before discovery output. If the global
        // character budget is reached, concrete review evidence survives.
        for request in readRequests(args["read"]) {
            guard let path = request["path"] as? String, !path.isEmpty else {
                add("Read", "Missing 'path' in read request.")
                continue
            }
            let from = max(1, integer(request["from_line"]) ?? 1)
            let lines = min(max(1, integer(request["lines"]) ?? defaultReadLines), maxReadLines)
            add("Read · \(path):\(from)+\(lines)",
                ProjectCache.shared.readTool(path: path, fromLine: from, maxLines: lines))
        }

        let requestedSearches = searchRequests(args["search"])
        let availableSearchSlots = max(0, maxOperations - operations)
        let executableSearches = Array(requestedSearches.prefix(availableSearchSlots))
        omittedOperations += max(0, requestedSearches.count - executableSearches.count)

        // Group equal scopes so every file is decoded/lowercased only once for
        // all queries in that scope. Result slots restore the requested order.
        struct SearchScope: Hashable { let path: String? }
        var grouped: [SearchScope: [(index: Int, query: String)]] = [:]
        var searchOutputs = [String?](repeating: nil, count: executableSearches.count)
        for (index, request) in executableSearches.enumerated() {
            if let query = request["query"] as? String, !query.isEmpty {
                grouped[SearchScope(path: request["path"] as? String), default: []]
                    .append((index, query))
            } else if let pattern = request["pattern"] as? String, !pattern.isEmpty {
                searchOutputs[index] = ProjectCache.shared.searchPatternTool(
                    pattern: pattern, pathPrefix: request["path"] as? String
                )
            } else {
                searchOutputs[index] = "Missing 'query' or 'pattern' in search request."
            }
        }
        for (scope, group) in grouped {
            let results = ProjectCache.shared.searchTools(
                queries: group.map(\.query), pathPrefix: scope.path
            )
            for (item, result) in zip(group, results) {
                searchOutputs[item.index] = result
            }
        }
        for (index, request) in executableSearches.enumerated() {
            let label: String
            if let query = request["query"] as? String, !query.isEmpty {
                label = query
            } else if let pattern = request["pattern"] as? String, !pattern.isEmpty {
                label = "pattern \(pattern)"
            } else {
                add("Search", searchOutputs[index] ?? "Missing 'query' or 'pattern' in search request.")
                continue
            }
            let path = request["path"] as? String
            let scope = path.map { " in \($0)" } ?? ""
            add("Search · \(label)\(scope)", searchOutputs[index] ?? "No results for '\(label)'.")
        }

        for pattern in stringArray(args["find"]) {
            add("Find · \(pattern)", ProjectCache.shared.findTool(pattern: pattern))
        }

        for path in stringArray(args["lists"]) {
            add("List · \(path.isEmpty ? "." : path)", ProjectCache.shared.listTool(path: path))
        }

        if let requestedDepth = integer(args["tree_depth"]) {
            let depth = min(max(1, requestedDepth), 6)
            add("Tree · depth \(depth)", ProjectCache.shared.treeTool(maxDepth: depth))
        }

        guard !sections.isEmpty else {
            return "No inspection requested. Supply changes, tree_depth, lists, find, search, or read."
        }
        if omittedOperations > 0 {
            sections.append("## Limit\n\(omittedOperations) operation(s) omitted; project_inspect accepts at most \(maxOperations) per call.")
        }

        let requestedBudget = integer(args["max_chars"]) ?? defaultOutputCharacters
        let budget = min(max(8_000, requestedBudget), maxOutputCharacters)
        let joined = sections.joined(separator: "\n\n")
        guard joined.count > budget else { return joined }
        let marker = "\n\n... [project_inspect truncated at \(budget) characters; narrow ranges or use one evidence follow-up]"
        return String(joined.prefix(max(0, budget - marker.count))) + marker
    }

    private static func gitSummary() -> String {
        let status = GitTool.run(argsLine: "status --short")
        let stat = GitTool.run(argsLine: "diff HEAD --no-ext-diff --stat")
        return "git status --short:\n\(status)\n\ngit diff HEAD --stat:\n\(stat)"
    }

    private static func gitDiff() -> String {
        let status = GitTool.run(argsLine: "status --short")
        let diff = GitTool.run(argsLine: "diff HEAD --no-ext-diff --unified=12")
        return "git status --short:\n\(status)\n\ngit diff HEAD --unified=12:\n\(diff)"
    }

    private static func integer(_ value: Any?) -> Int? {
        if let number = value as? NSNumber { return number.intValue }
        if let string = value as? String { return Int(string) }
        return nil
    }

    private static func stringArray(_ value: Any?) -> [String] {
        (value as? [Any])?.compactMap { $0 as? String } ?? []
    }

    /// Models do not always agree on whether a one-item batch should be a bare
    /// string, object, array, or even a JSON array encoded as a string. Normalize
    /// all those shapes before executing anything; malformed strings remain
    /// ordinary scalar values and get the usual path/query validation.
    private static func batchElements(_ value: Any?) -> [Any] {
        guard let value else { return [] }
        if let array = value as? [Any] { return array }
        if let string = value as? String {
            let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
            if (trimmed.hasPrefix("[") || trimmed.hasPrefix("{")),
               let data = trimmed.data(using: .utf8),
               let decoded = try? JSONSerialization.jsonObject(with: data) {
                return batchElements(decoded)
            }
        }
        return [value]
    }

    private static func readRequests(_ value: Any?) -> [[String: Any]] {
        batchElements(value).compactMap { element in
            if let path = element as? String { return ["path": path] }
            return element as? [String: Any]
        }
    }

    private static func searchRequests(_ value: Any?) -> [[String: Any]] {
        batchElements(value).compactMap { element in
            if let query = element as? String { return ["query": query] }
            return element as? [String: Any]
        }
    }
}
