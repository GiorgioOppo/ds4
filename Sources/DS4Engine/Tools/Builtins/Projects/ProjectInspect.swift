import Foundation
import DS4Core

/// High-density, read-only project inspection. Local models pay a full
/// prefill/decode round for every tool result, so independent discovery and
/// evidence reads belong in one bounded request instead of one call per file.
extension ToolRegistry {
    static let projectInspect = BuiltinTool(
        spec: ToolSpec(
            name: "project_inspect",
            description: "Batch READ-ONLY project inspection in one call: optional Git changes, tree, folder listings, path finds, content searches, and multiple file ranges. Prefer this over one project_* call per file. At most 12 operations and 48,000 output characters.",
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
                  "type": "array",
                  "maxItems": 8,
                  "items": {
                    "type": "object",
                    "properties": {
                      "query": {"type": "string"},
                      "path": {"type": "string", "description": "Optional relative folder or file."}
                    },
                    "required": ["query"]
                  }
                },
                "read": {
                  "type": "array",
                  "maxItems": 8,
                  "items": {
                    "type": "object",
                    "properties": {
                      "path": {"type": "string"},
                      "from_line": {"type": "integer", "minimum": 1},
                      "lines": {"type": "integer", "minimum": 1, "maximum": 240}
                    },
                    "required": ["path"]
                  },
                  "description": "Independent indexed file ranges; default 160 lines each, hard-capped at 240."
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
    private static let maxOperations = 12
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
        for request in objectArray(args["read"]) {
            guard let path = request["path"] as? String, !path.isEmpty else {
                add("Read", "Missing 'path' in read request.")
                continue
            }
            let from = max(1, integer(request["from_line"]) ?? 1)
            let lines = min(max(1, integer(request["lines"]) ?? defaultReadLines), maxReadLines)
            add("Read · \(path):\(from)+\(lines)",
                ProjectCache.shared.readTool(path: path, fromLine: from, maxLines: lines))
        }

        for request in objectArray(args["search"]) {
            guard let query = request["query"] as? String, !query.isEmpty else {
                add("Search", "Missing 'query' in search request.")
                continue
            }
            let path = request["path"] as? String
            let scope = path.map { " in \($0)" } ?? ""
            add("Search · \(query)\(scope)",
                ProjectCache.shared.searchTool(query: query, pathPrefix: path))
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

        guard operations > 0 else {
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

    private static func objectArray(_ value: Any?) -> [[String: Any]] {
        (value as? [Any])?.compactMap { $0 as? [String: Any] } ?? []
    }
}
