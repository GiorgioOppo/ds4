import Foundation
import DS4Core

extension ToolRegistry {
    static let projectSearch = BuiltinTool(
        spec: ToolSpec(name: "project_search",
                       description: "Search a text (case-insensitive) across the imported project; returns file:line matches. Optional 'path' restricts the search to a subfolder or single file.",
                       parametersJSON: #"{"type":"object","properties":{"query":{"type":"string"},"path":{"type":"string","description":"optional subfolder or file to search in (relative)"}},"required":["query"]}"#),
        run: { argsJSON in
            guard let q = stringArg(argsJSON, "query") else { return "Missing 'query' argument." }
            return ProjectCache.shared.searchTool(query: q, pathPrefix: stringArg(argsJSON, "path"))
        })
}
