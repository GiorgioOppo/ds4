import Foundation
import DS4Core

extension ToolRegistry {
    static let projectSearch = BuiltinTool(
        spec: ToolSpec(name: "project_search",
                       description: "Search a text (case-insensitive) across the imported project; returns file:line matches.",
                       parametersJSON: #"{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}"#),
        run: { argsJSON in
            guard let q = stringArg(argsJSON, "query") else { return "Argomento 'query' mancante." }
            return ProjectCache.shared.searchTool(query: q)
        })
}
