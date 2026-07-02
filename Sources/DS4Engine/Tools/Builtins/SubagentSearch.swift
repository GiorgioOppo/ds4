import Foundation
import DS4Core

extension ToolRegistry {
    /// Find loadable sub-agent targets: project files whose name/content match.
    static let subagentSearch = BuiltinTool(
        spec: ToolSpec(name: "subagent_search",
                       description: "Search for loadable sub-agent targets: project files matching by content. Returns file:line entries from which to derive the path passed to subagent_run.",
                       parametersJSON: #"{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}"#),
        run: { argsJSON in
            guard let q = stringArg(argsJSON, "query") else { return "Missing 'query' argument." }
            return ProjectCache.shared.searchTool(query: q)
        })
}
