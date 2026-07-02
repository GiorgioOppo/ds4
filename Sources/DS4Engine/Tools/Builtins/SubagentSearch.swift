import Foundation
import DS4Core

extension ToolRegistry {
    /// Find loadable sub-agent targets: project files whose name/content match.
    static let subagentSearch = BuiltinTool(
        spec: ToolSpec(name: "subagent_search",
                       description: "Cerca i target caricabili come sub-agent: file del progetto che corrispondono (per contenuto). Restituisce 'file:riga' da cui ricavare il percorso da passare a subagent_run.",
                       parametersJSON: #"{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}"#),
        run: { argsJSON in
            guard let q = stringArg(argsJSON, "query") else { return "Argomento 'query' mancante." }
            return ProjectCache.shared.searchTool(query: q)
        })
}
