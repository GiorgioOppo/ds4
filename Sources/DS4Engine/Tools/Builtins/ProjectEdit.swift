import Foundation
import DS4Core

extension ToolRegistry {
    static let projectEdit = BuiltinTool(
        spec: ToolSpec(name: "project_edit",
                       description: "Replace ONE exact occurrence of 'find' with 'replace' in a project file. 'find' must match exactly (incl. indentation) and be unique in the file — include surrounding lines to disambiguate.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string"},"find":{"type":"string"},"replace":{"type":"string"}},"required":["path","find","replace"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Argomento 'path' mancante." }
            guard let f = stringArg(argsJSON, "find") else { return "Argomento 'find' mancante." }
            let r = stringArg(argsJSON, "replace") ?? ""
            return ProjectCache.shared.editTool(path: p, find: f, replace: r)
        })
}
