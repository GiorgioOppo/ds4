import Foundation
import DS4Core

extension ToolRegistry {
    /// Find files by NAME/PATH (project_search searches file CONTENTS).
    static let projectFind = BuiltinTool(
        spec: ToolSpec(name: "project_find",
                       description: "Find project files by name/path (case-insensitive; '*' matches any run of characters, e.g. '*Tests*.swift' or 'Sources/*View*'). Complements project_search, which searches file contents.",
                       parametersJSON: #"{"type":"object","properties":{"pattern":{"type":"string","description":"substring or *-wildcard pattern matched against relative paths"}},"required":["pattern"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "pattern") else { return "Missing 'pattern' argument." }
            return ProjectCache.shared.findTool(pattern: p)
        })
}
