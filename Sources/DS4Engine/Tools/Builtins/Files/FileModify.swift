import Foundation
import DS4Core

extension ToolRegistry {
    /// MODIFY (replace) a line range.
    static let fileModify = BuiltinTool(
        spec: ToolSpec(name: "file_modify",
                       description: "Modify a file by replacing lines [from_line, to_line] (1-based, inclusive) with 'content' (to_line omitted = one line; empty content = delete those lines). The file must exist. Prefer project_edit for exact text replacements.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string","description":"path relative to the project root"},"content":{"type":"string","description":"replacement lines (empty = delete)"},"from_line":{"type":"number","description":"first line to replace, 1-based"},"to_line":{"type":"number","description":"last included line, 1-based (optional = from_line)"}},"required":["path","content","from_line"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Missing 'path' argument." }
            guard let c = stringArg(argsJSON, "content") else { return "Missing 'content' argument." }
            guard let f = intArg(argsJSON, "from_line") else { return "Missing 'from_line' argument." }
            return ProjectCache.shared.modifyLinesTool(path: p, content: c, fromLine: f, toLine: intArg(argsJSON, "to_line"))
        })
}
