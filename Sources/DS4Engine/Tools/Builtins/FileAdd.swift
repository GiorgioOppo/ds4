import Foundation
import DS4Core

extension ToolRegistry {
    /// ADD lines (insert) without overwriting.
    static let fileAdd = BuiltinTool(
        spec: ToolSpec(name: "file_add",
                       description: "Add lines to a file without overwriting: inserts 'content' before 'at_line' (1-based); without 'at_line', appends at the end. Creates the file if it does not exist.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string","description":"path relative to the project root"},"content":{"type":"string","description":"lines to insert"},"at_line":{"type":"number","description":"insert before this 1-based line (optional: append)"}},"required":["path","content"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Missing 'path' argument." }
            guard let c = stringArg(argsJSON, "content") else { return "Missing 'content' argument." }
            return ProjectCache.shared.addLinesTool(path: p, content: c, atLine: intArg(argsJSON, "at_line"))
        })
}
