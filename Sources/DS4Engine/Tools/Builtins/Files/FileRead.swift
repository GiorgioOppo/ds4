import Foundation
import DS4Core

extension ToolRegistry {
    /// Read any file inside the project root (raw, not limited to the index),
    /// optionally a line range [from_line, to_line].
    static let fileRead = BuiltinTool(
        spec: ToolSpec(name: "file_read",
                       description: "Read any file inside the imported project root, even if it is not indexed (e.g. dotfiles). Without from_line/to_line, returns the whole file (24 KB cap — use line ranges for anything bigger); with from_line/to_line (1-based, inclusive), returns only those numbered lines.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string","description":"path relative to the project root"},"from_line":{"type":"number","description":"first line, 1-based (optional)"},"to_line":{"type":"number","description":"last included line, 1-based (optional)"}},"required":["path"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Missing 'path' argument." }
            return ProjectCache.shared.readFileTool(path: p,
                                                    fromLine: intArg(argsJSON, "from_line"),
                                                    toLine: intArg(argsJSON, "to_line"))
        })
}
