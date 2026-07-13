import Foundation
import DS4Core

extension ToolRegistry {
    /// DELETE one file inside the project root (never directories).
    static let fileDelete = BuiltinTool(
        spec: ToolSpec(name: "file_delete",
                       description: "Delete ONE file inside the imported project root (never directories). Destructive: use only when the task requires removing a file; in a git repo the file stays recoverable.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string","description":"path relative to the project root"}},"required":["path"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Missing 'path' argument." }
            return ProjectCache.shared.deleteFileTool(path: p)
        })
}
