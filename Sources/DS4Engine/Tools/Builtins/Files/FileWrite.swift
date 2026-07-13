import Foundation
import DS4Core

extension ToolRegistry {
    /// Create/overwrite the WHOLE file inside the project root.
    static let fileWrite = BuiltinTool(
        spec: ToolSpec(name: "file_write",
                       description: "Create or overwrite the whole file inside the imported project root (any extension; creates folders). Use file_add to add lines and file_modify to modify existing lines.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string","description":"path relative to the project root"},"content":{"type":"string","description":"full file content"}},"required":["path","content"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Missing 'path' argument." }
            guard let c = stringArg(argsJSON, "content") else { return "Missing 'content' argument." }
            return ProjectCache.shared.writeFileTool(path: p, content: c)
        })
}
