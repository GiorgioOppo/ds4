import Foundation
import DS4Core

extension ToolRegistry {
    static let projectWrite = BuiltinTool(
        spec: ToolSpec(name: "project_write",
                       description: "Create or overwrite a TEXT file inside the imported project. Use project_edit for small changes to existing files.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string","description":"relative path"},"content":{"type":"string","description":"full file content"}},"required":["path","content"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Argomento 'path' mancante." }
            guard let c = stringArg(argsJSON, "content") else { return "Argomento 'content' mancante." }
            return ProjectCache.shared.writeTool(path: p, content: c)
        })
}
