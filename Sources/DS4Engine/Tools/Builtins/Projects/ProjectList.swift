import Foundation
import DS4Core

// Project-exploration tools (read-only over the imported ProjectCache; results
// enter the chat ONLY when the model calls them, so the project import never
// alters the conversation memory).

extension ToolRegistry {
    static let projectList = BuiltinTool(
        spec: ToolSpec(name: "project_list",
                       description: "List files/folders of the imported project. Optional 'path' (relative) lists a subfolder; omit it for the root.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string"}}}"#),
        run: { argsJSON in
            let path = stringArg(argsJSON, "path") ?? ""
            return ProjectCache.shared.listTool(path: path)
        })
}
