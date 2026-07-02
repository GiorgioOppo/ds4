import Foundation
import DS4Core

extension ToolRegistry {
    static let projectRead = BuiltinTool(
        spec: ToolSpec(name: "project_read",
                       description: "Read a project file (about 120 lines per call, with line numbers). 'path' relative; optional 'from_line' to continue.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string"},"from_line":{"type":"number"}},"required":["path"]}"#),
        run: { argsJSON in
            guard let path = stringArg(argsJSON, "path") else { return "Argomento 'path' mancante." }
            let from = intArg(argsJSON, "from_line") ?? 1
            return ProjectCache.shared.readTool(path: path, fromLine: from)
        })
}
