import Foundation
import DS4Core

extension ToolRegistry {
    static let projectRead = BuiltinTool(
        spec: ToolSpec(name: "project_read",
                       description: "Read a project file with line numbers. 'path' relative; optional 'from_line' to continue; optional 'lines' to read up to 400 lines in ONE call (default 120) — read long files in few large chunks, not many small ones.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string"},"from_line":{"type":"number"},"lines":{"type":"number","description":"lines to read in this call, up to 400 (default 120)"}},"required":["path"]}"#),
        run: { argsJSON in
            guard let path = stringArg(argsJSON, "path") else { return "Missing 'path' argument." }
            let from = intArg(argsJSON, "from_line") ?? 1
            return ProjectCache.shared.readTool(path: path, fromLine: from,
                                                maxLines: intArg(argsJSON, "lines"))
        })
}
