import Foundation
import DS4Core

extension ToolRegistry {
    /// Count the lines of a file (for choosing line ranges).
    static let fileLines = BuiltinTool(
        spec: ToolSpec(name: "file_lines",
                       description: "Count the lines and bytes of a file inside the project root. Useful before file_read/file_modify/file_add to choose correct line ranges.",
                       parametersJSON: #"{"type":"object","properties":{"path":{"type":"string","description":"path relative to the project root"}},"required":["path"]}"#),
        run: { argsJSON in
            guard let p = stringArg(argsJSON, "path") else { return "Missing 'path' argument." }
            return ProjectCache.shared.lineCountTool(path: p)
        })
}
