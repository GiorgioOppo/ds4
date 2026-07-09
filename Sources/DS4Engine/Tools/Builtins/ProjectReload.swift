import Foundation
import DS4Core

extension ToolRegistry {
    /// Rebuild the active project's index from disk. The index is otherwise a
    /// snapshot taken at import: changes made OUTSIDE the tools (git stash,
    /// scripts, the user's editor adding files) are invisible to
    /// project_list/find/search until re-indexed.
    static let projectReload = BuiltinTool(
        spec: ToolSpec(name: "project_reload",
                       description: "Re-index the active project from disk. Use after changes made outside the project tools (scripts, the user's editor, git operations) so project_list/project_find/project_search see the current tree. The git tool already re-indexes automatically after 'stash'.",
                       parametersJSON: #"{"type":"object","properties":{}}"#),
        run: { _ in
            guard let info = ProjectCache.shared.reload() else { return "No project imported." }
            return "Re-indexed \"\(info.name)\": \(info.fileCount) text files, \(info.totalBytes / 1024) KB."
        })
}
