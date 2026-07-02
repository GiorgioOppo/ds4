import Foundation
import DS4Core

extension ToolRegistry {
    /// Whole-project overview in one call (project_list shows one level per call).
    static let projectTree = BuiltinTool(
        spec: ToolSpec(name: "project_tree",
                       description: "Compact tree of the imported project: directories with file counts plus root files, in a single call. Optional 'depth' (1-6, default 3). Use it first to orient, then project_list / project_read for details.",
                       parametersJSON: #"{"type":"object","properties":{"depth":{"type":"number","description":"directory depth to show, 1-6 (default 3)"}}}"#),
        run: { argsJSON in
            ProjectCache.shared.treeTool(maxDepth: intArg(argsJSON, "depth") ?? 3)
        })
}
