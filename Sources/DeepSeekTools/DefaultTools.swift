import Foundation

/// One-call helper to populate a `ToolRegistry` with the native tools
/// shipped by this package. Used by the GUI / CLI to wire a sensible
/// default; callers can still register additional tools (e.g.
/// MCP-backed ones) on the same registry afterwards.
///
/// `includeShell` / `includeNetwork` / `includeRepoClone` let the
/// caller opt out of the most invasive tools — for example, a
/// `.plan`-only agent often wants the network off too. The `lsp`
/// tool is *not* registered by default because its implementation
/// is a stub (`ToolError.notImplemented`); pass `includeStubs: true`
/// to surface it anyway.
public enum DefaultTools {
    public static func standard(planStore: PlanStore,
                                includeShell: Bool = true,
                                includeNetwork: Bool = true,
                                includeRepoClone: Bool = true,
                                includeStubs: Bool = false,
                                shellUsesSandbox: Bool = false) -> [Tool] {
        var tools: [Tool] = [
            ReadTool(),
            WriteTool(),
            EditTool(),
            GlobTool(),
            GrepTool(),
            ApplyPatchTool(),
            RepoOverviewTool(),
            PlanTool(store: planStore),
            TaskTool(store: planStore),
            TodoTool(store: planStore),
        ]
        if includeShell {
            tools.append(ShellTool(useSandbox: shellUsesSandbox))
        }
        if includeNetwork {
            tools.append(WebFetchTool())
            tools.append(WebSearchTool())
        }
        if includeRepoClone {
            tools.append(RepoCloneTool())
        }
        if includeStubs {
            tools.append(LSPTool())
        }
        return tools
    }
}
