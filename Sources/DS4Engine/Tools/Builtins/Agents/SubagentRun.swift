import Foundation
import DS4Core

extension ToolRegistry {
    /// Delegate a focused task to an isolated sub-agent. EXECUTED BY THE ENGINE
    /// (InferenceService.runSubAgent), which intercepts this call so the sub-agent
    /// runs in a separate context; this sentinel only applies if a non-engine path
    /// (HTTP server / distributed) emits the call, where sub-agents are unsupported.
    static let subagentRun = BuiltinTool(
        spec: ToolSpec(name: "subagent_run",
                       description: "Run an isolated sub-agent on a target (project file path, or \"project\" for the whole project) with a self-contained question. With 'agent' (id from agents_list), it assumes that role. Optionally request a minimal 'tools' subset. Role and requested tools are always restricted by this parent agent's configured delegation scope; they cannot expand its permissions. The sub-agent returns only its answer.",
                       parametersJSON: #"{"type":"object","properties":{"target":{"type":"string","description":"relative file path, or \"project\""},"question":{"type":"string","description":"self-contained task; the sub-agent cannot see the parent chat"},"agent":{"type":"string","description":"optional role id from agents_list"},"tools":{"type":"array","items":{"type":"string"},"description":"optional minimal subset of the parent agent's configured delegable tools"}},"required":["target","question"]}"#),
        run: { _ in #"{"note":"subagent_run is handled by the engine and is not available in this context"}"# })
}
