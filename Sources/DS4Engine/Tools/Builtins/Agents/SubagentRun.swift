import Foundation
import DS4Core

extension ToolRegistry {
    /// Delegate a focused task to an isolated sub-agent. EXECUTED BY THE ENGINE
    /// (InferenceService.runSubAgent), which intercepts this call so the sub-agent
    /// runs in a separate context; this sentinel only applies if a non-engine path
    /// (HTTP server / distributed) emits the call, where sub-agents are unsupported.
    static let subagentRun = BuiltinTool(
        spec: ToolSpec(name: "subagent_run",
                       description: "Run an isolated sub-agent on a target (project file path, or \"project\" for the whole project) with a question. The sub-agent has the target content in context and returns only the answer. With 'agent' (id from agents_list), it assumes that role and tools. Alternatively pass a minimal 'tools' set. Precedence: tools > role tools > read-only. Grantable tools: every built-in except the orchestration ones (agents_list, subagent_search, subagent_run) — e.g. project_tree, project_list, project_find, project_read, project_search, project_edit, project_write, file_read, file_lines, file_write, file_add, file_modify, file_delete, git, web_search, web_fetch, now, calculator.",
                       parametersJSON: #"{"type":"object","properties":{"target":{"type":"string","description":"relative file path, or \"project\""},"question":{"type":"string","description":"task or question for the sub-agent"},"agent":{"type":"string","description":"id of an agent (from agents_list): the sub-agent assumes its role and tools. Optional."},"tools":{"type":"array","items":{"type":"string"},"description":"optional override: minimal set of granted tools. If absent, uses the 'agent' role tools, otherwise read-only."}},"required":["target","question"]}"#),
        run: { _ in #"{"note":"subagent_run is handled by the engine and is not available in this context"}"# })
}
