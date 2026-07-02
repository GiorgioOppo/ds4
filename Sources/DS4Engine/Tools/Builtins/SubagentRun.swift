import Foundation
import DS4Core

extension ToolRegistry {
    /// Delegate a focused task to an isolated sub-agent. EXECUTED BY THE ENGINE
    /// (InferenceService.runSubAgent), which intercepts this call so the sub-agent
    /// runs in a separate context; this sentinel only applies if a non-engine path
    /// (HTTP server / distributed) emits the call, where sub-agents are unsupported.
    static let subagentRun = BuiltinTool(
        spec: ToolSpec(name: "subagent_run",
                       description: "Run an isolated sub-agent on a target with a question. Target can be a project file path, \"project\" for the whole imported project, or \"task\" for non-project work such as online research. With 'agent' (id from agents_list), it assumes that role and its tools, for example agent=\"ricerca\" for web research. Alternatively pass a minimal 'tools' set. Precedence: valid explicit tools > role tools > read-only project defaults.",
                       parametersJSON: #"{"type":"object","properties":{"target":{"type":"string","description":"relative file path, \"project\" for the whole imported project, or \"task\" for non-project work"},"question":{"type":"string","description":"self-contained task or question for the sub-agent"},"agent":{"type":"string","description":"id of an agent from agents_list, for example \"ricerca\" for online research. Optional."},"tools":{"type":"array","items":{"type":"string"},"description":"optional override: minimal set of granted tools. If absent, uses the 'agent' role tools, otherwise read-only project defaults."}},"required":["target","question"]}"#),
        run: { _ in #"{"note":"subagent_run is handled by the engine and is not available in this context"}"# })
}
