import Foundation
import DS4Core

// Sub-agent tools (delegate a focused task to an isolated context).

extension ToolRegistry {
    /// List the available agents (roles) and the tools each one has — so the
    /// orchestrator can pick the right minimal tool set to grant a sub-agent.
    static let agentsList = BuiltinTool(
        spec: ToolSpec(name: "agents_list",
                       description: "List the available agents (roles) and the tools each one has (id · name · tools). Use it to choose which tools to grant to a sub-agent (the 'tools' parameter of subagent_run) based on the role best suited to the task.",
                       parametersJSON: #"{"type":"object","properties":{}}"#),
        run: { _ in AgentRegistry.shared.describe() })
}
