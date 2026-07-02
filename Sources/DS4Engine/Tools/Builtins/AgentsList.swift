import Foundation
import DS4Core

// Sub-agent tools (delegate a focused task to an isolated context).

extension ToolRegistry {
    /// List the available agents (roles) and the tools each one has — so the
    /// orchestrator can pick the right minimal tool set to grant a sub-agent.
    static let agentsList = BuiltinTool(
        spec: ToolSpec(name: "agents_list",
                       description: "Elenca gli agenti (ruoli) disponibili e i tool che ciascuno ha a disposizione (id · nome · tool). Usalo per scegliere quali tool concedere a un sub-agent (parametro 'tools' di subagent_run) in base al ruolo adatto al compito.",
                       parametersJSON: #"{"type":"object","properties":{}}"#),
        run: { _ in AgentRegistry.shared.describe() })
}
