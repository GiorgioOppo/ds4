import Foundation
import DS4Core
import DS4Metal

extension InferenceService {
// MARK: - Agents (roles) + per-agent expert usage

    /// The active agent's id — keys the persisted usage profile.

    /// Switch the conversation to `agent`: persists the outgoing agent's usage
    /// profile, swaps in the new agent's one, drops the slot-cache pools (they
    /// re-warm lazily with the NEW profile), declares the agent's tools and
    /// starts a fresh conversation with its role (system prompt).
    public func setAgent(_ agent: AgentProfile, tools: [ToolSpec]) {
        // Profilo usage e pool si toccano SOLO se l'agente cambia davvero:
        // passare tra due chat con lo stesso ruolo non deve buttare via i GB
        // di slot-cache calda (l'invalidate li faceva ricostruire al primo
        // messaggio, ogni volta).
        if agent.id != agentId {
            saveExpertUsage()
            agentId = agent.id
            decoder.usage?.replace(with: Self.usageDataSeeded(modelName: modelName, agentId: agentId))
            decoder.slotCache?.invalidate()
            warmedUp = false   // pool da ricostruire: un warmup() successivo li riscalda
        }
        self.tools = tools
        resetConversation(systemPrompt: agent.systemPrompt.isEmpty ? nil : agent.systemPrompt)
    }
}

