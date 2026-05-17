import Foundation

/// DEMO: chat concreta che usa `DemoEchoAgent` come backend per
/// produrre le risposte e — opzionalmente — pubblica gli
/// envelope di Question/Answer su un `PluginRegistry`.
///
/// Mostra l'integrazione fra i componenti di Phase A:
///   `Chat` → `Question` (envelope) → `AgentBase.step` (stream di
///   envelope) → `Answer` (envelope) → `publish` → `Plugin.observe`.
public final class DemoEchoChat: ChatBase, @unchecked Sendable {
    public let agent: DemoEchoAgent
    public let plugins: PluginRegistry?

    public init(agent: DemoEchoAgent = DemoEchoAgent(),
                plugins: PluginRegistry? = nil,
                title: String = "Demo Echo Chat") {
        self.agent = agent
        self.plugins = plugins
        super.init(title: title)
    }

    public override func answer(question: Question) async throws -> Answer {
        let input = AgentInput(messages: chat.agentMessages,
                               thinkingMode: .chat)
        var assembled = ""
        for try await env in agent.step(input: input) {
            if let tok = env as? AgentTokenEnvelope {
                assembled.append(tok.text)
            }
        }
        return Answer(content: assembled,
                      questionID: question.envelopeID)
    }

    public override func publish(envelope: any MessageEnvelope) async {
        await plugins?.broadcast(envelope)
    }
}
