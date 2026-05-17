import Foundation

/// Envelope token incrementale prodotto da un agent.
public struct AgentTokenEnvelope: MessageEnvelope, Sendable {
    public let envelopeID: UUID
    public let timestamp: Date
    public let kind: String = "agent.token"
    public let schemaVersion: Int = 1

    public let agentID: UUID
    public let text: String

    public init(agentID: UUID,
                text: String,
                envelopeID: UUID = UUID(),
                timestamp: Date = Date()) {
        self.agentID = agentID
        self.text = text
        self.envelopeID = envelopeID
        self.timestamp = timestamp
    }
}

/// Envelope di fine generazione di un agent.
public struct AgentDoneEnvelope: MessageEnvelope, Sendable {
    public let envelopeID: UUID
    public let timestamp: Date
    public let kind: String = "agent.done"
    public let schemaVersion: Int = 1

    public let agentID: UUID

    public init(agentID: UUID,
                envelopeID: UUID = UUID(),
                timestamp: Date = Date()) {
        self.agentID = agentID
        self.envelopeID = envelopeID
        self.timestamp = timestamp
    }
}

/// DEMO: agent finto che restituisce l'input prefissato da
/// `"echo: "`, emettendo un `AgentTokenEnvelope` per ogni
/// carattere e un `AgentDoneEnvelope` finale.
///
/// Nessun Transformer, nessun Metal: pura Swift, utile come
/// fixture di test e come verifica che il contratto `AgentBase`
/// + `AgentEventStream` regga.
public final class DemoEchoAgent: AgentBase, @unchecked Sendable {
    public override func step(input: AgentInput) -> AgentEventStream {
        let agentID = self.id
        return AgentEventStream { continuation in
            let last = input.messages.last?.content ?? ""
            for ch in "echo: \(last)" {
                continuation.yield(
                    AgentTokenEnvelope(agentID: agentID,
                                       text: String(ch)))
            }
            continuation.yield(AgentDoneEnvelope(agentID: agentID))
            continuation.finish()
        }
    }
}
