import Foundation

/// Una singola domanda dentro una chat. Vista tipata di dominio
/// sui messaggi con `role == .user` o `role == .system`.
/// Conforme a `MessageEnvelope` con `kind = "chat.question"` per
/// poter essere routato verso plugin observer senza adattatori.
public struct Question: MessageEnvelope, Sendable {
    public let envelopeID: UUID
    public let timestamp: Date
    public let kind: String = "chat.question"
    public let schemaVersion: Int = 1

    public let content: String
    public let role: ChatRole
    public let metadata: [String: String]

    public init(content: String,
                role: ChatRole = .user,
                metadata: [String: String] = [:],
                envelopeID: UUID = UUID(),
                timestamp: Date = Date()) {
        self.content = content
        self.role = role
        self.metadata = metadata
        self.envelopeID = envelopeID
        self.timestamp = timestamp
    }
}
