import Foundation

/// Una risposta dentro una chat. Vista tipata di dominio sui
/// messaggi con `role == .assistant`. Conforme a
/// `MessageEnvelope` con `kind = "chat.answer"`. Tiene il link
/// `questionID` alla `Question` originaria per correlazione.
///
/// `isPartial` è true quando la `Answer` è un delta intermedio
/// dello stream; il client può accumulare delta partial finché
/// non riceve quello con `isPartial == false`.
public struct Answer: MessageEnvelope, Sendable {
    public let envelopeID: UUID
    public let timestamp: Date
    public let kind: String = "chat.answer"
    public let schemaVersion: Int = 1

    public let content: String
    public let reasoning: String?
    public let toolCalls: [ChatToolCall]
    public let isPartial: Bool
    public let metadata: [String: String]
    public let questionID: UUID

    public init(content: String,
                questionID: UUID,
                reasoning: String? = nil,
                toolCalls: [ChatToolCall] = [],
                isPartial: Bool = false,
                metadata: [String: String] = [:],
                envelopeID: UUID = UUID(),
                timestamp: Date = Date()) {
        self.content = content
        self.questionID = questionID
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.isPartial = isPartial
        self.metadata = metadata
        self.envelopeID = envelopeID
        self.timestamp = timestamp
    }
}
