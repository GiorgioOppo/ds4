import Foundation

/// Un turno di chat: domanda + (opzionalmente) risposta. Il
/// turno è "in volo" finché `answer == nil`.
public struct ChatTurn: Sendable {
    public let question: Question
    public let answer: Answer?

    public init(question: Question, answer: Answer? = nil) {
        self.question = question
        self.answer = answer
    }
}

/// Conversazione completa: lista ordinata di turni Q/A più
/// metadata identificativi. Struct per value semantics — la
/// chat è copiabile, snapshottabile, serializzabile.
///
/// **Non** conforme a `MessageEnvelope`: è uno stato persistente,
/// non un evento di trasporto. I singoli `Question`/`Answer`
/// dentro i turni lo sono già.
public struct Chat: Sendable {
    public let id: UUID
    public var title: String
    public let createdAt: Date
    public var turns: [ChatTurn]

    public init(id: UUID = UUID(),
                title: String = "Untitled",
                createdAt: Date = Date(),
                turns: [ChatTurn] = []) {
        self.id = id
        self.title = title
        self.createdAt = createdAt
        self.turns = turns
    }

    /// Convenience: la storia come sequenza `[AgentChatMessage]`
    /// (formato consumabile da `AgentBase.step(input:)`).
    public var agentMessages: [AgentChatMessage] {
        turns.flatMap { turn -> [AgentChatMessage] in
            var out: [AgentChatMessage] = [
                AgentChatMessage(role: turn.question.role,
                                 content: turn.question.content),
            ]
            if let a = turn.answer {
                out.append(AgentChatMessage(
                    role: .assistant,
                    content: a.content,
                    reasoning: a.reasoning,
                    toolCalls: a.toolCalls))
            }
            return out
        }
    }
}

/// Adapter bidirezionale: qualunque sorgente che fornisca
/// `(role, content, toolCalls)` per turno può essere importata
/// come `Chat`. Definito con un piccolo protocollo tampone per
/// non far dipendere DeepSeekTools da DeepSeekUI — in Phase B
/// `StoredMessage` di DeepSeekUI può conformarsi via extension
/// per ottenere il bridge gratis.
public protocol ChatTurnSource: Sendable {
    var sourceRole: ChatRole { get }
    var sourceContent: String { get }
    var sourceReasoning: String? { get }
    var sourceToolCalls: [ChatToolCall] { get }
}

extension Chat {
    /// Costruisce una `Chat` da una sequenza eterogenea di
    /// turni-sorgente. `.user`/`.system` aprono un nuovo turno;
    /// l'`.assistant` successivo lo chiude. Messaggi orfani
    /// diventano turni con `answer == nil`.
    public static func from<S: Sequence>(_ source: S,
                                          id: UUID = UUID(),
                                          title: String = "Imported")
    -> Chat where S.Element == any ChatTurnSource {
        var turns: [ChatTurn] = []
        var openQ: Question? = nil
        for m in source {
            switch m.sourceRole {
            case .user, .system:
                if let q = openQ {
                    turns.append(ChatTurn(question: q, answer: nil))
                }
                openQ = Question(content: m.sourceContent, role: m.sourceRole)
            case .assistant:
                if let q = openQ {
                    let a = Answer(content: m.sourceContent,
                                   questionID: q.envelopeID,
                                   reasoning: m.sourceReasoning,
                                   toolCalls: m.sourceToolCalls)
                    turns.append(ChatTurn(question: q, answer: a))
                    openQ = nil
                }
            }
        }
        if let q = openQ {
            turns.append(ChatTurn(question: q, answer: nil))
        }
        return Chat(id: id, title: title, turns: turns)
    }
}
