import Foundation
import DS4Engine
import DS4Core

// MARK: - Persistence mapping (UIMessage <-> StoredMessage)

extension ChatRole {
    var persistedString: String {
        switch self {
        case .system: return "system"
        case .user: return "user"
        case .assistant: return "assistant"
        case .tool: return "tool"
        }
    }
    init(persisted: String) {
        switch persisted {
        case "user": self = .user
        case "assistant": self = .assistant
        case "tool": self = .tool
        default: self = .system
        }
    }
}

extension StoredMessage {
    init(from m: UIMessage) {
        self.role = m.role.persistedString
        self.reasoning = m.reasoning
        self.text = m.text
        self.attachments = m.attachments
        self.toolCalls = m.toolCalls.map { StoredToolCall(id: $0.id, name: $0.name, argumentsJSON: $0.argumentsJSON) }
        self.subAgent = m.subAgent.map {
            StoredSubAgent(target: $0.target, question: $0.question, answer: $0.answer, steps: $0.steps)
        }
    }
}

extension UIMessage {
    init(stored s: StoredMessage) {
        self.init(role: ChatRole(persisted: s.role),
                  reasoning: s.reasoning,
                  text: s.text,
                  toolStreamText: "",
                  toolCalls: s.toolCalls.map { ToolCall(id: $0.id, name: $0.name, argumentsJSON: $0.argumentsJSON) },
                  attachments: s.attachments,
                  subAgent: s.subAgent.map {
                      InferenceService.SubAgentRun(target: $0.target, question: $0.question,
                                                   answer: $0.answer, steps: $0.steps)
                  })
    }
}

extension ChatStore {
    /// Rebuild engine turns from the visible transcript to re-prime a reopened chat.
    /// Attachments (one-shot context) are not restored; tool results are re-fed by
    /// their displayed content so the model keeps the thread.
    static func chatTurns(from messages: [UIMessage]) -> [ChatTurn] {
        var turns: [ChatTurn] = []
        for m in messages {
            switch m.role {
            case .user:
                turns.append(.user(m.text))
            case .assistant:
                if m.text.isEmpty && m.toolCalls.isEmpty { continue }
                turns.append(.assistant(text: m.text, toolCalls: m.toolCalls))
            case .tool:
                let content = m.subAgent?.answer ?? m.text
                turns.append(.toolResult(callId: m.toolCalls.first?.id ?? "", name: "", content: content))
            case .system:
                continue
            }
        }
        return turns
    }
}

