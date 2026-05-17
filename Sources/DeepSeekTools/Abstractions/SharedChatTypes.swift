import Foundation

/// Ruoli locali per i messaggi di chat dentro DeepSeekTools.
/// Duplicano deliberatamente l'enum `Role` di DeepSeekKit per
/// mantenere il target Tools privo di dipendenza dall'engine
/// (vedi `Package.swift:54-61`). Le extension di bridging vivono
/// in DeepSeekUI/Phase B.
public enum ChatRole: String, Sendable, Codable {
    case user
    case assistant
    case system
}

/// Tool call rappresentata come dato puro: nome, argomenti JSON
/// stringa, id opzionale. Duplica `ToolCall` di DeepSeekKit per
/// la stessa ragione di layering.
public struct ChatToolCall: Sendable, Codable {
    public let name: String
    public let args: String
    public let id: String?

    public init(name: String, args: String, id: String? = nil) {
        self.name = name
        self.args = args
        self.id = id
    }
}

/// Messaggio passato all'agent come parte di un `AgentInput`.
/// Equivalente domain-neutral di `Message` di DeepSeekKit.
public struct AgentChatMessage: Sendable {
    public let role: ChatRole
    public let content: String
    public let reasoning: String?
    public let toolCalls: [ChatToolCall]
    public let toolOutputs: [String]

    public init(role: ChatRole,
                content: String,
                reasoning: String? = nil,
                toolCalls: [ChatToolCall] = [],
                toolOutputs: [String] = []) {
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.toolOutputs = toolOutputs
    }
}

/// Modalità di reasoning richiesta all'agent. Equivalente di
/// `ThinkingMode` di DeepSeekKit.
public enum AgentThinkingMode: String, Sendable, Codable {
    case chat       // non-think
    case high       // think high
    case max        // think max
}
