import Foundation

public enum Role: String, Sendable {
    case user
    case assistant
    case system
}

public struct ToolCall: Sendable {
    public var name: String
    public var args: String      // JSON-encoded
    public var id: String?
    public init(name: String, args: String, id: String? = nil) {
        self.name = name
        self.args = args
        self.id = id
    }
}

public struct Message: Sendable {
    public var role: Role
    public var content: String
    public var reasoningContent: String?    // contents of `<think>...</think>`
    public var toolCalls: [ToolCall]

    public init(role: Role, content: String,
                reasoningContent: String? = nil,
                toolCalls: [ToolCall] = []) {
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
    }
}

public enum ThinkingMode: String, Sendable {
    case chat       // Non-think
    case high       // Think High
    case max        // Think Max
}
