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
    /// One result string per entry in `toolCalls`, captured after
    /// the host actually executed the call. Wrapped by
    /// `EncodingDSV4.encodeMessages` into the V4 native
    /// `<｜tool▁outputs▁begin｜>…<｜tool▁outputs▁end｜>` block right
    /// after this assistant turn's `<eos>`. Empty when no execution
    /// happened (yet) — the message renders as a normal assistant
    /// turn in that case.
    public var toolOutputs: [String]

    public init(role: Role, content: String,
                reasoningContent: String? = nil,
                toolCalls: [ToolCall] = [],
                toolOutputs: [String] = []) {
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
        self.toolOutputs = toolOutputs
    }
}

public enum ThinkingMode: String, Sendable {
    case chat       // Non-think
    case high       // Think High
    case max        // Think Max
}
