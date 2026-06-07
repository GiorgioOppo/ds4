/// A chat role in DeepSeek's prompt format.
public enum ChatRole: String, Sendable {
    case system
    case user
    case assistant
}

/// One conversation message (the durable transcript the engine re-renders each
/// turn). Reasoning text is not stored: only the visible answer carries into
/// later turns, matching how the chat transcript is rebuilt.
public struct ChatMessage: Sendable {
    public var role: ChatRole
    public var content: String

    public init(role: ChatRole, content: String) {
        self.role = role
        self.content = content
    }
}

/// An event emitted while the model generates a turn.
public enum DS4Event: Sendable {
    /// A chunk of chain-of-thought text (between <think> and </think>).
    case reasoning(String)
    /// A chunk of visible answer text.
    case text(String)
}
