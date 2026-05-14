import Foundation
import DeepSeekKit

/// Codable mirrors of `DeepSeekKit.Message` / `Role` / `ToolCall` so we
/// can persist conversations without adding Codable to the kit. Trivial
/// translation in both directions.
struct StoredMessage: Codable, Identifiable, Hashable {
    let id: UUID
    var role: StoredRole
    var content: String
    var reasoningContent: String?
    var toolCalls: [StoredToolCall]
    /// Number of model tokens this message contributes once rendered
    /// through the chat template. Populated by the send / generate
    /// path; nil for messages written before this field existed.
    /// Purely informational (the UI shows it as "234 tok" in the
    /// future); the canonical token stream lives in
    /// `Conversation.encodedTokens`.
    var tokenCount: Int?

    init(id: UUID = UUID(), role: StoredRole, content: String,
         reasoningContent: String? = nil,
         toolCalls: [StoredToolCall] = [],
         tokenCount: Int? = nil) {
        self.id = id
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
        self.tokenCount = tokenCount
    }

    static func from(_ m: Message, id: UUID = UUID()) -> StoredMessage {
        StoredMessage(id: id,
                       role: StoredRole(m.role),
                       content: m.content,
                       reasoningContent: m.reasoningContent,
                       toolCalls: m.toolCalls.map(StoredToolCall.init))
    }

    func asKitMessage() -> Message {
        Message(role: role.asKitRole(),
                 content: content,
                 reasoningContent: reasoningContent,
                 toolCalls: toolCalls.map { $0.asKitToolCall() })
    }
}

enum StoredRole: String, Codable, Hashable {
    case user, assistant, system
    init(_ r: Role) {
        switch r {
        case .user: self = .user
        case .assistant: self = .assistant
        case .system: self = .system
        }
    }
    func asKitRole() -> Role {
        switch self {
        case .user: return .user
        case .assistant: return .assistant
        case .system: return .system
        }
    }
}

struct StoredToolCall: Codable, Hashable {
    var name: String
    var args: String
    var id: String?
    init(_ tc: ToolCall) {
        self.name = tc.name; self.args = tc.args; self.id = tc.id
    }
    func asKitToolCall() -> ToolCall {
        ToolCall(name: name, args: args, id: id)
    }
}

/// One chat thread. Title is auto-generated from the first user
/// message; falls back to "New Chat" until the user types something.
struct Conversation: Codable, Identifiable, Hashable {
    let id: UUID
    var title: String
    var createdAt: Date
    var modelDirPath: String      // captured at chat creation
    var messages: [StoredMessage]
    /// Optional reference to a `Project` in `ProjectLibrary`. When
    /// set, the chat surface shows the project's name in the toolbar.
    /// Step 3 of the KV cache pipeline will turn this reference into
    /// an actual prefill rehydration of the project's documents.
    /// `Codable` sees this as Optional and reads existing on-disk
    /// `.json` files (written before this field existed) as nil, so
    /// no migration is needed.
    var projectID: UUID?
    /// Tokenized form of the entire conversation up to (and
    /// including) the last assistant turn's eos. This is the
    /// canonical chat history — text bodies in `messages` are a
    /// view onto it. Letting the disk format carry the tokens means
    /// the next turn only needs to BPE-encode the *delta* (new user
    /// message + trailing assistant marker) instead of re-rendering
    /// + re-tokenizing the whole transcript.
    /// Nil until the first turn has run (or after invalidation —
    /// see `lastEncodedMode`).
    var encodedTokens: [Int32]?
    /// The `ThinkingMode` raw value that produced `encodedTokens`.
    /// The trailing assistant marker (and, for `.max`, the injected
    /// system prefix) depend on mode, so a mid-conversation change
    /// invalidates the cached prompt and forces a full re-encode.
    var lastEncodedMode: String?

    init(id: UUID = UUID(),
         title: String = "New Chat",
         createdAt: Date = .now,
         modelDirPath: String,
         messages: [StoredMessage] = [],
         projectID: UUID? = nil,
         encodedTokens: [Int32]? = nil,
         lastEncodedMode: String? = nil) {
        self.id = id; self.title = title; self.createdAt = createdAt
        self.modelDirPath = modelDirPath; self.messages = messages
        self.projectID = projectID
        self.encodedTokens = encodedTokens
        self.lastEncodedMode = lastEncodedMode
    }

    mutating func retitleIfNeeded() {
        guard title == "New Chat",
              let first = messages.first(where: { $0.role == .user })?.content,
              !first.isEmpty else { return }
        // First 40 chars of the first user turn, single-line.
        let stripped = first
            .components(separatedBy: .newlines)
            .first?
            .trimmingCharacters(in: .whitespaces) ?? "New Chat"
        title = String(stripped.prefix(40))
    }
}
