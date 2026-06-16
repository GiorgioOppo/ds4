import Foundation
import DS4Engine
import DS4Core

/// A persisted chat: metadata + the full transcript. One JSON file per chat under
/// Application Support/DwarfStar/chats, so conversations survive app restarts and
/// the user can keep several side by side. The live UI transcript (`UIMessage`)
/// holds engine value types we don't want to entangle with Codable, so it is
/// mapped to/from these plain Codable mirrors.
struct ChatSession: Codable, Identifiable {
    let id: String
    var title: String
    var agentId: String
    /// The extra system prompt the user typed for this chat (added to the role).
    var systemNote: String
    var modelName: String
    var createdAt: Date
    var updatedAt: Date
    var messages: [StoredMessage]

    init(id: String = UUID().uuidString, title: String = ChatSession.untitled,
         agentId: String, systemNote: String = "", modelName: String = "") {
        self.id = id
        self.title = title
        self.agentId = agentId
        self.systemNote = systemNote
        self.modelName = modelName
        let now = Date()
        self.createdAt = now
        self.updatedAt = now
        self.messages = []
    }

    static let untitled = "Nuova chat"
}

/// Codable mirror of `UIMessage`.
struct StoredMessage: Codable {
    var role: String                 // system | user | assistant | tool
    var reasoning: String
    var text: String
    var attachments: [String]
    var toolCalls: [StoredToolCall]
    var subAgent: StoredSubAgent?
}

struct StoredToolCall: Codable {
    var id: String
    var name: String
    var argumentsJSON: String
}

struct StoredSubAgent: Codable {
    var target: String
    var question: String
    var answer: String
    var steps: [String]
}

/// On-disk store: one JSON file per chat under Application Support/DwarfStar/chats.
/// Robust to corrupt / foreign files (they are skipped, never throw to the caller).
enum ChatSessionStore {
    static var directory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let dir = base.appendingPathComponent("DwarfStar/chats", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// All persisted chats, newest first. Skips anything that doesn't decode.
    static func loadAll() -> [ChatSession] {
        let fm = FileManager.default
        guard let urls = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else { return [] }
        let dec = JSONDecoder()
        var out: [ChatSession] = []
        for url in urls where url.pathExtension == "json" {
            if let data = try? Data(contentsOf: url), let s = try? dec.decode(ChatSession.self, from: data) {
                out.append(s)
            }
        }
        return out.sorted { $0.updatedAt > $1.updatedAt }
    }

    static func save(_ session: ChatSession) {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted]
        guard let data = try? enc.encode(session) else { return }
        try? data.write(to: directory.appendingPathComponent("\(session.id).json"), options: .atomic)
    }

    static func delete(_ id: String) {
        try? FileManager.default.removeItem(at: directory.appendingPathComponent("\(id).json"))
    }
}
