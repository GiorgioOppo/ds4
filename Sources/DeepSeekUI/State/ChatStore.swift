import Foundation
import SwiftUI
import DeepSeekKit

enum GenerationPhase: Equatable {
    case idle
    case streaming(buffer: String, status: String)
    case error(String)
}

/// Multi-chat store. Owns `[Conversation]` indexed by id, a selection,
/// per-conversation generation state, and the shared `InferenceService`.
/// Persists each conversation as one JSON file under
/// `~/Library/Application Support/<appName>/conversations/`.
@MainActor
final class ChatStore: ObservableObject {
    @Published var conversations: [Conversation] = []
    @Published var selectedID: UUID?
    @Published private(set) var phases: [UUID: GenerationPhase] = [:]

    let service: InferenceService
    let modelDirPath: String

    private let saveDebounce: TimeInterval = 0.5
    private var pendingSaves: [UUID: Task<Void, Never>] = [:]

    init(modelDirPath: String, service: InferenceService) {
        self.modelDirPath = modelDirPath
        self.service = service
        loadFromDisk()
        if conversations.isEmpty {
            let first = Conversation(modelDirPath: modelDirPath)
            conversations.append(first)
            selectedID = first.id
            scheduleSave(first.id)
        } else {
            selectedID = conversations.first?.id
        }
    }

    // ----- selection / mutation -----

    var selectedConversation: Conversation? {
        guard let id = selectedID else { return nil }
        return conversations.first(where: { $0.id == id })
    }

    func newChat() {
        let c = Conversation(modelDirPath: modelDirPath)
        conversations.insert(c, at: 0)
        selectedID = c.id
        scheduleSave(c.id)
    }

    func delete(_ id: UUID) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }) else { return }
        conversations.remove(at: idx)
        phases.removeValue(forKey: id)
        // Best-effort: remove the on-disk file.
        if let url = try? PersistencePaths.conversationURL(id: id) {
            try? FileManager.default.removeItem(at: url)
        }
        if selectedID == id {
            selectedID = conversations.first?.id
        }
    }

    func phase(of id: UUID) -> GenerationPhase {
        phases[id] ?? .idle
    }

    // ----- streaming -----

    /// Send a user message to the currently-selected conversation.
    func send(text: String,
               mode: ThinkingMode,
               options: SamplingOptions,
               maxTokens: Int) {
        guard let id = selectedID,
              let idx = conversations.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if case .streaming = phase(of: id) { return }

        conversations[idx].messages.append(StoredMessage(role: .user, content: trimmed))
        conversations[idx].retitleIfNeeded()
        phases[id] = .streaming(buffer: "", status: "Encoding prompt…")
        scheduleSave(id)

        let history = conversations[idx].messages.map { $0.asKitMessage() }
        let placeholderId = UUID()
        conversations[idx].messages.append(
            StoredMessage(id: placeholderId, role: .assistant, content: ""))

        Task {
            do {
                for try await event in service.generate(
                    history: history, mode: mode,
                    options: options, maxTokens: maxTokens)
                {
                    apply(event: event, to: id, placeholderId: placeholderId)
                }
            } catch {
                phases[id] = .error((error as? LocalizedError)?.errorDescription
                                     ?? error.localizedDescription)
                scheduleSave(id)
            }
        }
    }

    func cancel() {
        service.cancelCurrent()
    }

    private func apply(event: GenerationEvent,
                        to id: UUID,
                        placeholderId: UUID) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }) else { return }
        switch event {
        case .token(let piece):
            guard case .streaming(let buffer, _) = phase(of: id) else { return }
            let newBuffer = buffer + piece
            phases[id] = .streaming(buffer: newBuffer, status: "")
            if let mIdx = conversations[idx].messages.firstIndex(
                where: { $0.id == placeholderId }) {
                conversations[idx].messages[mIdx].content = newBuffer
            }

        case .status(let s):
            if case .streaming(let buffer, _) = phase(of: id) {
                phases[id] = .streaming(buffer: buffer, status: s)
            }

        case .done(let final):
            if let mIdx = conversations[idx].messages.firstIndex(
                where: { $0.id == placeholderId }) {
                conversations[idx].messages[mIdx] = StoredMessage(
                    id: placeholderId,
                    role: .assistant,
                    content: final.content,
                    reasoningContent: final.reasoningContent,
                    toolCalls: final.toolCalls.map(StoredToolCall.init))
            }
            phases[id] = .idle
            scheduleSave(id)
        }
    }

    // ----- persistence -----

    private func loadFromDisk() {
        guard let dir = try? PersistencePaths.conversationsDir() else { return }
        let files = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var loaded: [Conversation] = []
        for url in files where url.pathExtension == "json" {
            if let data = try? Data(contentsOf: url),
               let c = try? decoder.decode(Conversation.self, from: data) {
                loaded.append(c)
            }
        }
        loaded.sort { $0.createdAt > $1.createdAt }
        conversations = loaded
    }

    private func scheduleSave(_ id: UUID) {
        pendingSaves[id]?.cancel()
        pendingSaves[id] = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(nanoseconds: UInt64(self.saveDebounce * 1_000_000_000))
            if Task.isCancelled { return }
            await self.flushSave(id)
        }
    }

    private func flushSave(_ id: UUID) async {
        guard let c = conversations.first(where: { $0.id == id }) else { return }
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(c)
            let url = try PersistencePaths.conversationURL(id: id)
            try data.write(to: url, options: .atomic)
        } catch {
            // Surface persistence errors as a status note on the
            // conversation; non-fatal — the in-memory state still
            // reflects everything correctly.
            phases[id] = .error("Save failed: \(error.localizedDescription)")
        }
    }
}
