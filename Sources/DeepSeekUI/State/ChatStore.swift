import Foundation
import SwiftUI
import DeepSeekKit

/// Throughput readings the UI shows alongside the streamed text. Both
/// fields stay at 0 until the corresponding phase produces a sample.
struct GenerationMetrics: Equatable {
    var promptTokens: Int = 0
    var prefillElapsed: TimeInterval = 0
    var prefillTokPerMin: Double = 0
    var generatedTokens: Int = 0
    var generationElapsed: TimeInterval = 0
    var generationTokPerMin: Double = 0
}

enum GenerationPhase: Equatable {
    case idle
    /// The prefill forward is running. `promptTokens` is fixed, `startTime`
    /// lets the UI animate elapsed seconds while waiting (prefill is a
    /// single synchronous forward that doesn't stream intermediate
    /// progress, so live elapsed is the most useful liveness signal).
    case prefilling(promptTokens: Int, startTime: Date)
    case streaming(buffer: String, status: String, metrics: GenerationMetrics)
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

    /// Attach (or detach when `pid == nil`) a project to a conversation
    /// and persist. The reference is metadata for now; the chat path
    /// reads it only for the toolbar label until Step 3 of the KV
    /// pipeline pulls the project's documents into the prefill.
    func setProject(_ pid: UUID?, for id: UUID) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }) else { return }
        guard conversations[idx].projectID != pid else { return }
        conversations[idx].projectID = pid
        scheduleSave(id)
    }

    func delete(_ id: UUID) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }) else { return }
        conversations.remove(at: idx)
        phases.removeValue(forKey: id)
        // Best-effort: remove the on-disk files. Both the JSON
        // transcript and the persistent KV cache live next to each
        // other; wipe both so a future conversation reusing this
        // UUID never inherits stale prefill bytes.
        if let url = try? PersistencePaths.conversationURL(id: id) {
            try? FileManager.default.removeItem(at: url)
        }
        if let url = try? PersistencePaths.kvCacheURL(id: id) {
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
    ///
    /// Fast path: when the conversation already carries an
    /// `encodedTokens` cache produced under the same mode, we only
    /// BPE-encode the new user turn (the "delta": `<User>text<Assistant>
    /// <think_marker>`) and concatenate. The model's prefill therefore
    /// re-runs over the full prompt — that part is unavoidable since
    /// `releaseCache()` is called per turn — but the *tokenization*
    /// of the history happens once and is reused turn after turn.
    /// On mode change or first turn the prompt is rebuilt from
    /// scratch via `tokenizeFullHistory`.
    func send(text: String,
               mode: ThinkingMode,
               options: SamplingOptions,
               maxTokens: Int) {
        guard let id = selectedID,
              let idx = conversations.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        switch phase(of: id) {
        case .streaming, .prefilling: return
        default: break
        }

        let modeRaw = mode.rawValue
        let userMessage = StoredMessage(role: .user, content: trimmed)
        conversations[idx].messages.append(userMessage)
        conversations[idx].retitleIfNeeded()
        phases[id] = .streaming(buffer: "", status: "Encoding prompt…",
                                 metrics: GenerationMetrics())
        scheduleSave(id)

        let placeholderId = UUID()
        conversations[idx].messages.append(
            StoredMessage(id: placeholderId, role: .assistant, content: ""))

        // Snapshot the bits we need off the main actor.
        let cachedPrefix = conversations[idx].encodedTokens
        let cachedMode   = conversations[idx].lastEncodedMode
        let canReusePrefix = (cachedPrefix != nil && cachedMode == modeRaw)
        let historyForFullEncode = conversations[idx].messages.map {
            $0.asKitMessage()
        }

        Task {
            do {
                let promptTokens = try await buildPromptTokens(
                    canReusePrefix: canReusePrefix,
                    cachedPrefix: cachedPrefix,
                    userText: trimmed,
                    mode: mode,
                    fullHistory: historyForFullEncode)
                for try await event in service.generateFromPrompt(
                    promptTokens: promptTokens,
                    mode: mode,
                    options: options,
                    maxTokens: maxTokens)
                {
                    apply(event: event,
                           to: id,
                           placeholderId: placeholderId,
                           userMessageId: userMessage.id,
                           mode: modeRaw)
                }
            } catch {
                phases[id] = .error((error as? LocalizedError)?.errorDescription
                                     ?? error.localizedDescription)
                scheduleSave(id)
            }
        }
    }

    /// Assemble the BPE prompt either by extending the cached prefix
    /// with the new turn's delta, or — when the cache is missing /
    /// stale — by re-encoding the whole history once.
    private func buildPromptTokens(canReusePrefix: Bool,
                                    cachedPrefix: [Int32]?,
                                    userText: String,
                                    mode: ThinkingMode,
                                    fullHistory: [Message]) async throws
    -> [Int32] {
        if canReusePrefix, let prefix = cachedPrefix {
            guard let delta = await service.tokenizeUserTurnDelta(
                userText, mode: mode)
            else {
                throw NSError(domain: "ChatStore", code: 10, userInfo: [
                    NSLocalizedDescriptionKey:
                        "Tokenizer unavailable (no model loaded?)."
                ])
            }
            return prefix + delta
        }
        // Cold path / mode changed: full re-encode of the entire
        // conversation. Drops the last message (the just-appended
        // empty assistant placeholder), then asks for the trailing
        // assistant marker via mode.
        let history = fullHistory.dropLast()  // remove the empty placeholder
        guard let tokens = await service.tokenizeFullHistory(
            Array(history), mode: mode)
        else {
            throw NSError(domain: "ChatStore", code: 11, userInfo: [
                NSLocalizedDescriptionKey:
                    "Tokenizer unavailable (no model loaded?)."
            ])
        }
        return tokens
    }


    func cancel() {
        service.cancelCurrent()
    }

    private func apply(event: GenerationEvent,
                        to id: UUID,
                        placeholderId: UUID,
                        userMessageId: UUID,
                        mode: String) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }) else { return }
        switch event {
        case .token(let piece):
            // Once tokens start flowing we leave `.prefilling` if we
            // were still in it (e.g. very small prompts where
            // prefillDone arrived after the first token sample). Carry
            // the metrics forward.
            let m = currentMetrics(of: id)
            let buffer = currentBuffer(of: id)
            let newBuffer = buffer + piece
            phases[id] = .streaming(buffer: newBuffer, status: "", metrics: m)
            if let mIdx = conversations[idx].messages.firstIndex(
                where: { $0.id == placeholderId }) {
                conversations[idx].messages[mIdx].content = newBuffer
            }

        case .status(let s):
            let m = currentMetrics(of: id)
            let buffer = currentBuffer(of: id)
            phases[id] = .streaming(buffer: buffer, status: s, metrics: m)

        case .prefillStart(let promptTokens):
            phases[id] = .prefilling(promptTokens: promptTokens,
                                      startTime: Date())

        case .prefillDone(let promptTokens, let elapsed, let tokPerMin):
            var m = currentMetrics(of: id)
            m.promptTokens = promptTokens
            m.prefillElapsed = elapsed
            m.prefillTokPerMin = tokPerMin
            phases[id] = .streaming(buffer: currentBuffer(of: id),
                                     status: "", metrics: m)

        case .generationProgress(let generated, let elapsed, let tokPerMin):
            var m = currentMetrics(of: id)
            m.generatedTokens = generated
            m.generationElapsed = elapsed
            m.generationTokPerMin = tokPerMin
            phases[id] = .streaming(buffer: currentBuffer(of: id),
                                     status: "", metrics: m)

        case .done(let final, let promptTokens, let generatedTokens):
            // Finalize the assistant placeholder with the parsed
            // structure (reasoning + tool calls split out).
            if let mIdx = conversations[idx].messages.firstIndex(
                where: { $0.id == placeholderId }) {
                conversations[idx].messages[mIdx] = StoredMessage(
                    id: placeholderId,
                    role: .assistant,
                    content: final.content,
                    reasoningContent: final.reasoningContent,
                    toolCalls: final.toolCalls.map(StoredToolCall.init),
                    tokenCount: generatedTokens.count)
            }
            // Stamp the user message with the share of the prompt it
            // contributed. Approximation: prompt length minus the
            // bytes already accounted for by previous messages.
            if let uIdx = conversations[idx].messages.firstIndex(
                where: { $0.id == userMessageId }) {
                let prevPromptTokens = conversations[idx].encodedTokens?.count ?? 0
                conversations[idx].messages[uIdx].tokenCount =
                    max(promptTokens.count - prevPromptTokens, 0)
            }
            // Persist the canonical tokenized history: full prompt
            // plus everything we just sampled. Next turn's send()
            // will append a delta to this — no full re-encode.
            conversations[idx].encodedTokens = promptTokens + generatedTokens
            conversations[idx].lastEncodedMode = mode

            phases[id] = .idle
            scheduleSave(id)
        }
    }

    private func currentMetrics(of id: UUID) -> GenerationMetrics {
        if case .streaming(_, _, let m) = phase(of: id) { return m }
        return GenerationMetrics()
    }

    private func currentBuffer(of id: UUID) -> String {
        if case .streaming(let b, _, _) = phase(of: id) { return b }
        return ""
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
