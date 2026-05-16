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
    /// Library references handed down from the App scene. Used at
    /// "first turn of a chat with a project attached" time to pull
    /// the project's pre-tokenized files and assemble the prompt
    /// with native repo/file delimiter tokens. Both are weak-ish
    /// (unowned semantics implicit via @MainActor reference holding)
    /// but kept as strong refs to avoid lifetime surprises — the
    /// libraries live for the whole app session.
    let documents: DocumentLibrary
    let projects: ProjectLibrary
    /// Live MCP pool. Used by `send` to snapshot the current tool
    /// schemas and fold them into the system block of the prompt.
    /// Snapshot-on-send semantics means a server enabled mid-chat
    /// won't appear to the model until the cached prefix is
    /// invalidated (mode change or new chat) — same trade-off the
    /// project-context path already makes.
    let mcpPool: MCPClientPool

    private let saveDebounce: TimeInterval = 0.5
    private var pendingSaves: [UUID: Task<Void, Never>] = [:]
    /// Per-conversation count of how many tool-call -> generate
    /// continuations have fired since the latest user turn. Cleared
    /// the moment a `.done` arrives without tool calls (= final
    /// reply) or the cap is hit. The cap prevents the obvious
    /// failure mode of a model that keeps calling tools forever.
    private var toolRoundtrips: [UUID: Int] = [:]
    private let maxToolRoundtripsPerTurn: Int = 8
    /// Sampler parameters captured at the start of `send`; reused
    /// by `runToolCallsAndContinue` so a tool-output continuation
    /// inherits the same temperature / topK / topP / maxTokens
    /// the user picked for the current turn.
    private var lastSamplingOptions: [UUID: (opts: SamplingOptions,
                                              maxTokens: Int)] = [:]

    init(modelDirPath: String,
         service: InferenceService,
         documents: DocumentLibrary,
         projects: ProjectLibrary,
         mcpPool: MCPClientPool) {
        self.modelDirPath = modelDirPath
        self.service = service
        self.documents = documents
        self.projects = projects
        self.mcpPool = mcpPool
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
        // Tool-call continuations (M3b) re-use these — capture them
        // before the Task that drives the model run is spawned so
        // the value is stable for the whole multi-roundtrip turn.
        lastSamplingOptions[id] = (options, maxTokens)
        toolRoundtrips[id] = 0
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
        // Snapshot the MCP tool catalogue once, off the main actor's
        // critical path. The Task below treats this as immutable
        // for the duration of the generation — same snapshot
        // semantics the project-context path uses.
        let toolSchemasJSON = mcpPool.toolSchemasJSON()

        // First-turn-with-project info: only the very first user
        // turn of a chat (encodedTokens == nil) ever pays the cost
        // of injecting the project's whole token stream. Every
        // subsequent turn extends the cached prefix via the normal
        // fast-path delta.
        let projectContext: FirstTurnProjectContext? = {
            guard cachedPrefix == nil,
                  let pid = conversations[idx].projectID,
                  let project = projects.project(id: pid) else { return nil }
            let docs = documents.documents(for: pid)
            guard !docs.isEmpty else { return nil }
            // Snapshot path + tokens off the main actor. tokens(of:)
            // does a small disk read per file; expected to take ms,
            // not seconds, even for a multi-hundred-file project —
            // each `.tokens` blob is just the Int32 payload.
            var files: [(path: String, tokens: [Int32])] = []
            files.reserveCapacity(docs.count)
            for d in docs {
                guard let toks = try? documents.tokens(of: d.id) else { continue }
                files.append((path: d.displayPath ?? d.name, tokens: toks))
            }
            return FirstTurnProjectContext(name: project.name, files: files)
        }()

        Task {
            do {
                let promptTokens = try await buildPromptTokens(
                    canReusePrefix: canReusePrefix,
                    cachedPrefix: cachedPrefix,
                    userText: trimmed,
                    mode: mode,
                    fullHistory: historyForFullEncode,
                    projectContext: projectContext,
                    toolSchemasJSON: toolSchemasJSON)
                // Snapshot the pending turn on disk immediately. If
                // the app dies during the prefill (which can take
                // minutes for a project context), the user's prompt
                // is preserved and the resume path can still pick
                // up — without any tokens emitted yet.
                if let cIdx = conversations.firstIndex(where: { $0.id == id }) {
                    conversations[cIdx].pendingTurn = PendingTurn(
                        assistantMessageID: placeholderId,
                        promptTokens: promptTokens,
                        generatedTokens: [],
                        mode: modeRaw)
                    scheduleSave(id)
                }
                for try await event in service.generateForConversation(
                    promptTokens: promptTokens,
                    conversationID: id,
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

    /// Restart a generation that was cut short (crash, app-quit,
    /// power loss). The conversation already carries a `pendingTurn`
    /// snapshot describing the prompt the model saw and every token
    /// sampled before the interruption; we feed that prompt+tokens
    /// back into the model as a fresh prefill so the KV cache is
    /// rebuilt, then the decode loop picks up from
    /// `startPos == promptTokens.count + generatedTokens.count`.
    ///
    /// Sampling parameters are taken from current Settings (we
    /// don't persist them per-turn) — the partial text already on
    /// screen ensures resumed sampling stays consistent enough that
    /// the user can keep reading without seeing a regressed style.
    func resumePendingTurn(of id: UUID,
                             options: SamplingOptions,
                             maxTokens: Int) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }),
              let pt = conversations[idx].pendingTurn else { return }
        switch phase(of: id) {
        case .streaming, .prefilling: return
        default: break
        }
        // Re-derive ThinkingMode from the persisted raw value. Falls
        // back to .chat for forward-compat (a future enum case we
        // don't recognise yet shouldn't lock the user out of
        // resuming an interrupted reply).
        let mode = ThinkingMode(rawValue: pt.mode) ?? .chat
        // Stitch prompt + already-generated ids into one prompt:
        // the model will prefill the whole thing, the decode loop
        // then samples token #(generatedTokens.count + 1).
        let resumedPrompt = pt.promptTokens + pt.generatedTokens

        // Restore the live streaming UI state from the persisted
        // partial content. Tokens-per-minute resets to zero — we
        // don't carry it across restarts.
        let buffer: String
        if let mIdx = conversations[idx].messages.firstIndex(
            where: { $0.id == pt.assistantMessageID }) {
            buffer = conversations[idx].messages[mIdx].content
        } else {
            buffer = ""
        }
        phases[id] = .streaming(buffer: buffer,
                                 status: "Resuming…",
                                 metrics: GenerationMetrics())

        let placeholderId = pt.assistantMessageID
        let modeRaw = pt.mode
        // The most recent user message — used by `apply` to stamp
        // tokenCount once the turn finishes.
        let userMessageId = conversations[idx].messages
            .last(where: { $0.role == .user })?.id ?? placeholderId

        Task {
            do {
                for try await event in service.generateForConversation(
                    promptTokens: resumedPrompt,
                    conversationID: id,
                    mode: mode,
                    options: options,
                    maxTokens: maxTokens)
                {
                    apply(event: event,
                           to: id,
                           placeholderId: placeholderId,
                           userMessageId: userMessageId,
                           mode: modeRaw)
                }
            } catch {
                phases[id] = .error((error as? LocalizedError)?.errorDescription
                                     ?? error.localizedDescription)
                scheduleSave(id)
            }
        }
    }

    /// Carries the snapshotted project context across the actor hop
    /// into `buildPromptTokens`. Constructed only when the very
    /// first turn of a chat with an attached project is about to
    /// fire (cachedPrefix == nil, projectID != nil, library has
    /// indexed files).
    private struct FirstTurnProjectContext {
        let name: String
        let files: [(path: String, tokens: [Int32])]
    }

    /// Assemble the BPE prompt:
    ///   - fast path: extend cached prefix with the new turn's delta;
    ///   - first turn with a project attached: emit native repo / file
    ///     delimiter ids and splice in each file's pre-tokenized stream;
    ///   - cold path: full re-encode of the entire history.
    ///
    /// `toolSchemasJSON`, when non-nil, only matters on the
    /// project-first-turn and cold-path branches — those produce
    /// the system block from scratch. The delta fast path inherits
    /// whatever system block was in the cached prefix.
    private func buildPromptTokens(canReusePrefix: Bool,
                                    cachedPrefix: [Int32]?,
                                    userText: String,
                                    mode: ThinkingMode,
                                    fullHistory: [Message],
                                    projectContext: FirstTurnProjectContext?,
                                    toolSchemasJSON: String?
    ) async throws -> [Int32] {
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
        // First turn with a project: try the structured-context
        // path. Falls through to the plain full-encode if the
        // tokenizer can't resolve every special id we need.
        if let ctx = projectContext {
            let history = fullHistory.dropLast()
            let systemText = history
                .first(where: { $0.role == .system })?
                .content ?? ""
            if let tokens = await service.tokenizeFirstTurnWithProject(
                systemText: systemText,
                projectName: ctx.name,
                files: ctx.files,
                userText: userText,
                mode: mode,
                toolSchemasJSON: toolSchemasJSON)
            {
                return tokens
            }
        }
        // Cold path / mode changed: full re-encode of the entire
        // conversation. Drops the last message (the just-appended
        // empty assistant placeholder), then asks for the trailing
        // assistant marker via mode.
        let history = fullHistory.dropLast()
        guard let tokens = await service.tokenizeFullHistory(
            Array(history), mode: mode,
            toolSchemasJSON: toolSchemasJSON)
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

    /// User chose to throw away the interrupted assistant turn. The
    /// partial content stays in the transcript (so they don't lose
    /// what they already read), but the model's view of the chat
    /// rolls back to the last completed turn — i.e. `encodedTokens`
    /// is untouched and the next `send` starts fresh.
    func discardPendingTurn(of id: UUID) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }),
              conversations[idx].pendingTurn != nil else { return }
        conversations[idx].pendingTurn = nil
        scheduleSave(id)
    }

    private func apply(event: GenerationEvent,
                        to id: UUID,
                        placeholderId: UUID,
                        userMessageId: UUID,
                        mode: String) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }) else { return }
        switch event {
        case .token(let piece, let tokenId):
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
            // Append the raw id to the on-disk pendingTurn so a
            // crash mid-stream replays this token exactly. The save
            // is debounced (~500 ms) so we don't pay disk I/O at
            // the model's sampling rate.
            conversations[idx].pendingTurn?.generatedTokens.append(tokenId)
            scheduleSave(id)

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
            // The generation completed cleanly; drop the crash-
            // recovery snapshot so the UI stops offering Resume.
            conversations[idx].pendingTurn = nil

            // Model asked to call one or more tools? Run them, splice
            // the outputs back into the prompt, and let the model
            // continue. The iteration guard stops the obvious
            // infinite-recursion failure mode (model keeps emitting
            // calls without ever producing a final reply).
            if !final.toolCalls.isEmpty,
               toolRoundtrips[id, default: 0] < maxToolRoundtripsPerTurn {
                toolRoundtrips[id, default: 0] += 1
                runToolCallsAndContinue(
                    conversationID: id,
                    finishedMessageID: placeholderId,
                    calls: final.toolCalls,
                    mode: mode)
                scheduleSave(id)
                return
            }
            toolRoundtrips[id] = nil
            lastSamplingOptions[id] = nil

            phases[id] = .idle
            scheduleSave(id)
        }
    }

    /// Dispatch every tool call the model just emitted to MCPClientPool,
    /// stash the outputs on the originating assistant message, then
    /// build the `<eos>…<tool_outputs>…<Assistant>` delta and feed
    /// the model the result of its own work so it can finish the
    /// reply (or call more tools, bounded by `maxToolRoundtripsPerTurn`).
    private func runToolCallsAndContinue(conversationID id: UUID,
                                          finishedMessageID: UUID,
                                          calls: [ToolCall],
                                          mode: ThinkingMode) {
        // Surface that we're between turns — the buffer is empty
        // (the previous placeholder is finalised), and the next
        // streaming view-state will be created by the
        // generateForConversation stream below.
        phases[id] = .streaming(buffer: "",
                                 status: "Running \(calls.count) tool\(calls.count == 1 ? "" : "s")…",
                                 metrics: currentMetrics(of: id))

        // Capture references the detached Task needs.
        let pool = self.mcpPool

        Task { [weak self] in
            guard let self else { return }
            // Run sequentially — most MCP servers are single-process
            // and an interleaved fan-out doesn't buy much. Errors
            // are flattened into the output text by `invokeQualified`
            // so this loop never throws.
            var outputs: [String] = []
            outputs.reserveCapacity(calls.count)
            for call in calls {
                let result = await pool.invokeQualified(
                    call.name, argsJSON: call.args)
                outputs.append(result)
            }

            // Persist the outputs on the assistant turn that asked
            // for them — that's what the chat template's
            // tool_outputs block hangs off.
            if let cIdx = self.conversations.firstIndex(where: { $0.id == id }),
               let mIdx = self.conversations[cIdx].messages.firstIndex(
                where: { $0.id == finishedMessageID })
            {
                self.conversations[cIdx].messages[mIdx].toolOutputs = outputs
            }

            // Build the delta to feed the model. Same shape as the
            // user-turn fast path, but instead of `<User>...` we
            // splice the tool_outputs block between `<eos>` and the
            // re-opened `<Assistant>` turn. The cache prefix already
            // includes everything up to (and including) the call-
            // emitting assistant content, so cacheImage will match
            // and the model continues with only the new delta
            // tokens to chew on.
            guard let cachedPrefix = self.conversations.first(where: { $0.id == id })?
                .encodedTokens else {
                self.phases[id] = .error(
                    "Tool-call continuation: missing cached prefix")
                self.toolRoundtrips[id] = nil
                self.scheduleSave(id)
                return
            }
            guard let delta = await self.service.tokenizeToolOutputsDelta(
                callNames: calls.map(\.name),
                outputs: outputs,
                mode: mode)
            else {
                self.phases[id] = .error(
                    "Tokenizer unavailable while building tool-output delta")
                self.toolRoundtrips[id] = nil
                self.scheduleSave(id)
                return
            }
            let promptTokens = cachedPrefix + delta

            // Append a fresh placeholder for the model's next reply
            // and re-arm pendingTurn so a crash here is recoverable.
            let newPlaceholderId = UUID()
            if let cIdx = self.conversations.firstIndex(where: { $0.id == id }) {
                self.conversations[cIdx].messages.append(
                    StoredMessage(id: newPlaceholderId,
                                   role: .assistant,
                                   content: ""))
                self.conversations[cIdx].pendingTurn = PendingTurn(
                    assistantMessageID: newPlaceholderId,
                    promptTokens: promptTokens,
                    generatedTokens: [],
                    mode: mode.rawValue)
                self.scheduleSave(id)
            }

            // Reuse the same sampling parameters the user picked for
            // this turn (captured in `lastSamplingOptions` at the
            // start of `send`). Fallback to sensible-but-tame
            // defaults if somehow the slot was cleared mid-flight.
            let stored = self.lastSamplingOptions[id]
            let opts = stored?.opts ?? SamplingOptions(
                temperature: 0.7,
                topK: 0, topP: 1.0,
                repetitionPenalty: 1.0)
            let maxTok = stored?.maxTokens ?? 4096
            do {
                for try await event in self.service.generateForConversation(
                    promptTokens: promptTokens,
                    conversationID: id,
                    mode: mode,
                    options: opts,
                    maxTokens: maxTok)
                {
                    self.apply(event: event,
                                to: id,
                                placeholderId: newPlaceholderId,
                                userMessageId: finishedMessageID,
                                mode: mode.rawValue)
                }
            } catch {
                self.phases[id] = .error(
                    (error as? LocalizedError)?.errorDescription
                    ?? error.localizedDescription)
                self.toolRoundtrips[id] = nil
                self.scheduleSave(id)
            }
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
