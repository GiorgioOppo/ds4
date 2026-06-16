import SwiftUI
import AppKit
import UniformTypeIdentifiers
import DS4Engine
import DS4Core

/// A message as shown in the UI: reasoning and visible answer are kept apart so
/// the chain-of-thought can be collapsed. Assistant messages may carry tool
/// calls; tool results are shown as `.tool` messages.
struct UIMessage: Identifiable {
    let id = UUID()
    let role: ChatRole
    var reasoning: String = ""
    var text: String
    var toolStreamText: String = ""   // raw tool markup shown live while it generates
    var toolCalls: [ToolCall] = []
    /// Names of text files imported with this (user) message — shown as badges; the
    /// full content was folded into the turn actually sent to the model.
    var attachments: [String] = []
    /// Set on a `.tool` message that reports an isolated sub-agent run (question,
    /// answer, and a collapsible trace of its internal steps).
    var subAgent: InferenceService.SubAgentRun?
}

/// A text file staged in the composer: its full content is folded into the next
/// user turn sent to the model; the transcript shows only the filename + size.
struct ChatAttachment: Identifiable, Equatable {
    let id = UUID()
    let name: String
    let content: String
    var bytes: Int { content.utf8.count }
}

/// Main-thread view model. Owns the `InferenceService` actor and mirrors its
/// streamed output into observable UI state.
@MainActor
@Observable
final class ChatStore {
    enum Phase: Equatable {
        case needsModel
        case loading
        case ready
        case failed(String)
    }

    // Configuration (editable before loading). Defaults adapt to dev vs bundle.
    // NOTE: the pure-Swift engine always runs the SSD-streaming path (mmap no-copy
    // + per-token expert gather); the old C-engine streaming/RAM-mode toggles were
    // dead and have been removed.
    /// Shared app settings: model path / context / mode are owned by the
    /// Impostazioni screen; this store (like every other controller) proxies them.
    let settings: AppSettings
    var modelPath: String {
        get { settings.modelPath }
        set { settings.modelPath = newValue }
    }
    var contextSize: Int {
        get { settings.contextSize }
        set { settings.contextSize = newValue }
    }
    var scriptDir = AppEnvironment.resourceDir   // download_model.sh / gguf

    init(settings: AppSettings) {
        self.settings = settings
        AgentRegistry.shared.set(agents)   // didSet doesn't fire for the initial value
        _ = setenv("DS4_RAW_RING", rawRingEnabled ? "1" : "0", 1)   // apply the persisted value at startup
        // Restore the persisted chats (newest first). Always keep at least one so
        // there is an active conversation to write into.
        sessions = ChatSessionStore.loadAll()
        if let first = sessions.first {
            activeSessionId = first.id
        } else {
            let s = ChatSession(agentId: selectedAgentId, systemNote: systemPrompt)
            sessions = [s]
            activeSessionId = s.id
        }
    }
    var systemPrompt = ""
    /// Expert slot-cache slots per layer (0 = off). Wired memory ≈ 6,9 MB/slot ×
    /// 43 layer on the 2-bit model. Applied on the NEXT model load.
    var expertCacheSlots: Int = UserDefaults.standard.integer(forKey: "DS4ExpertCacheSlots") {
        didSet { UserDefaults.standard.set(expertCacheSlots, forKey: "DS4ExpertCacheSlots") }
    }

    // Disk KV cache (ds4_kvstore model): checkpoints completed generations and
    // restores matching prefixes on cold starts. Applied on the NEXT model load.
    // ON by default (8 GB budget) so conversations are checkpointed and re-prefill
    // is avoided across reloads; the explicit user choice is then persisted.
    var diskKVEnabled: Bool = (UserDefaults.standard.object(forKey: "DS4DiskKV") as? Bool) ?? true {
        didSet { UserDefaults.standard.set(diskKVEnabled, forKey: "DS4DiskKV") }
    }
    var diskKVBudgetMB: Int = UserDefaults.standard.object(forKey: "DS4DiskKVBudgetMB") as? Int ?? 8192 {
        didSet { UserDefaults.standard.set(diskKVBudgetMB, forKey: "DS4DiskKVBudgetMB") }
    }
    /// Raw-KV ring buffer (experimental): keep only the nSWA attention window in RAM
    /// instead of the full context, so the KV RAM is constant. Sets the engine env
    /// var; applied on the NEXT model load.
    var rawRingEnabled: Bool = (UserDefaults.standard.object(forKey: "DS4RawRing") as? Bool) ?? false {
        didSet {
            UserDefaults.standard.set(rawRingEnabled, forKey: "DS4RawRing")
            _ = setenv("DS4_RAW_RING", rawRingEnabled ? "1" : "0", 1)
        }
    }
    /// Application Support/DwarfStar/kv-cache (shared by chat and HTTP server).
    static var diskKVDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("DwarfStar/kv-cache", isDirectory: true)
    }

    // Tuning tab state.
    var tuningInfo: InferenceService.TuningInfo?

    // Agents (roles). Selecting one starts a fresh chat with its role and swaps
    // the per-agent expert-usage profile (the cache re-warms with ITS experts).
    // Editable + persisted; edits apply on the next new chat / agent switch.
    var agents: [AgentProfile] = ChatStore.loadAgents() {
        didSet { AgentRegistry.shared.set(agents) }   // keep the engine-side agents_list tool in sync
    }
    var selectedAgentId: String = UserDefaults.standard.string(forKey: "DS4SelectedAgent") ?? "generale" {
        didSet { UserDefaults.standard.set(selectedAgentId, forKey: "DS4SelectedAgent") }
    }
    var selectedAgent: AgentProfile { agents.first { $0.id == selectedAgentId } ?? agents[0] }

    static func loadAgents() -> [AgentProfile] {
        if let data = UserDefaults.standard.data(forKey: "DS4Agents"),
           var arr = try? JSONDecoder().decode([AgentProfile].self, from: data), !arr.isEmpty {
            // New DEFAULT agents (e.g. "code") must appear even for users with a
            // persisted list: append the missing ones without touching edits.
            for d in AgentProfile.defaults where !arr.contains(where: { $0.id == d.id }) {
                arr.append(d)
            }
            return arr
        }
        return AgentProfile.defaults
    }

    func saveAgents() {
        if let data = try? JSONEncoder().encode(agents) {
            UserDefaults.standard.set(data, forKey: "DS4Agents")
        }
    }

    func isDefaultAgent(_ id: String) -> Bool { AgentProfile.defaults.contains { $0.id == id } }

    func addAgent() {
        let id = "custom-\(UUID().uuidString.prefix(8))"
        agents.append(AgentProfile(id: id, name: "Nuovo agente", icon: "person.fill.questionmark",
                                   systemPrompt: "", toolNames: []))
        saveAgents()
    }

    func deleteAgent(_ id: String) {
        guard !isDefaultAgent(id), agents.count > 1 else { return }
        agents.removeAll { $0.id == id }
        if selectedAgentId == id { selectAgent(agents[0].id) }
        saveAgents()
    }

    func restoreDefaultAgents() {
        agents = AgentProfile.defaults
        if !agents.contains(where: { $0.id == selectedAgentId }) { selectAgent(agents[0].id) }
        saveAgents()
    }

    /// Agents as pretty JSON (for export/sharing between machines).
    func exportAgentsData() -> Data? {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try? enc.encode(agents)
    }

    /// Merge agents from JSON: matching ids are updated, new ones appended.
    /// Returns how many agents were imported (0 = invalid file).
    @discardableResult
    func importAgents(from data: Data) -> Int {
        guard let imported = try? JSONDecoder().decode([AgentProfile].self, from: data),
              !imported.isEmpty else { return 0 }
        for a in imported {
            if let i = agents.firstIndex(where: { $0.id == a.id }) { agents[i] = a }
            else { agents.append(a) }
        }
        saveAgents()
        return imported.count
    }

    /// The agent with the user's extra system prompt appended (if any).
    private func resolvedAgent() -> AgentProfile {
        var agent = selectedAgent
        let extra = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !extra.isEmpty {
            agent.systemPrompt = agent.systemPrompt.isEmpty ? extra : agent.systemPrompt + "\n\n" + extra
        }
        return agent
    }

    /// Apply the agent to the running service: fresh chat with its role + tools,
    /// per-agent usage profile swapped in, slot-cache re-warmed.
    private func applyAgent() {
        guard let service else { return }
        let agent = resolvedAgent()
        toolsEnabled = !agent.toolNames.isEmpty
        enabledToolNames = Set(agent.toolNames)
        let tools = toolsEnabled ? ToolRegistry.specs(enabled: enabledToolNames) : []
        Task {
            await service.setAgent(agent, tools: tools)
            await service.setCompactTools(compactTools)
            refreshTuningInfo()
        }
    }

    func selectAgent(_ id: String) {
        selectedAgentId = id
        startNewChat()   // a role switch starts a fresh persisted chat with that role
    }

    // Discovered GGUF files on disk.
    var discoveredModels: [DiscoveredModel] = []

    // Last applied preset explanation, shown in the load screen.
    var presetNote: String?

    // Sampling. Temperature is user-tunable (lower = more focused, less drift —
    // helps on aggressively-quantized models). Persisted across launches.
    var temperature: Double = UserDefaults.standard.object(forKey: "DS4Temperature") as? Double ?? 0.6 {
        didSet { UserDefaults.standard.set(temperature, forKey: "DS4Temperature") }
    }
    /// Repetition penalty (>1 discourages repeats; breaks the repeat-loop collapse
    /// on quantized models). 1.0 = off. Persisted.
    var repetitionPenalty: Double = UserDefaults.standard.object(forKey: "DS4RepPenalty") as? Double ?? 1.1 {
        didSet { UserDefaults.standard.set(repetitionPenalty, forKey: "DS4RepPenalty") }
    }
    private var sampling: SamplingParams {
        SamplingParams(temperature: Float(temperature), repetitionPenalty: Float(repetitionPenalty))
    }

    // Tools.
    var toolsEnabled = false
    var enabledToolNames: Set<String> = Set(ToolRegistry.builtins.map { $0.spec.name })
    /// Compact tool declaration (just name(params)) — fewer prefill tokens. On by
    /// default for local inference.
    var compactTools = true
    /// Tool calls awaiting a manually-entered result (non-built-in tools).
    var pendingManualCalls: [ToolCall] = []
    /// Drives the manual-results sheet (set when `pendingManualCalls` is filled).
    var awaitingManualResults = false
    private var partialAutoOutputs: [ToolOutput] = []
    /// Tool-loop bound. Illimitato su richiesta: il loop si ferma comunque quando
    /// il contesto è pieno o l'utente preme Stop.
    private var toolRounds = 0
    private var maxToolRounds: Int { .max }

    // Live state.
    var phase: Phase = .needsModel
    var info: ModelInfo?
    var messages: [UIMessage] = []
    var input = ""
    /// Text files staged for the next message (folded into the user turn on send).
    var attachments: [ChatAttachment] = []
    /// Transient composer note (e.g. a file that couldn't be decoded as text).
    var attachmentNote: String?
    /// Rough token estimate of the staged attachments (≈4 chars/token); nil if none.
    /// Used to warn before they overflow the context window.
    var attachmentTokenEstimate: Int? {
        guard !attachments.isEmpty else { return nil }
        return attachments.reduce(0) { $0 + $1.content.count } / 4
    }
    var think = false
    var isGenerating = false
    var status = ""          // live prefill/decode progress
    /// Tokens committed to the KV (≈ context used); drives the near-full warning.
    var contextUsed = 0

    // MARK: - Chat sessions (persistent, multiple)

    /// All persisted chats, newest first. The active one's transcript is mirrored
    /// in `messages`; the others live on disk and are loaded on demand.
    var sessions: [ChatSession] = []
    /// Id of the chat currently shown in `messages`.
    var activeSessionId: String = ""
    /// True when the engine's KV already holds the active chat. False right after a
    /// persisted chat is restored: the next send re-primes the engine from the
    /// visible history (the disk-KV cache restores the prefix), after which turns
    /// are incremental again.
    private var enginePrimed = true

    private var service: InferenceService?
    private var generation: Task<Void, Never>?

    var isReady: Bool { if case .ready = phase { return true } else { return false } }
    var availableTools: [ToolSpec] { ToolRegistry.builtins.map(\.spec) }

    private var bookmarkRestored = false

    /// Under the App Sandbox, re-open the last user-picked GGUF via its persisted
    /// security-scoped bookmark (starts access). No-op if none / already restored.
    func restoreModelBookmark() {
        guard !bookmarkRestored else { return }
        bookmarkRestored = true
        if let path = ModelPicker.restoreBookmark() { modelPath = path }
    }

    /// Scan the configured directories for GGUF files.
    func scanModels() {
        let gguf = (scriptDir as NSString).appendingPathComponent("gguf")
        discoveredModels = ModelCatalog.scan(directories: [scriptDir, gguf])
    }

    /// Apply the preset recommended for the detected RAM.
    func applyRecommendedPreset() {
        scanModels()
        let preset = HardwarePresets.forRAM(MemoryInfo.physicalBytes)
        contextSize = preset.contextSize

        var note = preset.summary
        if preset.prefersTwoBit {
            if let twoBit = discoveredModels.first(where: { HardwarePresets.isTwoBit($0.name) }) {
                modelPath = twoBit.path
                note += " Selezionato il modello 2-bit: \(twoBit.name)."
            } else {
                note += " Nessun modello 2-bit trovato: scaricalo con il pulsante “Scarica…” (target q2-imatrix) o `./download_model.sh q2-imatrix`."
            }
        }
        presetNote = note
    }

    /// Open the model off the main thread, then flip to `.ready`.
    func load() {
        guard phase != .loading else { return }
        phase = .loading
        let path = modelPath, ctx = contextSize
        let cacheSlots = expertCacheSlots
        let kvDir = diskKVEnabled ? Self.diskKVDirectory : nil
        let kvBudget = diskKVBudgetMB
        Task.detached(priority: .userInitiated) {
            do {
                let svc = try InferenceService(modelPath: path,
                                               contextSize: ctx,
                                               systemPrompt: nil,   // set by applyAgent below
                                               expertCacheSlots: cacheSlots > 0 ? cacheSlots : nil)
                await svc.setDiskKV(directory: kvDir, budgetMB: kvBudget)
                let info = await svc.modelInfo()
                await MainActor.run {
                    self.service = svc
                    self.info = info
                    self.phase = .ready
                    self.activate(self.activeSessionId)   // load the active chat + apply its role
                }
            } catch {
                await MainActor.run { self.phase = .failed("\(error)") }
            }
        }
    }

    /// Push the current tool selection to the engine (call after toggling tools).
    func syncTools() {
        guard let service else { return }
        let tools = toolsEnabled ? ToolRegistry.specs(enabled: enabledToolNames) : []
        let compact = compactTools
        Task { await service.setTools(tools); await service.setCompactTools(compact) }
    }

    private var thinkMode: DS4ThinkMode { think ? .high : .none }

    /// Send the current input (+ any imported text files) and stream the reply,
    /// running the tool loop. Attachments are folded into the turn sent to the
    /// model; the transcript shows just the typed text and the filenames.
    func send() {
        let typed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let service, !isGenerating, !(typed.isEmpty && attachments.isEmpty) else { return }
        let atts = attachments
        let text = Self.composeUserText(typed: typed, attachments: atts)
        input = ""
        attachments = []
        attachmentNote = nil
        // If the engine doesn't hold this (reopened) chat yet, re-feed the prior
        // turns on this first send. Capture them BEFORE appending the new rows.
        let primed = enginePrimed
        let history = primed ? [] : Self.chatTurns(from: messages)
        let sys = primed ? nil : (resolvedAgent().systemPrompt.isEmpty ? nil : resolvedAgent().systemPrompt)
        enginePrimed = true
        messages.append(UIMessage(role: .user, text: typed, attachments: atts.map(\.name)))
        let index = appendAssistant()
        isGenerating = true
        toolRounds = 0                     // fresh user turn resets the tool-loop guard
        persistActiveSession()             // checkpoint the user turn right away

        let mode = thinkMode
        let params = sampling             // capture: `self` is weak inside the Task
        generation = Task { [weak self] in
            let stream = primed
                ? await service.send(userText: text, thinkMode: mode, sampling: params, maxTokens: 4096)
                : await service.sendWithHistory(history, userText: text, systemPrompt: sys,
                                                thinkMode: mode, sampling: params, maxTokens: 4096)
            await self?.consume(stream, into: index)
            let continued = await self?.handleToolCalls(assistantIndex: index) ?? false
            if !continued { await MainActor.run { self?.finishIfIdle() } }
        }
    }

    /// Submit manually-entered results for the pending (non-built-in) tool calls.
    func submitManualResults(_ contents: [String: String]) {
        guard let service, !pendingManualCalls.isEmpty else { return }
        var outputs = partialAutoOutputs
        for c in pendingManualCalls {
            let content = contents[c.id] ?? ""
            outputs.append(ToolOutput(callId: c.id, name: c.name, content: content))
            messages.append(UIMessage(role: .tool, text: "\(c.name) → \(content)"))
        }
        pendingManualCalls = []
        partialAutoOutputs = []
        awaitingManualResults = false
        continueWithToolOutputs(outputs, service: service)
    }

    /// Abandon the pending manual tool calls without answering them. The calls
    /// stay in the committed KV (the model emitted them), so the next user turn
    /// follows an unanswered call — the model generally copes, but we surface the
    /// abandonment in the transcript so the state is visible.
    func cancelManualResults() {
        if !pendingManualCalls.isEmpty {
            let names = pendingManualCalls.map(\.name).joined(separator: ", ")
            messages.append(UIMessage(role: .tool, text: "✗ risultati non forniti per: \(names)"))
        }
        pendingManualCalls = []
        partialAutoOutputs = []
        awaitingManualResults = false
        finishIfIdle()
    }

    func stop() { generation?.cancel() }

    // MARK: - Text-file attachments

    /// Present an open panel for one or more text files and stage their contents.
    /// Honors the App Sandbox: each pick grants security-scoped access for the
    /// one-shot read (entitlement: files.user-selected.read-write).
    func pickAndAttachFiles() {
        attachmentNote = nil
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.title = "Importa file di testo"
        panel.prompt = "Importa"
        // Prefer text types; allow any file (.data) so odd extensions can still be
        // picked — non-text content simply fails to decode and is reported.
        panel.allowedContentTypes = [.text, .plainText, .sourceCode, .json, .xml,
                                     .commaSeparatedText, .log, .data]
        guard panel.runModal() == .OK else { return }
        importFiles(panel.urls)
    }

    /// Read each URL as text (UTF-8, then Latin-1) and stage it; collect failures.
    func importFiles(_ urls: [URL]) {
        var failed: [String] = []
        for url in urls {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            guard let text = Self.readText(url) else { failed.append(url.lastPathComponent); continue }
            let name = url.lastPathComponent
            // Re-importing the identical file is a no-op (avoid duplicate context).
            if !attachments.contains(where: { $0.name == name && $0.content == text }) {
                attachments.append(ChatAttachment(name: name, content: text))
            }
        }
        if !failed.isEmpty {
            attachmentNote = "Non leggibili come testo: \(failed.joined(separator: ", "))"
        }
    }

    func removeAttachment(_ id: UUID) { attachments.removeAll { $0.id == id } }

    /// Decode a file as text: UTF-8 first, then Latin-1 (covers most legacy files).
    static func readText(_ url: URL) -> String? {
        if let s = try? String(contentsOf: url, encoding: .utf8) { return s }
        return try? String(contentsOf: url, encoding: .isoLatin1)
    }

    /// Fold staged attachments + the typed message into the text sent to the model.
    /// Each file is delimited so the model can tell content apart from the prompt.
    static func composeUserText(typed: String, attachments: [ChatAttachment]) -> String {
        guard !attachments.isEmpty else { return typed }
        var parts: [String] = attachments.map {
            "--- File allegato: \($0.name) ---\n\($0.content)\n--- fine: \($0.name) ---"
        }
        if !typed.isEmpty { parts.append(typed) }
        return parts.joined(separator: "\n\n")
    }

    // MARK: - Tuning tab

    func refreshTuningInfo() {
        guard let service else { tuningInfo = nil; return }
        Task { tuningInfo = await service.tuningInfo() }
    }

    func saveExpertUsage() {
        guard let service else { return }
        Task { await service.saveExpertUsage(); refreshTuningInfo() }
    }

    func resetExpertUsage() {
        guard let service else { return }
        Task { await service.resetExpertUsage(); refreshTuningInfo() }
    }

    // MARK: - Sessions (create / switch / delete / rename / persist)

    func newChat() { startNewChat() }

    /// Make a fresh persisted chat with the current role active. Reuses the current
    /// chat if it's still empty (so flipping the agent before sending anything
    /// doesn't pile up blank chats).
    private func startNewChat() {
        generation?.cancel()
        isGenerating = false
        status = ""
        clearTransientTurnState()
        contextUsed = 0
        enginePrimed = true
        if let i = sessions.firstIndex(where: { $0.id == activeSessionId }),
           messages.isEmpty, sessions[i].messages.isEmpty {
            sessions[i].agentId = selectedAgentId
            sessions[i].systemNote = systemPrompt
            ChatSessionStore.save(sessions[i])
        } else {
            persistActiveSession()
            let session = ChatSession(agentId: selectedAgentId, systemNote: systemPrompt,
                                      modelName: info?.name ?? "")
            sessions.insert(session, at: 0)
            activeSessionId = session.id
            messages.removeAll()
            ChatSessionStore.save(session)
        }
        applyAgent()                      // role + tools + usage profile + resetConversation
    }

    /// Switch to an existing chat: persist the current one, then restore the target.
    func switchSession(_ id: String) {
        guard id != activeSessionId else { return }
        persistActiveSession()
        activate(id)
    }

    func deleteSession(_ id: String) {
        let wasActive = (id == activeSessionId)
        ChatSessionStore.delete(id)
        sessions.removeAll { $0.id == id }
        guard wasActive else { return }
        if let next = sessions.first { activate(next.id) } else { startNewChat() }
    }

    func renameSession(_ id: String, to title: String) {
        guard let i = sessions.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        sessions[i].title = trimmed.isEmpty ? ChatSession.untitled : trimmed
        ChatSessionStore.save(sessions[i])
    }

    /// Restore a session into the live UI and reset the engine to its role (without
    /// persisting the previous one — callers do that first when needed). A non-empty
    /// chat must re-prime on the next send, since the engine no longer holds its KV.
    private func activate(_ id: String) {
        guard let target = sessions.first(where: { $0.id == id }) else { return }
        generation?.cancel()
        isGenerating = false
        status = ""
        clearTransientTurnState()
        activeSessionId = id
        messages = target.messages.map { UIMessage(stored: $0) }
        contextUsed = 0
        systemPrompt = target.systemNote
        if target.agentId != selectedAgentId, agents.contains(where: { $0.id == target.agentId }) {
            selectedAgentId = target.agentId
        }
        enginePrimed = messages.isEmpty
        applyAgent()                      // reset engine to the role; first send re-primes
    }

    /// Snapshot the live transcript into the active session and write it to disk.
    /// Trailing/empty assistant placeholders are dropped so a chat interrupted
    /// mid-generation doesn't reopen with a blank bubble.
    private func persistActiveSession() {
        guard let i = sessions.firstIndex(where: { $0.id == activeSessionId }) else { return }
        let kept = messages.filter {
            !($0.role == .assistant && $0.text.isEmpty && $0.reasoning.isEmpty
              && $0.toolCalls.isEmpty && $0.subAgent == nil)
        }
        sessions[i].messages = kept.map { StoredMessage(from: $0) }
        sessions[i].agentId = selectedAgentId
        sessions[i].systemNote = systemPrompt
        if let name = info?.name { sessions[i].modelName = name }
        sessions[i].updatedAt = Date()
        if sessions[i].title == ChatSession.untitled {
            sessions[i].title = Self.deriveTitle(from: messages)
        }
        ChatSessionStore.save(sessions[i])
    }

    private func clearTransientTurnState() {
        attachments = []
        attachmentNote = nil
        pendingManualCalls = []
        partialAutoOutputs = []
        awaitingManualResults = false
        toolRounds = 0
    }

    /// First non-empty user line, for an auto title.
    private static func deriveTitle(from messages: [UIMessage]) -> String {
        guard let first = messages.first(where: { $0.role == .user && !$0.text.isEmpty }) else {
            return ChatSession.untitled
        }
        let line = first.text.split(separator: "\n").first.map(String.init) ?? first.text
        return String(line.prefix(48))
    }

    // MARK: - Internals

    /// Parse the (target, question, agent, tools) arguments of a `subagent_run`
    /// call. `tools` accepts a JSON array or a comma/space-separated string (some
    /// models quote list arguments).
    private static func subAgentArgs(_ json: String) -> (target: String, question: String, agent: String, tools: [String]) {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return ("", "", "", []) }
        var tools: [String] = []
        if let arr = obj["tools"] as? [Any] { tools = arr.compactMap { $0 as? String } }
        else if let s = obj["tools"] as? String {
            tools = s.split(whereSeparator: { $0 == "," || $0 == " " }).map(String.init)
        }
        return ((obj["target"] as? String) ?? "", (obj["question"] as? String) ?? "",
                (obj["agent"] as? String) ?? "", tools)
    }

    private func appendAssistant() -> Int {
        messages.append(UIMessage(role: .assistant, text: ""))
        return messages.count - 1
    }

    private func finishIfIdle() {
        if pendingManualCalls.isEmpty {
            isGenerating = false
            status = ""
            refreshContextUsage()
            persistActiveSession()        // checkpoint the completed turn
        }
    }

    /// Refresh the committed-token count (context usage) from the engine.
    private func refreshContextUsage() {
        guard let service else { contextUsed = 0; return }
        Task { contextUsed = await service.committedTokens() }
    }

    /// Drain one generation stream into the assistant message at `index`.
    private func consume(_ stream: AsyncThrowingStream<GenEvent, Error>, into index: Int) async {
        do {
            for try await event in stream {
                guard index < messages.count else { break }
                switch event {
                case .reasoning(let r): messages[index].reasoning += r
                case .text(let t): messages[index].text += t
                case .toolStream(let s): messages[index].toolStreamText += s
                case .toolCall(let calls):
                    messages[index].toolCalls = calls
                    // The block closed: drop the raw live markup; the formatted card
                    // (ToolCallView) takes over.
                    messages[index].toolStreamText = ""
                    // When the model spelled the DSML markup out as plain text (it
                    // streamed into the bubble), strip the parsed block + any leaked
                    // malformed markup from view.
                    if !messages[index].text.isEmpty {
                        let visible = ToolCallParser.parse(messages[index].text, markup: .dsv4).visibleText
                        messages[index].text = ToolCallParser.stripLeakedMarkup(visible, markup: .dsv4)
                    }
                case .progress(let p): status = p
                }
            }
            // The stream ended: the raw live markup was ephemeral feedback — drop it
            // (a parsed call shows as a card; an unparsable block is surfaced as text).
            // Also scrub any malformed tool markup the model emitted as text (degraded
            // 2-bit output) so the final bubble shows clean prose.
            if index < messages.count {
                messages[index].toolStreamText = ""
                messages[index].text = ToolCallParser.stripLeakedMarkup(messages[index].text, markup: .dsv4)
            }
        } catch is CancellationError {
            // User-initiated stop: keep the partial text, no error banner.
        } catch {
            let tail = EngineLog.shared.tail()
            if index < messages.count {
                messages[index].text += "\n[errore: \(error)]"
                if !tail.isEmpty { messages[index].text += "\n\n--- log motore ---\n\(tail)" }
            }
        }
    }

    /// Execute the tool calls the assistant emitted: auto-run built-ins, collect
    /// manual ones, and continue the conversation. Returns true if generation
    /// continues (a continuation was spawned or we're awaiting manual input) — in
    /// which case the caller must NOT mark generation finished.
    private func handleToolCalls(assistantIndex index: Int) async -> Bool {
        guard let service, index < messages.count else { return false }
        let calls = messages[index].toolCalls
        guard !calls.isEmpty else { return false }

        toolRounds += 1
        if toolRounds > maxToolRounds {
            messages.append(UIMessage(role: .tool, text: "⚠️ troppi round di tool (\(maxToolRounds)) — interrotto."))
            return false
        }

        var outputs: [ToolOutput] = []
        var manual: [ToolCall] = []
        for c in calls {
            // subagent_run runs ON the engine (it drives the decoder in an isolated
            // context): the main KV only commits this call + the returned answer.
            if c.name == "subagent_run" {
                let (target, question, agent, tools) = Self.subAgentArgs(c.argumentsJSON)
                status = "sub-agent su \(target)…"
                let run: InferenceService.SubAgentRun
                do {
                    run = try await service.runSubAgent(target: target, question: question, agent: agent, tools: tools)
                } catch is CancellationError {
                    run = InferenceService.SubAgentRun(target: target, question: question,
                                                       answer: "(sub-agent interrotto)", steps: [])
                } catch {
                    run = InferenceService.SubAgentRun(target: target, question: question,
                                                       answer: "Errore sub-agent: \(error)", steps: [])
                }
                messages.append(UIMessage(role: .tool, text: "", subAgent: run))
                outputs.append(ToolOutput(callId: c.id, name: c.name, content: run.answer))
                continue
            }
            if let out = ToolRegistry.execute(c) {
                outputs.append(out)
                messages.append(UIMessage(role: .tool, text: "\(c.name) → \(out.content)"))
            } else {
                manual.append(c)
            }
        }

        if !manual.isEmpty {
            partialAutoOutputs = outputs
            pendingManualCalls = manual
            awaitingManualResults = true
            return true
        }
        continueWithToolOutputs(outputs, service: service)
        return true
    }

    /// Feed tool outputs back and stream the model's continuation (which may emit
    /// further tool calls — the loop repeats, bounded by maxToolRounds).
    private func continueWithToolOutputs(_ outputs: [ToolOutput], service: InferenceService) {
        let index = appendAssistant()
        isGenerating = true
        let mode = thinkMode
        let params = sampling             // capture: `self` is weak inside the Task
        generation = Task { [weak self] in
            let stream = await service.provideToolResults(outputs, thinkMode: mode,
                                                          sampling: params, maxTokens: 4096)
            await self?.consume(stream, into: index)
            let continued = await self?.handleToolCalls(assistantIndex: index) ?? false
            if !continued { await MainActor.run { self?.finishIfIdle() } }
        }
    }
}

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
