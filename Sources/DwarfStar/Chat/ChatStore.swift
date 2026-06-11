import SwiftUI
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
    var modelPath = AppEnvironment.defaultModelPath
    var scriptDir = AppEnvironment.resourceDir   // download_model.sh / gguf
    var contextSize = 8192
    var systemPrompt = ""
    /// Expert slot-cache slots per layer (0 = off). Wired memory ≈ 6,9 MB/slot ×
    /// 43 layer on the 2-bit model. Applied on the NEXT model load.
    var expertCacheSlots: Int = UserDefaults.standard.integer(forKey: "DS4ExpertCacheSlots") {
        didSet { UserDefaults.standard.set(expertCacheSlots, forKey: "DS4ExpertCacheSlots") }
    }

    // Tuning tab state.
    var tuningInfo: InferenceService.TuningInfo?

    // Agents (roles). Selecting one starts a fresh chat with its role and swaps
    // the per-agent expert-usage profile (the cache re-warms with ITS experts).
    // Editable + persisted; edits apply on the next new chat / agent switch.
    var agents: [AgentProfile] = ChatStore.loadAgents()
    var selectedAgentId: String = UserDefaults.standard.string(forKey: "DS4SelectedAgent") ?? "generale" {
        didSet { UserDefaults.standard.set(selectedAgentId, forKey: "DS4SelectedAgent") }
    }
    var selectedAgent: AgentProfile { agents.first { $0.id == selectedAgentId } ?? agents[0] }

    static func loadAgents() -> [AgentProfile] {
        if let data = UserDefaults.standard.data(forKey: "DS4Agents"),
           let arr = try? JSONDecoder().decode([AgentProfile].self, from: data), !arr.isEmpty {
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
        generation?.cancel()
        isGenerating = false
        status = ""
        messages.removeAll()
        pendingManualCalls = []
        partialAutoOutputs = []
        awaitingManualResults = false
        toolRounds = 0
        applyAgent()
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
    private var sampling: SamplingParams { SamplingParams(temperature: Float(temperature)) }

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
    /// Guard against a tool loop (model re-calling instead of answering).
    private var toolRounds = 0
    private let maxToolRounds = 6

    // Live state.
    var phase: Phase = .needsModel
    var info: ModelInfo?
    var messages: [UIMessage] = []
    var input = ""
    var think = false
    var isGenerating = false
    var status = ""          // live prefill/decode progress

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
        Task.detached(priority: .userInitiated) {
            do {
                let svc = try InferenceService(modelPath: path,
                                               contextSize: ctx,
                                               systemPrompt: nil,   // set by applyAgent below
                                               expertCacheSlots: cacheSlots > 0 ? cacheSlots : nil)
                let info = await svc.modelInfo()
                await MainActor.run {
                    self.service = svc
                    self.info = info
                    self.phase = .ready
                    self.applyAgent()   // role + tools + per-agent usage profile
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

    /// Send the current input and stream the reply, running the tool loop.
    func send() {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let service, !text.isEmpty, !isGenerating else { return }
        input = ""
        messages.append(UIMessage(role: .user, text: text))
        let index = appendAssistant()
        isGenerating = true
        toolRounds = 0                     // fresh user turn resets the tool-loop guard

        let mode = thinkMode
        generation = Task { [weak self] in
            let stream = await service.send(userText: text, thinkMode: mode,
                                            sampling: sampling, maxTokens: 4096)
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

    func newChat() {
        guard let service else { return }
        generation?.cancel()              // stop any in-flight generation/tool loop
        isGenerating = false
        status = ""
        let agentSys = resolvedAgent().systemPrompt
        let sys = agentSys.isEmpty ? nil : agentSys
        messages.removeAll()
        pendingManualCalls = []
        partialAutoOutputs = []
        awaitingManualResults = false
        toolRounds = 0
        Task { await service.resetConversation(systemPrompt: sys) }
    }

    // MARK: - Internals

    private func appendAssistant() -> Int {
        messages.append(UIMessage(role: .assistant, text: ""))
        return messages.count - 1
    }

    private func finishIfIdle() {
        if pendingManualCalls.isEmpty { isGenerating = false; status = "" }
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
        generation = Task { [weak self] in
            let stream = await service.provideToolResults(outputs, thinkMode: mode,
                                                          sampling: sampling, maxTokens: 4096)
            await self?.consume(stream, into: index)
            let continued = await self?.handleToolCalls(assistantIndex: index) ?? false
            if !continued { await MainActor.run { self?.finishIfIdle() } }
        }
    }
}
