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

    // Discovered GGUF files on disk.
    var discoveredModels: [DiscoveredModel] = []

    // Last applied preset explanation, shown in the load screen.
    var presetNote: String?

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
        let sys = systemPrompt
        let tools = toolsEnabled ? ToolRegistry.specs(enabled: enabledToolNames) : []
        let compact = compactTools
        Task.detached(priority: .userInitiated) {
            do {
                let svc = try InferenceService(modelPath: path,
                                               contextSize: ctx,
                                               systemPrompt: sys.isEmpty ? nil : sys)
                await svc.setTools(tools)
                await svc.setCompactTools(compact)
                let info = await svc.modelInfo()
                await MainActor.run {
                    self.service = svc
                    self.info = info
                    self.phase = .ready
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
                                            sampling: SamplingParams(), maxTokens: 4096)
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

    func newChat() {
        guard let service else { return }
        generation?.cancel()              // stop any in-flight generation/tool loop
        isGenerating = false
        status = ""
        let sys = systemPrompt.isEmpty ? nil : systemPrompt
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
                case .toolCall(let calls):
                    messages[index].toolCalls = calls
                    // When the model spelled the DSML markup out as plain text (it
                    // streamed into the bubble), strip the parsed block from view.
                    if !messages[index].text.isEmpty {
                        messages[index].text = ToolCallParser
                            .parse(messages[index].text, markup: .dsv4).visibleText
                    }
                case .progress(let p): status = p
                }
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
                                                          sampling: SamplingParams(), maxTokens: 4096)
            await self?.consume(stream, into: index)
            let continued = await self?.handleToolCalls(assistantIndex: index) ?? false
            if !continued { await MainActor.run { self?.finishIfIdle() } }
        }
    }
}
