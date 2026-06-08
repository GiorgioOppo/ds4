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
    var modelPath = AppEnvironment.defaultModelPath
    var scriptDir = AppEnvironment.resourceDir   // download_model.sh / gguf
    var contextSize = 8192
    var systemPrompt = ""
    var streamingEnabled = false
    var streamingCacheSpec = ""     // e.g. "32GB"; empty = auto
    /// Tier-A per-layer streaming: tells the engine not to pin or warm the
    /// model so macOS pages out cold layers. Required on tight-RAM machines.
    var minimumRAMMode = false
    /// Tier-B per-layer streaming: drive decode one layer at a time and call
    /// MADV_DONTNEED on each finished layer's weights. Costs latency per token
    /// but caps the resident working-set to roughly one layer.
    var perLayerStreaming = false

    // Discovered GGUF files on disk.
    var discoveredModels: [DiscoveredModel] = []

    // Last applied preset explanation, shown in the load screen.
    var presetNote: String?

    // Tools.
    var toolsEnabled = false
    var enabledToolNames: Set<String> = Set(ToolRegistry.builtins.map { $0.spec.name })
    /// Tool calls awaiting a manually-entered result (non-built-in tools).
    var pendingManualCalls: [ToolCall] = []
    /// Drives the manual-results sheet (set when `pendingManualCalls` is filled).
    var awaitingManualResults = false
    private var partialAutoOutputs: [ToolOutput] = []

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
        streamingEnabled = preset.streaming
        streamingCacheSpec = preset.cacheSpec
        contextSize = preset.contextSize
        minimumRAMMode = preset.minimumRAMMode
        perLayerStreaming = preset.perLayerStreaming

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
        let streaming = StreamingOptions(enabled: streamingEnabled,
                                         cacheSpec: streamingCacheSpec.isEmpty ? nil : streamingCacheSpec)
        let minRAM = minimumRAMMode
        let perLayer = perLayerStreaming
        let tools = toolsEnabled ? ToolRegistry.specs(enabled: enabledToolNames) : []
        Task.detached(priority: .userInitiated) {
            do {
                let svc = try InferenceService(modelPath: path,
                                               contextSize: ctx,
                                               systemPrompt: sys.isEmpty ? nil : sys,
                                               streaming: streaming,
                                               minimumRAMMode: minRAM,
                                               perLayerStreaming: perLayer)
                await svc.setTools(tools)
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
        Task { await service.setTools(tools) }
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

        let mode = thinkMode
        generation = Task { [weak self] in
            let stream = await service.send(userText: text, thinkMode: mode,
                                            sampling: SamplingParams(), maxTokens: 4096)
            await self?.consume(stream, into: index)
            await self?.handleToolCalls(assistantIndex: index)
            await MainActor.run { self?.finishIfIdle() }
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

    func cancelManualResults() {
        pendingManualCalls = []
        partialAutoOutputs = []
        awaitingManualResults = false
        finishIfIdle()
    }

    func stop() { generation?.cancel() }

    func newChat() {
        guard let service else { return }
        let sys = systemPrompt.isEmpty ? nil : systemPrompt
        messages.removeAll()
        pendingManualCalls = []
        partialAutoOutputs = []
        awaitingManualResults = false
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
                case .toolCall(let calls): messages[index].toolCalls = calls
                case .progress(let p): status = p
                }
            }
        } catch {
            let tail = EngineLog.shared.tail()
            if index < messages.count {
                messages[index].text += "\n[errore: \(error)]"
                if !tail.isEmpty { messages[index].text += "\n\n--- log motore ---\n\(tail)" }
            }
        }
    }

    /// Execute the tool calls the assistant emitted: auto-run built-ins, collect
    /// manual ones, and continue the conversation when all results are ready.
    private func handleToolCalls(assistantIndex index: Int) async {
        guard let service, index < messages.count else { return }
        let calls = messages[index].toolCalls
        guard !calls.isEmpty else { return }

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
            // Pause for manual entry; the auto results so far are kept.
            partialAutoOutputs = outputs
            pendingManualCalls = manual
            awaitingManualResults = true
            return
        }
        continueWithToolOutputs(outputs, service: service)
    }

    /// Feed tool outputs back and stream the model's continuation (which may emit
    /// further tool calls — the loop repeats).
    private func continueWithToolOutputs(_ outputs: [ToolOutput], service: InferenceService) {
        let index = appendAssistant()
        isGenerating = true
        let mode = thinkMode
        generation = Task { [weak self] in
            let stream = await service.provideToolResults(outputs, thinkMode: mode,
                                                          sampling: SamplingParams(), maxTokens: 4096)
            await self?.consume(stream, into: index)
            await self?.handleToolCalls(assistantIndex: index)
            await MainActor.run { self?.finishIfIdle() }
        }
    }
}
