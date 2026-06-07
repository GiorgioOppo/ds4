import SwiftUI
import DS4Engine

/// A message as shown in the UI: reasoning and visible answer are kept apart so
/// the chain-of-thought can be collapsed.
struct UIMessage: Identifiable {
    let id = UUID()
    let role: ChatRole
    var reasoning: String = ""
    var text: String
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

    // Live state.
    var phase: Phase = .needsModel
    var info: ModelInfo?
    var messages: [UIMessage] = []
    var input = ""
    var think = false
    var isGenerating = false
    var status = ""          // live prefill/decode progress (e.g. "prefill 3/11", "12 token · 1.4 tok/s")

    private var service: InferenceService?
    private var generation: Task<Void, Never>?

    var isReady: Bool { if case .ready = phase { return true } else { return false } }

    /// Scan the configured directories for GGUF files.
    func scanModels() {
        let gguf = (scriptDir as NSString).appendingPathComponent("gguf")
        discoveredModels = ModelCatalog.scan(directories: [scriptDir, gguf])
    }

    /// Apply the preset recommended for the detected RAM: sets streaming, cache,
    /// and context, prefers a 2-bit model if one is on disk, and notes how to get
    /// one otherwise.
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
        Task.detached(priority: .userInitiated) {
            do {
                let svc = try InferenceService(modelPath: path,
                                               contextSize: ctx,
                                               systemPrompt: sys.isEmpty ? nil : sys,
                                               streaming: streaming,
                                               minimumRAMMode: minRAM,
                                               perLayerStreaming: perLayer)
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

    /// Send the current input and stream the reply into the last message.
    func send() {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let service, !text.isEmpty, !isGenerating else { return }
        input = ""
        messages.append(UIMessage(role: .user, text: text))
        messages.append(UIMessage(role: .assistant, text: ""))
        let index = messages.count - 1
        isGenerating = true
        let thinkMode: DS4ThinkMode = think ? .high : .none

        generation = Task { [weak self] in
            let stream = await service.send(userText: text,
                                            thinkMode: thinkMode,
                                            sampling: SamplingParams(),
                                            maxTokens: 4096)
            do {
                for try await event in stream {
                    guard let self, index < self.messages.count else { break }
                    switch event {
                    case .reasoning(let r): self.messages[index].reasoning += r
                    case .text(let t): self.messages[index].text += t
                    case .progress(let p): self.status = p
                    }
                }
            } catch {
                let tail = EngineLog.shared.tail()
                self?.messages[index].text += "\n[errore: \(error)]"
                if !tail.isEmpty {
                    self?.messages[index].text += "\n\n--- log motore ---\n\(tail)"
                }
            }
            self?.isGenerating = false
            self?.status = ""
        }
    }

    func stop() {
        generation?.cancel()
    }

    func newChat() {
        guard let service else { return }
        let sys = systemPrompt.isEmpty ? nil : systemPrompt
        messages.removeAll()
        Task { await service.resetConversation(systemPrompt: sys) }
    }
}
