import SwiftUI
import AppKit
import UniformTypeIdentifiers
import DS4Engine
import DS4Core

extension ChatStore {
    /// Under the App Sandbox, re-open the last user-picked GGUF via its persisted
    /// security-scoped bookmark (starts access). No-op if none / already restored.
    func restoreModelBookmark() {
        guard !bookmarkRestored else { return }
        bookmarkRestored = true
        if let path = ModelPicker.restoreBookmark() { modelPath = path }
        // Folder grant (sandbox): re-arm access to the model's directory so the
        // sidecar caches next to the GGUF stay readable across launches.
        ModelPicker.restoreFolderBookmark()
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
                note += " Selected 2-bit model: \(twoBit.name)."
            } else {
                note += " No 2-bit model found: download it with the Download button (target q2-imatrix) or `./download_model.sh q2-imatrix`."
            }
        }
        presetNote = note
    }

    /// Open the model off the main thread, then flip to `.ready`.

    func load() {
        guard phase != .loading else { return }
        phase = .loading
        loadFraction = 0
        loadStage = ""
        LoadProgress.shared.reset()
        // Poll del progresso finché il load è in corso (si auto-cancella).
        let poller = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                let s = LoadProgress.shared.snapshot
                self?.loadFraction = s.fraction
                self?.loadStage = s.stage
                try? await Task.sleep(nanoseconds: 120_000_000)
            }
        }
        let path = modelPath, ctx = contextSize
        let cacheSlots = expertCacheSlots
        let kvDir = diskKVEnabled ? Self.diskKVDirectory : nil
        let kvBudgetTokens = diskKVBudgetKTok * 1000
        Task.detached(priority: .userInitiated) {
            defer { poller.cancel() }
            do {
                // Il motore si costruisce su un thread GCD CLASSICO, non sul
                // thread del pool cooperativo di Swift Concurrency di questo
                // Task: il load usa DispatchQueue.concurrentPerform ovunque
                // (riquantizzazione Q4, lettura cache, fill esperti) e chiamato
                // da un thread cooperativo può degradare a esecuzione quasi
                // seriale — un core al 100% e la riquantizzazione in ore. La
                // demo CLI, che carica dal main thread, ha sempre avuto il
                // fan-out pieno: stesso contesto anche qui.
                let svc = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<InferenceService, Error>) in
                    DispatchQueue.global(qos: .userInitiated).async {
                        do {
                            cont.resume(returning: try InferenceService(
                                modelPath: path,
                                contextSize: ctx,
                                systemPrompt: nil,   // set by applyAgent below
                                expertCacheSlots: cacheSlots > 0 ? cacheSlots : nil))
                        } catch {
                            cont.resume(throwing: error)
                        }
                    }
                }
                await svc.setDiskKV(directory: kvDir, budgetTokens: kvBudgetTokens)
                let info = await svc.modelInfo()
                await MainActor.run {
                    self.service = svc
                    self.info = info
                    self.phase = .ready
                    self.activate(self.activeSessionId)   // load the active chat + apply its role
                }
                // Warmup in background A UI GIÀ PRONTA: paga ORA il costo
                // una-tantum della prima generazione (creazione pool esperti
                // + fill top-usage, ~GB da SSD, kernel Metal freddi) invece
                // che sul primo messaggio. Un send immediato si accoda al
                // warmup sull'actor: mai più lento di prima, di norma il
                // primo token passa da ~5-7s a ~1s.
                await svc.warmup()
            } catch {
                await MainActor.run { self.phase = .failed("\(error)") }
            }
        }
    }

    /// Push the current tool selection to the engine (call after toggling tools).
    /// Also re-run when an MCP server (dis)connects: the declared set includes
    /// MCP specs, which exist only while their server is connected.
    func syncTools() {
        guard let service else { return }
        let tools = toolsEnabled ? ToolRegistry.autoSpecs(enabled: enabledToolNames) : []
        let compact = compactTools
        Task { await service.setTools(tools); await service.setCompactTools(compact) }
    }

    var thinkMode: DS4ThinkMode { think ? .high : .none }

}
