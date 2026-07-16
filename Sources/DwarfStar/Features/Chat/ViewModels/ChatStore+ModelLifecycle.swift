import SwiftUI
import AppKit
import UniformTypeIdentifiers
import DS4Engine
import DS4Core

extension ChatStore {
    /// Select a runnable entry found by the catalog scanner, preserving an exact
    /// matching sandbox bookmark and neutralizing an unrelated stale one.
    @discardableResult
    func selectCatalogModel(path: String) -> Bool {
        guard phase != .loading else { return false }
        guard ModelPicker.acceptRunnableGGUF(path: path) else { return false }
        ModelPicker.rememberCatalogGGUF(path: path)
        commitModelSelection(path: path)
        scanModels()
        return true
    }

    /// Commit a path already validated and bookmarked by `ModelPicker.pickGGUF`.
    func selectPickedModel(path: String) {
        guard phase != .loading else { return }
        commitModelSelection(path: path)
        scanModels()
    }

    /// A path change cannot keep advertising the old in-memory engine as ready.
    /// Stop in-flight work, persist the transcript and release the old backend;
    /// Settings will then offer Load Model for the newly selected GGUF.
    private func commitModelSelection(path: String) {
        guard modelPath != path else { return }
        if isGenerating { stop() }
        if service != nil {
            persistActiveSession()
        }
        service = nil
        info = nil
        inspectedModelDescriptor = nil
        phase = .needsModel
        enginePrimed = messages.isEmpty
        contextUsed = 0
        modelPath = path
    }

    /// Refresh architecture/capability metadata without allocating Metal or
    /// constructing a decoder. Stale results are discarded if the user changes
    /// the selected file while inspection is running.
    func inspectSelectedModel(path explicitPath: String? = nil) async {
        let path = explicitPath ?? modelPath
        guard !path.isEmpty else {
            inspectedModelDescriptor = nil
            return
        }
        guard modelPath != path || inspectedModelDescriptor == nil else { return }
        let descriptor = await Task.detached(priority: .utility) {
            try? InferenceService.inspectModel(path: path)
        }.value
        guard modelPath == path else { return }
        inspectedModelDescriptor = descriptor
    }

    /// Under the App Sandbox, re-open the last user-picked GGUF via its persisted
    /// security-scoped bookmark (starts access). No-op if none / already restored.
    func restoreModelBookmark() {
        guard !bookmarkRestored else { return }
        bookmarkRestored = true
        // Models downloaded by DwarfStar live in its Application Support
        // container and need no security-scoped bookmark. In particular, do not
        // let an older manual-picker bookmark overwrite a newer catalog choice.
        if !AppEnvironment.isManagedModelPath(modelPath),
           let path = ModelPicker.restoreBookmark() {
            modelPath = path
        }
        // Folder grant (sandbox): re-arm access to the model's directory so the
        // sidecar caches next to the GGUF stay readable across launches.
        ModelPicker.restoreFolderBookmark()
    }

    /// Scan the configured directories for GGUF files.
    func scanModels() {
        let gguf = (scriptDir as NSString).appendingPathComponent("gguf")
        var directories = [
            AppEnvironment.modelDownloadDirectory.path,
            scriptDir,
            gguf,
        ]
        if !modelPath.isEmpty {
            directories.append((modelPath as NSString).deletingLastPathComponent)
        }
        discoveredModels = ModelCatalog.scan(directories: directories)
    }

    /// Apply the preset recommended for the detected RAM.
    func applyRecommendedPreset() {
        scanModels()
        let preset = HardwarePresets.forRAM(MemoryInfo.physicalBytes)
        contextSize = preset.contextSize

        var note = preset.summary
        if preset.prefersTwoBit {
            if let twoBit = discoveredModels.first(where: { HardwarePresets.isTwoBit($0.name) }) {
                if selectCatalogModel(path: twoBit.path) {
                    note += " Selected 2-bit model: \(twoBit.name)."
                } else {
                    note += " Il file 2-bit trovato non è caricabile dal runtime corrente."
                }
            } else {
                note += " Nessun modello 2-bit trovato: apri Scarica modelli e scegli DeepSeek V4 Flash IQ2XXS."
            }
        }
        presetNote = note
    }

    /// Open the model off the main thread, then flip to `.ready`.

    func load() {
        // Rebuilding a decoder while it is generating keeps the old model and
        // its Metal buffers alive during the new allocation. On memory-bound
        // Macs that can exhaust swap and make the app appear to crash.
        guard phase != .loading, !isGenerating else { return }
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
                // Populate capability-driven UI even when backend construction
                // subsequently fails (for example a recognized Qwen GGUF).
                let inspected = try InferenceService.inspectModel(path: path)
                await MainActor.run {
                    if self.modelPath == path { self.inspectedModelDescriptor = inspected }
                }
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
        // Before the first prompt (also when a persisted chat still needs to be
        // re-primed), this selection is exactly what the model will see. Once a
        // live KV exists, setTools intentionally applies only to the next chat,
        // so keep the execution policy tied to the already-declared set.
        if messages.isEmpty || !enginePrimed {
            activeConversationToolNames = Set(tools.map(\.name))
        }
        let compact = compactTools
        Task { await service.setTools(tools); await service.setCompactTools(compact) }
    }

    var thinkMode: DS4ThinkMode { think ? .high : .none }

}
