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
    /// Stop in-flight work and hide it immediately, but retain the service until
    /// its GPU/I/O/DiskKV work has drained. A following `load()` either observes
    /// that completed teardown or joins the same service before allocating.
    private func commitModelSelection(path: String) {
        guard modelPath != path else { return }
        if isGenerating { stop() }
        let serviceToRetire = service
        if serviceToRetire != nil {
            persistActiveSession()
        }
        loadedEngineSignature = nil
        info = nil
        inspectedModelDescriptor = nil
        phase = .needsModel
        enginePrimed = messages.isEmpty
        contextUsed = 0
        modelPath = path

        if let serviceToRetire {
            Task { [weak self] in
                guard let self else {
                    await serviceToRetire.quiesceForTeardown()
                    return
                }
                // A role-specific warmup may already have captured the service.
                // Join it first, then establish the final teardown boundary.
                _ = await self.waitForEngineSetup()
                await serviceToRetire.quiesceForTeardown()
                // A new load may have installed another service while this task
                // was suspended; never clear a newer owner.
                if self.service === serviceToRetire {
                    self.service = nil
                }
            }
        }
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
        guard phase != .loading, !isGenerating, !benchRunning,
              EngineActivityGate.shared.activeOwner == nil else { return }
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
        let glmResident = glmResidentLayers
        let glmExperts = glmActiveExperts
        // Knob GLM aggiuntivi: il motore li legge dall'ambiente (stessa
        // strada della demo CLI), quindi la GUI li fissa QUI, prima della
        // costruzione del servizio. 0 = default del motore.
        _ = setenv("DS4_GLM_EXPERT_ARENA",
                   glmExpertArena > 0 ? String(glmExpertArena) : "24", 1)
        _ = setenv("DS4_GLM_STREAM_SLOTS",
                   glmStreamSlots > 0 ? String(glmStreamSlots) : "3", 1)
        _ = setenv("DS4_GLM_MTLIO", glmMetalIOEnabled ? "1" : "0", 1)
        _ = setenv("DS4_GLM_SPEC_EXPERTS",
                   glmSpeculativeExperts ? "1" : "0", 1)
        _ = setenv("DS4_GLM_LAYERQ4", glmUseQ4Sidecar ? "1" : "0", 1)
        // Knob delle ottimizzazioni misurate in sessione: il motore li
        // rilegge a ogni init (GLM52DispatchKnobs.refresh), quindi un
        // toggle qui ha effetto al reload.
        _ = setenv("DS4_GLM_FUSE", glmFuseEnabled ? "1" : "0", 1)
        _ = setenv("DS4_GLM_MOE_BATCH", glmMoEBatchEnabled ? "1" : "0", 1)
        _ = setenv("DS4_GLM_GPU_ROUTER", glmGpuRouterEnabled ? "1" : "0", 1)
        _ = setenv("DS4_GLM_MLOCK", glmMlockEnabled ? "1" : "0", 1)
        _ = setenv("DS4_GLM_READ_SPLIT",
                   glmReadSplit > 0 ? String(glmReadSplit) : "4", 1)
        _ = setenv("DS4_GLM_NSG", glmNSG > 0 ? String(glmNSG) : "4", 1)
        let loadEngineSignature = machineAutoTuneEngineSignature()
        let kvDir = diskKVEnabled ? Self.diskKVDirectory : nil
        let kvBudgetTokens = diskKVBudgetKTok * 1000
        Task.detached(priority: .userInitiated) {
            defer { poller.cancel() }
            do {
                // A role change/warmup may still retain the current service.
                // Drain it before dropping the last store reference, then give
                // Metal/VM a short window to release wired buffers. This avoids
                // ever constructing two model-sized engines at once.
                _ = await self.waitForEngineSetup()
                let previousService = await MainActor.run { self.service }
                await previousService?.quiesceForTeardown()
                await MainActor.run {
                    self.service = nil
                    self.glmService = nil
                    self.loadedEngineSignature = nil
                    self.info = nil
                }
                try await Task.sleep(nanoseconds: 4_000_000_000)

                // Populate capability-driven UI even when backend construction
                // subsequently fails (for example a recognized Qwen GGUF).
                let inspected = try InferenceService.inspectModel(path: path)
                await MainActor.run {
                    if self.modelPath == path { self.inspectedModelDescriptor = inspected }
                }
                // GLM 5.2: chat served by the GLM resident/streaming engine,
                // not the DeepSeek loop. Deliberately minimal surface: no
                // disk KV, roles, tools, benchmark or auto-tune — greedy
                // chat with layer-major prefill.
                if inspected.architecture
                       == GLM52BackendDefinition.supportedArchitecture,
                   GLM52BackendDefinition.runtimeEnabled {
                    // Sidecar GLM: riusa quello accanto al GGUF quando c'è,
                    // altrimenti redirigi la build in Application Support
                    // (in sandbox la cartella del modello non è scrivibile).
                    Self.prepareGLMSidecarEnvironment(modelPath: path)
                    let glm = try await withCheckedThrowingContinuation {
                        (cont: CheckedContinuation<GLM52ChatService, Error>) in
                        DispatchQueue.global(qos: .userInitiated).async {
                            do {
                                cont.resume(returning: try GLM52ChatService(
                                    modelPath: path, contextSize: ctx,
                                    systemPrompt: nil,
                                    residentLayers: glmResident > 0
                                        ? glmResident : nil,
                                    activeExperts: glmExperts > 0
                                        ? glmExperts : nil,
                                    diskKVDirectory: kvDir?.path))
                            } catch {
                                cont.resume(throwing: error)
                            }
                        }
                    }
                    let glmInfo = await glm.modelInfo()
                    // Warmup reale (un token attraverso lo stack streamato):
                    // il primo token utente non paga i costi una-tantum.
                    _ = await glm.warmup()
                    await MainActor.run {
                        self.glmService = glm
                        self.service = nil
                        self.info = glmInfo
                        self.enginePrimed = false
                        self.activate(self.activeSessionId)
                        self.phase = .ready
                    }
                    return
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
                // Warm the default profile while the UI is still in `.loading`.
                // The active role may replace the profile below and perform one
                // additional, awaited warmup before `.ready` is published.
                await svc.warmup()
                await MainActor.run {
                    self.service = svc
                    self.loadedEngineSignature = loadEngineSignature
                    self.info = info
                    self.activate(self.activeSessionId)   // load the active chat + apply its role
                }
                guard await self.waitForEngineSetup() else {
                    await svc.quiesceForTeardown()
                    await MainActor.run {
                        if self.service === svc {
                            self.service = nil
                            self.loadedEngineSignature = nil
                            self.info = nil
                        }
                    }
                    throw NSError(
                        domain: "DwarfStar.EngineSetup",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey:
                            "warmup del ruolo attivo fallito"]
                    )
                }
                await MainActor.run {
                    guard self.modelPath == path, self.service === svc else { return }
                    self.phase = .ready
                }
            } catch {
                await MainActor.run {
                    self.loadedEngineSignature = nil
                    self.phase = .failed("\(error)")
                }
            }
        }
    }

    /// Push the current tool selection to the engine (call after toggling tools).
    /// Also re-run when an MCP server (dis)connects: the declared set includes
    /// MCP specs, which exist only while their server is connected.
    func syncTools() {
        guard EngineActivityGate.shared.activeOwner == nil else { return }
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
