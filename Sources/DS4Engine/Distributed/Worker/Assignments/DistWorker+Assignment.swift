import Foundation
import DS4Core

extension DistWorker {
    /// Handle an ASSIGN: validate, resolve the gguf (files distributed by the
    /// coordinator first, then local fallbacks), load (or reuse) the slice
    /// engine, reply READY — or an ERROR frame with the reason.
    func handleAssign(_ payload: Data, on conn: DistConnection,
                              resolvedFiles: [String: String]) async throws {
        guard let assign = DistAssign.decode(payload) else {
            try await conn.sendFrame(.error, Data("malformed ASSIGN frame".utf8))
            return
        }
        guard assign.layerStart >= 0, assign.layerStart <= assign.layerEnd,
              assign.contextSize > 0 else {
            let msg = "invalid ASSIGN: layers \(assign.layerStart)...\(assign.layerEnd) "
                + "and context \(assign.contextSize)"
            onLog(msg + "\n")
            try await conn.sendFrame(.error, Data(msg.utf8))
            return
        }
        // The coordinator distributes the files: whatever the OFFER resolved
        // (received into the managed store, or hash-matched locally) wins;
        // the plain path fallback remains for robustness.
        let sanitizedName = DistFileStore.sanitize(assign.modelName)
        guard let resolved = resolvedFiles[sanitizedName]
                ?? Self.resolveModelPath(requestedPath: assign.modelPath,
                                         modelName: assign.modelName,
                                         localHint: config.localModelPath) else {
            let msg = "gguf '\(assign.modelName)' not available on this worker "
                + "(not offered/transferred, and no hash-matching local file)"
            onLog(msg + "\n")
            try await conn.sendFrame(.error, Data(msg.utf8))
            return
        }
        // Validate the slice against the model that actually won resolution
        // (transferred/hash-matched/local), before allocating Metal or KV. A
        // worker may have a Flash local hint while the coordinator assigns Pro.
        let modelLayout: (nLayers: Int, nExperts: Int)
        do {
            modelLayout = try DistEngine.inspectLayout(modelPath: resolved)
        } catch {
            let msg = "cannot inspect assigned GGUF '\(assign.modelName)': \(error)"
            onLog(msg + "\n")
            try await conn.sendFrame(.error, Data(msg.utf8))
            return
        }
        guard assign.layerEnd < modelLayout.nLayers else {
            let msg = "invalid ASSIGN: layers \(assign.layerStart)...\(assign.layerEnd) "
                + "outside resolved \(modelLayout.nLayers)-layer model"
            onLog(msg + "\n")
            try await conn.sendFrame(.error, Data(msg.utf8))
            return
        }
        // v9: i knob di PERFORMANCE del coordinatore, applicati PRIMA del
        // load (l'env è letto alla creazione del motore). Doppia whitelist:
        // il filo non deve mai poter impostare variabili arbitrarie qui.
        let allowedKnobs = Set(Dist.perfKnobKeys)
        var appliedKnobs: [String] = []
        for (k, v) in assign.envKnobs where allowedKnobs.contains(k) && v.count <= 256 {
            _ = setenv(k, v, 1)
            appliedKnobs.append("\(k)=\(v)")
        }
        if !appliedKnobs.isEmpty {
            onLog("knob dal coordinatore: " + appliedKnobs.joined(separator: " ") + "\n")
        }
        // Sidecar/derived caches: coordinator-decided. The env is read at
        // ENGINE LOAD; pointing the cache-dir vars at the transferred files'
        // directory covers the case where the gguf resolved elsewhere (the
        // sibling rule would miss them).
        _ = setenv("DS4_EXPERT_BUNDLE", assign.useExpertBundle ? "1" : "0", 1)
        if assign.useExpertBundle,
           let bundlePath = resolvedFiles[sanitizedName + ".expbundle"] {
            _ = setenv("DS4_BUNDLE_DIR", (bundlePath as NSString).deletingLastPathComponent, 1)
        }
        _ = setenv("DS4_DENSE_Q4", assign.useDenseQ4 ? "1" : "0", 1)
        if assign.useDenseQ4,
           let q4Path = resolvedFiles[sanitizedName + ".q4dense"] {
            // The coordinator's requant cache: loaded in ~0.5 s instead of
            // re-requantizing for minutes on this worker.
            _ = setenv("DS4_Q4_CACHE_DIR", (q4Path as NSString).deletingLastPathComponent, 1)
        }
        let wanted = Assignment(resolvedModelPath: resolved, contextSize: assign.contextSize,
                                expertCacheSlots: assign.expertCacheSlots,
                                useExpertBundle: assign.useExpertBundle,
                                useDenseQ4: assign.useDenseQ4,
                                layerStart: assign.layerStart, layerEnd: assign.layerEnd,
                                hasOutput: assign.hasOutput)

        // Locked state transitions live in sync helpers: NSLock is not usable
        // directly inside an async function.
        var claim = claimAssignment(wanted)
        if case .inFlight = claim {
            // v8 retry: la connessione che ha chiesto QUESTO stesso load è
            // caduta e il coordinatore ha riconnesso. Il load prosegue per
            // conto suo: aggancialo — rilancia il progresso e attendi l'esito
            // invece di rispondere "busy" (che il coordinatore tratta come
            // fatale e affosserebbe la route per un blip di rete).
            onLog("assegnazione identica già in caricamento — mi aggancio al load in corso\n")
            let deadline = Date().addingTimeInterval(45 * 60)
            joinLoop: while Date() < deadline {
                try await Task.sleep(nanoseconds: 1_500_000_000)
                let s = LoadProgress.shared.snapshot
                if !s.stage.isEmpty {
                    try? await conn.sendFrame(.progress,
                                              Data(String(format: "%@ (%.0f%%)", s.stage, s.fraction * 100).utf8))
                }
                claim = claimAssignment(wanted)
                if case .inFlight = claim { continue }
                break joinLoop           // reuse (commit), load (il vecchio è fallito: slot nostro) o busy
            }
            if case .inFlight = claim {
                try await conn.sendFrame(.error, Data("load in corso da troppo tempo (timeout join)".utf8))
                return
            }
        }
        switch claim {
        case .reuse(let current):
            applyAncillary(assign, to: current)
            onLog("assegnazione invariata: layer \(wanted.layerStart)...\(wanted.layerEnd) — riuso il motore\n")
            try await conn.sendFrame(.ready, helloPayload().encoded())
            return
        case .busy:
            try await conn.sendFrame(.error, Data("worker busy loading a previous assignment".utf8))
            return
        case .inFlight:
            return                   // impossibile: gestito sopra (timeout incluso)
        case .load:
            break                    // old shard freed, `loadingAssignment` claimed
        }

        onLog("assegnazione: \(assign.modelName) · layer \(wanted.layerStart)...\(wanted.layerEnd)"
              + (wanted.hasOutput ? " +output" : "") + " · ctx \(wanted.contextSize)"
              + " · modello \(modelLayout.nLayers) layer/\(modelLayout.nExperts) esperti"
              + (wanted.expertCacheSlots > 0 ? " · \(wanted.expertCacheSlots) slot cache" : "")
              + " — carico il motore…\n")
        let t0 = Date()
        // The load is SILENT for minutes otherwise (mmap, Metal init, resident
        // copies, sidecar/Q4 REQUANT builds on first run): mirror the phase
        // breadcrumbs + LoadProgress into the worker log AND relay them to the
        // coordinator as `progress` frames, so BOTH logs show what is running.
        LoadProgress.shared.reset()
        let report: @Sendable (String) -> Void = { [onLog] text in
            onLog("caricamento: \(text)\n")
        }
        let progressConn = conn
        let poller = Task { [report] in
            var lastStage = ""
            var lastPct = -10.0
            while !Task.isCancelled {
                let s = LoadProgress.shared.snapshot
                let pct = s.fraction * 100
                if !s.stage.isEmpty, s.stage != lastStage || pct - lastPct >= 10 {
                    lastStage = s.stage; lastPct = pct
                    let text = String(format: "%@ (%.0f%%)", s.stage, pct)
                    report(text)
                    try? await progressConn.sendFrame(.progress, Data(text.utf8))
                }
                try? await Task.sleep(nanoseconds: 1_500_000_000)
            }
        }
        defer { poller.cancel() }
        do {
            // Off this connection's task: the load (mmap + Metal init) runs for
            // minutes while frames could arrive. On a GCD thread, NOT a Swift
            // Concurrency cooperative thread: the load fans out everywhere with
            // DispatchQueue.concurrentPerform (Q4 requant, cache reads, expert
            // fills), which from a cooperative thread can degrade to near-SERIAL
            // execution — one core pegged and the requant taking hours instead
            // of minutes. Phase breadcrumbs cover what LoadProgress does not.
            let slots = wanted.expertCacheSlots
            let loaded = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<DistEngine, Error>) in
                DispatchQueue.global(qos: .userInitiated).async { [report] in
                    do {
                        cont.resume(returning: try DistEngine(
                            modelPath: resolved, contextSize: wanted.contextSize,
                            expertCacheSlots: slots > 0 ? slots : nil,
                            kvLayers: wanted.layerStart..<(wanted.layerEnd + 1),
                            onLoadLog: report))
                    } catch {
                        cont.resume(throwing: error)
                    }
                }
            }
            poller.cancel()              // no progress frames after READY is submitted
            commitAssignment(loaded, wanted)
            applyAncillary(assign, to: loaded)
            onLog(String(format: "motore pronto in %.1fs\n", Date().timeIntervalSince(t0)))
            try await conn.sendFrame(.ready, helloPayload().encoded())
        } catch {
            poller.cancel()
            releaseAssignmentClaim()
            let msg = "engine load failed: \(error)"
            onLog(msg + "\n")
            try await conn.sendFrame(.error, Data(msg.utf8))
        }
    }

    // Synchronous locked state transitions for handleAssign (the compiler
    // forbids direct NSLock use in async bodies — a suspension while holding
    // the lock would deadlock).

    private enum AssignmentClaim { case reuse(DistEngine), busy, inFlight, load }

    /// Reuse the loaded engine, report busy, join an identical in-flight load,
    /// or claim the load slot (freeing the old shard FIRST so its memory is
    /// gone before the new one arrives).
    private func claimAssignment(_ wanted: Assignment) -> AssignmentClaim {
        stateLock.lock(); defer { stateLock.unlock() }
        if assignment == wanted, let current = engine { return .reuse(current) }
        if loadingAssignment {
            // v8 retry: se la STESSA assegnazione è già in caricamento (la
            // connessione che l'ha chiesta è caduta a metà load, il retry del
            // coordinatore riconnette), il nuovo tentativo si AGGANCIA al load
            // in corso invece di ricevere "busy" — un blip di rete durante i
            // minuti del load non deve affossare la route.
            return pendingAssignment == wanted ? .inFlight : .busy
        }
        loadingAssignment = true
        pendingAssignment = wanted
        engine = nil
        assignment = nil
        return .load
    }

    private func commitAssignment(_ loaded: DistEngine, _ wanted: Assignment) {
        stateLock.lock()
        engine = loaded
        assignment = wanted
        loadingAssignment = false
        pendingAssignment = nil
        stateLock.unlock()
    }

    private func releaseAssignmentClaim() {
        stateLock.lock(); loadingAssignment = false; pendingAssignment = nil; stateLock.unlock()
    }

    /// Apply the parts of an ASSIGN that do NOT require a reload: the usage
    /// imatrix (slot-cache pre-warm) and the disk-KV budget. Runs on every
    /// ASSIGN, including the engine-reuse path.
    private func applyAncillary(_ assign: DistAssign, to engine: DistEngine) {
        let dir = Self.shardStateDirectory(modelName: assign.modelName,
                                           layerStart: assign.layerStart,
                                           layerEnd: assign.layerEnd)
        // Usage: this shard's own refined profile wins (it already contains a
        // coordinator seed from a previous session); else the ASSIGN's blob.
        let file = dir.appendingPathComponent("usage.json")
        if let own = try? Data(contentsOf: file) {
            engine.loadUsage(own)
            onLog("imatrix: profilo locale dello shard caricato\n")
        } else if !assign.usageJSON.isEmpty {
            engine.loadUsage(assign.usageJSON)
            onLog("imatrix: profilo del coordinatore caricato (\(assign.usageJSON.count) byte)\n")
        }
        engine.setDiskKV(directory: assign.diskKVBudgetTokens > 0
                             ? dir.appendingPathComponent("kv", isDirectory: true) : nil,
                         budgetTokens: assign.diskKVBudgetTokens)
        if assign.diskKVBudgetTokens > 0 {
            onLog("disk KV shard: budget \(assign.diskKVBudgetTokens) token\n")
        }
        stateLock.lock(); usageFile = file; stateLock.unlock()
    }

}
