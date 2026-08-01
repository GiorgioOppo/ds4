import Foundation

extension DistWorker {
    func currentShard() -> ExpertShardEngine? {
        stateLock.lock(); defer { stateLock.unlock() }
        return expertShard
    }
    private func claimShardLoad() -> Bool {
        stateLock.lock(); defer { stateLock.unlock() }
        if loadingShard { return false }
        loadingShard = true
        expertShard = nil            // libera il vecchio shard PRIMA del nuovo load
        return true
    }
    private func commitShard(_ s: ExpertShardEngine?) {
        stateLock.lock(); expertShard = s; loadingShard = false; stateLock.unlock()
    }
    /// EXPERT ASSIGN (Fase B, scissione verticale): carica lo shard di esperti
    /// definito dalla mask. Alternativo all'assegnazione a layer; il modello
    /// arriva dalla stessa distribuzione file v8 (offer/need/resume).
    func handleExpertAssign(_ payload: Data, on conn: DistConnection,
                                    resolvedFiles: [String: String]) async throws {
        guard let assign = DistExpertAssign.decode(payload) else {
            try await conn.sendFrame(.error, Data("malformed EXPERT ASSIGN frame".utf8))
            return
        }
        let sanitizedName = DistFileStore.sanitize(assign.modelName)
        guard let resolved = resolvedFiles[sanitizedName]
                ?? Self.resolveModelPath(requestedPath: "", modelName: assign.modelName,
                                         localHint: config.localModelPath) else {
            let msg = "gguf '\(assign.modelName)' not available on this worker"
            onLog(msg + "\n")
            try await conn.sendFrame(.error, Data(msg.utf8))
            return
        }
        // Knob di performance del coordinatore (stessa whitelist di handleAssign).
        let allowedKnobs = Set(Dist.perfKnobKeys)
        for (k, v) in assign.envKnobs where allowedKnobs.contains(k) && v.count <= 256 {
            _ = setenv(k, v, 1)
        }
        // `useExpertBundle` is accepted only for compatibility with older
        // coordinators; expert shards now read directly from the GGUF.
        guard claimShardLoad() else {
            try await conn.sendFrame(.error, Data("worker busy loading an expert shard".utf8))
            return
        }
        onLog("expert-shard: \(assign.modelName) — carico lo shard…\n")
        do {
            let report: @Sendable (String) -> Void = { [onLog] t in onLog("caricamento shard: \(t)\n") }
            let maskData = assign.expertMask
            let slots = assign.expertCacheSlots
            let usageJSON = assign.usageJSON
            // Su thread GCD, non sul pool cooperativo (stessa ragione del load
            // del motore a slice qui sopra: fan-out concurrentPerform pieni).
            let shard = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<ExpertShardEngine, Error>) in
                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        cont.resume(returning: try ExpertShardEngine(
                            modelPath: resolved, expertMask: maskData,
                            expertCacheSlots: slots, usageJSON: usageJSON,
                            onLoadLog: report))
                    } catch {
                        cont.resume(throwing: error)
                    }
                }
            }
            commitShard(shard)
            onLog("expert-shard pronto: \(shard.ownedCount)/\(shard.nExperts) esperti · \(shard.nLayers) layer\n")
            // READY describes the shard that was actually loaded, not the
            // worker's optional local hint (and not a stale horizontal job).
            let ready = DistHello(modelName: assign.modelName,
                                  layerStart: 0, layerEnd: shard.nLayers - 1,
                                  hasOutput: false, nLayers: shard.nLayers,
                                  contextSize: 0)
            try await conn.sendFrame(.ready, ready.encoded())
        } catch {
            commitShard(nil)
            onLog("expert-shard: load fallito (\(error))\n")
            try await conn.sendFrame(.error, Data("expert shard load failed: \(error)".utf8))
        }
    }

}
