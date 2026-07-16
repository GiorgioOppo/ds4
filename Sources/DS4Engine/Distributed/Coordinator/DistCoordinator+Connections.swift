import Foundation
import DS4Core

extension DistCoordinator {
    /// Contiguous, near-equal split of `nLayers` across `workers` peers, in
    /// peer-list order (the user controls placement by ordering the list); the
    /// first `nLayers % workers` slices get one extra layer. The LAST slice
    /// also runs the output head.
    static func partition(nLayers: Int, workers: Int) throws -> [(start: Int, end: Int)] {
        guard workers > 0 else { throw DistError.sliceGap("no workers") }
        guard workers <= nLayers else {
            throw DistError.sliceGap("more workers (\(workers)) than layers (\(nLayers)): remove peers")
        }
        let base = nLayers / workers, rem = nLayers % workers
        var out: [(Int, Int)] = []
        var start = 0
        for i in 0..<workers {
            let len = base + (i < rem ? 1 : 0)
            out.append((start, start + len - 1))
            start += len
        }
        return out
    }

    // MARK: Session

    /// Establish the route, with the COORDINATOR defining each worker's job:
    /// split the layers across the peers (in list order), then per peer —
    /// HELLO (version check) → ASSIGN (gguf + context + cache slots + slice) →
    /// READY (worker loaded its engine). Then validate contiguous coverage and
    /// (forwarding only) start the return listener.
    public func connect(onLog: @escaping @Sendable (String) -> Void) async throws {
        let slices = try Self.partition(nLayers: engine.nLayers, workers: config.peers.count)
        let modelName = (config.modelPath as NSString).lastPathComponent
        // Everything the workers need comes from HERE: build the file offer
        // once (gguf + sidecar, identified by size + SHA-256; the hash is
        // computed once per machine and answered from the cache afterwards).
        let offer = try buildFileOffer(onLog: onLog)
        // Setup dei peer IN PARALLELO: trasferimento file e caricamento del
        // motore di ogni worker procedono INSIEME, non uno alla volta — il
        // tempo di attivazione della route passa da Σ(setup dei worker) a
        // max(setup). Con i file già distribuiti (dal secondo avvio) N worker
        // si preparano nel tempo di UNO; a freddo i trasferimenti condividono
        // la banda del coordinatore, ma il load (mmap, Metal, requant) di un
        // worker si sovrappone comunque al transfer degli altri. L'ordine
        // della route resta quello della lista peer (raccolta per indice); le
        // righe di log si intrecciano, ma ogni messaggio porta già l'host.
        var results = [(conn: DistConnection, entry: DistRouteEntry)?](repeating: nil,
                                                                       count: config.peers.count)
        try await withThrowingTaskGroup(of: (Int, DistConnection, DistRouteEntry).self) { group in
            for (i, p) in config.peers.enumerated() {
                let slice = slices[i]
                let hasOutput = (i == config.peers.count - 1)
                group.addTask { [self] in
                    let (conn, entry) = try await setupPeer(p, slice: slice, hasOutput: hasOutput,
                                                            offer: offer, modelName: modelName,
                                                            onLog: onLog)
                    return (i, conn, entry)
                }
            }
            for try await (i, conn, entry) in group { results[i] = (conn, entry) }
        }
        for r in results {
            guard let r else { throw DistError.badFrame }  // impossibile: gruppo completo o throw
            conns.append(r.conn)
            entries.append(r.entry)
        }
        // By construction the route is ordered and contiguous; keep the check
        // as a safety net against partition bugs.
        var expected = 0
        for e in entries {
            guard e.layerStart == expected else { throw DistError.sliceGap("expected layer \(expected), found \(e.layerStart)") }
            expected = e.layerEnd + 1
        }
        guard expected == engine.nLayers else {
            throw DistError.sliceGap("coverage \(expected)/\(engine.nLayers) layers: the route must cover 0...\(engine.nLayers - 1) contiguously (missing from \(expected) onward)")
        }
        if config.forward {
            let l = DistReturnListener()
            try l.start(port: config.returnPort)
            returnListener = l
            returnIter = l.results.makeAsyncIterator()
            onLog("return listener on :\(config.returnPort)\n")
        }
        onLog("route complete: \(engine.nLayers) layers on \(entries.count) workers"
              + (config.forward ? " · worker-to-worker forwarding" : " · relay") + "\n")
    }

    /// Setup di un peer con RETRY: un disturbo di rete a metà trasferimento
    /// (o del load) non affossa la route al primo colpo — si riconnette e
    /// riprova fino a 3 volte. Il progresso NON si perde: il worker tiene il
    /// suo `.part`, lo convalida con la catena di checkpoint dell'offer e
    /// chiede di riprendere dall'ultimo blocco buono (v8), quindi ogni
    /// tentativo ritrasmette al massimo 256 MB. Gli errori SEMANTICI
    /// (versione, slice sbagliato, errore riportato dal worker) non si
    /// ritentano: rifarebbero la stessa fine.
    private func setupPeer(_ p: Peer, slice: (start: Int, end: Int), hasOutput: Bool,
                           offer: [DistFileEntry], modelName: String,
                           onLog: @escaping @Sendable (String) -> Void) async throws
        -> (DistConnection, DistRouteEntry) {
        let maxAttempts = 3
        var lastError: Error = DistError.badFrame
        for attempt in 1...maxAttempts {
            do {
                return try await setupPeerOnce(p, slice: slice, hasOutput: hasOutput,
                                               offer: offer, modelName: modelName, onLog: onLog)
            } catch let e as DistError {
                switch e {
                case .versionMismatch, .sliceGap, .remote: throw e   // semantico: non ritentare
                default: lastError = e
                }
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                lastError = error                                    // trasporto: ritentabile
            }
            guard attempt < maxAttempts else { break }
            onLog("peer \(p.host):\(p.port): tentativo \(attempt)/\(maxAttempts) fallito (\(lastError)) "
                  + "— riprovo tra 2s dall'ultimo checkpoint trasferito…\n")
            try await Task.sleep(nanoseconds: 2_000_000_000)
        }
        onLog("peer \(p.host):\(p.port): setup fallito dopo \(maxAttempts) tentativi\n")
        throw lastError
    }

    /// UN tentativo di setup: connessione → HELLO (versione) → offerta e
    /// trasferimento dei file mancanti (dagli offset di resume dichiarati dal
    /// worker) → ASSIGN dello slice → attesa READY. Non tocca lo stato
    /// condiviso (conns/entries: li aggiorna il chiamante, in ordine di
    /// route) — pensato per girare in PARALLELO su tutti i peer. In errore la
    /// connessione viene chiusa prima di rilanciare.
    private func setupPeerOnce(_ p: Peer, slice: (start: Int, end: Int), hasOutput: Bool,
                               offer: [DistFileEntry], modelName: String,
                               onLog: @escaping @Sendable (String) -> Void) async throws
        -> (DistConnection, DistRouteEntry) {
        onLog("connessione a \(p.host):\(p.port)…\n")
        let conn = try await DistConnection.connect(host: p.host, port: p.port,
                                                    queue: queue, onState: onLog)
        // Cancellazione del task group (un altro peer ha esaurito i retry):
        // readFrame/sendFrame NON sono cancellation-aware — senza questo
        // handler un fratello bloccato sull'attesa del READY (minuti di load)
        // terrebbe in ostaggio il gruppo, e se poi RIUSCISSE la sua
        // connessione verrebbe scartata senza mai essere chiusa.
        // conn.cancel() fa completare la receive pendente con errore → il
        // do/catch qui sotto chiude e rilancia.
        return try await withTaskCancellationHandler {
            try await setupPeerBody(p, conn: conn, slice: slice, hasOutput: hasOutput,
                                    offer: offer, modelName: modelName, onLog: onLog)
        } onCancel: {
            conn.cancel()
        }
    }

    /// HELLO (versione) + distribuzione file (offer/need/trasferimento con
    /// ripresa v8): la parte di setup COMUNE a pipeline e verticale.
    func handshakeAndTransfer(_ p: Peer, conn: DistConnection,
                                      offer: [DistFileEntry],
                                      onLog: @escaping @Sendable (String) -> Void) async throws {
        let (type, payload) = try await conn.readFrame()
        if type == .error {
            throw DistError.remote(String(decoding: payload, as: UTF8.self))
        }
        guard type == .hello, let h = DistHello.decode(payload) else { throw DistError.badFrame }
        guard h.version == Dist.protocolVersion else {
            throw DistError.versionMismatch(
                "worker \(p.host):\(p.port) speaks protocol v\(h.version), this build v\(Dist.protocolVersion) — update all nodes to the same DwarfStar build")
        }
        // FILE DISTRIBUTION: offer the manifest; the worker answers with
        // what it is missing (hash-verified locally) and only that is
        // streamed — the huge transfer happens on the FIRST round only.
        try await conn.sendFrame(.fileOffer, DistFileOffer(entries: offer).encoded())
        let (nType, nPayload) = try await readControl(conn)
        guard nType == .fileNeed, let need = DistFileNeed.decode(nPayload) else {
            throw DistError.badFrame
        }
        if need.indices.isEmpty {
            onLog("file: \(p.host):\(p.port) ha già tutto (hash verificati)\n")
        }
        for (j, index) in need.indices.enumerated() {
            guard index >= 0, index < offer.count else { throw DistError.badFrame }
            let resume = j < need.offsets.count ? need.offsets[j] : 0
            try await sendFile(offer[index], index: index, to: conn, peer: p,
                               from: resume, onLog: onLog)
        }
    }

    private func setupPeerBody(_ p: Peer, conn: DistConnection,
                               slice: (start: Int, end: Int), hasOutput: Bool,
                               offer: [DistFileEntry], modelName: String,
                               onLog: @escaping @Sendable (String) -> Void) async throws
        -> (DistConnection, DistRouteEntry) {
        do {
            try await handshakeAndTransfer(p, conn: conn, offer: offer, onLog: onLog)

            // ASSIGN this peer its slice; the worker loads (or reuses) its
            // engine and replies READY. The last slice also runs the head.
            // The usage imatrix seed pre-warms the worker's slot cache with
            // this model's richest local profile (same seeding as local chat).
            let (ls, le) = (slice.start, slice.end)
            let usage = InferenceService.usageDataSeeded(modelName: modelName,
                                                         agentId: "generale") ?? Data()
            // v9: il worker eredita i knob di PERFORMANCE del coordinatore
            // (whitelist Dist.perfKnobKeys). Senza, uno shard con i default di
            // fabbrica girava senza dense stream/mlock/pread: 0.37 tok/s
            // misurati contro i 2.7 dello stesso hardware configurato.
            let env = ProcessInfo.processInfo.environment
            let knobs: [(key: String, value: String)] = Dist.perfKnobKeys.compactMap { k in
                env[k].map { (key: k, value: $0) }
            }
            let assign = DistAssign(modelPath: config.modelPath, modelName: modelName,
                                    contextSize: config.contextSize,
                                    expertCacheSlots: config.workerCacheSlots,
                                    diskKVBudgetTokens: config.diskKVBudgetTokens,
                                    useExpertBundle: config.useExpertBundle,
                                    useDenseQ4: config.useDenseQ4,
                                    layerStart: ls, layerEnd: le, hasOutput: hasOutput,
                                    usageJSON: usage, envKnobs: knobs)
            onLog("assign: \(p.host):\(p.port) -> layers \(ls)...\(le)\(hasOutput ? " +output" : "") · \(modelName) · ctx \(config.contextSize)\n")
            onLog("attendo il caricamento del motore sul worker \(p.host) (minuti alla prima esecuzione: "
                  + "mmap + Metal + eventuali sidecar; progresso nel log del tab Worker)…\n")
            try await conn.sendFrame(.assign, assign.encoded())
            // Await READY, relaying the worker's load-progress frames into
            // THIS log (Q4 requant, sidecar builds and Metal init are minutes
            // of silence otherwise — "stuck" and "working" must be tellable
            // apart from the coordinator alone).
            var rType: Dist.MsgType
            var rPayload: Data
            while true {
                (rType, rPayload) = try await conn.readFrame()
                if rType == .progress {
                    onLog("worker \(p.host): " + String(decoding: rPayload, as: UTF8.self) + "\n")
                    continue
                }
                if rType == .error {
                    throw DistError.remote("\(p.host):\(p.port): " + String(decoding: rPayload, as: UTF8.self))
                }
                break
            }
            guard rType == .ready, let ready = DistHello.decode(rPayload), ready.assigned else {
                throw DistError.badFrame
            }
            guard ready.layerStart == ls, ready.layerEnd == le, ready.hasOutput == hasOutput,
                  ready.contextSize == config.contextSize,
                  ready.nLayers == engine.nLayers else {
                throw DistError.sliceGap(
                    "worker \(p.host):\(p.port) loaded \(ready.nLayers)-layer model, "
                    + "slice \(ready.layerStart)...\(ready.layerEnd); expected \(engine.nLayers) layers, \(ls)...\(le)")
            }
            if ready.modelName != modelName {
                onLog("warning: worker \(p.host) loaded '\(ready.modelName)' != '\(modelName)'\n")
            }
            onLog("route: \(p.host):\(p.port) -> layers \(ls)...\(le)\(hasOutput ? " +output" : "") pronto\n")
            return (conn, DistRouteEntry(host: p.host, port: p.port, layerStart: ls,
                                         layerEnd: le, hasOutput: hasOutput))
        } catch {
            conn.cancel()
            throw error
        }
    }

    public func disconnect() {
        for c in conns { c.cancel() }
        conns = []; entries = []
        for p in expertPeers { p.conn.cancel() }
        expertPeers = []; verticalEngine = nil
        committedIds = []; kvValid = false; lastDiskStoreCount = 0
        returnListener?.stop(); returnListener = nil; returnIter = nil
    }
}
