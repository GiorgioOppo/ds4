import Foundation
import DS4Core

/// The distributed COORDINATOR: owns the embedding, the sampling loop and the
/// conversation. `connect()` establishes a persistent route to the workers,
/// ASSIGNING each one its job (gguf, settings, layer slice); each `send(...)`
/// re-renders the WHOLE conversation and runs it across the cluster, streaming
/// the reply. Re-rendering keeps correctness trivially checkable — and when
/// the render EXACTLY extends the prefix committed by the last clean turn, the
/// prefill covers only the suffix (in-memory reuse), else a disk restore is
/// negotiated across the shards. Any mismatch falls back to a cold prefill.
///
/// Transports: RELAY (default; coordinator round-trips each chunk through every
/// worker) or FORWARDING (`forward:true`; workers pass the HC state worker→worker
/// and the terminal worker replies to this coordinator's return listener).
public final class DistCoordinator: @unchecked Sendable {
    public struct Peer: Sendable {
        public var host: String
        public var port: UInt16
        public init(host: String, port: UInt16) { self.host = host; self.port = port }
    }
    public struct Config: Sendable {
        public var modelPath: String
        public var contextSize: Int
        public var peers: [Peer]
        public var activationBits: Int
        public var prefillChunk: Int
        public var forward: Bool
        public var returnHost: String
        public var returnPort: UInt16
        /// Expert slot-cache budget ASSIGNed to each worker (0 = disabled).
        public var workerCacheSlots: Int
        /// Disk-KV token budget ASSIGNed to each worker's shard (0 = disabled).
        public var diskKVBudgetTokens: Int
        /// Workers should use the expert-bundle sidecar (the coordinator's
        /// bundle is offered for transfer when it exists on disk).
        public var useExpertBundle: Bool
        /// Workers should use the Q4 dense requant; the coordinator's cache
        /// file is offered so no worker pays the minutes-long requant again.
        public var useDenseQ4: Bool
        public init(modelPath: String, contextSize: Int, peers: [Peer], activationBits: Int,
                    prefillChunk: Int = 32, forward: Bool = false,
                    returnHost: String = "", returnPort: UInt16 = 9099,
                    workerCacheSlots: Int = 0, diskKVBudgetTokens: Int = 0,
                    useExpertBundle: Bool = false, useDenseQ4: Bool = false) {
            self.modelPath = modelPath; self.contextSize = contextSize
            self.peers = peers; self.activationBits = activationBits
            self.prefillChunk = max(1, prefillChunk); self.forward = forward
            self.returnHost = returnHost; self.returnPort = returnPort
            self.workerCacheSlots = max(0, workerCacheSlots)
            self.diskKVBudgetTokens = max(0, diskKVBudgetTokens)
            self.useExpertBundle = useExpertBundle
            self.useDenseQ4 = useDenseQ4
        }
    }

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

    private let engine: DistEngine
    private let config: Config
    private let queue = DispatchQueue(label: "ds4.dist.coord")

    // Persistent session state (set by connect, used by send).
    private var conns: [DistConnection] = []
    private var entries: [DistRouteEntry] = []
    private var returnListener: DistReturnListener?
    private var returnIter: AsyncStream<DistResult>.Iterator?
    /// Per-turn session id, echoed by workers in every RESULT. A turn abandoned
    /// mid-chunk (Stop) leaves its reply in a TCP/listener buffer; the next
    /// turn's id differs, so the stale frame is discarded instead of being
    /// mistaken for the new turn's first reply. Sends are serialized by the
    /// caller (one chat turn / benchmark at a time), so a plain counter is enough.
    private var sessionCounter: UInt32 = 0

    /// KV continuity across turns. `committedIds` are the tokens whose KV every
    /// worker shard holds after the last CLEAN turn; `kvValid` mirrors the local
    /// engine's dirty-until-clean rule (false during a turn, after Stop/error,
    /// and around a benchmark — the NSA compressor is recurrent and cannot
    /// rewind, so a partial turn invalidates the whole prefix).
    private var committedIds: [Int] = []
    private var kvValid = false
    /// Tokens covered by the last disk checkpoint (interval gate, like local).
    private var lastDiskStoreCount = 0

    public var routeSummary: String { "\(engine.nLayers) layers · \(entries.count) workers" }

    public init(config: Config) throws {
        self.config = config
        self.engine = try DistEngine(modelPath: config.modelPath, contextSize: config.contextSize,
                                     kvLayers: 0..<0)   // pure coordinator: embed + head only
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

    /// Setup completo di UN peer: connessione → HELLO (versione) → offerta e
    /// trasferimento dei file mancanti → ASSIGN dello slice → attesa READY.
    /// Non tocca lo stato condiviso (conns/entries: li aggiorna il chiamante,
    /// in ordine di route) — pensato per girare in PARALLELO su tutti i peer.
    /// In errore la connessione viene chiusa prima di rilanciare.
    private func setupPeer(_ p: Peer, slice: (start: Int, end: Int), hasOutput: Bool,
                           offer: [DistFileEntry], modelName: String,
                           onLog: @escaping @Sendable (String) -> Void) async throws
        -> (DistConnection, DistRouteEntry) {
        onLog("connessione a \(p.host):\(p.port)…\n")
        let conn = try await DistConnection.connect(host: p.host, port: p.port,
                                                    queue: queue, onState: onLog)
        do {
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
            for index in need.indices {
                guard index >= 0, index < offer.count else { throw DistError.badFrame }
                try await sendFile(offer[index], index: index, to: conn, peer: p, onLog: onLog)
            }

            // ASSIGN this peer its slice; the worker loads (or reuses) its
            // engine and replies READY. The last slice also runs the head.
            // The usage imatrix seed pre-warms the worker's slot cache with
            // this model's richest local profile (same seeding as local chat).
            let (ls, le) = (slice.start, slice.end)
            let usage = InferenceService.usageDataSeeded(modelName: modelName,
                                                         agentId: "generale") ?? Data()
            let assign = DistAssign(modelPath: config.modelPath, modelName: modelName,
                                    contextSize: config.contextSize,
                                    expertCacheSlots: config.workerCacheSlots,
                                    diskKVBudgetTokens: config.diskKVBudgetTokens,
                                    useExpertBundle: config.useExpertBundle,
                                    useDenseQ4: config.useDenseQ4,
                                    layerStart: ls, layerEnd: le, hasOutput: hasOutput,
                                    usageJSON: usage)
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
                  ready.contextSize == config.contextSize else {
                throw DistError.sliceGap(
                    "worker \(p.host):\(p.port) loaded layers \(ready.layerStart)...\(ready.layerEnd) instead of \(ls)...\(le)")
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
        committedIds = []; kvValid = false; lastDiskStoreCount = 0
        returnListener?.stop(); returnListener = nil; returnIter = nil
    }

    // MARK: File distribution (the workers receive everything from here)

    /// The distributable file set: the gguf, plus the expert-bundle sidecar
    /// when enabled and present (next to the gguf, else in DS4_BUNDLE_DIR).
    /// Hashes come from the persistent cache; the first run streams the file
    /// once to compute them (logged — it can take minutes on a 100+ GB gguf).
    private func buildFileOffer(onLog: @Sendable (String) -> Void) throws -> [DistFileEntry] {
        let ggufName = (config.modelPath as NSString).lastPathComponent
        guard let (size, _) = DistFileHash.stat(config.modelPath),
              let sha = DistFileHash.cachedOrCompute(path: config.modelPath, onLog: onLog) else {
            throw DistError.sliceGap("cannot read/hash the gguf at \(config.modelPath)")
        }
        var entries = [DistFileEntry(kind: .gguf, name: ggufName, size: size, sha256: sha)]
        if config.useExpertBundle,
           let bundle = findLocalPath(kind: .expertBundle),
           let (bSize, _) = DistFileHash.stat(bundle),
           let bSha = DistFileHash.cachedOrCompute(path: bundle, onLog: onLog) {
            entries.append(DistFileEntry(kind: .expertBundle, name: ggufName + ".expbundle",
                                         size: bSize, sha256: bSha))
        }
        // The Q4 requant cache is derived and deterministic: ~1.4 GB on the
        // wire beats minutes of re-requant on every worker.
        if config.useDenseQ4,
           let q4 = findLocalPath(kind: .q4Dense),
           let (qSize, _) = DistFileHash.stat(q4),
           let qSha = DistFileHash.cachedOrCompute(path: q4, onLog: onLog) {
            entries.append(DistFileEntry(kind: .q4Dense, name: ggufName + ".q4dense",
                                         size: qSize, sha256: qSha))
        }
        return entries
    }

    /// Stream one offered file to a worker: sequential 4 MB chunks (F_NOCACHE
    /// read — the transfer must not evict the coordinator's hot page cache),
    /// then DONE, then the worker's hash-verified ack.
    private func sendFile(_ entry: DistFileEntry, index: Int, to conn: DistConnection,
                          peer: Peer, onLog: @Sendable (String) -> Void) async throws {
        guard let path = entry.kind == .gguf ? config.modelPath : findLocalPath(kind: entry.kind) else {
            throw DistError.sliceGap("offered file \(entry.name) no longer found locally")
        }
        onLog("file: invio \(entry.name) (\(entry.size / 1_048_576) MB) a \(peer.host):\(peer.port)…\n")
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw DistError.sliceGap("cannot open \(path)") }
        defer { close(fd) }
        _ = fcntl(fd, F_NOCACHE, 1)

        var offset: UInt64 = 0
        var buf = [UInt8](repeating: 0, count: Dist.fileChunkBytes)
        let t0 = Date()
        var lastLogged: UInt64 = 0
        while offset < entry.size {
            try Task.checkCancellation()
            let n = buf.withUnsafeMutableBytes { read(fd, $0.baseAddress, Dist.fileChunkBytes) }
            guard n > 0 else { throw DistError.sliceGap("read failed at \(offset) of \(path)") }
            let chunk = DistFileChunk(index: index, offset: offset, data: Data(buf[0..<n]))
            try await conn.sendFrame(.fileChunk, chunk.encoded())
            offset += UInt64(n)
            if offset - lastLogged >= 1_073_741_824 {          // progress every GB
                lastLogged = offset
                let mbps = Double(offset) / 1_048_576 / max(0.001, Date().timeIntervalSince(t0))
                onLog(String(format: "file: %@ %.0f%% (%.0f MB/s)\n", entry.name,
                             Double(offset) / Double(entry.size) * 100, mbps))
            }
        }
        try await conn.sendFrame(.fileDone, DistFileDone(index: index).encoded())
        let (aType, aPayload) = try await readControl(conn)
        guard aType == .fileAck, let ack = DistKV.decodeAck(aPayload) else { throw DistError.badFrame }
        guard ack.ok else { throw DistError.remote("\(peer.host):\(peer.port): \(ack.message)") }
        onLog(String(format: "file: %@ trasferito in %.0fs\n", entry.name, Date().timeIntervalSince(t0)))
    }

    /// Where a derived file lives on the coordinator: next to the gguf, else
    /// in the app-owned cache directory (same lookup order as the engine).
    private func findLocalPath(kind: DistFileEntry.Kind) -> String? {
        let name = (config.modelPath as NSString).lastPathComponent
        let (ext, dirEnv): (String, String)
        switch kind {
        case .expertBundle: (ext, dirEnv) = (".expbundle", "DS4_BUNDLE_DIR")
        case .q4Dense:      (ext, dirEnv) = (".q4dense", "DS4_Q4_CACHE_DIR")
        case .gguf:         return config.modelPath
        }
        var candidates = [config.modelPath + ext]
        if let dir = ProcessInfo.processInfo.environment[dirEnv], !dir.isEmpty {
            candidates.append(dir + "/" + name + ext)
        }
        return candidates.first { FileManager.default.fileExists(atPath: $0) }
    }

    // MARK: Cluster KV continuity

    /// Read a KV-control reply on a relay connection, skipping stale RESULT
    /// frames an abandoned turn may have left in the socket buffer.
    private func readControl(_ conn: DistConnection) async throws -> (Dist.MsgType, Data) {
        while true {
            let (type, payload) = try await conn.readFrame()
            if type == .result { continue }                       // stale turn reply: drop
            if type == .progress { continue }                     // informational only
            if type == .error { throw DistError.remote(String(decoding: payload, as: UTF8.self)) }
            return (type, payload)
        }
    }

    /// Negotiate a disk restore for `ids`: intersect every worker's stored
    /// prefix lengths, pick the longest ALL shards can restore, and restore it
    /// everywhere. Returns the restored length, or nil (cold prefill — any
    /// partially restored shard is overwritten by the pos-0 prefill).
    private func negotiateRestore(ids: [Int], onLog: @Sendable (String) -> Void) async -> Int? {
        do {
            var common: Set<Int>?
            for conn in conns {
                try await conn.sendFrame(.kvQuery, DistKV.encodeTokens(ids))
                let (type, payload) = try await readControl(conn)
                guard type == .kvLengths, let lengths = DistKV.decodeLengths(payload) else { return nil }
                common = common.map { $0.intersection(lengths) } ?? Set(lengths)
                if common?.isEmpty == true { return nil }
            }
            guard let best = common?.max(), best > 0 else { return nil }
            let prefix = Array(ids.prefix(best))
            for (i, conn) in conns.enumerated() {
                try await conn.sendFrame(.kvRestore, DistKV.encodeTokens(prefix))
                let (type, payload) = try await readControl(conn)
                guard type == .kvAck, let ack = DistKV.decodeAck(payload), ack.ok else {
                    onLog("restore KV fallito su \(entries[i].host):\(entries[i].port) — prefill da zero\n")
                    return nil
                }
            }
            onLog("KV ripristinato da disco su \(conns.count) worker (\(best) token)\n")
            return best
        } catch {
            onLog("negoziazione restore KV fallita: \(error) — prefill da zero\n")
            return nil
        }
    }

    /// Checkpoint the committed prefix on every worker (interval-gated by the
    /// caller). The workers export synchronously and write in the background.
    private func broadcastSave(ids: [Int], cold: Bool, onLog: @Sendable (String) -> Void) async {
        do {
            for conn in conns {
                try await conn.sendFrame(.kvSave, DistKV.encodeSave(tokens: ids, cold: cold))
                let (type, payload) = try await readControl(conn)
                guard type == .kvAck, let ack = DistKV.decodeAck(payload), ack.ok else {
                    onLog("checkpoint KV rifiutato da un worker" +
                          ((DistKV.decodeAck(payload)?.message).map { ": \($0)" } ?? "") + "\n")
                    return
                }
            }
            lastDiskStoreCount = ids.count
            onLog("checkpoint KV cluster: \(ids.count) token su \(conns.count) worker\n")
        } catch {
            onLog("checkpoint KV fallito: \(error)\n")
        }
    }

    // MARK: One chat turn

    /// Render the whole conversation (with the agent's tools declared), run it
    /// across the cluster (fresh KV), and stream the assistant reply: reasoning
    /// via `onReasoning`, visible text via `onToken`. Returns the parsed DSML
    /// tool calls (empty when the model answered directly) — the caller executes
    /// them locally and continues with `.toolResult` turns.
    @discardableResult
    public func send(turns: [ChatTurn], tools: [ToolSpec] = [], think: Bool, maxTokens: Int,
                     sampling: SamplingParams,
                     onLog: @Sendable (String) -> Void,
                     onProgress: @Sendable (String) -> Void = { _ in },
                     onReasoning: @Sendable (String) -> Void,
                     onToken: @Sendable (String) -> Void) async throws -> [ToolCall] {
        guard !entries.isEmpty else { throw DistError.closed }
        sessionCounter &+= 1
        let session = sessionCounter
        let ids = engine.chatPromptIds(turns: turns, tools: tools, think: think)
        guard ids.count < config.contextSize else { throw DistError.sliceGap("prompt exceeds context") }

        // KV continuity: when the re-rendered conversation EXACTLY extends the
        // prefix committed by the last clean turn, prefill only the suffix
        // (the workers still hold that KV). Otherwise try a disk restore
        // negotiated across all shards. Exact-match checks make both paths
        // opportunistic and safe: any mismatch falls back to a cold prefill.
        var startPos = 0
        if kvValid, !committedIds.isEmpty, ids.count > committedIds.count,
           ids.starts(with: committedIds) {
            startPos = committedIds.count
            onLog("KV riusato in memoria: \(startPos) token già nel cluster\n")
        } else if config.diskKVBudgetTokens > 0 {
            if let restored = await negotiateRestore(ids: ids, onLog: onLog) { startPos = restored }
        }
        kvValid = false                            // dirty-until-clean (like local)
        onLog("prefill \(ids.count - startPos) di \(ids.count) tokens (chunk \(config.prefillChunk))...\n")

        // PREFILL the (remaining) prompt in chunks. The FIRST chunk carries
        // turnStart (session adoption on the workers); posBase 0 also resets
        // their compressor state — a mid-context start keeps/restored state.
        var pos = startPos
        var lastLogits: [Float] = []
        var start = startPos
        var firstChunk = true
        while start < ids.count {
            try Task.checkCancellation()
            let end = min(start + config.prefillChunk, ids.count)
            var hcs: [[Float]] = []
            for (i, id) in ids[start..<end].enumerated() {
                hcs.append(try engine.embed(token: id, pos: pos + i))
            }
            if pos == 0, let h0 = hcs.first {
                let n = (h0.reduce(0) { $0 + $1 * $1 }).squareRoot()
                onLog(String(format: "diag: |embed| = %.2f (hc=%d float)\n", n, h0.count))
            }
            if let logits = try await runChunk(session: session, hcs: hcs, posBase: pos,
                                               tokens: Array(ids[start..<end]),
                                               wantLogits: end == ids.count, turnStart: firstChunk) {
                lastLogits = logits
            }
            firstChunk = false
            pos += end - start
            start = end
            onProgress("prefill \(pos)/\(ids.count) tokens...")
        }
        guard !lastLogits.isEmpty else { throw DistError.badFrame }
        // Diagnose where the pipeline breaks: a sane top token here = prefill OK,
        // problem in decode; garbage here = problem in embed/slice/head/wire.
        if let mx = lastLogits.indices.max(by: { lastLogits[$0] < lastLogits[$1] }) {
            let finite = lastLogits.filter { $0.isFinite }.count
            onLog(String(format: "diag: prefill top=%d (%@) logit=%.2f · %d/%d finite\n",
                         mx, engine.tokenText(mx), lastLogits[mx], finite, lastLogits.count))
        }

        // DECODE token-by-token, splitting reasoning (<think>…</think>) from text
        // and buffering DSML tool-call markup (never shown; parsed at the end —
        // same scheme as the local InferenceService, incl. the held '<' opener).
        var rng = sampling.seed
        var produced = 0
        var producedIds: [Int] = []              // fed tokens (KV committed on the workers)
        var inReasoning = think
        var inTool = false
        var pendingLT = false                    // a held trailing '<' (may open <｜DSML｜…)
        var visible = ""
        var toolText = ""
        var recentIds = Array(ids.suffix(sampling.repeatLastN))
        let t0 = Date()
        func emit(_ s: String) {
            if inReasoning { onReasoning(s) } else { visible += s; onToken(s) }
        }
        while produced < maxTokens {
            try Task.checkCancellation()
            let next = engine.sample(lastLogits, params: sampling, recent: recentIds[...], rng: &rng)
            if next == engine.eosId { break }
            if !inTool, next == engine.dsmlId {
                if pendingLT { pendingLT = false; toolText += "<" }   // the '<' belonged to the opener
                inTool = true
                toolText += engine.tokenText(next)
            } else if inTool {
                toolText += engine.tokenText(next)
            } else if next == engine.thinkEndId { inReasoning = false }
            else if next == engine.thinkStartId { inReasoning = true }
            else {
                if pendingLT { pendingLT = false; emit("<") }         // plain '<' after all
                let s = engine.tokenText(next)
                if s.hasSuffix("<") {
                    let head = String(s.dropLast())
                    if !head.isEmpty { emit(head) }
                    pendingLT = true
                } else {
                    emit(s)
                }
            }
            recentIds.append(next)
            if recentIds.count > sampling.repeatLastN { recentIds.removeFirst() }
            let hc = try engine.embed(token: next, pos: pos)
            guard let logits = try await runChunk(session: session, hcs: [hc], posBase: pos,
                                                  tokens: [next], wantLogits: true) else {
                throw DistError.badFrame
            }
            lastLogits = logits
            producedIds.append(next)             // its KV is now committed cluster-wide
            pos += 1; produced += 1
            let elapsed = Date().timeIntervalSince(t0)
            onProgress(String(format: "%d tokens · %.2f tok/s", produced,
                              elapsed > 0 ? Double(produced) / elapsed : 0))
        }
        if pendingLT, !inTool { emit("<") }
        let dt = Date().timeIntervalSince(t0)
        onLog("[\(produced) tokens · \(String(format: "%.2f", dt > 0 ? Double(produced) / dt : 0)) tok/s]\n")

        // CLEAN completion: the cluster KV now holds prompt + fed tokens — the
        // next turn can extend it in memory; checkpoint to disk interval-gated.
        committedIds = ids + producedIds
        kvValid = true
        if config.diskKVBudgetTokens > 0,
           committedIds.count >= 128,
           committedIds.count - lastDiskStoreCount >= 256 {
            await broadcastSave(ids: committedIds, cold: lastDiskStoreCount == 0, onLog: onLog)
        }

        guard inTool else { return engine.parseToolCalls(visible).calls }
        return engine.parseToolCalls(visible + toolText).calls
    }

    // MARK: Benchmark

    /// Distributed benchmark: prefill a synthetic prompt of `contextTokens` tokens
    /// across the already-connected cluster and decode `genTokens` from it, returning
    /// prefill / generation throughput at that context frontier. Reuses the live
    /// route (no reconnect) and runs with a fresh cluster KV (posBase 0 resets the
    /// workers), so it must NOT overlap a chat `send`. Mirrors
    /// `InferenceService.benchmark` so local and distributed numbers are comparable.
    public func benchmark(contextTokens: Int, genTokens: Int) async throws -> InferenceService.BenchPoint {
        guard !entries.isEmpty else { throw DistError.closed }
        sessionCounter &+= 1
        let session = sessionCounter
        kvValid = false          // the run rewrites the cluster KV from pos 0
        let ctx = max(8, min(contextTokens, config.contextSize - genTokens - 4))
        // Synthetic prompt: BOS + tiled filler. Output quality is irrelevant for
        // timing; the per-token work (embed · slice forward · expert gather) is the same.
        var ids: [Int] = [engine.bosId]
        let filler = engine.tokenize("The quick brown fox jumps over the lazy dog. ")
        let pad = filler.isEmpty ? [engine.eosId] : filler
        var i = 0
        while ids.count < ctx { ids.append(pad[i % pad.count]); i += 1 }
        ids = Array(ids.prefix(ctx))

        // PREFILL the whole prompt in chunks (posBase 0 resets the workers' KV).
        let t0 = Date()
        var pos = 0
        var lastLogits: [Float] = []
        var start = 0
        var firstChunk = true
        while start < ids.count {
            try Task.checkCancellation()
            let end = min(start + config.prefillChunk, ids.count)
            var hcs: [[Float]] = []
            for (k, id) in ids[start..<end].enumerated() {
                hcs.append(try engine.embed(token: id, pos: pos + k))
            }
            if let logits = try await runChunk(session: session, hcs: hcs, posBase: pos,
                                               tokens: Array(ids[start..<end]),
                                               wantLogits: end == ids.count, turnStart: firstChunk) {
                lastLogits = logits
            }
            firstChunk = false
            pos += end - start
            start = end
        }
        let prefillDt = Date().timeIntervalSince(t0)
        guard !lastLogits.isEmpty else { throw DistError.badFrame }

        // DECODE genTokens token-by-token (content discarded; only timing matters).
        var rng: UInt64 = 0xD54
        let samp = SamplingParams()
        var produced = 0
        let g0 = Date()
        while produced < genTokens {
            try Task.checkCancellation()
            let next = engine.sample(lastLogits, params: samp, rng: &rng)
            let hc = try engine.embed(token: next, pos: pos)
            guard let logits = try await runChunk(session: session, hcs: [hc], posBase: pos,
                                                  tokens: [next], wantLogits: true) else {
                throw DistError.badFrame
            }
            lastLogits = logits
            pos += 1; produced += 1
        }
        let genDt = Date().timeIntervalSince(g0)
        let kv = UInt64(engine.nLayers) * UInt64(ctx) * UInt64(engine.headDim) * 4
        return InferenceService.BenchPoint(
            contextTokens: ctx,
            prefillTps: prefillDt > 0 ? Double(ctx) / prefillDt : 0,
            genTps: genDt > 0 && produced > 0 ? Double(produced) / genDt : 0,
            kvBytes: kv)
    }

    /// One chunk through the pipeline; returns the last token's logits if `wantLogits`.
    /// Results are matched on the `session` echo: anything from an older session
    /// (a turn abandoned mid-chunk) is discarded and the read repeats. ERROR
    /// frames from workers surface with the worker's own message.
    private func runChunk(session: UInt32, hcs: [[Float]], posBase: Int, tokens: [Int],
                          wantLogits: Bool, turnStart: Bool = false) async throws -> [Float]? {
        var flags: Dist.WorkFlags = []
        if posBase == 0 { flags.insert(.resetSession) }
        if turnStart { flags.insert(.turnStart) }
        if config.forward {
            var f = flags
            if wantLogits { f.insert(.outputLogits) }
            let work = DistWork(session: session, pos: posBase, nTokens: hcs.count,
                                layerStart: entries[0].layerStart, layerEnd: entries[0].layerEnd,
                                flags: f, hcBits: config.activationBits, route: entries, routeIndex: 0,
                                returnHost: config.returnHost, returnPort: config.returnPort,
                                hc: hcs.flatMap { $0 }, tokenIds: tokens.map(Int32.init))
            try await conns[0].sendFrame(.work, work.encoded())
            while true {
                guard let res = await returnIter?.next() else { throw DistError.closed }
                guard res.session == session else { continue }   // stale turn: drop
                return res.kind == .logits ? res.values : nil
            }
        }
        var states = hcs
        let stateLen = engine.hcStateCount
        for (i, e) in entries.enumerated() {
            var f = flags
            if i == entries.count - 1, wantLogits { f.insert(.outputLogits) }
            let work = DistWork(session: session, pos: posBase, nTokens: states.count,
                                layerStart: e.layerStart, layerEnd: e.layerEnd,
                                flags: f, hcBits: config.activationBits,
                                hc: states.flatMap { $0 }, tokenIds: tokens.map(Int32.init))
            try await conns[i].sendFrame(.work, work.encoded())
            var res: DistResult
            while true {
                let (type, payload) = try await conns[i].readFrame()
                if type == .progress { continue }                // informational only
                if type == .error { throw DistError.remote(String(decoding: payload, as: UTF8.self)) }
                guard type == .result, let r = DistResult.decode(payload) else { throw DistError.badFrame }
                guard r.session == session else { continue }     // stale turn: drop
                res = r
                break
            }
            if res.kind == .logits { return res.values }
            // The worker echoes exactly nTokens states; anything else is a bug
            // upstream — fail the turn, never slice a short array.
            guard res.values.count == states.count * stateLen else { throw DistError.badFrame }
            states = (0..<states.count).map { Array(res.values[$0 * stateLen..<($0 + 1) * stateLen]) }
        }
        return nil
    }
}
