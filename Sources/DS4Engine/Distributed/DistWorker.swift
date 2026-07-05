import Foundation
@preconcurrency import Network

/// A distributed WORKER: starts IDLE (listening, no model loaded) and lets the
/// COORDINATOR define its whole job via ASSIGN — which gguf, context size,
/// expert-cache budget, and which contiguous layer slice to own. On ASSIGN the
/// worker resolves the gguf locally, loads (or reuses) its slice engine and
/// replies READY; then it serves WORK frames (run the slice, or the output
/// head when flagged) and answers with RESULT frames.
public final class DistWorker: @unchecked Sendable {
    public struct Config: Sendable {
        public var port: UInt16
        /// Local gguf hint: the worker tries the coordinator's path verbatim
        /// first, then a file with the assigned NAME next to this hint, then
        /// the hint itself when the filename matches (per-Mac disk layouts).
        public var localModelPath: String
        public init(port: UInt16, localModelPath: String) {
            self.port = port; self.localModelPath = localModelPath
        }
    }

    private let config: Config
    private let onLog: @Sendable (String) -> Void
    private let queue = DispatchQueue(label: "ds4.dist.worker")
    private let gate = DistGate()
    private var listener: NWListener?

    /// The coordinator-defined job this worker currently serves.
    struct Assignment: Equatable, Sendable {
        var resolvedModelPath: String
        var contextSize: Int
        var expertCacheSlots: Int
        var layerStart: Int
        var layerEnd: Int
        var hasOutput: Bool
    }

    /// Engine + assignment, set by ASSIGN. Guarded by `stateLock`: WORK frames
    /// can arrive on a different connection (forwarding) than the assigning one.
    private let stateLock = NSLock()
    private var engine: DistEngine?
    private var assignment: Assignment?
    private var loadingAssignment = false
    /// Where this shard persists its usage imatrix (slice-keyed: counts are
    /// collected only for the owned layers). nil until assigned.
    private var usageFile: URL?

    /// App Support home for worker-side per-shard state (usage + disk KV),
    /// keyed by model AND slice: a checkpoint holds only the shard's layers,
    /// so a changed slice must never see another slice's entries.
    static func shardStateDirectory(modelName: String, layerStart: Int, layerEnd: Int) -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DwarfStar/dist-worker/\(modelName)-\(layerStart)-\(layerEnd)",
                                    isDirectory: true)
    }

    /// Persist the usage collected so far (cheap JSON; called between turns).
    private func persistUsage() {
        stateLock.lock()
        let file = usageFile
        let data = engine?.usageData()
        stateLock.unlock()
        guard let file, let data else { return }
        try? FileManager.default.createDirectory(at: file.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        try? data.write(to: file)
    }

    /// One TURN at a time, enforced at the session level (NOT per connection:
    /// in forwarding mode a worker legitimately holds one connection from the
    /// coordinator AND one from the previous worker). A pos==0 chunk ADOPTS its
    /// session id as current; any chunk from a different session is refused
    /// with an ERROR frame — a competing coordinator fails loudly instead of
    /// silently resetting the active turn's KV shard.
    private let sessionLock = NSLock()
    private var currentSession: UInt32?

    private func admit(_ work: DistWork) -> Bool {
        sessionLock.lock(); defer { sessionLock.unlock() }
        // turnStart (not pos==0): with KV reuse/restore a turn may begin
        // mid-context, and it is still the legitimate start of a new turn.
        if work.flags.contains(.turnStart) { currentSession = work.session; return true }
        return currentSession == work.session
    }

    public init(config: Config, onLog: @escaping @Sendable (String) -> Void) {
        self.config = config
        self.onLog = onLog
    }

    public func start() throws {
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        guard let port = NWEndpoint.Port(rawValue: config.port) else { throw DistError.badPort }
        let l = try NWListener(using: params, on: port)
        l.stateUpdateHandler = { [onLog, config] state in
            switch state {
            case .ready: onLog("worker in ascolto su :\(config.port) — in attesa dell'assegnazione dal coordinatore\n")
            case .failed(let e): onLog("worker listener failed: \(e)\n")
            default: break
            }
        }
        l.newConnectionHandler = { [weak self] c in self?.accept(c) }
        l.start(queue: queue)
        listener = l
    }

    public func stop() {
        persistUsage()               // keep what this shard learned across sessions
        listener?.cancel(); listener = nil
        stateLock.lock(); engine = nil; assignment = nil; usageFile = nil; stateLock.unlock()
    }

    /// Resolve the gguf named by an ASSIGN to a LOCAL file: the coordinator's
    /// path verbatim (shared disk layouts), else a file with the same name in
    /// the local hint's directory, else the hint itself when the name matches.
    static func resolveModelPath(requestedPath: String, modelName: String,
                                 localHint: String,
                                 exists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) })
        -> String? {
        if !requestedPath.isEmpty, exists(requestedPath) { return requestedPath }
        if !modelName.isEmpty {
            let sibling = ((localHint as NSString).deletingLastPathComponent as NSString)
                .appendingPathComponent(modelName)
            if exists(sibling) { return sibling }
            if (localHint as NSString).lastPathComponent == modelName, exists(localHint) {
                return localHint
            }
        }
        return nil
    }

    private func accept(_ c: NWConnection) {
        c.start(queue: queue)
        let conn = DistConnection(c)
        Task { [weak self] in await self?.serve(conn) }
    }

    /// Snapshot for HELLO/READY: the active assignment, or the idle state.
    private func helloPayload() -> DistHello {
        stateLock.lock(); defer { stateLock.unlock() }
        if let engine, let a = assignment {
            return DistHello(modelName: engine.modelName, layerStart: a.layerStart,
                             layerEnd: a.layerEnd, hasOutput: a.hasOutput,
                             nLayers: engine.nLayers, contextSize: a.contextSize)
        }
        return .idle(localModelName: (config.localModelPath as NSString).lastPathComponent,
                     nLayers: DistEngine.modelLayers)
    }

    /// Current (engine, assignment) if ready to serve WORK; nil while idle/loading.
    private func activeEngine() -> (engine: DistEngine, assignment: Assignment)? {
        stateLock.lock(); defer { stateLock.unlock() }
        guard let engine, let assignment, !loadingAssignment else { return nil }
        return (engine, assignment)
    }

    private func serve(_ conn: DistConnection) async {
        onLog("connessione in ingresso\n")
        // Outbound connections (next-hop worker / coordinator return), per session.
        var downstream: [String: DistConnection] = [:]
        defer { for c in downstream.values { c.cancel() } }

        // `expectHello`: next-hop workers greet new connections with a HELLO frame
        // (consume it once); the coordinator's return listener does not.
        func outbound(_ host: String, _ port: UInt16, expectHello: Bool) async throws -> DistConnection {
            let key = "\(host):\(port)"
            if let c = downstream[key] { return c }
            let c = try DistConnection.connect(host: host, port: port, queue: queue)
            if expectHello { _ = try await c.readFrame() }
            downstream[key] = c
            return c
        }

        do {
            try await conn.sendFrame(.hello, helloPayload().encoded())

            while true {
                let (type, payload) = try await conn.readFrame()
                if type == .assign {
                    try await handleAssign(payload, on: conn)
                    continue
                }
                if type == .kvQuery || type == .kvRestore || type == .kvSave {
                    try await handleKV(type, payload, on: conn)
                    continue
                }
                guard type == .work, let work = DistWork.decode(payload) else { continue }

                guard let (engine, active) = activeEngine() else {
                    let msg = "worker not ready: no assignment loaded (send ASSIGN first)"
                    onLog(msg + "\n")
                    try await conn.sendFrame(.error, Data(msg.utf8))
                    continue
                }
                // Validate the chunk BEFORE touching the engine: sizes come from
                // the network, and a mismatch would otherwise crash the process
                // (out-of-bounds slicing) or touch KV the shard never allocated.
                let stateLen = engine.hcStateCount
                let n = work.nTokens
                guard n >= 1, work.hc.count == n * stateLen,
                      work.layerStart == active.layerStart, work.layerEnd == active.layerEnd,
                      work.pos >= 0, work.pos + n <= active.contextSize else {
                    let msg = "invalid WORK frame: nTokens=\(work.nTokens) hc=\(work.hc.count) "
                        + "(state \(stateLen)) layers \(work.layerStart)...\(work.layerEnd) "
                        + "(assigned \(active.layerStart)...\(active.layerEnd)) pos \(work.pos)"
                    onLog(msg + "\n")
                    try await conn.sendFrame(.error, Data(msg.utf8))
                    continue
                }
                guard admit(work) else {
                    let msg = "refused WORK for session \(work.session): another turn is active on this worker"
                    onLog(msg + "\n")
                    try await conn.sendFrame(.error, Data(msg.utf8))
                    continue
                }
                // A new turn boundary: persist the usage the PREVIOUS turn
                // accumulated (cheap JSON, same cadence as the local engine).
                if work.flags.contains(.turnStart) { persistUsage() }

                // Serialize compute: one chunk at a time against the shard.
                // The chunk's hc holds nTokens states; split, run, re-concat.
                let outStates: [[Float]] = try await gate.run {
                    var hcs: [[Float]] = []
                    hcs.reserveCapacity(n)
                    for i in 0..<n { hcs.append(Array(work.hc[i*stateLen..<(i+1)*stateLen])) }
                    return try engine.forwardSliceBatch(hcs: hcs, posBase: work.pos,
                                                        start: work.layerStart, end: work.layerEnd)
                }
                if work.pos == 0, let outHC = outStates.first {
                    func nrm(_ a: [Float]) -> Float { (a.reduce(0) { $0 + $1 * $1 }).squareRoot() }
                    let inHC = Array(work.hc[0..<stateLen])
                    onLog(String(format: "diag: layer %d…%d  |in|=%.2f  |out|=%.2f\n",
                                 work.layerStart, work.layerEnd, nrm(inHC), nrm(outHC)))
                }

                let isTerminal = work.route.isEmpty || work.routeIndex >= work.route.count - 1
                if isTerminal {
                    // Terminal hop: produce logits for the chunk's LAST token if asked,
                    // else hidden states (relay) / a bare ack (forwarding flow control).
                    // Every result ECHOES the work's session id (stale-reply guard).
                    let result: DistResult
                    if work.flags.contains(.outputLogits) {
                        result = DistResult(session: work.session, kind: .logits, bits: 32,
                                            values: try engine.head(hc: outStates[n-1]))
                    } else if work.route.isEmpty {
                        result = DistResult(session: work.session, kind: .hidden, bits: work.hcBits,
                                            values: outStates.flatMap { $0 })
                    } else {
                        result = DistResult(session: work.session, kind: .ack, bits: 32, values: [])
                    }
                    if work.route.isEmpty {
                        try await conn.sendFrame(.result, result.encoded())     // relay: reply upstream
                    } else {
                        let back = try await outbound(work.returnHost, work.returnPort, expectHello: false)
                        try await back.sendFrame(.result, result.encoded())     // forwarding: reply to coordinator
                    }
                } else {
                    // Forward the chunk to the next hop in the route.
                    let nextIdx = work.routeIndex + 1
                    let next = work.route[nextIdx]
                    let fwd = DistWork(session: work.session, pos: work.pos, nTokens: n,
                                       layerStart: next.layerStart, layerEnd: next.layerEnd,
                                       flags: work.flags, hcBits: work.hcBits,
                                       route: work.route, routeIndex: nextIdx,
                                       returnHost: work.returnHost, returnPort: work.returnPort,
                                       hc: outStates.flatMap { $0 })
                    let c = try await outbound(next.host, next.port, expectHello: true)
                    try await c.sendFrame(.work, fwd.encoded())
                }
            }
        } catch {
            onLog("sessione chiusa: \(error)\n")
            conn.cancel()
        }
    }

    /// Handle an ASSIGN: validate, resolve the gguf locally, load (or reuse)
    /// the slice engine, reply READY — or an ERROR frame with the reason.
    private func handleAssign(_ payload: Data, on conn: DistConnection) async throws {
        guard let assign = DistAssign.decode(payload) else {
            try await conn.sendFrame(.error, Data("malformed ASSIGN frame".utf8))
            return
        }
        guard assign.layerStart >= 0, assign.layerStart <= assign.layerEnd,
              assign.layerEnd < DistEngine.modelLayers, assign.contextSize > 0 else {
            let msg = "invalid ASSIGN: layers \(assign.layerStart)...\(assign.layerEnd) "
                + "(model has \(DistEngine.modelLayers)), context \(assign.contextSize)"
            onLog(msg + "\n")
            try await conn.sendFrame(.error, Data(msg.utf8))
            return
        }
        guard let resolved = Self.resolveModelPath(requestedPath: assign.modelPath,
                                                   modelName: assign.modelName,
                                                   localHint: config.localModelPath) else {
            let msg = "gguf '\(assign.modelName)' not found on this worker "
                + "(tried the coordinator's path and next to '\(config.localModelPath)')"
            onLog(msg + "\n")
            try await conn.sendFrame(.error, Data(msg.utf8))
            return
        }
        let wanted = Assignment(resolvedModelPath: resolved, contextSize: assign.contextSize,
                                expertCacheSlots: assign.expertCacheSlots,
                                layerStart: assign.layerStart, layerEnd: assign.layerEnd,
                                hasOutput: assign.hasOutput)

        stateLock.lock()
        if assignment == wanted, let current = engine {
            stateLock.unlock()
            applyAncillary(assign, to: current)
            onLog("assegnazione invariata: layer \(wanted.layerStart)...\(wanted.layerEnd) — riuso il motore\n")
            try await conn.sendFrame(.ready, helloPayload().encoded())
            return
        }
        if loadingAssignment {
            stateLock.unlock()
            try await conn.sendFrame(.error, Data("worker busy loading a previous assignment".utf8))
            return
        }
        loadingAssignment = true
        engine = nil                 // free the old shard BEFORE loading the new one
        assignment = nil
        stateLock.unlock()

        onLog("assegnazione: \(assign.modelName) · layer \(wanted.layerStart)...\(wanted.layerEnd)"
              + (wanted.hasOutput ? " +output" : "") + " · ctx \(wanted.contextSize)"
              + (wanted.expertCacheSlots > 0 ? " · \(wanted.expertCacheSlots) slot cache" : "")
              + " — carico il motore…\n")
        let t0 = Date()
        do {
            // Detached: the load (mmap + Metal init) runs for minutes and must
            // not sit on this connection's task while frames could arrive.
            let slots = wanted.expertCacheSlots
            let loaded = try await Task.detached(priority: .userInitiated) {
                try DistEngine(modelPath: resolved, contextSize: wanted.contextSize,
                               expertCacheSlots: slots > 0 ? slots : nil,
                               kvLayers: wanted.layerStart..<(wanted.layerEnd + 1))
            }.value
            stateLock.lock()
            engine = loaded
            assignment = wanted
            loadingAssignment = false
            stateLock.unlock()
            applyAncillary(assign, to: loaded)
            onLog(String(format: "motore pronto in %.1fs\n", Date().timeIntervalSince(t0)))
            try await conn.sendFrame(.ready, helloPayload().encoded())
        } catch {
            stateLock.lock(); loadingAssignment = false; stateLock.unlock()
            let msg = "engine load failed: \(error)"
            onLog(msg + "\n")
            try await conn.sendFrame(.error, Data(msg.utf8))
        }
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

    /// Serve one KV control frame: query stored prefixes, restore an exact
    /// checkpoint, or checkpoint the current shard state.
    private func handleKV(_ type: Dist.MsgType, _ payload: Data, on conn: DistConnection) async throws {
        guard let (engine, _) = activeEngine() else {
            let msg = "worker not ready: no assignment loaded"
            switch type {
            case .kvQuery: try await conn.sendFrame(.kvLengths, DistKV.encodeLengths([]))
            default: try await conn.sendFrame(.kvAck, DistKV.encodeAck(ok: false, message: msg))
            }
            return
        }
        switch type {
        case .kvQuery:
            guard let ids = DistKV.decodeTokens(payload) else {
                try await conn.sendFrame(.kvLengths, DistKV.encodeLengths([]))
                return
            }
            try await conn.sendFrame(.kvLengths,
                                     DistKV.encodeLengths(engine.storedPrefixLengths(of: ids)))
        case .kvRestore:
            guard let ids = DistKV.decodeTokens(payload) else {
                try await conn.sendFrame(.kvAck, DistKV.encodeAck(ok: false, message: "malformed kvRestore"))
                return
            }
            // Through the gate: the restore writes the shard's KV buffers and
            // must not interleave with compute.
            let ok = await gate.run { engine.restoreKV(tokens: ids) }
            if ok { onLog("KV shard ripristinato da disco (\(ids.count) token)\n") }
            try await conn.sendFrame(.kvAck, DistKV.encodeAck(
                ok: ok, message: ok ? "" : "no checkpoint for \(ids.count) tokens"))
        case .kvSave:
            guard let (ids, cold) = DistKV.decodeSave(payload) else {
                try await conn.sendFrame(.kvAck, DistKV.encodeAck(ok: false, message: "malformed kvSave"))
                return
            }
            // Export under the gate (state must hold still); the disk write
            // itself streams in the background (SnapshotBox).
            await gate.run { engine.saveKV(tokens: ids, cold: cold) }
            onLog("KV shard: checkpoint \(ids.count) token avviato\n")
            try await conn.sendFrame(.kvAck, DistKV.encodeAck(ok: true))
        default:
            break
        }
    }
}

/// Serializes async closures (the shard runs one step at a time).
actor DistGate {
    func run<T: Sendable>(_ body: @Sendable () throws -> T) async rethrows -> T { try body() }
}
