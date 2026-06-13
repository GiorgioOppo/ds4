import Foundation
import DS4Core

/// The distributed COORDINATOR: owns the embedding, the sampling loop and the
/// conversation. `connect()` establishes a persistent route to the workers
/// (validated for full contiguous layer coverage); each `send(...)` re-renders
/// the WHOLE conversation and runs it across the cluster, streaming the reply.
/// Re-rendering per turn (stateless) keeps the cluster KV trivially consistent —
/// no cross-turn rewind to coordinate over the network.
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
        public init(modelPath: String, contextSize: Int, peers: [Peer], activationBits: Int,
                    prefillChunk: Int = 32, forward: Bool = false,
                    returnHost: String = "", returnPort: UInt16 = 9099) {
            self.modelPath = modelPath; self.contextSize = contextSize
            self.peers = peers; self.activationBits = activationBits
            self.prefillChunk = max(1, prefillChunk); self.forward = forward
            self.returnHost = returnHost; self.returnPort = returnPort
        }
    }

    private let engine: DistEngine
    private let config: Config
    private let queue = DispatchQueue(label: "ds4.dist.coord")

    // Persistent session state (set by connect, used by send).
    private var conns: [DistConnection] = []
    private var entries: [DistRouteEntry] = []
    private var returnListener: DistReturnListener?
    private var returnIter: AsyncStream<DistResult>.Iterator?

    public var routeSummary: String { "\(engine.nLayers) layer · \(entries.count) worker" }

    public init(config: Config) throws {
        self.config = config
        self.engine = try DistEngine(modelPath: config.modelPath, contextSize: config.contextSize,
                                     kvLayers: 0..<0)   // pure coordinator: embed + head only
    }

    // MARK: Session

    /// Connect to every worker, read their HELLO, assemble + validate a contiguous
    /// route covering all layers, and (forwarding only) start the return listener.
    public func connect(onLog: @Sendable (String) -> Void) async throws {
        for p in config.peers {
            let conn = try DistConnection.connect(host: p.host, port: p.port, queue: queue)
            let (type, payload) = try await conn.readFrame()
            guard type == .hello, let h = DistHello.decode(payload) else { throw DistError.badFrame }
            if h.modelName != engine.modelName {
                onLog("attenzione: worker \(p.host) ha modello '\(h.modelName)' ≠ '\(engine.modelName)'\n")
            }
            conns.append(conn)
            entries.append(DistRouteEntry(host: p.host, port: p.port, layerStart: h.layerStart,
                                          layerEnd: h.layerEnd, hasOutput: h.hasOutput))
            onLog("route: \(p.host):\(p.port) → layer \(h.layerStart)…\(h.layerEnd)\(h.hasOutput ? " +output" : "")\n")
        }
        // Sort by layerStart, keep conns aligned, validate contiguous full coverage.
        let order = entries.indices.sorted { entries[$0].layerStart < entries[$1].layerStart }
        entries = order.map { entries[$0] }
        conns = order.map { conns[$0] }
        var expected = 0
        for e in entries {
            guard e.layerStart == expected else { throw DistError.sliceGap("atteso layer \(expected), trovato \(e.layerStart)") }
            expected = e.layerEnd + 1
        }
        guard expected == engine.nLayers else {
            throw DistError.sliceGap("copertura \(expected)/\(engine.nLayers) layer — la route deve coprire 0…\(engine.nLayers - 1) in modo contiguo (manca da \(expected) in poi)")
        }
        if config.forward {
            let l = DistReturnListener()
            try l.start(port: config.returnPort)
            returnListener = l
            returnIter = l.results.makeAsyncIterator()
            onLog("listener di ritorno su :\(config.returnPort)\n")
        }
        onLog("route completa: \(engine.nLayers) layer su \(entries.count) worker"
              + (config.forward ? " · inoltro worker→worker" : " · relay") + "\n")
    }

    public func disconnect() {
        for c in conns { c.cancel() }
        conns = []; entries = []
        returnListener?.stop(); returnListener = nil; returnIter = nil
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
        let ids = engine.chatPromptIds(turns: turns, tools: tools, think: think)
        guard ids.count < config.contextSize else { throw DistError.sliceGap("prompt oltre il contesto") }
        onLog("prefill \(ids.count) token (chunk \(config.prefillChunk))…\n")

        // PREFILL the whole prompt in chunks, fresh KV (posBase 0 resets workers).
        var pos = 0
        var lastLogits: [Float] = []
        var start = 0
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
            if let logits = try await runChunk(hcs: hcs, posBase: pos, wantLogits: end == ids.count) {
                lastLogits = logits
            }
            pos += end - start
            start = end
            onProgress("prefill \(pos)/\(ids.count) token…")
        }
        guard !lastLogits.isEmpty else { throw DistError.badFrame }
        // Diagnose where the pipeline breaks: a sane top token here = prefill OK,
        // problem in decode; garbage here = problem in embed/slice/head/wire.
        if let mx = lastLogits.indices.max(by: { lastLogits[$0] < lastLogits[$1] }) {
            let finite = lastLogits.filter { $0.isFinite }.count
            onLog(String(format: "diag: prefill top=%d (%@) logit=%.2f · %d/%d finiti\n",
                         mx, engine.tokenText(mx), lastLogits[mx], finite, lastLogits.count))
        }

        // DECODE token-by-token, splitting reasoning (<think>…</think>) from text
        // and buffering DSML tool-call markup (never shown; parsed at the end —
        // same scheme as the local InferenceService, incl. the held '<' opener).
        var rng = sampling.seed
        var produced = 0
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
            guard let logits = try await runChunk(hcs: [hc], posBase: pos, wantLogits: true) else {
                throw DistError.badFrame
            }
            lastLogits = logits
            pos += 1; produced += 1
            let elapsed = Date().timeIntervalSince(t0)
            onProgress(String(format: "%d token · %.2f tok/s", produced,
                              elapsed > 0 ? Double(produced) / elapsed : 0))
        }
        if pendingLT, !inTool { emit("<") }
        let dt = Date().timeIntervalSince(t0)
        onLog("[\(produced) token · \(String(format: "%.2f", dt > 0 ? Double(produced) / dt : 0)) tok/s]\n")
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
        while start < ids.count {
            try Task.checkCancellation()
            let end = min(start + config.prefillChunk, ids.count)
            var hcs: [[Float]] = []
            for (k, id) in ids[start..<end].enumerated() {
                hcs.append(try engine.embed(token: id, pos: pos + k))
            }
            if let logits = try await runChunk(hcs: hcs, posBase: pos, wantLogits: end == ids.count) {
                lastLogits = logits
            }
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
            guard let logits = try await runChunk(hcs: [hc], posBase: pos, wantLogits: true) else {
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
    private func runChunk(hcs: [[Float]], posBase: Int, wantLogits: Bool) async throws -> [Float]? {
        var flags: Dist.WorkFlags = []
        if posBase == 0 { flags.insert(.resetSession) }
        if config.forward {
            var f = flags
            if wantLogits { f.insert(.outputLogits) }
            let work = DistWork(pos: posBase, nTokens: hcs.count,
                                layerStart: entries[0].layerStart, layerEnd: entries[0].layerEnd,
                                flags: f, hcBits: config.activationBits, route: entries, routeIndex: 0,
                                returnHost: config.returnHost, returnPort: config.returnPort,
                                hc: hcs.flatMap { $0 })
            try await conns[0].sendFrame(.work, work.encoded())
            guard let res = await returnIter?.next() else { throw DistError.closed }
            return res.kind == .logits ? res.values : nil
        }
        var states = hcs
        let stateLen = engine.hcStateCount
        for (i, e) in entries.enumerated() {
            var f = flags
            if i == entries.count - 1, wantLogits { f.insert(.outputLogits) }
            let work = DistWork(pos: posBase, nTokens: states.count, layerStart: e.layerStart,
                                layerEnd: e.layerEnd, flags: f, hcBits: config.activationBits,
                                hc: states.flatMap { $0 })
            try await conns[i].sendFrame(.work, work.encoded())
            let (type, payload) = try await conns[i].readFrame()
            guard type == .result, let res = DistResult.decode(payload) else { throw DistError.badFrame }
            if res.kind == .logits { return res.values }
            states = (0..<states.count).map { Array(res.values[$0 * stateLen..<($0 + 1) * stateLen]) }
        }
        return nil
    }
}
