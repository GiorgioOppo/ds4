import Foundation

/// The distributed COORDINATOR: owns the embedding, the sampling loop and the
/// prompt. It connects to each worker, reads their HELLO to assemble + validate a
/// contiguous route covering every layer, then drives generation:
///
/// - PREFILL runs in chunks (`prefillChunk` tokens per WORK frame): the network
///   round trip is amortized over the chunk instead of paid per token.
/// - DECODE runs token-by-token (chunks of 1) and asks the terminal hop for logits.
///
/// Two transports:
/// - RELAY (default): the coordinator round-trips each chunk through every worker
///   in layer order. Simple, no inbound connectivity needed on the coordinator.
/// - FORWARDING (`forward: true`): the chunk carries the whole route; workers pass
///   the HC state worker→worker and the TERMINAL worker replies to the
///   coordinator's return listener (`returnHost:returnPort`) — half the hops.
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
        public var activationBits: Int   // 32 / 16 / 8
        public var prefillChunk: Int     // tokens per WORK frame during prefill
        public var forward: Bool         // worker→worker forwarding
        public var returnHost: String    // this coordinator's address, as workers see it
        public var returnPort: UInt16    // return listener port (forwarding only)
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

    public init(config: Config) throws {
        self.config = config
        // Pure coordinator: embed + head only — no KV/compressor allocation.
        self.engine = try DistEngine(modelPath: config.modelPath, contextSize: config.contextSize,
                                     kvLayers: 0..<0)
    }

    /// Run one prompt across the cluster, streaming generated token text via `onToken`.
    public func generate(system: String?, prompt: String, maxTokens: Int, sampling: SamplingParams,
                         onLog: @Sendable (String) -> Void,
                         onToken: @Sendable (String) -> Void) async throws {
        // Connect the route.
        var conns: [DistConnection] = []
        var entries: [DistRouteEntry] = []
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
        defer { for c in conns { c.cancel() } }

        // Sort by layerStart, keep conns aligned, validate contiguous full coverage.
        let order = entries.indices.sorted { entries[$0].layerStart < entries[$1].layerStart }
        entries = order.map { entries[$0] }
        conns = order.map { conns[$0] }
        var expected = 0
        for e in entries {
            guard e.layerStart == expected else {
                throw DistError.sliceGap("atteso layer \(expected), trovato \(e.layerStart)")
            }
            expected = e.layerEnd + 1
        }
        guard expected == engine.nLayers else {
            throw DistError.sliceGap("copertura \(expected)/\(engine.nLayers) layer — la route deve coprire 0…\(engine.nLayers - 1) in modo contiguo (manca da \(expected) in poi)")
        }
        onLog("route completa: \(engine.nLayers) layer su \(entries.count) worker"
              + (config.forward ? " · inoltro worker→worker" : " · relay") + "\n")

        // Forwarding: start the return listener the terminal worker replies to.
        var returnListener: DistReturnListener?
        var returnIter: AsyncStream<DistResult>.Iterator?
        if config.forward {
            let l = DistReturnListener()
            try l.start(port: config.returnPort)
            returnListener = l
            returnIter = l.results.makeAsyncIterator()
            onLog("listener di ritorno su :\(config.returnPort)\n")
        }
        defer { returnListener?.stop() }

        /// One chunk through the pipeline; returns the last token's logits if `wantLogits`.
        func runChunk(hcs: [[Float]], posBase: Int, wantLogits: Bool) async throws -> [Float]? {
            var flags: Dist.WorkFlags = []
            if posBase == 0 { flags.insert(.resetSession) }
            if config.forward {
                var f = flags
                if wantLogits { f.insert(.outputLogits) }
                let work = DistWork(pos: posBase, nTokens: hcs.count,
                                    layerStart: entries[0].layerStart, layerEnd: entries[0].layerEnd,
                                    flags: f, hcBits: config.activationBits,
                                    route: entries, routeIndex: 0,
                                    returnHost: config.returnHost, returnPort: config.returnPort,
                                    hc: hcs.flatMap { $0 })
                try await conns[0].sendFrame(.work, work.encoded())
                guard let res = await returnIter?.next() else { throw DistError.closed }
                return res.kind == .logits ? res.values : nil
            }
            // Relay: round-trip each hop through the coordinator.
            var states = hcs
            let stateLen = engine.hcStateCount
            for (i, e) in entries.enumerated() {
                var f = flags
                let last = i == entries.count - 1
                if last && wantLogits { f.insert(.outputLogits) }
                let work = DistWork(pos: posBase, nTokens: states.count,
                                    layerStart: e.layerStart, layerEnd: e.layerEnd,
                                    flags: f, hcBits: config.activationBits,
                                    hc: states.flatMap { $0 })
                try await conns[i].sendFrame(.work, work.encoded())
                let (type, payload) = try await conns[i].readFrame()
                guard type == .result, let res = DistResult.decode(payload) else { throw DistError.badFrame }
                if res.kind == .logits { return res.values }
                states = (0..<states.count).map { Array(res.values[$0*stateLen..<($0+1)*stateLen]) }
            }
            return nil
        }

        // PREFILL in chunks.
        let ids = engine.chatPromptIds(system: system, prompt: prompt)
        onLog("prefill \(ids.count) token (chunk \(config.prefillChunk))…\n")
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
            let wantLogits = end == ids.count       // logits only for the prompt's last token
            if let logits = try await runChunk(hcs: hcs, posBase: pos, wantLogits: wantLogits) {
                lastLogits = logits
            }
            pos += end - start
            start = end
            onLog("prefill \(pos)/\(ids.count)\n")
        }
        guard !lastLogits.isEmpty else { throw DistError.badFrame }

        // DECODE token-by-token.
        var rng = sampling.seed
        var produced = 0
        // Tail of the token stream (prompt + generated) for the repetition penalty.
        var recentIds = Array(ids.suffix(sampling.repeatLastN))
        let t0 = Date()
        while produced < maxTokens {
            try Task.checkCancellation()
            let next = engine.sample(lastLogits, params: sampling, recent: recentIds[...], rng: &rng)
            if next == engine.eosId { break }
            onToken(engine.tokenText(next))
            recentIds.append(next)
            if recentIds.count > sampling.repeatLastN { recentIds.removeFirst() }
            let hc = try engine.embed(token: next, pos: pos)
            guard let logits = try await runChunk(hcs: [hc], posBase: pos, wantLogits: true) else {
                throw DistError.badFrame
            }
            lastLogits = logits
            pos += 1; produced += 1
        }
        let dt = Date().timeIntervalSince(t0)
        onLog("\n[completato: \(produced) token · \(String(format: "%.2f", dt > 0 ? Double(produced)/dt : 0)) tok/s]\n")
    }
}
