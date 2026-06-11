import Foundation

/// The distributed COORDINATOR: owns the embedding, the sampling loop and the
/// prompt. It connects to each worker, reads their HELLO to assemble + validate a
/// contiguous route covering every layer, then per token: embed → relay the HC
/// state through the workers in layer order → receive logits (from the worker that
/// owns the output head, or run the head locally) → sample.
///
/// This first version RELAYS each hop through the coordinator (coordinator→worker→
/// coordinator→next worker…) and runs prefill token-by-token: simple and correct,
/// not yet pipelined. Worker-to-worker forwarding + chunked prefill are follow-ups.
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
        public init(modelPath: String, contextSize: Int, peers: [Peer], activationBits: Int) {
            self.modelPath = modelPath; self.contextSize = contextSize
            self.peers = peers; self.activationBits = activationBits
        }
    }

    private let engine: DistEngine
    private let config: Config
    private let queue = DispatchQueue(label: "ds4.dist.coord")

    public init(config: Config) throws {
        self.config = config
        self.engine = try DistEngine(modelPath: config.modelPath, contextSize: config.contextSize)
    }

    /// Run one prompt across the cluster, streaming generated token text via `onToken`.
    public func generate(system: String?, prompt: String, maxTokens: Int, sampling: SamplingParams,
                         onLog: @Sendable (String) -> Void,
                         onToken: @Sendable (String) -> Void) async throws {
        var route: [(conn: DistConnection, entry: DistRouteEntry)] = []
        for p in config.peers {
            let conn = try DistConnection.connect(host: p.host, port: p.port, queue: queue)
            let (type, payload) = try await conn.readFrame()
            guard type == .hello, let h = DistHello.decode(payload) else { throw DistError.badFrame }
            if h.modelName != engine.modelName {
                onLog("attenzione: worker \(p.host) ha modello '\(h.modelName)' ≠ '\(engine.modelName)'\n")
            }
            route.append((conn, DistRouteEntry(host: p.host, port: p.port, layerStart: h.layerStart,
                                               layerEnd: h.layerEnd, hasOutput: h.hasOutput)))
            onLog("route: \(p.host):\(p.port) → layer \(h.layerStart)…\(h.layerEnd)\(h.hasOutput ? " +output" : "")\n")
        }
        defer { for r in route { r.conn.cancel() } }

        route.sort { $0.entry.layerStart < $1.entry.layerStart }
        var expected = 0
        for r in route {
            guard r.entry.layerStart == expected else {
                throw DistError.sliceGap("atteso layer \(expected), trovato \(r.entry.layerStart)")
            }
            expected = r.entry.layerEnd + 1
        }
        guard expected == engine.nLayers else {
            throw DistError.sliceGap("copertura \(expected)/\(engine.nLayers) layer — route incompleta")
        }
        onLog("route completa: \(engine.nLayers) layer su \(route.count) worker\n")

        // Prefill (token-by-token through the pipeline).
        let ids = engine.chatPromptIds(system: system, prompt: prompt)
        onLog("prefill \(ids.count) token…\n")
        var pos = 0
        var lastLogits: [Float] = []
        for id in ids {
            try Task.checkCancellation()
            lastLogits = try await step(token: id, pos: pos, route: route)
            pos += 1
        }

        // Decode.
        var rng = sampling.seed
        var produced = 0
        while produced < maxTokens {
            try Task.checkCancellation()
            let next = engine.sample(lastLogits, params: sampling, rng: &rng)
            if next == engine.eosId { break }
            onToken(engine.tokenText(next))
            lastLogits = try await step(token: next, pos: pos, route: route)
            pos += 1; produced += 1
        }
        onLog("\n[completato: \(produced) token]\n")
    }

    /// One forward through the whole pipeline for `token` at `pos`, returning logits.
    private func step(token: Int, pos: Int,
                      route: [(conn: DistConnection, entry: DistRouteEntry)]) async throws -> [Float] {
        var hc = try engine.embed(token: token, pos: pos)
        let nKeys = pos + 1
        for r in route {
            var flags: Dist.WorkFlags = []
            if r.entry.hasOutput { flags.insert(.outputLogits) }
            if pos == 0 { flags.insert(.resetSession) }
            let work = DistWork(pos: pos, nKeys: nKeys, layerStart: r.entry.layerStart,
                                layerEnd: r.entry.layerEnd, flags: flags, token: token,
                                hcBits: config.activationBits, hc: hc)
            try await r.conn.sendFrame(.work, work.encoded())
            let (type, payload) = try await r.conn.readFrame()
            guard type == .result, let res = DistResult.decode(payload) else { throw DistError.badFrame }
            if res.kind == .logits { return res.values }
            hc = res.values
        }
        // No worker owned the output head → run it locally on the coordinator.
        return try engine.head(hc: hc)
    }
}
