import Foundation
import DS4Core

extension DistCoordinator {
    // MARK: - Verticale (expert parallelism, Fase C — docs/EXPERT_PARALLELISM.md)

    /// Partizione round-robin: esperto e → worker e % n (mask dimensionata da
    /// nExperts, copertura esatta). Il bilanciamento per CARICO dalla usage
    /// imatrix è Fase D; il round-robin distribuisce già bene gli esperti
    /// caldi (id sparsi su tutto lo spazio).
    static func partitionExperts(nExperts: Int, workers: Int) -> [Data] {
        guard nExperts > 0, workers > 0 else { return [] }
        var masks = [Data](repeating: Data(repeating: 0, count: (nExperts + 7) / 8),
                           count: workers)
        for e in 0..<nExperts {
            masks[e % workers][e / 8] |= UInt8(1 << (e % 8))
        }
        return masks
    }

    /// Strict mask decoder shared by route construction/tests. It rejects both
    /// the wrong byte length and non-zero padding bits so malformed ownership
    /// can never silently leave an expert unassigned or multiply assigned.
    static func decodeExpertMask(_ data: Data, nExperts: Int) -> [Bool]? {
        guard nExperts > 0, data.count == (nExperts + 7) / 8 else { return nil }
        if nExperts % 8 != 0, let last = data.last {
            let validMask = UInt8((1 << (nExperts % 8)) - 1)
            guard last & ~validMask == 0 else { return nil }
        }
        return (0..<nExperts).map { e in
            (data[data.startIndex + e / 8] >> UInt8(e % 8)) & 1 == 1
        }
    }

    /// Stabilisce la route VERTICALE: ogni peer riceve i file (v8, ripresa a
    /// checkpoint inclusa) e uno shard di ESPERTI (expertAssign); poi il
    /// coordinatore carica il backbone denso locale con la FFN routed
    /// instradata sui peer. PREREQUISITO misurato (vedi doc): RTT < 1 ms —
    /// ~41 round-trip sincroni per token.
    public func connectVertical(onLog: @escaping @Sendable (String) -> Void) async throws {
        let modelName = (config.modelPath as NSString).lastPathComponent
        let offer = try buildFileOffer(onLog: onLog)
        let n = config.peers.count
        guard n >= 1 else { throw DistError.sliceGap("verticale: serve almeno un worker") }
        let nExperts = engine.nExperts
        let masks = Self.partitionExperts(nExperts: nExperts, workers: n)
        guard masks.count == n else { throw DistError.sliceGap("verticale: partizione esperti vuota") }
        var setup = [DistConnection?](repeating: nil, count: n)
        try await withThrowingTaskGroup(of: (Int, DistConnection).self) { group in
            for (i, p) in config.peers.enumerated() {
                let mask = masks[i]
                group.addTask { [self] in
                    (i, try await setupExpertPeer(p, mask: mask, offer: offer,
                                                  modelName: modelName, onLog: onLog))
                }
            }
            for try await (i, conn) in group { setup[i] = conn }
        }
        expertPeers = setup.enumerated().compactMap { i, conn in
            guard let conn,
                  let bools = Self.decodeExpertMask(masks[i], nExperts: nExperts) else { return nil }
            return (conn, bools)
        }
        guard expertPeers.count == n else { throw DistError.badFrame }
        for expert in 0..<nExperts {
            guard expertPeers.reduce(0, { $0 + ($1.mask[expert] ? 1 : 0) }) == 1 else {
                throw DistError.sliceGap(
                    "verticale: esperto \(expert) non ha esattamente un proprietario")
            }
        }
        onLog("carico il backbone denso locale (route/attention/KV/head)…\n")
        // Cache esperti locale spenta: gli esperti vivono sui worker e il
        // gather locale non parte mai (remoteExperts sostituisce quel ramo;
        // anche il look-ahead speculativo è disattivato dal decoder).
        // Su thread GCD, non sul pool cooperativo: il load del backbone fa
        // fan-out con concurrentPerform (incluso l'eventuale requant Q4).
        let cfg = config
        let engine = try await Self.onGCD {
            try DistEngine(modelPath: cfg.modelPath, contextSize: cfg.contextSize,
                           expertCacheSlots: 0, kvLayers: nil,
                           onLoadLog: { onLog("backbone: \($0)\n") })
        }
        let peers = expertPeers
        let bits = config.activationBits == 16 ? 16 : 32
        engine.setRemoteExperts { [self] layer, ids, weights, activation in
            try routedPartialBlocking(layer: layer, ids: ids, weights: weights,
                                      activation: activation, peers: peers, bits: bits)
        }
        verticalEngine = engine
        onLog("route verticale completa: \(n) shard · \(nExperts) esperti · \(engine.nLayers) layer + backbone locale\n")
    }

    /// Setup di UN peer verticale: handshake+file condivisi, poi EXPERT ASSIGN
    /// (mask + knob del coordinatore + usage) e attesa READY con progresso.
    private func setupExpertPeer(_ p: Peer, mask: Data, offer: [DistFileEntry],
                                 modelName: String,
                                 onLog: @escaping @Sendable (String) -> Void) async throws -> DistConnection {
        onLog("connessione (verticale) a \(p.host):\(p.port)…\n")
        let conn = try await DistConnection.connect(host: p.host, port: p.port,
                                                    queue: queue, onState: onLog)
        return try await withTaskCancellationHandler {
            do {
                try await handshakeAndTransfer(p, conn: conn, offer: offer, onLog: onLog)
                let env = ProcessInfo.processInfo.environment
                let knobs: [(key: String, value: String)] = Dist.perfKnobKeys.compactMap { k in
                    env[k].map { (key: k, value: $0) }
                }
                let usage = InferenceService.usageDataSeeded(modelName: modelName,
                                                             agentId: "generale") ?? Data()
                let assign = DistExpertAssign(modelName: modelName,
                                              expertCacheSlots: config.workerCacheSlots,
                                              useExpertBundle: config.useExpertBundle,
                                              expertMask: mask, envKnobs: knobs, usageJSON: usage)
                let owned = mask.reduce(0) { $0 + $1.nonzeroBitCount }
                onLog("expert-assign: \(p.host):\(p.port) — \(owned) esperti\n")
                try await conn.sendFrame(.expertAssign, assign.encoded())
                while true {
                    let (rType, rPayload) = try await conn.readFrame()
                    if rType == .progress {
                        onLog("worker \(p.host): " + String(decoding: rPayload, as: UTF8.self) + "\n")
                        continue
                    }
                    if rType == .error {
                        throw DistError.remote("\(p.host):\(p.port): " + String(decoding: rPayload, as: UTF8.self))
                    }
                    guard rType == .ready,
                          let ready = DistHello.decode(rPayload), ready.assigned else {
                        throw DistError.badFrame
                    }
                    guard ready.nLayers == engine.nLayers,
                          ready.layerStart == 0,
                          ready.layerEnd == engine.nLayers - 1 else {
                        throw DistError.sliceGap(
                            "worker verticale \(p.host):\(p.port) ha caricato \(ready.nLayers) layer "
                            + "(\(ready.layerStart)...\(ready.layerEnd)), attesi \(engine.nLayers)")
                    }
                    break
                }
                onLog("shard \(p.host):\(p.port) pronto (\(owned) esperti)\n")
                return conn
            } catch {
                conn.cancel()
                throw error
            }
        } onCancel: {
            conn.cancel()
        }
    }

    /// Ponte SINCRONO decode-thread → rete: partiziona i 6 id per proprietario,
    /// spedisce le expertWork in PARALLELO ai peer coinvolti, somma le parziali.
    /// Bloccante (semaforo): il decode gira su un thread detached, MAI sul
    /// cooperative pool (benchmarkVertical lo documenta).
    private func routedPartialBlocking(layer: Int, ids: [Int32], weights: [Float],
                                       activation: [Float],
                                       peers: [(conn: DistConnection, mask: [Bool])],
                                       bits: Int) throws -> [Float] {
        let nE = activation.count
        seqLock.lock(); seqCounter &+= 1; let seq = seqCounter; seqLock.unlock()
        let actData: Data
        if bits == 16 {
            var halves = [Float16](repeating: 0, count: nE)
            for i in 0..<nE { halves[i] = Float16(activation[i]) }
            actData = halves.withUnsafeBytes { Data($0) }
        } else {
            actData = activation.withUnsafeBytes { Data($0) }
        }
        var requests: [(conn: DistConnection, ids: [Int32], w: [Float])] = []
        var covered = 0
        for peer in peers {
            var myIds: [Int32] = [], myW: [Float] = []
            for (j, id) in ids.enumerated()
            where id >= 0 && Int(id) < peer.mask.count && peer.mask[Int(id)] {
                myIds.append(id); myW.append(j < weights.count ? weights[j] : 0)
            }
            if !myIds.isEmpty { requests.append((peer.conn, myIds, myW)); covered += myIds.count }
        }
        guard covered == ids.filter({ $0 >= 0 }).count else {
            throw DistError.sliceGap("verticale: esperti selezionati senza proprietario (partizione rotta)")
        }
        let sem = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var outcome: Result<[Float], Error> = .failure(DistError.closed)
        let reqs = requests
        Task {
            do {
                var acc = [Float](repeating: 0, count: nE)
                try await withThrowingTaskGroup(of: [Float].self) { group in
                    for r in reqs {
                        group.addTask {
                            let work = DistExpertWork(seq: seq, layer: layer, ids: r.ids,
                                                      weights: r.w, activation: actData, bits: bits)
                            try await r.conn.sendFrame(.expertWork, work.encoded())
                            while true {
                                let (t, p) = try await r.conn.readFrame()
                                if t == .expertSum, let s = DistExpertSum.decode(p), s.seq == seq {
                                    return try Self.floats(from: s.partial, bits: s.bits, count: nE)
                                }
                                if t == .error {
                                    throw DistError.remote(String(decoding: p, as: UTF8.self))
                                }
                                // progress/altro: ignora e continua a leggere
                            }
                        }
                    }
                    for try await part in group {
                        for i in 0..<nE { acc[i] += part[i] }
                    }
                }
                outcome = .success(acc)
            } catch { outcome = .failure(error) }
            sem.signal()
        }
        sem.wait()
        return try outcome.get()
    }

    /// Payload f32/f16 → [Float] (esattamente `count` valori; memcpy staging —
    /// una Data di rete può essere una slice non allineata).
    static func floats(from d: Data, bits: Int, count: Int) throws -> [Float] {
        if bits == 32 {
            guard d.count == count * 4 else { throw DistError.badFrame }
            var out = [Float](repeating: 0, count: count)
            _ = out.withUnsafeMutableBytes { dst in
                d.withUnsafeBytes { src in memcpy(dst.baseAddress!, src.baseAddress!, count * 4) }
            }
            return out
        }
        guard d.count == count * 2 else { throw DistError.badFrame }
        var halves = [Float16](repeating: 0, count: count)
        _ = halves.withUnsafeMutableBytes { dst in
            d.withUnsafeBytes { src in memcpy(dst.baseAddress!, src.baseAddress!, count * 2) }
        }
        return halves.map { Float($0) }
    }
}
