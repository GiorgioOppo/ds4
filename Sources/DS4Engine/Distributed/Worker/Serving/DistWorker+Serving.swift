import Foundation
import DS4Core

extension DistWorker {
    func serve(_ conn: DistConnection) async {
        onLog("connessione in ingresso\n")
        // Outbound connections (next-hop worker / coordinator return), per session.
        var downstream: [String: DistConnection] = [:]
        defer { for c in downstream.values { c.cancel() } }

        // `expectHello`: next-hop workers greet new connections with a HELLO frame
        // (consume it once); the coordinator's return listener does not.
        func outbound(_ host: String, _ port: UInt16, expectHello: Bool) async throws -> DistConnection {
            let key = "\(host):\(port)"
            if let c = downstream[key] { return c }
            let c = try await DistConnection.connect(host: host, port: port, queue: queue,
                                                     readyTimeout: 10, onState: onLog)
            if expectHello { _ = try await c.readFrame() }
            downstream[key] = c
            return c
        }

        // File distribution state for THIS connection: the offer's entries,
        // where each resolved locally, and the file currently being received.
        var offerEntries: [DistFileEntry] = []
        var resolvedFiles: [String: String] = [:]      // sanitized name → local path
        var incoming: IncomingFile?
        var pendingResume: [Int: UInt64] = [:]         // v8: offer index → offset di ripresa promesso
        var transferError: String?                     // first failure, reported at fileDone

        do {
            try await conn.sendFrame(.hello, helloPayload().encoded())

            while true {
                let (type, payload) = try await conn.readFrame()
                if type == .fileOffer {
                    guard let offer = DistFileOffer.decode(payload) else { continue }
                    // Una nuova offerta azzera lo stato di trasferimento della
                    // connessione: un incoming/errore rimasti da un'offerta
                    // precedente non devono interferire (e l'incoming va
                    // sospeso PRIMA che resumePoint tronchi il suo .part).
                    incoming?.suspend(); incoming = nil
                    transferError = nil
                    offerEntries = offer.entries
                    var needs: [Int] = []
                    var offsets: [UInt64] = []
                    pendingResume.removeAll()
                    for (i, entry) in offer.entries.enumerated() {
                        if let local = resolveOffered(entry) {
                            resolvedFiles[DistFileStore.sanitize(entry.name)] = local
                            onLog("file: \(entry.name) già presente (hash ok) — \(local)\n")
                        } else if let promoted = promoteCompletePart(entry, onLog: onLog) {
                            // v8 passo 1: .part a size piena e hash INTERO valido
                            // → promosso sul posto, niente da trasferire.
                            resolvedFiles[DistFileStore.sanitize(entry.name)] = promoted
                        } else {
                            // v8 passo 2: hash intero fallito o .part parziale —
                            // la catena di checkpoint dice fin dove il
                            // trasferimento era arrivato e si riprende da lì.
                            let resume = resumePoint(entry, onLog: onLog)
                            needs.append(i)
                            offsets.append(resume)
                            pendingResume[i] = resume
                            if resume > 0 {
                                onLog("file: \(entry.name) parziale — riprendo da \(resume / 1_048_576) di \(entry.size / 1_048_576) MB\n")
                            } else {
                                onLog("file: \(entry.name) mancante (\(entry.size / 1_048_576) MB) — richiedo il trasferimento\n")
                            }
                        }
                    }
                    try await conn.sendFrame(.fileNeed, DistFileNeed(indices: needs, offsets: offsets).encoded())
                    continue
                }
                if type == .fileChunk {
                    // Failures are remembered and reported ONCE at fileDone —
                    // the coordinator only reads the ack there, and one bad
                    // chunk must not pile an ack per remaining chunk.
                    guard let chunk = DistFileChunk.decode(payload),
                          chunk.index >= 0, chunk.index < offerEntries.count else {
                        transferError = transferError ?? "malformed fileChunk frame"
                        continue
                    }
                    // Il primo chunk di un file deve arrivare ESATTAMENTE
                    // all'offset di ripresa dichiarato nel fileNeed (0 senza .part).
                    if incoming == nil, transferError == nil,
                       chunk.offset == (pendingResume[chunk.index] ?? 0) {
                        incoming = IncomingFile(entry: offerEntries[chunk.index],
                                                index: chunk.index,
                                                resumeFrom: pendingResume[chunk.index] ?? 0,
                                                onLog: onLog)
                    }
                    if let file = incoming, file.index == chunk.index, file.append(chunk) {
                        continue
                    }
                    incoming?.suspend(); incoming = nil
                    transferError = transferError ?? "out-of-order or unexpected chunk"
                    continue
                }
                if type == .fileDone {
                    defer { transferError = nil }
                    // v8, caso limite: il .part copriva GIÀ tutto il file (catena
                    // verificata) → il coordinatore non manda alcun chunk e il
                    // DONE arriva senza un IncomingFile: crealo qui a offset
                    // pieno, così finalize() ne verifica hash e lo promuove.
                    if incoming == nil, transferError == nil,
                       let done = DistFileDone.decode(payload),
                       done.index >= 0, done.index < offerEntries.count,
                       let resume = pendingResume[done.index],
                       resume == offerEntries[done.index].size {
                        incoming = IncomingFile(entry: offerEntries[done.index], index: done.index,
                                                resumeFrom: resume, onLog: onLog)
                    }
                    guard let done = DistFileDone.decode(payload),
                          let file = incoming, file.index == done.index, transferError == nil else {
                        incoming?.suspend(); incoming = nil
                        try await conn.sendFrame(.fileAck, DistKV.encodeAck(
                            ok: false, message: transferError ?? "DONE without a matching transfer"))
                        continue
                    }
                    incoming = nil
                    switch file.finalize() {
                    case .success(let path):
                        resolvedFiles[DistFileStore.sanitize(file.entry.name)] = path
                        onLog("file: \(file.entry.name) ricevuto e verificato\n")
                        try await conn.sendFrame(.fileAck, DistKV.encodeAck(ok: true))
                    case .failure(let reason):
                        onLog("file: \(file.entry.name) SCARTATO: \(reason)\n")
                        try await conn.sendFrame(.fileAck, DistKV.encodeAck(ok: false, message: reason))
                    }
                    continue
                }
                if type == .assign {
                    try await handleAssign(payload, on: conn, resolvedFiles: resolvedFiles)
                    continue
                }
                if type == .expertAssign {
                    try await handleExpertAssign(payload, on: conn, resolvedFiles: resolvedFiles)
                    continue
                }
                if type == .expertWork {
                    guard let req = DistExpertWork.decode(payload) else {
                        try await conn.sendFrame(.error, Data("malformed EXPERT WORK frame".utf8))
                        continue
                    }
                    guard let shard = currentShard() else {
                        try await conn.sendFrame(.error,
                                                 Data("worker not ready: no expert shard loaded (send EXPERT ASSIGN first)".utf8))
                        continue
                    }
                    do {
                        // Sincrona (gather SSD + un cb GPU, ~ms): una richiesta
                        // alla volta per connessione — il parallelismo del
                        // verticale è TRA i worker.
                        let sum = try shard.partial(req)
                        try await conn.sendFrame(.expertSum, sum.encoded())
                    } catch {
                        try await conn.sendFrame(.error, Data("expertWork(seq \(req.seq)): \(error)".utf8))
                    }
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
                                                        start: work.layerStart, end: work.layerEnd,
                                                        tokens: work.tokenIds.isEmpty ? nil
                                                            : work.tokenIds.map(Int.init))
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
                                       hc: outStates.flatMap { $0 }, tokenIds: work.tokenIds)
                    let c = try await outbound(next.host, next.port, expectHello: true)
                    try await c.sendFrame(.work, fwd.encoded())
                }
            }
        } catch {
            incoming?.suspend()               // KEEP the half-received .part: the
                                              // next offer resumes it via the
                                              // checkpoint chain (v8)
            onLog("sessione chiusa: \(error)\n")
            conn.cancel()
        }
    }

}
