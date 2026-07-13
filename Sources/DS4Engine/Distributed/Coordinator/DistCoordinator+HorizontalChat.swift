import Foundation
import DS4Core

extension DistCoordinator {
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

    /// One chunk through the pipeline; returns the last token's logits if `wantLogits`.
    /// Results are matched on the `session` echo: anything from an older session
    /// (a turn abandoned mid-chunk) is discarded and the read repeats. ERROR
    /// frames from workers surface with the worker's own message.
    func runChunk(session: UInt32, hcs: [[Float]], posBase: Int, tokens: [Int],
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
