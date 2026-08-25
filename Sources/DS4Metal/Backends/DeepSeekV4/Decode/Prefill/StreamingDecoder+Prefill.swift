import Foundation
import Metal
import DS4Core

private enum PrefillPoolFailure {
    case cancelled
    case error(String)
}

/// Void-specialized pool boundary. Keeping this separate from the generic
/// result carrier avoids optimized code assigning `()` through `T?`; that path
/// produced an invalid Objective-C retain inside nested expert-group pools.
@inline(never)
private func withPrefillVoidAutoreleasePool(_ body: () throws -> Void) throws {
    var failure: PrefillPoolFailure?
    autoreleasepool {
        do {
            try body()
        } catch is CancellationError {
            failure = .cancelled
        } catch {
            failure = .error(String(reflecting: error))
        }
    }
    switch failure {
    case .cancelled:
        throw CancellationError()
    case .error(let message):
        throw MetalError.unsupported("prefill autoreleasepool: \(message)")
    case nil:
        return
    }
}

/// Foundation's throwing `autoreleasepool` can let an Objective-C-backed
/// `Error` outlive the pool that owns its NSError box in optimized builds.
/// Capture its text while the pool is still alive and only throw Swift value
/// types after the drain. The explicit result slot also retains successful
/// Objective-C-backed return values (notably GraphContext) before draining.
@inline(never)
private func withPrefillAutoreleasePool<T>(_ body: () throws -> T) throws -> T {
    var output: T?
    var failure: PrefillPoolFailure?
    autoreleasepool {
        do {
            output = try body()
        } catch is CancellationError {
            failure = .cancelled
        } catch {
            failure = .error(String(reflecting: error))
        }
    }
    switch failure {
    case .cancelled:
        throw CancellationError()
    case .error(let message):
        throw MetalError.unsupported("prefill autoreleasepool: \(message)")
    case nil:
        guard let output else {
            throw MetalError.unsupported("prefill autoreleasepool returned no value")
        }
        return output
    }
}

extension StreamingDecoder {
    /// once** (per chunk) instead of once per token, so the dominant weight I/O is
    /// amortized over all the chunk's tokens. Numerically **identical** to calling
    /// `forward()` for tokens 0..N-1 in order — same ops, same per-token order,
    /// same KV-cache and NSA-compressor evolution — just reordered (layer outer,
    /// token inner) so the mmap'd weights stay hot across tokens. The prompt is
    /// split into chunks of `chunk` tokens to bound activation memory (≈ 2·chunk
    /// HC buffers); KV cache and the recurrent compressor carry across chunks.
    /// Populates the KV cache for positions startPos..startPos+N-1 and returns the
    /// LAST token's logits. With `startPos > 0` the call is **incremental**: it does
    /// NOT reset the recurrent compressor and continues the KV cache from the given
    /// position (the caller guarantees positions 0..startPos-1 are already valid) —
    /// this is what enables KV reuse across turns (prefill only the new suffix).
    public func prefill(tokens: [Int], startPos: Int = 0, chunk: Int = 512) throws -> [Float] {
        precondition(!tokens.isEmpty)
        if startPos == 0 { for c in compStates { try c?.reset(rt) }; for c in indexStates { try c?.reset(rt) } }   // fresh sequence
        var lastHC: GPUTensor?
        var start = 0
        // DS4_PREFILL_CHUNK: token per chunk (default 512). Un chunk piu' largo
        // ammortizza meglio i costi per-chunk (ogni chunk ricarica i densi di
        // TUTTI i layer: ~6 GB con DENSE_STREAM) al prezzo di ~160 KB/token di
        // attivazioni transienti in piu'.
        let envChunk = ProcessInfo.processInfo.environment["DS4_PREFILL_CHUNK"].flatMap(Int.init)
        let step = max(1, envChunk ?? chunk)
        do {
            while start < tokens.count {
                let end = min(start + step, tokens.count)
                // Drain the ObjC autorelease pool per chunk: Metal command buffers /
                // encoders are autoreleased, and a long prefill inside one pool scope
                // accumulates them all — transient footprint grows with the prompt.
                let hiddens = try withPrefillAutoreleasePool {
                    try prefillRange(tokens, start: start, end: end, posBase: startPos)
                }
                lastHC = hiddens.last
                start = end
            }
        } catch {
            // The per-token path (n==1 chunks) commits its routed FFN async: a
            // cancellation/gather error must never escape with a cb in flight
            // over state the caller will tear down (same invariant as forward).
            drainFullLayerGather()
            drainFFN()
            throw error
        }
        drainFullLayerGather()   // by construction none survives a clean chunk — defensive
        drainFFN()   // don't hand a stale in-flight handle past the prefill
        profile.forwards += tokens.count
        return try outputHead(lastHC!)
    }

    /// Teacher-forced, layer-major prefill that reports the `k` highest-scoring
    /// predictions after selected INPUT positions. `scoreRange` is relative to
    /// `tokens` (not to the absolute KV positions): prediction `i` is produced
    /// after consuming `tokens[i]`.
    ///
    /// The normal prefill already keeps one final hidden state per token inside
    /// a chunk. This variant applies the output head only to the overlap with
    /// `scoreRange`, scans its shared logits buffer in place, and immediately
    /// discards it. It therefore avoids both decode-style token-by-token weight
    /// streaming and `scoredTokens x vocab` logits storage. The callback fires
    /// once per prefill chunk with a range aligned one-to-one with the nested
    /// candidate arrays.
    @discardableResult
    public func prefillTopK(
        tokens: [Int],
        startPos: Int = 0,
        chunk: Int = 512,
        k: Int,
        scoreRange: Range<Int>? = nil,
        onPredictions: (_ inputRange: Range<Int>, _ tokenIds: [[Int]]) throws -> Void = { _, _ in }
    ) throws -> [[Int]] {
        precondition(!tokens.isEmpty)
        precondition(k > 0, "prefillTopK: k deve essere positivo")
        precondition(startPos >= 0 && startPos + tokens.count <= maxKeys,
                     "prefillTopK: oltre il KV")
        let scored = scoreRange ?? tokens.indices
        precondition(scored.lowerBound >= tokens.startIndex
                     && scored.upperBound <= tokens.endIndex,
                     "prefillTopK: scoreRange fuori dagli input")

        if startPos == 0 {
            for c in compStates { try c?.reset(rt) }
            for c in indexStates { try c?.reset(rt) }
        }
        let envChunk = ProcessInfo.processInfo.environment["DS4_PREFILL_CHUNK"].flatMap(Int.init)
        let step = max(1, envChunk ?? chunk)
        var predictions: [[Int]] = []
        predictions.reserveCapacity(scored.count)
        var start = 0
        do {
            while start < tokens.count {
                try Task.checkCancellation()
                let end = min(start + step, tokens.count)
                let hiddens = try withPrefillAutoreleasePool {
                    try prefillRange(tokens, start: start, end: end, posBase: startPos)
                }
                let overlapStart = max(start, scored.lowerBound)
                let overlapEnd = min(end, scored.upperBound)
                if overlapStart < overlapEnd {
                    let overlap = overlapStart..<overlapEnd
                    var block: [[Int]] = []
                    block.reserveCapacity(overlap.count)
                    for inputIndex in overlap {
                        try Task.checkCancellation()
                        block.append(try outputHeadTopK(hiddens[inputIndex - start], k: k))
                    }
                    predictions.append(contentsOf: block)
                    try onPredictions(overlap, block)
                }
                start = end
            }
        } catch {
            // Match `prefill`: no command buffer may escape an interrupted
            // layer/output traversal while its tensors are being released.
            drainFullLayerGather()
            drainFFN()
            throw error
        }
        drainFullLayerGather()
        drainFFN()
        profile.forwards += tokens.count
        return predictions
    }

    /// Compatibility wrapper for callers that only need the greedy candidate.
    @discardableResult
    public func prefillTop1(
        tokens: [Int],
        startPos: Int = 0,
        chunk: Int = 512,
        scoreRange: Range<Int>? = nil,
        onPredictions: (_ inputRange: Range<Int>, _ tokenIds: [Int]) throws -> Void = { _, _ in }
    ) throws -> [Int] {
        var predictions: [Int] = []
        predictions.reserveCapacity(scoreRange?.count ?? tokens.count)
        _ = try prefillTopK(tokens: tokens, startPos: startPos, chunk: chunk, k: 1,
                           scoreRange: scoreRange) { range, candidates in
            let top1 = candidates.map { $0[0] }
            predictions.append(contentsOf: top1)
            try onPredictions(range, top1)
        }
        return predictions
    }

    /// Process one prompt chunk [start, end) layer-major at absolute positions
    /// posBase+start … . Weights for each layer are loaded once and applied to all
    /// the chunk's tokens (in order). On the expert-gather path the routed-FFN
    /// phase is BATCHED: each unique expert is gathered once per group instead of
    /// 6 per token. Returns every final HC state in the chunk: normal prefill
    /// retains only `.last`, while teacher forcing scores a selected subset.
    private func prefillRange(_ tokens: [Int], start: Int, end: Int,
                              posBase: Int) throws -> [GPUTensor] {
        // Size once for the last (largest) causal position in this chunk.  All
        // earlier tokens then reuse the same high-water allocation.
        try prepareLiveContext(nKeys: posBase + end)
        let n = end - start
        dsparkStage0Runtime?.beginCapture(position: posBase + end - 1)
        dsparkStage0Runtime?.beginBatchCapture(
            startPosition: posBase + start, nTokens: n)
        let hcDim = d.nHC * d.nEmbd
        // Slab-backed HC ping-pong: same per-token views as the historical
        // separate buffers, but the batched phase-B tail can write a whole
        // run's HC outputs with ONE hcExpand4 dispatch on the slab.
        var curPair = try PrefillStage.slabViews(rt, n: n, rowBytes: hcDim * 4, rowCount: hcDim)
        var otherPair = try PrefillStage.slabViews(rt, n: n, rowBytes: hcDim * 4, rowCount: hcDim)
        var cur = curPair.views
        var other = otherPair.views
        let chunkTokens = Array(tokens[start..<end])
        try embedTokensBatch(chunkTokens, into: cur)
        // Batched-attention staging: run capacity = the route batch, KV span
        // capacity = the chunk-end scratch high-water (raw window + comp rows)
        // plus the extra span rows a multi-token run adds (nq-1) and one pad
        // block of margin. prepareLiveContext already sized attentionRows for
        // this chunk's last position.
        let routeBatch = prefillRouteBatch
        let flashBatch: (nq: Int, maxKv: Int)? = (prefillBatchAttn && routeBatch >= 2)
            ? (nq: min(routeBatch, n), maxKv: scratch.attentionRows + routeBatch + 64) : nil
        // Full-layer phase B uses GLOBAL expert ids: the mm map buffers must
        // cover the whole expert set, not just the union cap.
        let stageUnion = (prefillFullLayer && n >= prefillFullLayerMin)
            ? max(maxUnionExperts, d.nExperts) : maxUnionExperts
        // Default ON after the Release WMO fix in batchedExpertLayer and the
        // slab-row ownership redesign. `=0` keeps the historical per-token
        // phase-B path available for parity/performance diagnostics.
        let expertBatchEnabled = ProcessInfo.processInfo.environment["DS4_PREFILL_EXPERT_BATCH"] != "0"
        let stage: PrefillStage? = (expertBatchEnabled && expertGather != nil && n > 1)
            ? try PrefillStage(rt, n: n, d: d, mmPath: prefillMM, maxUnion: stageUnion,
                               flashBatch: flashBatch) : nil
        // SPIA del percorso batchato: i gate del run flash hanno condizioni di
        // CAPACITÀ (ring raw, staging KV) che, se violate, non producono alcun
        // errore — degradano alla coda per-token e basta. È così che il ring a
        // nSWA secchi ha tenuto spento l'intero prefill batchato in GUI per
        // giorni a 1/3 della velocità, invisibile nei log. Qui il caso viene
        // diagnosticato UNA volta con il motivo, invece di restare muto.
        let flashRunsBefore = profile.prefillFlashRuns
        defer {
            if n >= 64, prefillBatchAttn, routeBatch >= 2,
               profile.prefillFlashRuns == flashRunsBefore, !warnedNoFlashRuns {
                warnedNoFlashRuns = true
                let rawRows = rawCaches.first.map { $0.count / d.headDim } ?? 0
                let needed = min(routeBatch, n) + d.nSWA - 1
                DS4Log.info("prefill", "percorso batchato MAI attivo su un chunk da \(n) token — "
                    + "ring raw \(rawRows) righe, servono \(needed) (route-batch \(routeBatch) + nSWA \(d.nSWA) - 1)"
                    + (rawRows > 0 && needed > rawRows
                       ? "  ⇒ il ring e' troppo corto: abbassa DS4_PREFILL_ROUTE_BATCH o allarga il ring"
                       : "  ⇒ causa diversa dalla capacita' del ring (quant densi Q4? staging KV?)"))
            }
        }
        for i in 0..<nLayers {
            // Per-layer pool drain: the layer weights and per-token command
            // buffers are autoreleased ObjC objects — without this they pile up
            // for the whole chunk instead of freeing at each EVICT.
            try withPrefillVoidAutoreleasePool {
                try Task.checkCancellation()
                prefillLayerProgress?(i, nLayers, n)     // progresso vivo per la GUI
                let w = try layerProvider(i)            // LOAD layer i ONCE for all chunk tokens
                if i + 1 < nLayers { prefetch?(i + 1) }   // read-ahead next layer (overlaps its I/O)
                let layerRope = ropeParams(layer: i)
                if let gather = expertGather, n > 1, let stage {
                    try batchedExpertLayer(i, w: w, layerRope: layerRope, cur: cur, other: other,
                                           otherSlab: otherPair.slab,
                                           n: n, posBase: posBase + start, tokens: chunkTokens,
                                           gather: gather, stage: stage)
                } else {
                    for j in 0..<n {
                        let pos = posBase + start + j     // attends KV[0..pos] (incl. earlier chunks/turns)
                        try runLayer(i, w: w, layerRope: layerRope, cur: cur[j], other: other[j],
                                     pos: pos, nKeys: pos + 1, token: chunkTokens[j])
                    }
                }
                swap(&cur, &other)                       // w drops here -> EVICT
                swap(&curPair, &otherPair)
                if let dspark = dsparkStage0Runtime {
                    do {
                        try dspark.capture(layer: i, hiddenHC: cur[n - 1])
                        try dspark.captureBatch(
                            layer: i, hiddenHC: curPair.slab, nTokens: n)
                    }
                    catch { disableDSparkAfterFailure(error) }
                }
            }
        }
        if let dspark = dsparkStage0Runtime {
            do {
                _ = try dspark.finishCapture()
                _ = dspark.finishBatchCapture()
            }
            catch { disableDSparkAfterFailure(error) }
        }
        // The alternate ping-pong side can be released immediately. `cur`
        // leaves this helper so the caller can select the last hidden (normal
        // prefill) or score multiple positions (teacher forcing).
        other.removeAll(keepingCapacity: false)
        return cur
    }

    /// Fase B self-speculative (docs/SELF-SPECULATIVE.md): UN passo batch
    /// full-config sui K token candidati con logit PER POSIZIONE — lo stesso
    /// giro layer-major del prefill (pesi di ogni layer caricati una volta,
    /// unione degli esperti dedupata da batchedExpertLayer), ma l'output head
    /// è applicato a OGNI hidden finale (~8 ms/token: trascurabile a K ≤ 8).
    /// Scrive KV raw e stato compressori full-config per le posizioni
    /// startPos..<startPos+K; il chiamante gestisce snapshot/restore attorno
    /// al round (specSnapshot/specRestore) e il rollback dei rifiutati.
    /// Mai a inizio sequenza (il prompt passa dal prefill): startPos > 0.
    public func specVerifyStep(tokens: [Int], startPos: Int) throws -> [[Float]] {
        precondition(!tokens.isEmpty && startPos > 0)
        precondition(startPos + tokens.count <= maxKeys, "specVerifyStep: oltre il KV")
        try prepareLiveContext(nKeys: startPos + tokens.count)
        let n = tokens.count
        let hcDim = d.nHC * d.nEmbd
        var cur: [GPUTensor] = try (0..<n).map { _ in try .zeros(rt, floatCount: hcDim) }
        var other: [GPUTensor] = try (0..<n).map { _ in try .zeros(rt, floatCount: hcDim) }
        try embedTokensBatch(tokens, into: cur)
        dsparkStage0Runtime?.beginCapture(position: startPos + n - 1)
        dsparkStage0Runtime?.beginBatchCapture(
            startPosition: startPos, nTokens: n)
        // Fase V1 (DS4_SPEC_VERIFY_BATCH, default on): dove possibile la route/
        // attention dell'INTERA finestra va in UN command buffer per layer
        // (encodeRouteInto + snapshot blit, la fase A del prefill batchato — una
        // sync per layer invece di una per token) e la FFN routed è servita
        // dalla SLOT-CACHE per token (hit dal pool, niente unione da disco: il
        // motivo per cui questa funzione evitava batchedExpertLayer). Stessi
        // dispatch nello stesso ordine per token -> numerica identica al
        // percorso per-token; i layer non idonei (indexer attivo nella
        // finestra, quant fuori classe, activeExperts ridotti) ricadono sul
        // giro per-token storico.
        let expertBatchEnabled = ProcessInfo.processInfo.environment["DS4_PREFILL_EXPERT_BATCH"] != "0"
        let stage: PrefillStage? = (expertBatchEnabled && specVerifyBatch && n > 1 && remoteExperts == nil
                                    && expertGather != nil && slotCache != nil
                                    && !profileRoute && d.activeExperts >= d.k)
            ? try PrefillStage(rt, n: n, d: d, mmPath: false, maxUnion: d.k) : nil
        do {
            for i in 0..<nLayers {
                try withPrefillVoidAutoreleasePool {
                    try Task.checkCancellation()
                    let w = try layerProvider(i)
                    if i + 1 < nLayers { prefetch?(i + 1) }
                    let layerRope = ropeParams(layer: i)
                    if let stage, let cache = slotCache,
                       ((w.gateQuant == d.gateQuant && w.upQuant == d.upQuant && w.downQuant == d.downQuant)
                        || cache.supports(layer: i)),
                       specWindowIndexerInactive(i, startPos: startPos, n: n) {
                        try specVerifyBatchedLayer(i, w: w, layerRope: layerRope, cur: cur, other: other,
                                                   n: n, startPos: startPos, tokens: tokens,
                                                   cache: cache, stage: stage)
                    } else {
                        // Percorso per-token storico (runLayer): resta il
                        // riferimento di parità e il fallback dei layer non
                        // batchabili.
                        for j in 0..<n {
                            let pos = startPos + j
                            try runLayer(i, w: w, layerRope: layerRope, cur: cur[j], other: other[j],
                                         pos: pos, nKeys: pos + 1, token: tokens[j])
                        }
                    }
                    swap(&cur, &other)
                    if let dspark = dsparkStage0Runtime {
                        do {
                            try dspark.capture(layer: i, hiddenHC: cur[n - 1])
                            try dspark.captureBatchRows(layer: i, hiddenHC: cur)
                        } catch {
                            disableDSparkAfterFailure(error)
                        }
                    }
                }
            }
        } catch {
            drainFFN()   // stesso invariante di prefill: mai un cb in volo oltre l'errore
            dsparkStage0Runtime?.abortCapture()
            dsparkStage0Runtime?.abortBatchCapture()
            throw error
        }
        drainFFN()
        if let dspark = dsparkStage0Runtime {
            do {
                _ = try dspark.finishCapture()
                _ = dspark.finishBatchCapture()
            } catch {
                disableDSparkAfterFailure(error)
            }
        }
        profile.forwards += n
        return try cur.map { try outputHead($0) }
    }

    /// La finestra speculativa può usare encodeRouteInto su questo layer?
    /// (richiede indexer INATTIVO per ogni posizione, controllato
    /// prospetticamente come nel prefill batchato.)
    private func specWindowIndexerInactive(_ i: Int, startPos: Int, n: Int) -> Bool {
        var extra = 0
        for j in 0..<n {
            let pos = startPos + j
            if indexerActive(i, pos: pos, extraRows: extra) { return false }
            if let idx = indexStates[i], (pos + 1) % idx.ratio == 0 { extra += 1 }
        }
        return true
    }

    /// Un layer di verifica speculativa in forma batchata. Fase A: le route dei
    /// token della finestra (causali: il token j vede il KV scritto da 0..j in
    /// QUESTO layer) encodate in UN command buffer, con lo snapshot per-token
    /// di FFN input + selezione blit-copiato prima che il token successivo
    /// sovrascriva lo scratch — una sync per layer. Fase B: FFN condivisa +
    /// routed per token, esperti dalla SLOT-CACHE. Le FFN sono serializzate al
    /// confine di token (join del c2 precedente prima dell'acquire): acquire
    /// back-to-back sullo stesso layer con FFN in volo potrebbe evictare slot
    /// ancora letti dalla GPU — un hazard che il percorso per-token non ha mai
    /// esposto alla cache. La shared FFN (commit async) copre la bolla.
    private func specVerifyBatchedLayer(_ i: Int, w: LayerWeights, layerRope: RopeParams,
                                        cur: [GPUTensor], other: [GPUTensor], n: Int,
                                        startPos: Int, tokens: [Int],
                                        cache: ExpertSlotCache, stage: PrefillStage) throws {
        if remoteExperts == nil { kickLookahead(after: i, token: tokens[0]) }
        // Fase A: route batchata (stessi encode, stesso ordine del per-token).
        var t = Date()
        clearMaskIfDirty()
        let c = GraphContext(rt); try c.begin()
        for j in 0..<n {
            let pos = startPos + j
            try encodeRouteInto(c, i, w: w, layerRope: layerRope, curHc: cur[j],
                                pos: pos, nKeys: pos + 1, token: tokens[j])
            try c.blitCopies([
                (scratch.cur, 0, stage.cur[j], 0, d.nEmbd * 4),
                (scratch.afterAttn, 0, stage.attn[j], 0, d.nHC * d.nEmbd * 4),
                (scratch.split, 0, stage.split[j], 0, 24 * 4),
                (scratch.selected, 0, stage.ids[j], 0, d.k * 4),
                (scratch.rw, 0, stage.rw[j], 0, d.k * 4),
            ])
        }
        c.commit()               // UNA sync per layer per l'intera finestra
        profile.routeS += Date().timeIntervalSince(t)
        // Fase B: FFN per token dalla slot-cache. stage.ids[j] viene RISCRITTO
        // con gli SLOT del pool dopo la lettura CPU della selezione (buffer
        // per-token: nessun aliasing né race — lo stesso schema del prefill).
        var prev: GraphContext?
        for j in 0..<n {
            let (ids, _) = selection(sel: stage.ids[j], weights: stage.rw[j], layer: i)
            // Shared FFN subito (non dipende dalla selezione): sulla queue
            // in-order esegue dopo il c2 del token precedente, e copre con
            // GPU utile l'attesa/acquire CPU qui sotto.
            let c1 = GraphContext(rt)
            do { try c1.begin(); try c1.decodeSharedFFN(w: w, s: scratch, d: d, cur: stage.cur[j]) }
            catch { prev?.waitCompleted(); throw error }
            c1.commitAsync()
            // Join del token precedente PRIMA dell'acquire: il suo c2 legge
            // ancora il pool di questo layer.
            prev?.waitCompleted()
            t = Date()
            let h0 = cache.hits, m0 = cache.misses, w0 = cache.warmed, p0 = cache.prefilled
            let hb0 = cache.hitBytes, mb0 = cache.missBytes
            let wb0 = cache.warmedBytes, pb0 = cache.prefilledBytes
            let acquired: (pool: ExpertSlotCache.LayerPool, slots: [Int32],
                           lease: ExpertSlotCache.GPUReadLease)
            do { acquired = try cache.acquireLeased(layer: i, ids: ids) }
            catch { c1.waitCompleted(); throw error }
            let (pool, slots, _) = acquired
            let poolLease = acquired.lease
            profile.gatherS += Date().timeIntervalSince(t)
            profile.expertHits += cache.hits - h0
            profile.expertMisses += cache.misses - m0
            profile.expertWarmed += cache.warmed - w0
            profile.expertPrefilled += cache.prefilled - p0
            profile.expertHitBytes += cache.hitBytes - hb0
            profile.expertMissBytes += cache.missBytes - mb0
            profile.expertWarmedBytes += cache.warmedBytes - wb0
            profile.expertPrefilledBytes += cache.prefilledBytes - pb0
            profile.gatherBytes += (cache.missBytes - mb0) + (cache.warmedBytes - wb0)
            _ = slots.withUnsafeBytes {
                memcpy(stage.ids[j].buffer.contents() + stage.ids[j].byteOffset,
                       $0.baseAddress!, $0.count)
            }
            t = Date()
            let c2 = GraphContext(rt)
            do {
                try c2.begin()
                try c2.decodeRoutedExperts(w: w, s: scratch, d: d, gateExp: pool.gate,
                                           upExp: pool.up, downExp: pool.down,
                                           ids: stage.ids[j], outHc: other[j],
                                           cur: stage.cur[j], afterAttn: stage.attn[j],
                                           split: stage.split[j], rw: stage.rw[j],
                                           expertStride: pool.expertStride ?? slotCacheStride)
            } catch {
                poolLease.release()
                c1.waitCompleted()
                throw error
            }
            c2.onComplete { poolLease.release() }
            commitFFN(c2)
            prev = c2
            profile.expertsS += Date().timeIntervalSince(t)
            profile.layers += 1
        }
    }

    /// Max experts gathered per group in the batched prefill (bounds the packed
    /// union tensors' transient memory: ~7 MB/expert on the 2-bit model). Env
    /// override: DS4_PREFILL_UNION. Never below d.k.
    ///
    /// Default 192, misurato su M1 Pro: ogni gruppo rilegge la SUA unione dal
    /// disco (con DS4_EXPERT_PREAD il F_NOCACHE esclude la page cache), quindi
    /// i byte/token del prefill scalano ~ union/tokens-per-gruppo. A 64 il
    /// gather leggeva ~1.7 GB/token (≈257 esperti!) saturando l'SSD; a 192 i
    /// gruppi coprono ~3× piu' token a parita' di unione. Costo: ~1.3 GB per
    /// tensore packed × 2 (pipeline) di memoria transiente — su macchine
    /// strette abbassare via env.
    private var maxUnionExperts: Int {
        let v = ProcessInfo.processInfo.environment["DS4_PREFILL_UNION"].flatMap(Int.init) ?? 192
        return max(d.k, v)
    }

    /// One prefill layer over all chunk tokens with BATCHED expert I/O.
    /// Phase A — routes run sequentially per token (attention is causal: token j
    /// attends KV written by tokens 0..j in this same layer), saving each token's
    /// FFN inputs (attn-normed cur, residual, HC split) and its expert selection.
    /// Phase B — tokens are grouped; each group's UNION of selected experts is
    /// gathered ONCE and every token's FFN runs over it with remapped ids.
    /// Numerically identical to the per-token path (a token's FFN does not feed
    /// other tokens within the layer); only the expert I/O is deduplicated:
    /// ≤ min(6·tokens, 256) expert reads per layer instead of 6·tokens.
    // Swift 6.3 WMO/LICM miscompiles the Objective-C-backed tensor temporaries
    // in this very large mixed CPU/Metal routine (Release-only objc_retain at
    // 0x100000020; Debug and the per-token fallback are stable). The actual GPU
    // kernels remain fully optimized; this only keeps Swift's command-encoding
    // orchestration at the reliable optimization level.
    @_optimize(none)
    private func batchedExpertLayer(_ i: Int, w: LayerWeights, layerRope: RopeParams,
                                    cur: [GPUTensor], other: [GPUTensor], otherSlab: GPUTensor,
                                    n: Int, posBase: Int,
                                    tokens: [Int],
                                    gather: @escaping (Int, [Int32]) throws -> (GPUTensor, GPUTensor, GPUTensor),
                                    stage: PrefillStage) throws {
        // Phase A: routes. Attention is causal WITHIN the layer (token j attends
        // KV written by tokens 0..j), so the routes stay token-SEQUENTIAL — but
        // they don't need a CPU round-trip each: runs of prefillRouteBatch tokens
        // are encoded into ONE command buffer, each token's scratch snapshot
        // (FFN inputs + router selection) blit-copied GPU-side before the next
        // token overwrites it, and the CPU reads all the selections after a
        // single wait. Indexer-active tokens (CPU top-k mid-route) and
        // DS4_PROFILE_ROUTE fall back to the per-token path.
        // Snapshot once per layer. ProcessInfo.environment materializes a new
        // dictionary, so never query it in the per-token loop below.
        let routeBatch = prefillRouteBatch
        var idsT: [[Int32]] = [], rwT: [[Float]] = []
        idsT.reserveCapacity(n); rwT.reserveCapacity(n)
        // LEVA 8: with the full-layer phase B, selections can stay GPU-resident
        // (stage.ids/rw slabs are already in mul_mm_id's token-major layout) —
        // no per-run CPU readback, no restaging, no snapshot blits (slabDirect).
        // Requires top-k routing at full k (distinct ids by construction) and
        // an indexer that can NOT activate inside this chunk (the per-token
        // fallback would bypass the stage.ids writes).
        let fullLayer = prefillFullLayer && n >= prefillFullLayerMin
        let slabDirect = fullLayer && prefillDenseMM && prefillDenseEligible(w)
        let mmOKLayer = prefillMM && stage.mm != nil
            && w.gateQuant == .iq2_xxs && w.upQuant == .iq2_xxs && w.downQuant == .q2_K
            && d.k == 6 && d.nEmbd % 256 == 0 && d.expertFfn % 256 == 0
        let mayIndexer = indexStates[i] != nil
            && indexerActive(i, pos: posBase + n, extraRows: n / 4 + 2)
        let residentIds = prefillResidentIds && fullLayer && mmOKLayer
            && d.activeExperts >= d.k && !mayIndexer
        // ASYNC phase-A pipeline (depth 1): run r's cb is committed WITHOUT a
        // wait; run r+1 is encoded while r executes on the GPU; r's selections
        // are read only at the drain. The CPU mask fill alternates buffers by
        // parity (the only CPU-written resource); everything else is GPU-side
        // and the in-order queue serializes it. The C engine encodes the whole
        // chunk up front — this recovers the same overlap run by run.
        var pendingRun: (ctx: GraphContext, range: Range<Int>)?
        var maskParity = true
        func drainPendingRun() {
            guard let p = pendingRun else { return }
            let tw = Date()
            p.ctx.waitCompleted()
            profile.routeS += Date().timeIntervalSince(tw)
            if residentIds {
                profile.layers += p.range.count   // selections stay on the GPU
            } else {
                for jj in p.range {
                    let (ids, rw) = selection(sel: stage.ids[jj], weights: stage.rw[jj], layer: i)
                    idsT.append(ids); rwT.append(rw)
                    profile.layers += 1
                }
            }
            pendingRun = nil
        }
        // Error paths must never leave a run's cb in flight over state the
        // caller tears down (same invariant as drainFFN).
        defer { pendingRun?.ctx.waitCompleted() }
        var j = 0
        while j < n {
            // Extent of the batchable run starting at j: consecutive tokens for
            // which the indexer stays INACTIVE — its compressed-row count grows
            // deterministically with pos, so activation is checked prospectively
            // (extraRows) for the whole run before encoding any of it.
            var jEnd = j
            if routeBatch > 1 && !profileRoute {
                var extraRows = 0
                while jEnd < n && (jEnd - j) < routeBatch {
                    let pos = posBase + jEnd
                    if indexerActive(i, pos: pos, extraRows: extraRows) { break }
                    if let idx = indexStates[i], (pos + 1) % idx.ratio == 0 { extraRows += 1 }
                    jEnd += 1
                }
            }
            if jEnd <= j, !profileRoute, !profilePrefill, gpuIndexerTopK, d.indexedAttn,
               prefillDenseMM, prefillDenseEligible(w),
               let fb = stage.flash, compStates[i] != nil, indexStates[i] != nil,
               w.idxQB != nil, w.idxProj != nil {
                // LEVA 9 v2: run batchato con indexer ATTIVO — l'attivazione è
                // monotona in pos (da qui in poi tutto il chunk è idoneo);
                // l'estensione rispetta capacità del run, ring raw e righe di
                // score per query. Niente maschera CPU né staging F16: la
                // pipeline async non alterna la parità della mask.
                var iEnd = min(n, j + max(2, min(routeBatch, fb.nq)))
                let rawRowsI = rawCaches[i].count / d.headDim
                while iEnd > j + 1 {
                    let rawLo0i = max(0, posBase + j + 1 - d.nSWA)
                    let span = posBase + iEnd - rawLo0i
                    let scoreBound = (indexStates[i]?.count ?? 0) + (iEnd - j) / 4 + 2
                    if span <= rawRowsI && scoreBound <= fb.maxKv { break }
                    iEnd -= 1
                }
                if iEnd - j >= 2 {
                    let t = Date()
                    let ctx = try withPrefillAutoreleasePool { () -> GraphContext in
                        try Task.checkCancellation()
                        return try encodeIndexedFlashRun(fb, i, w: w, layerRope: layerRope,
                                                         cur: cur, stage: stage,
                                                         j: j, jEnd: iEnd, posBase: posBase,
                                                         tokens: tokens, slabDirect: slabDirect)
                    }
                    ctx.commitAsync()
                    profile.routeS += Date().timeIntervalSince(t)
                    drainPendingRun()               // r-1: GPU done under our encoding
                    pendingRun = (ctx, j..<iEnd)
                    j = iEnd
                    continue
                }
            }
            if jEnd <= j, profileRoute || !gpuIndexerTopK {
                // Per-token path (diagnostica DS4_PROFILE_ROUTE): reads and
                // writes the shared scratch CPU-side — join the pipeline first.
                drainPendingRun()
                try withPrefillVoidAutoreleasePool {
                    try Task.checkCancellation()
                    let pos = posBase + j
                    let t = Date()
                    let route = try encodeRoute(i, w: w, layerRope: layerRope, curHc: cur[j], pos: pos, nKeys: pos + 1,
                                                token: tokens[j])
                    // Fallback per-token: selezione E scratch (cur/afterAttn/split)
                    // letti CPU-side subito sotto — semantica bloccante come prima.
                    route.waitCompleted()
                    profile.routeS += Date().timeIntervalSince(t)
                    let (ids, rw) = readRouteSelection(layer: i)
                    idsT.append(ids); rwT.append(rw)
                    copyFloats(from: scratch.cur, to: stage.cur[j], count: d.nEmbd)
                    copyFloats(from: scratch.afterAttn, to: stage.attn[j], count: d.nHC * d.nEmbd)
                    copyFloats(from: scratch.split, to: stage.split[j], count: 24)
                    if let mm = stage.mm {
                        memcpy(mm.curMat.buffer.contents() + mm.curMat.byteOffset + j * d.nEmbd * 4,
                               scratch.cur.buffer.contents() + scratch.cur.byteOffset, d.nEmbd * 4)
                    }
                    profile.layers += 1
                }
                j += 1
                continue
            } else if jEnd <= j {
                // LEVA 9 v1 — coda indexer-attiva a GRUPPI (l'attivazione è
                // monotona in pos: da qui in poi tutto il chunk è per-token).
                // Stessi encode nello stesso ordine del fallback storico, ma
                // R token in UN command buffer con snapshot blit negli slab
                // (lo schema di specVerifyBatchedLayer): la sync per token
                // diventa una sync per gruppo. Il route per token dentro il
                // cb è autonomo (mask_one o ramo indicizzato riscrivono per
                // intero il loro scratch; l'in-order queue serializza).
                drainPendingRun()
                let gEnd = min(n, j + max(1, routeBatch))
                try withPrefillVoidAutoreleasePool {
                    try Task.checkCancellation()
                    let t = Date()
                    clearMaskIfDirty()
                    let c = GraphContext(rt); try c.begin()
                    for jj in j..<gEnd {
                        let pos = posBase + jj
                        try encodeRouteGroupedInto(c, i, w: w, layerRope: layerRope, curHc: cur[jj],
                                                   pos: pos, nKeys: pos + 1, token: tokens[jj])
                        var blits: [(GPUTensor, Int, GPUTensor, Int, Int)] = [
                            (scratch.cur, 0, stage.cur[jj], 0, d.nEmbd * 4),
                            (scratch.afterAttn, 0, stage.attn[jj], 0, d.nHC * d.nEmbd * 4),
                            (scratch.split, 0, stage.split[jj], 0, 24 * 4),
                            (scratch.selected, 0, stage.ids[jj], 0, d.k * 4),
                            (scratch.rw, 0, stage.rw[jj], 0, d.k * 4),
                        ]
                        if let mm = stage.mm {
                            blits.append((scratch.cur, 0, mm.curMat, jj * d.nEmbd * 4, d.nEmbd * 4))
                        }
                        try c.blitCopies(blits)
                    }
                    c.commit()               // UNA sync per l'intero gruppo
                    profile.routeS += Date().timeIntervalSince(t)
                    for jj in j..<gEnd {
                        let (ids, rw) = selection(sel: stage.ids[jj], weights: stage.rw[jj], layer: i)
                        idsT.append(ids); rwT.append(rw)
                        profile.layers += 1
                    }
                }
                j = gEnd
                continue
            }
            let t = Date()
            // Batched multi-query attention (DS4_PREFILL_BATCH_ATTN): eligible
            // when the run fits the staging capacity AND the union raw span
            // [first window start, last pos] is fully resident — with the
            // DS4_RAW_RING cache the newest stores overwrite the oldest rows,
            // so a span longer than the ring must use the per-token path
            // (each token attends BEFORE later tokens overwrite its window).
            let nqRun = jEnd - j
            let posFirst = posBase + j
            let rawLo0 = max(0, posFirst + 1 - d.nSWA)
            let nRawSpan = posBase + jEnd - rawLo0
            let rawRows = rawCaches[i].count / d.headDim
            let compBound = (compStates[i]?.count ?? 0) + nqRun / max(1, compStates[i]?.ratio ?? 4) + 1
            // The flash run pays for itself through the dense GEMMs (measured:
            // attention-only batching LOSES to the historical packed loop).
            // Default: take it only where the dense path applies; layers with
            // Q4-requantized dense weights use the historical loop. With
            // DS4_PREFILL_DENSE_MM=0 the lever-1 mode stays reachable for A/B.
            if let fb = stage.flash, nqRun >= 2, nqRun <= fb.nq,
               nRawSpan <= rawRows, nRawSpan + compBound <= fb.maxKv,
               prefillDenseEligible(w) || !prefillDenseMM {
                let ctx = try withPrefillAutoreleasePool { () -> GraphContext in
                    try Task.checkCancellation()
                    return try encodeFlashRun(fb, i, w: w, layerRope: layerRope, cur: cur, stage: stage,
                                              j: j, jEnd: jEnd, posBase: posBase, tokens: tokens,
                                              maskParity: maskParity, slabDirect: slabDirect)
                }
                if profilePrefill {
                    // Profiling: synchronous, phases accumulated per run.
                    ctx.commit()
                    accumulateRoutePhases(ctx, nil)
                    profile.routeS += Date().timeIntervalSince(t)
                    for jj in j..<jEnd {
                        let (ids, rw) = selection(sel: stage.ids[jj], weights: stage.rw[jj], layer: i)
                        idsT.append(ids); rwT.append(rw)
                        profile.layers += 1
                    }
                    j = jEnd
                    continue
                }
                ctx.commitAsync()
                maskParity.toggle()
                profile.routeS += Date().timeIntervalSince(t)
                drainPendingRun()               // r-1: GPU done under our encoding
                pendingRun = (ctx, j..<jEnd)
                j = jEnd
                continue
            } else {
                drainPendingRun()
                try withPrefillVoidAutoreleasePool {
                    try Task.checkCancellation()
                    clearMaskIfDirty()
                    let c = GraphContext(rt); try c.begin()
                    for jj in j..<jEnd {
                        let pos = posBase + jj
                        try encodeRouteInto(c, i, w: w, layerRope: layerRope, curHc: cur[jj],
                                            pos: pos, nKeys: pos + 1, token: tokens[jj])
                        // Snapshot ids/weights into stage.ids/rw too: phase B reads
                        // them back and REWRITES both buffers (remapped + padded)
                        // strictly after this buffer completes — no aliasing.
                        var copies: [(src: GPUTensor, srcOff: Int, dst: GPUTensor, dstOff: Int, bytes: Int)] = [
                            (scratch.cur, 0, stage.cur[jj], 0, d.nEmbd * 4),
                            (scratch.afterAttn, 0, stage.attn[jj], 0, d.nHC * d.nEmbd * 4),
                            (scratch.split, 0, stage.split[jj], 0, 24 * 4),
                            (scratch.selected, 0, stage.ids[jj], 0, d.k * 4),
                            (scratch.rw, 0, stage.rw[jj], 0, d.k * 4),
                        ]
                        if let mm = stage.mm {
                            copies.append((scratch.cur, 0, mm.curMat, jj * d.nEmbd * 4, d.nEmbd * 4))
                        }
                        try c.blitCopies(copies)
                    }
                    c.commit()
                }
            }
            profile.routeS += Date().timeIntervalSince(t)
            for jj in j..<jEnd {
                let (ids, rw) = selection(sel: stage.ids[jj], weights: stage.rw[jj], layer: i)
                idsT.append(ids); rwT.append(rw)
                profile.layers += 1
            }
            j = jEnd
        }
        drainPendingRun()   // last run of the layer: phase B needs every selection

        // ── Phase B, FULL-LAYER variant (DS4_PREFILL_FULL_LAYER): with a long
        // chunk, stream the WHOLE routed layer once and run every token
        // against it with GLOBAL ids — no unions, no remap, sequential I/O,
        // and the next layer's slab reads under this layer's FFNs. The C
        // engine's prefill shape (its "prefill expert reserve").
        if fullLayer {
            let actMat = slabDirect ? stage.curSlab : stage.mm?.curMat
            try fullLayerPhaseB(i, w: w, other: other, otherSlab: otherSlab,
                                n: n, gather: gather,
                                stage: stage, idsT: idsT, rwT: rwT,
                                residentIds: residentIds, actMat: actMat)
            return
        }

        // Phase B: group consecutive tokens while the union stays under the cap,
        // gather each group's union once, run every token's FFN with remapped ids.
        let cap = maxUnionExperts
        var groups: [(tokens: Range<Int>, union: [Int32])] = []
        var j0 = 0
        while j0 < n {
            var union: [Int32] = []
            var seen = Set<Int32>()
            var j1 = j0
            while j1 < n {
                let fresh = idsT[j1].filter { !seen.contains($0) }
                if !union.isEmpty && union.count + fresh.count > cap { break }
                for id in fresh { union.append(id); seen.insert(id) }
                j1 += 1
            }
            groups.append((tokens: j0..<j1, union: union))
            j0 = j1
        }

        // PIPELINE: every group's union is known up front (phase A did all the
        // routes), so group g+1's expert I/O runs on a background queue WHILE
        // group g's FFNs run on the GPU. Deterministic — no speculation: we
        // read exactly the experts the router selected. The background work
        // touches only the read-only mmap and creates fresh Metal buffers
        // (MTLDevice is thread-safe) — disjoint from the FFN scratch.
        let bg = PrefillGather(layer: i, gather: gather)
        var pending: PrefillGather.Pending? = nil
        defer { pending?.join() }   // never leave a background gather running on error/cancel
        for (gi, group) in groups.enumerated() {
            try withPrefillVoidAutoreleasePool {
                var t = Date()
                let g: GPUTensor, u: GPUTensor, dn: GPUTensor
                if let p = pending {
                    pending = nil
                    (g, u, dn) = try p.wait()   // residual only: the I/O ran during the previous group's FFNs
                } else {
                    (g, u, dn) = try gather(i, group.union)   // first group: nothing to overlap yet
                }
                profile.gatherS += Date().timeIntervalSince(t)   // EXPOSED (non-overlapped) I/O time
                profile.gatherBytes += g.byteLength + u.byteLength + dn.byteLength
                if gi + 1 < groups.count { pending = bg.start(groups[gi + 1].union) }
                var posOf: [Int32: Int32] = [:]
                for (p, id) in group.union.enumerated() { posOf[id] = Int32(p) }
                // mul_mm_id path (DS4_PREFILL_MM): expert weights read once per
                // tile for ALL the group's tokens. Requirements: Flash quants,
                // full k DISTINCT selections per token (map0 encodes the slot
                // as a sum over matches), dims multiple of 256, and enough
                // tokens to amortize the matmul setup.
                let gTok = group.tokens.count
                let useMM = prefillMM && stage.mm != nil && gTok >= 8
                    && w.gateQuant == .iq2_xxs && w.upQuant == .iq2_xxs && w.downQuant == .q2_K
                    && d.k == 6 && d.nEmbd % 256 == 0 && d.expertFfn % 256 == 0
                    && group.tokens.allSatisfy { idsT[$0].count == d.k }
                if useMM, let mm = stage.mm {
                    // CPU staging BEFORE the command buffer: group-local rows of
                    // remapped (union-relative) ids + route weights.
                    let idsPtr = (mm.idsMat.buffer.contents() + mm.idsMat.byteOffset)
                        .bindMemory(to: Int32.self, capacity: gTok * d.k)
                    let wPtr = (mm.wMat.buffer.contents() + mm.wMat.byteOffset)
                        .bindMemory(to: Float.self, capacity: gTok * d.k)
                    for (tl, j) in group.tokens.enumerated() {
                        for s in 0..<d.k {
                            idsPtr[tl * d.k + s] = posOf[idsT[j][s]]!
                            wPtr[tl * d.k + s] = rwT[j][s]
                        }
                    }
                    try Task.checkCancellation()
                    t = Date()
                    let c2 = GraphContext(rt); try c2.begin()
                    try c2.encodeMoEMap0(ids: mm.idsMat, htpe: mm.htpe, hids: mm.hids,
                                         nTok: gTok, kPerTok: d.k, nExperts: group.union.count)
                    try c2.encodeMMIdPairSwiGLUIQ2(gate: g, up: u, act: mm.curMat,
                                                   actBase: group.tokens.lowerBound * d.nEmbd * 4,
                                                   htpe: mm.htpe, hids: mm.hids,
                                                   mid: mm.mid, weights: mm.wMat,
                                                   nTok: gTok, kPerTok: d.k,
                                                   nExperts: group.union.count,
                                                   inDim: d.nEmbd, ffnDim: d.expertFfn,
                                                   clamp: d.swigluClamp)
                    try c2.encodeMMIdDownQ2K(down: dn, mid: mm.mid,
                                             htpe: mm.htpe, hids: mm.hids, out: mm.down6,
                                             nTok: gTok, kPerTok: d.k,
                                             nExperts: group.union.count,
                                             ffnDim: d.expertFfn, outDim: d.nEmbd)
                    // SHARED-expert FFN, batched per-quant (see the helper).
                    let sharedMM = try encodeSharedFFNBatched(
                        c2, w: w, mm: mm,
                        actBase: group.tokens.lowerBound * d.nEmbd * 4, gTok: gTok)
                    if sharedMM {
                        // Batched tail: sum6 + shared add + HC expand per run.
                        try encodeRoutedTailBatched(c2, mm: mm, stage: stage,
                                                    otherSlab: otherSlab,
                                                    tokens: group.tokens)
                    } else {
                        // Per-token shared fallback (Q4 dims ruled it out).
                        let tokBytes = d.k * d.nEmbd * 4
                        for (tl, j) in group.tokens.enumerated() {
                            try c2.decodeSharedFFN(w: w, s: scratch, d: d, cur: stage.cur[j])
                            try c2.blitCopies([(src: mm.down6, srcOff: tl * tokBytes,
                                                dst: scratch.down6, dstOff: 0, bytes: tokBytes)])
                            try c2.decodeRoutedTail(s: scratch, d: d, outHc: other[j],
                                                    afterAttn: stage.attn[j], split: stage.split[j])
                        }
                    }
                    c2.commit()
                    profile.expertsS += Date().timeIntervalSince(t)
                } else if prefillFFNBatch {
                    // ONE command buffer for the whole group's FFNs. All the
                    // CPU staging happens BEFORE the commit (per-token ids/rw
                    // buffers — the shared s.rw can't be rewritten between
                    // tokens of one buffer). Selections shorter than k
                    // (DS4_ACTIVE_EXPERTS) are padded with slot 0 at weight 0:
                    // SwiGLU scales the padded rows by 0, so their down
                    // projection contributes exactly zero — same numerics.
                    for j in group.tokens {
                        var remapped = idsT[j].map { posOf[$0]! }
                        var weights = rwT[j]
                        while remapped.count < d.k { remapped.append(0); weights.append(0) }
                        let idsRow = stage.ids[j]
                        let rwRow = stage.rw[j]
                        remapped.withUnsafeBytes {
                            _ = memcpy(
                                idsRow.buffer.contents() + idsRow.byteOffset,
                                $0.baseAddress!, $0.count)
                        }
                        writeFloats(weights, into: rwRow)
                    }
                    try Task.checkCancellation()
                    t = Date()
                    let c2 = GraphContext(rt); try c2.begin()
                    for j in group.tokens {
                        let idsRow = stage.ids[j]
                        let curRow = stage.cur[j]
                        let attnRow = stage.attn[j]
                        let splitRow = stage.split[j]
                        let rwRow = stage.rw[j]
                        try c2.decodeExperts(w: w, s: scratch, d: d, gateExp: g, upExp: u, downExp: dn,
                                             ids: idsRow, outHc: other[j], activeK: d.k,
                                             cur: curRow, afterAttn: attnRow,
                                             split: splitRow, rw: rwRow)
                    }
                    c2.commit()
                    profile.expertsS += Date().timeIntervalSince(t)
                } else {
                    for j in group.tokens {
                        try Task.checkCancellation()
                        let K = idsT[j].count
                        let remapped = idsT[j].map { posOf[$0]! }
                        let idsBuf = try GPUTensor.bytes(rt, remapped.withUnsafeBytes { Array($0) },
                                                         elementCount: K)
                        writeFloats(rwT[j], into: scratch.rw)
                        zeroDown6(from: K)
                        t = Date()
                        let c2 = GraphContext(rt); try c2.begin()
                        try c2.decodeExperts(w: w, s: scratch, d: d, gateExp: g, upExp: u, downExp: dn,
                                             ids: idsBuf, outHc: other[j], activeK: K,
                                             cur: stage.cur[j], afterAttn: stage.attn[j], split: stage.split[j])
                        c2.commit()
                        profile.expertsS += Date().timeIntervalSince(t)
                    }
                }
                // g/u/dn drop here (pool drain) -> the group's packed union tensors are freed
            }
        }
    }

    /// Batched SHARED-expert FFN of the MM prefill paths: gate/up GEMM ->
    /// rows-swiglu at unit weight -> down GEMM, one matmul each for the whole
    /// run instead of 3 matvecs per token. Each projection dispatches on its
    /// RESIDENT quant (Q8_0, or Q4_K under DS4_SHARED_Q4 — misurato: senza,
    /// il per-token costava ~220 ms/token con lo stack decode). Returns false
    /// when Q4 dims rule it out — the caller then encodes the per-token
    /// shared FFN in its tail loop.
    @_optimize(none)   // stesso bug LICM di Swift 6.3 delle altre funzioni di encode
    private func encodeSharedFFNBatched(_ c2: GraphContext, w: LayerWeights,
                                        mm: PrefillStage.MMBuffers,
                                        act actIn: GPUTensor? = nil,
                                        actBase: Int, gTok: Int) throws -> Bool {
        let actMat = actIn ?? mm.curMat
        let q4Dims = d.nEmbd % 256 == 0 && d.sharedFfn % 256 == 0
        let anyQ4 = w.sharedGateQ4 || w.sharedUpQ4 || w.sharedDownQ4
        guard !anyQ4 || q4Dims else { return false }
        func proj(_ weight: GPUTensor, q4: Bool, act: GPUTensor, base: Int,
                  out: GPUTensor, inDim: Int, outDim: Int) throws {
            if q4 {
                try c2.encodeMMDenseQ4K(weight: weight, act: act, actBase: base,
                                        out: out, inDim: inDim, outDim: outDim, nTok: gTok)
            } else {
                try c2.encodeMMDenseQ8(weight: weight, act: act, actBase: base,
                                       out: out, inDim: inDim, outDim: outDim, nTok: gTok)
            }
        }
        try proj(w.sharedGate, q4: w.sharedGateQ4, act: actMat, base: actBase,
                 out: mm.sGate, inDim: d.nEmbd, outDim: d.sharedFfn)
        try proj(w.sharedUp, q4: w.sharedUpQ4, act: actMat, base: actBase,
                 out: mm.sUp, inDim: d.nEmbd, outDim: d.sharedFfn)
        try c2.moeSwiGLUWeight(gate: mm.sGate, up: mm.sUp, weights: mm.ones,
                               mid: mm.sMid, width: d.sharedFfn, rows: gTok,
                               clampValue: d.swigluClamp)
        try proj(w.sharedDown, q4: w.sharedDownQ4, act: mm.sMid, base: 0,
                 out: mm.sOut, inDim: d.sharedFfn, outDim: d.nEmbd)
        return true
    }

    /// Batched routed tail of the MM prefill paths: sum6 over each token's k
    /// weighted down rows (hcWeightedSum with UNIT weights and zero token
    /// stride — x·1.0 is exact, so the collapse is bit-identical to the
    /// per-token moeSum6), + the batched shared rows, then ONE hcExpand4 over
    /// the whole run writing straight into the output HC slab. Replaces
    /// ~5 dispatches + 2 blits PER TOKEN with 3 dispatches per run.
    /// mm.down6/sOut rows are RUN-local (from row 0); the HC slabs are
    /// CHUNK-global (offset by tokens.lowerBound).
    @_optimize(none)   // stesso bug LICM di Swift 6.3 delle altre funzioni di encode
    private func encodeRoutedTailBatched(_ c2: GraphContext, mm: PrefillStage.MMBuffers,
                                         stage: PrefillStage, otherSlab: GPUTensor,
                                         tokens: Range<Int>) throws {
        let gTok = tokens.count
        let hcDim = d.nHC * d.nEmbd
        // routed[t] = sum_k down6[t][k][*]  (unit-weight HC collapse, batched)
        try c2.hcWeightedSum(x: mm.down6, weights: mm.ones, out: mm.routedMat,
                             nEmbd: d.nEmbd, nHC: d.k, nTokens: gTok,
                             weightsTokenStride: 0)
        // ffnOut[t] = shared[t] + routed[t]  (same operand order as the
        // per-token decodeRoutedTail's add)
        try c2.add(mm.sOut, mm.routedMat, out: mm.routedMat, width: d.nEmbd, rows: gTok)
        // HC expand over the whole run into the output slab.
        let attnView = stage.attnSlab.subview(byteOffset: tokens.lowerBound * hcDim * 4,
                                              byteLength: gTok * hcDim * 4,
                                              count: gTok * hcDim)
        let splitView = stage.splitSlab.subview(byteOffset: tokens.lowerBound * 24 * 4,
                                                byteLength: gTok * 24 * 4,
                                                count: gTok * 24)
        let outView = otherSlab.subview(byteOffset: tokens.lowerBound * hcDim * 4,
                                        byteLength: gTok * hcDim * 4,
                                        count: gTok * hcDim)
        try c2.hcExpand4(blockOut: mm.routedMat, residual: attnView,
                         post: splitView, comb: splitView, blockAdd: nil,
                         out: outView, nEmbd: d.nEmbd, nTokens: gTok,
                         postByteOffset: 4 * 4, combByteOffset: 8 * 4,
                         splitTokenStride: 24 * 4)
    }

    /// LEVA 6 — Phase B with FULL-LAYER expert streaming: the whole routed
    /// layer is read ONCE (all d.nExperts experts, packed in id order so ids
    /// stay GLOBAL — no unions, no remap, no per-group re-reads) and every
    /// token's FFN runs against it; the NEXT layer's slab gathers on the
    /// background queue while this layer's FFNs run on the GPU (the C
    /// engine's double-buffered "prefill expert reserve", ~2x1.7 GiB
    /// transient). Pays when the chunk is long — bytes/token =
    /// layerBytes/chunkTokens — and the caller gates it on
    /// DS4_PREFILL_FULL_LAYER_MIN.
    @_optimize(none)   // stesso bug LICM di Swift 6.3 delle altre funzioni di encode
    private func fullLayerPhaseB(_ i: Int, w: LayerWeights, other: [GPUTensor],
                                 otherSlab: GPUTensor, n: Int,
                                 gather: @escaping (Int, [Int32]) throws -> (GPUTensor, GPUTensor, GPUTensor),
                                 stage: PrefillStage, idsT: [[Int32]], rwT: [[Float]],
                                 residentIds: Bool = false, actMat: GPUTensor? = nil) throws {
        let nE = d.nExperts
        let allIds = Array(0..<Int32(nE))
        var t = Date()
        let g: GPUTensor, u: GPUTensor, dn: GPUTensor
        if let pend = fullLayerPending, pend.layer == i {
            fullLayerPending = nil
            (g, u, dn) = try pend.pending.wait()   // I/O ran under the previous layer's FFNs
        } else {
            drainFullLayerGather()                 // stale pending (defensive)
            (g, u, dn) = try gather(i, allIds)     // first layer of the chunk: nothing to overlap
        }
        profile.gatherS += Date().timeIntervalSince(t)   // EXPOSED (non-overlapped) I/O time
        profile.gatherBytes += g.byteLength + u.byteLength + dn.byteLength
        // Kick the NEXT layer's slab now: its I/O runs under this layer's FFNs.
        if i + 1 < nLayers {
            fullLayerPending = (i + 1, PrefillGather(layer: i + 1, gather: gather).start(allIds))
        }
        // LEVA 8 (resident): ONE MM run over the whole chunk, ids/weights and
        // activations read straight from the GPU-resident stage slabs — no
        // CPU staging, no partitioning. Usage statistics are recorded from
        // the slab AFTER the commit (same data the readback produced).
        if residentIds {
            var t2 = Date()
            try withPrefillVoidAutoreleasePool {
                try fullLayerMMRun(w: w, other: other, otherSlab: otherSlab,
                                   stage: stage, idsT: idsT, rwT: rwT,
                                   gate: g, up: u, down: dn, tokens: 0..<n, nExperts: nE,
                                   residentSel: (ids: stage.idsSlab, rw: stage.rwSlab),
                                   act: actMat)
            }
            profile.expertsS += Date().timeIntervalSince(t2)
            if let usage {
                t2 = Date()
                let base = (stage.idsSlab.buffer.contents() + stage.idsSlab.byteOffset)
                    .bindMemory(to: Int32.self, capacity: n * d.k)
                for tk in 0..<n {
                    usage.record(layer: i, ids: Array(UnsafeBufferPointer(start: base + tk * d.k, count: d.k)))
                }
                profile.routeS += Date().timeIntervalSince(t2)
            }
            return
        }

        // Partition the chunk into maximal runs by MM eligibility (full-k
        // distinct selections); every run reads the SAME slab with global ids.
        let mmOK = prefillMM && stage.mm != nil
            && w.gateQuant == .iq2_xxs && w.upQuant == .iq2_xxs && w.downQuant == .q2_K
            && d.k == 6 && d.nEmbd % 256 == 0 && d.expertFfn % 256 == 0
        var j0 = 0
        while j0 < n {
            try Task.checkCancellation()
            let mmRun = mmOK && idsT[j0].count == d.k
            var j1 = j0 + 1
            while j1 < n && (mmOK && idsT[j1].count == d.k) == mmRun { j1 += 1 }
            t = Date()
            try withPrefillVoidAutoreleasePool {
                if mmRun && j1 - j0 >= 8 {
                    // `act` MUST follow the slabDirect source: with the
                    // snapshot blits gone, mm.curMat is stale (the resident
                    // A/B caught exactly this on the partition path).
                    try fullLayerMMRun(w: w, other: other, otherSlab: otherSlab,
                                       stage: stage, idsT: idsT, rwT: rwT,
                                       gate: g, up: u, down: dn, tokens: j0..<j1, nExperts: nE,
                                       act: actMat)
                } else {
                    try fullLayerMatvecRun(w: w, other: other, stage: stage, idsT: idsT, rwT: rwT,
                                           gate: g, up: u, down: dn, tokens: j0..<j1)
                }
            }
            profile.expertsS += Date().timeIntervalSince(t)
            j0 = j1
        }
        // g/u/dn drop here -> this layer's slab frees while the next one lands
    }

    /// Full-layer MM run: the union-path useMM block with GLOBAL ids (map0
    /// over the whole expert set — no posOf remap, no group staging).
    @_optimize(none)   // stesso bug LICM di Swift 6.3 delle altre funzioni di encode
    private func fullLayerMMRun(w: LayerWeights, other: [GPUTensor], otherSlab: GPUTensor,
                                stage: PrefillStage,
                                idsT: [[Int32]], rwT: [[Float]],
                                gate: GPUTensor, up: GPUTensor, down: GPUTensor,
                                tokens: Range<Int>, nExperts nE: Int,
                                residentSel: (ids: GPUTensor, rw: GPUTensor)? = nil,
                                act: GPUTensor? = nil) throws {
        guard let mm = stage.mm else { return }
        let gTok = tokens.count
        if residentSel == nil {
            let idsPtr = (mm.idsMat.buffer.contents() + mm.idsMat.byteOffset)
                .bindMemory(to: Int32.self, capacity: gTok * d.k)
            let wPtr = (mm.wMat.buffer.contents() + mm.wMat.byteOffset)
                .bindMemory(to: Float.self, capacity: gTok * d.k)
            for (tl, j) in tokens.enumerated() {
                for s in 0..<d.k {
                    idsPtr[tl * d.k + s] = idsT[j][s]        // GLOBAL ids
                    wPtr[tl * d.k + s] = rwT[j][s]
                }
            }
        }
        let idsBuf = residentSel?.ids ?? mm.idsMat
        let rwBuf = residentSel?.rw ?? mm.wMat
        let actBuf = act ?? mm.curMat
        let c2 = GraphContext(rt)
        if profilePrefill { c2.phaseTimes = [:] }   // DS4_PROFILE_PREFILL: experts split
        try c2.begin()
        try c2.encodeMoEMap0(ids: idsBuf, htpe: mm.htpe, hids: mm.hids,
                             nTok: gTok, kPerTok: d.k, nExperts: nE)
        try c2.encodeMMIdPairSwiGLUIQ2(gate: gate, up: up, act: actBuf,
                                       actBase: tokens.lowerBound * d.nEmbd * 4,
                                       htpe: mm.htpe, hids: mm.hids,
                                       mid: mm.mid, weights: rwBuf,
                                       nTok: gTok, kPerTok: d.k, nExperts: nE,
                                       inDim: d.nEmbd, ffnDim: d.expertFfn,
                                       clamp: d.swigluClamp)
        try c2.phase("pair")
        try c2.encodeMMIdDownQ2K(down: down, mid: mm.mid,
                                 htpe: mm.htpe, hids: mm.hids, out: mm.down6,
                                 nTok: gTok, kPerTok: d.k, nExperts: nE,
                                 ffnDim: d.expertFfn, outDim: d.nEmbd)
        try c2.phase("down")
        let sharedMM = try encodeSharedFFNBatched(
            c2, w: w, mm: mm, act: actBuf,
            actBase: tokens.lowerBound * d.nEmbd * 4, gTok: gTok)
        try c2.phase("shared")
        if sharedMM {
            // Batched tail: sum6 + shared add + HC expand over the whole run.
            try encodeRoutedTailBatched(c2, mm: mm, stage: stage, otherSlab: otherSlab,
                                        tokens: tokens)
        } else {
            // Per-token shared fallback (Q4 dims ruled the batched shared out).
            let tokBytes = d.k * d.nEmbd * 4
            for (tl, j) in tokens.enumerated() {
                try c2.decodeSharedFFN(w: w, s: scratch, d: d, cur: stage.cur[j])
                try c2.blitCopies([(src: mm.down6, srcOff: tl * tokBytes,
                                    dst: scratch.down6, dstOff: 0, bytes: tokBytes)])
                try c2.decodeRoutedTail(s: scratch, d: d, outHc: other[j],
                                        afterAttn: stage.attn[j], split: stage.split[j])
            }
        }
        try c2.phase("tail")
        c2.commit()
        if let pt = c2.phaseTimes {
            profile.prefillPairS += pt["pair", default: 0]
            profile.prefillDownS += pt["down", default: 0]
            profile.prefillSharedS += pt["shared", default: 0]
            profile.prefillTailS += pt["tail", default: 0]
        }
    }

    /// Full-layer per-token fallback (short selections, MM ineligible or runs
    /// under the MM minimum): decodeExperts against the full slab with GLOBAL
    /// ids, one command buffer per run. Selections shorter than k are padded
    /// with expert 0 at weight 0 (SwiGLU zeroes the padded rows — same
    /// numerics as the union path's padding).
    @_optimize(none)   // stesso bug LICM di Swift 6.3 delle altre funzioni di encode
    private func fullLayerMatvecRun(w: LayerWeights, other: [GPUTensor], stage: PrefillStage,
                                    idsT: [[Int32]], rwT: [[Float]],
                                    gate: GPUTensor, up: GPUTensor, down: GPUTensor,
                                    tokens: Range<Int>) throws {
        // CPU staging BEFORE the command buffer (per-token ids/rw buffers).
        for j in tokens {
            var ids = idsT[j]
            var weights = rwT[j]
            while ids.count < d.k { ids.append(0); weights.append(0) }
            _ = ids.withUnsafeBytes {
                memcpy(stage.ids[j].buffer.contents() + stage.ids[j].byteOffset,
                       $0.baseAddress!, $0.count)
            }
            writeFloats(weights, into: stage.rw[j])
        }
        try Task.checkCancellation()
        let c2 = GraphContext(rt); try c2.begin()
        for j in tokens {
            try c2.decodeExperts(w: w, s: scratch, d: d, gateExp: gate, upExp: up, downExp: down,
                                 ids: stage.ids[j], outHc: other[j], activeK: d.k,
                                 cur: stage.cur[j], afterAttn: stage.attn[j],
                                 split: stage.split[j], rw: stage.rw[j])
        }
        c2.commit()
    }

    /// Dense-GEMM eligibility for a layer: every dense weight of the route is
    /// Q8_0/F16 (no Q4 resident requant) so the whole run can go through the
    /// matrix kernels. Q4-requantized layers (DS4_DENSE_Q4/DS4_QKV_Q4) keep
    /// the historical per-token loop.
    private func prefillDenseEligible(_ w: LayerWeights) -> Bool {
        guard d.nHC == 4 && d.nLoraO % 64 == 0 && d.attnGroupDim % 32 == 0 else { return false }
        // Q4-requantized projections (DS4_DENSE_Q4/DS4_QKV_Q4) go through the
        // Q4_K dense GEMM, which needs a 256-aligned reduction dim.
        if w.qAQ4 && d.nEmbd % 256 != 0 { return false }
        if w.kvQ4 && d.nEmbd % 256 != 0 { return false }
        if w.qBQ4 && d.qRank % 256 != 0 { return false }
        if w.attnOutAQ4 && d.attnGroupDim % 256 != 0 { return false }
        if w.attnOutQ4 && d.attnLowDim % 256 != 0 { return false }
        return true
    }

    /// Dense GEMM on the projection's RESIDENT quant: Q8_0 by default, Q4_K
    /// when the DS4_DENSE_Q4/DS4_QKV_Q4 requant replaced the weight.
    @_optimize(none)   // stesso bug LICM di Swift 6.3 delle altre funzioni di encode
    private func encodeMMDense(_ c: GraphContext, weight: GPUTensor, q4: Bool,
                               act: GPUTensor, out: GPUTensor,
                               inDim: Int, outDim: Int, nTok: Int) throws {
        if q4 {
            try c.encodeMMDenseQ4K(weight: weight, act: act, actBase: 0, out: out,
                                   inDim: inDim, outDim: outDim, nTok: nTok)
        } else {
            try c.encodeMMDenseQ8(weight: weight, act: act, actBase: 0, out: out,
                                  inDim: inDim, outDim: outDim, nTok: nTok)
        }
    }

    /// Phase A run with BATCHED attention (DS4_PREFILL_BATCH_ATTN), one command
    /// buffer, one sync — the C prefill's shape adapted to the route batch:
    ///  A1 (per token, in order): pre-attention half (HC reduce, compressor
    ///     state update + possible comp-row emit, Q/KV projections, fp8 raw
    ///     store) — unchanged decodeRoutePre — then blit the token's Q row into
    ///     the batched Q matrix and its attention HC split into a per-token
    ///     buffer (later tokens' A1 overwrites both scratch slots).
    ///  A2 (once): ONE multi-query FlashAttention over the union span (raw SWA
    ///     window rows + comp rows, staged F16 after ALL the run's KV stores)
    ///     with a CPU-filled per-query mask reproducing exactly the per-token
    ///     visibility: causal + SWA window + comp rows emitted up to each token.
    ///  A3 (per token, in order): blit the token's heads row back and encode
    ///     the unchanged tail (inverse RoPE, output projection, pre-FFN HC,
    ///     router) + the phase-B snapshot blits.
    /// Same math over the same visible keys as the per-token path; only the
    /// attention accumulation order differs (simdgroup MMA blocks vs vec).
    // @_optimize(none): il pass LoopInvariantCodeMotion di Swift 6.3 crasha
    // (SIGSEGV in getBorrowIntroducers) ottimizzando questi encoder in -O.
    // Sono orchestrazione di dispatch Metal: il costo CPU è trascurabile.
    @_optimize(none)
    /// Encodes ONE run and returns its UNCOMMITTED context: the caller
    /// commits (async in the phase-A pipeline — run r+1's CPU encoding
    /// overlaps run r's GPU execution; the mask alternates by `maskParity`).
    private func encodeFlashRun(_ fb: PrefillStage.FlashBatch, _ i: Int, w: LayerWeights,
                                layerRope: RopeParams, cur: [GPUTensor], stage: PrefillStage,
                                j: Int, jEnd: Int, posBase: Int, tokens: [Int],
                                maskParity: Bool, slabDirect: Bool) throws -> GraphContext {
        let nq = jEnd - j
        let posFirst = posBase + j
        let rawLo0 = max(0, posFirst + 1 - d.nSWA)
        let nRawSpan = posBase + jEnd - rawLo0
        let denseMM = prefillDenseMM && prefillDenseEligible(w)
        let c = GraphContext(rt)
        if profilePrefill && denseMM { c.phaseTimes = [:] }   // DS4_PROFILE_PREFILL
        try c.begin()
        var nCompVis = [Int](repeating: 0, count: nq)

        // ── A1: pre-attention halves (split into helper methods — one big
        // function here crashes the release optimizer's LICM pass).
        if denseMM {
            var idxVis = [Int](repeating: 0, count: nq)   // inerti: indexer inattivo nei run flash
            try encodeDensePre(c, fb, i, w: w, layerRope: layerRope, cur: cur,
                               j: j, jEnd: jEnd, posBase: posBase, nCompVis: &nCompVis,
                               idxVis: &idxVis)
        } else {
            try encodePerTokenPre(c, fb, i, w: w, layerRope: layerRope, cur: cur,
                                  j: j, jEnd: jEnd, posBase: posBase, nCompVis: &nCompVis)
        }

        // ── A2: ONE multi-query FlashAttention over the whole run.
        let nComp = nCompVis[nq - 1]
        // CPU mask fill BEFORE commit. Parity-alternated buffer: the PREVIOUS
        // run may still be executing on the GPU (async pipeline) and reading
        // ITS mask — never this one.
        let maskBuf = maskParity ? fb.mask : fb.maskB
        let maskPtr = (maskBuf.buffer.contents() + maskBuf.byteOffset)
            .bindMemory(to: UInt16.self, capacity: nq * (nRawSpan + nComp))
        GraphContext.fillPrefillAttnMask(maskPtr, nQ: nq, posFirst: posFirst,
                                         rawStart: rawLo0, rawSpan: nRawSpan,
                                         window: d.nSWA, nCompVis: nCompVis, nComp: nComp)
        try c.flashAttnPrefill(q: fb.qMat, kvF32: rawCaches[i], kvF16: fb.kvF16,
                               mask: maskBuf, sinks: w.attnSinks, pad: fb.pad, blk: fb.blk,
                               heads: fb.heads, nHead: d.nHead, nQ: nq,
                               rawSpan: nRawSpan, rawStartRow: rawLo0,
                               comp: compStates[i]?.cache, nComp: nComp)
        try c.phase("attn")                       // DS4_PROFILE_PREFILL boundary

        // ── A3: post-attention tails.
        if denseMM {
            try encodeDenseTail(c, fb, w: w, layerRope: layerRope, stage: stage,
                                j: j, jEnd: jEnd, posBase: posBase, tokens: tokens,
                                slabDirect: slabDirect)
        } else {
            try encodePerTokenTail(c, fb, w: w, layerRope: layerRope, cur: cur, stage: stage,
                                   j: j, jEnd: jEnd, posBase: posBase, tokens: tokens)
        }
        profile.prefillFlashRuns += 1
        if denseMM { profile.prefillDenseRuns += 1 }
        return c                 // caller commits (pipeline) — no sync here
    }

    /// LEVA 9 v2 — run batchato con indexer ATTIVO (DS4_INDEXED_ATTN): la
    /// stessa A1 dense-GEMM dei run flash (encodeDensePre, che già batcha il
    /// compressore indexer e ne espone i conteggi idxVis), poi scoring
    /// batchato (GEMM query/pesi + score/top-k/sort per query, tutto su GPU)
    /// e UN dispatch di attention indicizzata multi-token al posto di
    /// maschera CPU + staging F16 + flash. La coda A3 è identica ai run
    /// flash. NUMERICA: stessa classe del percorso per-token indicizzato
    /// (kernel heads8 vs rb16: stesso set di righe, tiling diverso).
    // @_optimize(none): stesso bug LICM di Swift 6.3 degli altri encoder.
    @_optimize(none)
    private func encodeIndexedFlashRun(_ fb: PrefillStage.FlashBatch, _ i: Int, w: LayerWeights,
                                       layerRope: RopeParams, cur: [GPUTensor], stage: PrefillStage,
                                       j: Int, jEnd: Int, posBase: Int, tokens: [Int],
                                       slabDirect: Bool) throws -> GraphContext {
        let nq = jEnd - j
        let posFirst = posBase + j
        let rawLo0 = max(0, posFirst + 1 - d.nSWA)
        let nRawSpan = posBase + jEnd - rawLo0
        guard let idx = indexStates[i], let iqb = w.idxQB, let iproj = w.idxProj,
              let comp = compStates[i] else {
            throw MetalError.unsupported("encodeIndexedFlashRun senza scoring indexer")
        }
        let c = GraphContext(rt)
        if profilePrefill { c.phaseTimes = [:] }
        try c.begin()
        var nCompVis = [Int](repeating: 0, count: nq)
        var idxVis = [Int](repeating: 0, count: nq)
        try encodeDensePre(c, fb, i, w: w, layerRope: layerRope, cur: cur,
                           j: j, jEnd: jEnd, posBase: posBase, nCompVis: &nCompVis,
                           idxVis: &idxVis)
        // ── A2i: scoring batchato — proiezioni in GEMM sul run, poi per
        // query: score sulla SUA finestra di righe + top-K a indici + sort.
        let ih = d.nIndexerHeadDim, inH = d.nIndexerHead
        if w.idxQBF16 {
            try c.encodeMMDenseF16(weight: iqb, act: fb.qrNormMat, actBase: 0, out: fb.idxQMat,
                                   inDim: d.qRank, outDim: inH * ih, nTok: nq)
        } else {
            try encodeMMDense(c, weight: iqb, q4: false, act: fb.qrNormMat, out: fb.idxQMat,
                              inDim: d.qRank, outDim: inH * ih, nTok: nq)
        }
        try c.ropeTail(x: fb.idxQMat, nTok: nq, nHead: inH, headDim: ih, nRot: d.nRot,
                       nCtxOrig: layerRope.nCtxOrig, freqBase: layerRope.freqBase,
                       freqScale: layerRope.freqScale, extFactor: layerRope.extFactor,
                       attnFactor: layerRope.attnFactor, betaFast: layerRope.betaFast,
                       betaSlow: layerRope.betaSlow, pos0: posFirst, posStep: 1)
        try c.indexerHadamardFp4Enc(fb.idxQMat, rows: nq * inH, rowStrideBytes: ih * 4)
        try c.encodeMMDenseF16(weight: iproj, act: fb.curMat, actBase: 0, out: fb.idxWMat,
                               inDim: d.nEmbd, outDim: inH, nTok: nq)
        let scoreScale = 1.0 / Float(ih * inH).squareRoot()
        // v2.1 (port fedele del C): score TILED in una dispatch (visibilità
        // causale per token in-kernel, -inf oltre → righe uniformi), top-K
        // batch con argsort+merge, sort per id in una dispatch per run.
        let nScAll = idxVis[nq - 1]
        precondition(nScAll > 0 && nScAll <= fb.maxKv,
                     "leva 9 v2: score fuori capacità (\(nScAll)/\(fb.maxKv))")
        try c.indexerScoresTiledBatch(q: fb.idxQMat, weights: fb.idxWMat,
                                      indexComp: idx.cache, scores: fb.idxScoresMat,
                                      nComp: nScAll, nTokens: nq, pos0: posFirst,
                                      ratio: idx.ratio, nHead: inH, headDim: ih,
                                      scale: scoreScale)
        try c.indexerTopKIndicesBatch(scores: fb.idxScoresMat, nComp: nScAll,
                                      nTokens: nq, topK: d.indexerTopK,
                                      out: fb.idxTopKMat, scratch: fb.idxSortScratch)
        try c.sortTopKAsc(indices: fb.idxTopKMat, sorted: fb.idxTopKSortMat,
                          topK: d.indexerTopK, nTokens: nq)
        // ── A2ii: UN dispatch di attention indicizzata per l'intero run.
        let rawRows = rawCaches[i].count / d.headDim
        let physStart = ((rawLo0 % rawRows) + rawRows) % rawRows
        try c.indexedMixedAttentionBatch(q: fb.qMat, rawKv: rawCaches[i], comp: comp.cache,
                                         topk: fb.idxTopKSortMat, sinks: w.attnSinks,
                                         heads: fb.heads, nTokens: nq, nHead: d.nHead,
                                         nRaw: nRawSpan, rawCap: rawRows, rawStart: physStart,
                                         nComp: nCompVis[nq - 1], topK: d.indexerTopK,
                                         pos0: posFirst, window: d.nSWA, ratio: comp.ratio)
        try c.phase("attn")                       // DS4_PROFILE_PREFILL boundary
        // ── A3: coda dense identica ai run flash.
        try encodeDenseTail(c, fb, w: w, layerRope: layerRope, stage: stage,
                            j: j, jEnd: jEnd, posBase: posBase, tokens: tokens,
                            slabDirect: slabDirect)
        profile.prefillFlashRuns += 1
        profile.prefillDenseRuns += 1
        return c                 // caller commits (pipeline) — no sync here
    }

    /// Batched pre/post HC reduce: rms -> mixer GEMM -> Sinkhorn split ->
    /// collapse -> norm, all over the run's rows. Same dispatches as the
    /// per-token hcReduce with rows = nq.
    // @_optimize(none): il pass LoopInvariantCodeMotion di Swift 6.3 crasha
    // (SIGSEGV in getBorrowIntroducers) ottimizzando questi encoder in -O.
    // Sono orchestrazione di dispatch Metal: il costo CPU è trascurabile.
    @_optimize(none)
    private func encodeHCReduceBatched(_ c: GraphContext, _ fb: PrefillStage.FlashBatch,
                                       x: GPUTensor, mixerFn: GPUTensor, scale: GPUTensor,
                                       base: GPUTensor, norm: GPUTensor,
                                       split: GPUTensor, out: GPUTensor, nq: Int) throws {
        let hcDim = d.nHC * d.nEmbd
        try c.rmsNorm(x, weight: nil, out: fb.flatMat, rows: nq, n: hcDim, eps: rmsEps)
        try c.encodeMMDenseF16(weight: mixerFn, act: fb.flatMat, actBase: 0, out: fb.mixMat,
                               inDim: hcDim, outDim: 24, nTok: nq)
        if d.fusedHC {
            try c.hcSplitWeightedSumNorm4(mix: fb.mixMat, scale: scale, base: base, x: x,
                                          split: split, embd: fb.embdMat, normWeight: norm,
                                          normOut: out, nEmbd: d.nEmbd, nRows: nq,
                                          sinkhornIters: d.sinkhornIterations,
                                          eps: hcEps, normEps: rmsEps)
        } else {
            try c.hcSplitSinkhorn(mix: fb.mixMat, scale: scale, base: base, out: split,
                                  nRows: nq, sinkhornIters: d.sinkhornIterations, eps: hcEps)
            try c.hcWeightedSum(x: x, weights: split, out: fb.embdMat, nEmbd: d.nEmbd,
                                nHC: d.nHC, nTokens: nq, weightsTokenStride: 24 * 4)
            try c.rmsNorm(fb.embdMat, weight: norm, out: out, rows: nq, n: d.nEmbd, eps: rmsEps)
        }
    }

    /// Dense-GEMM A1: pack HC states, batched pre-attn HC reduce, per-token
    /// compressor recurrences, batched Q/KV GEMMs + norms + RoPE, per-token
    /// fp8 ring stores.
    // @_optimize(none): il pass LoopInvariantCodeMotion di Swift 6.3 crasha
    // (SIGSEGV in getBorrowIntroducers) ottimizzando questi encoder in -O.
    // Sono orchestrazione di dispatch Metal: il costo CPU è trascurabile.
    @_optimize(none)
    private func encodeDensePre(_ c: GraphContext, _ fb: PrefillStage.FlashBatch, _ i: Int,
                                w: LayerWeights, layerRope: RopeParams, cur: [GPUTensor],
                                j: Int, jEnd: Int, posBase: Int, nCompVis: inout [Int],
                                idxVis: inout [Int]) throws {
        let nq = jEnd - j
        let posFirst = posBase + j
        let hcDim = d.nHC * d.nEmbd
        let hasIdxState = w.idxKv != nil && w.idxGate != nil    // state-only gate (see encodeRoute)
        // ── A0: pack the run's input HC states into one token-major matrix.
        try c.blitCopies((0..<nq).map { r in
            (cur[j + r], 0, fb.hcMat, r * hcDim * 4, hcDim * 4)
        })
        // ── A1a: batched pre-attention HC reduce -> attn-normed rows + split.
        try encodeHCReduceBatched(c, fb, x: fb.hcMat, mixerFn: w.hcAttnFn, scale: w.attnScale,
                                  base: w.attnBase, norm: w.attnNorm,
                                  split: fb.splitA, out: fb.curMat, nq: nq)
        try c.phase("hc-ffn")                     // DS4_PROFILE_PREFILL boundary (pre-attn HC)
        // ── A1b: compressor recurrences. The kv/score PROJECTIONS are batched:
        // one GEMM per weight for the whole run instead of two matvecs per
        // token (~520 MB/token of aggregate weight re-reads across the
        // compressed layers); only the recurrent state update + emit stays
        // token-sequential. Widths (2·headDim / headDim / 2·nIndexerHeadDim)
        // are all multiples of 64 by construction.
        if let comp = compStates[i], let ckv = w.compKv, let cgate = w.compGate,
           let cape = w.compApe, let cnorm = w.compNorm {
            try encodeCompProjBatched(c, kv: ckv, gate: cgate, act: fb.curMat,
                                      kvOut: fb.compKvMat, scOut: fb.compScMat,
                                      width: comp.width, nq: nq, q8: w.compQ8)
            try encodeCompressorRunBatched(c, fb: fb, comp: comp,
                                           kvMat: fb.compKvMat, scMat: fb.compScMat,
                                           ape: cape, normW: cnorm, rope: layerRope,
                                           nq: nq, posFirst: posFirst,
                                           finalize: .fp8, nCompVis: &nCompVis)
        } else if let comp = compStates[i] {
            // Missing compressor weights: historical no-op semantics.
            for r in 0..<nq { nCompVis[r] = comp.count }
        }
        if hasIdxState, let idx = indexStates[i], let ikv = w.idxKv, let igate = w.idxGate,
           let iape = w.idxApe, let inorm = w.idxNorm {
            try encodeCompProjBatched(c, kv: ikv, gate: igate, act: fb.curMat,
                                      kvOut: fb.idxKvMat, scOut: fb.idxScMat,
                                      width: idx.width, nq: nq, q8: w.idxCompQ8)
            // idxVis: conteggi righe indexer visibili per query — inerti nei
            // run inattivi, input dello scoring batchato (leva 9 v2).
            try encodeCompressorRunBatched(c, fb: fb, comp: idx,
                                           kvMat: fb.idxKvMat, scMat: fb.idxScMat,
                                           ape: iape, normW: inorm, rope: layerRope,
                                           nq: nq, posFirst: posFirst,
                                           finalize: .indexerQat, nCompVis: &idxVis)
        }
        try c.phase("comp")                       // DS4_PROFILE_PREFILL boundary (NSA recurrences)
        // ── A1c: batched Q path (GEMM + norms + RoPE over all rows).
        try encodeMMDense(c, weight: w.qA, q4: w.qAQ4, act: fb.curMat, out: fb.qrMat,
                          inDim: d.nEmbd, outDim: d.qRank, nTok: nq)
        try c.rmsNorm(fb.qrMat, weight: w.qANorm, out: fb.qrNormMat, rows: nq, n: d.qRank, eps: rmsEps)
        try encodeMMDense(c, weight: w.qB, q4: w.qBQ4, act: fb.qrNormMat, out: fb.qMat,
                          inDim: d.qRank, outDim: d.qDim, nTok: nq)
        try c.rmsNorm(fb.qMat, weight: nil, out: fb.qMat, rows: nq * d.nHead, n: d.headDim, eps: rmsEps)
        try c.ropeTail(x: fb.qMat, nTok: nq, nHead: d.nHead, headDim: d.headDim, nRot: d.nRot,
                       nCtxOrig: layerRope.nCtxOrig, freqBase: layerRope.freqBase,
                       freqScale: layerRope.freqScale, extFactor: layerRope.extFactor,
                       attnFactor: layerRope.attnFactor, betaFast: layerRope.betaFast,
                       betaSlow: layerRope.betaSlow, pos0: posFirst, posStep: 1)
        try c.phase("q")                          // DS4_PROFILE_PREFILL boundary (Q GEMMs)
        // ── A1d: batched KV path + per-token fp8 ring store.
        try encodeMMDense(c, weight: w.kvW, q4: w.kvQ4, act: fb.curMat, out: fb.kvMat,
                          inDim: d.nEmbd, outDim: d.headDim, nTok: nq)
        try c.rmsNorm(fb.kvMat, weight: w.kvNorm, out: fb.kvMat, rows: nq, n: d.headDim, eps: rmsEps)
        try c.ropeTail(x: fb.kvMat, nTok: nq, nHead: 1, headDim: d.headDim, nRot: d.nRot,
                       nCtxOrig: layerRope.nCtxOrig, freqBase: layerRope.freqBase,
                       freqScale: layerRope.freqScale, extFactor: layerRope.extFactor,
                       attnFactor: layerRope.attnFactor, betaFast: layerRope.betaFast,
                       betaSlow: layerRope.betaSlow, pos0: posFirst, posStep: 1)
        // Store fp8 del ring in UN dispatch invece di nq (le righe del run
        // cadono in slot distinti per costruzione: il clamp di routeBatch
        // garantisce nq + nSWA - 1 <= rawRows, quindi nq < rawRows).
        let rawRows = rawCaches[i].count / d.headDim
        if prefillMicroBatch {
            try c.kvFP8StoreBatch(kv: fb.kvMat, rawCache: rawCaches[i], headDim: d.headDim,
                                  nRot: d.nRot, pos0: posFirst, rawRows: rawRows, nTok: nq)
        } else {
            for r in 0..<nq {
                let kvRow = fb.kvMat.subview(byteOffset: r * d.headDim * 4,
                                             byteLength: d.headDim * 4, count: d.headDim)
                try c.kvFP8Store(kv: kvRow, rawCache: rawCaches[i], headDim: d.headDim,
                                 nRot: d.nRot, rawRow: (posFirst + r) % rawRows)
            }
        }
        try c.phase("kv")                         // DS4_PROFILE_PREFILL boundary (KV GEMM + stores)
    }

    /// LEVA 7 — batched NSA compressor for one dense run: APE-bias the run's
    /// projected scores in place, assemble the position-ordered combined
    /// window (persistent-state head + run projections), pool EVERY emission
    /// in ONE dispatch, then batched norm/rope/finalize into the cache, and
    /// rebuild the persistent state to exactly what the per-token recurrence
    /// would have left (prev/cur lanes for ratio-4, position-keyed ring for
    /// ratio-128 — layouts from kernel_dsv4_compressor_store_one). Never-
    /// written head positions carry the state's -1e30 init, so their softmax
    /// weight underflows to zero exactly like the per-token ring reads.
    @_optimize(none)   // stesso bug LICM di Swift 6.3 delle altre funzioni di encode
    private func encodeCompressorRunBatched(_ c: GraphContext, fb: PrefillStage.FlashBatch,
                                            comp: CompressorState, kvMat: GPUTensor, scMat: GPUTensor,
                                            ape: GPUTensor, normW: GPUTensor, rope: RopeParams,
                                            nq: Int, posFirst: Int,
                                            finalize: GraphContext.CompressorFinalize,
                                            nCompVis: inout [Int]) throws {
        let h = comp.headDim, ratio = comp.ratio, width = comp.width
        let win = comp.poolRows                      // 8 (ratio-4) or ratio (128)
        let posLast = posFirst + nq - 1
        let rowB = width * 4
        // 1) APE bias, in place on the run's projected scores (store-time math).
        try c.compApeAddEnc(sc: scMat, ape: ape, width: width, ratio: ratio,
                            pos0: posFirst, nTok: nq)
        // 2) Combined window base position and state head.
        let p0: Int, headRows: Int
        var copies: [(src: GPUTensor, srcOff: Int, dst: GPUTensor, dstOff: Int, bytes: Int)] = []
        if ratio == 4 {
            let g0 = (posFirst / 4) * 4
            p0 = g0 - 4
            headRows = posFirst - p0                 // 4 prev + partial cur
            // prev lane: state rows 0..<4 = positions [g0-4, g0) in order.
            copies.append((comp.stateKv, 0, fb.compCombKv, 0, 4 * rowB))
            copies.append((comp.stateScore, 0, fb.compCombSc, 0, 4 * rowB))
            let partial = posFirst - g0
            if partial > 0 {                         // cur lane: rows 4..<4+partial
                copies.append((comp.stateKv, 4 * rowB, fb.compCombKv, 4 * rowB, partial * rowB))
                copies.append((comp.stateScore, 4 * rowB, fb.compCombSc, 4 * rowB, partial * rowB))
            }
        } else {
            p0 = posFirst - (win - 1)
            headRows = win - 1
            // Ring slot of position p is ((p % win) + win) % win; consecutive
            // positions are consecutive slots -> at most one wrap split.
            let s0 = ((p0 % win) + win) % win
            let len1 = min(headRows, win - s0)
            copies.append((comp.stateKv, s0 * rowB, fb.compCombKv, 0, len1 * rowB))
            copies.append((comp.stateScore, s0 * rowB, fb.compCombSc, 0, len1 * rowB))
            if headRows > len1 {
                copies.append((comp.stateKv, 0, fb.compCombKv, len1 * rowB, (headRows - len1) * rowB))
                copies.append((comp.stateScore, 0, fb.compCombSc, len1 * rowB, (headRows - len1) * rowB))
            }
        }
        // 3) Run projections behind the head (position-ordered).
        copies.append((kvMat, 0, fb.compCombKv, headRows * rowB, nq * rowB))
        copies.append((scMat, 0, fb.compCombSc, headRows * rowB, nq * rowB))
        try c.blitCopies(copies)
        // 4) Emissions of the run.
        let countBefore = comp.count
        let e0 = ((posFirst + ratio) / ratio) * ratio - 1
        let nEmit = e0 > posLast ? 0 : (posLast - e0) / ratio + 1
        if nEmit > 0 {
            precondition(comp.count + nEmit <= comp.cacheCapacity,
                         "compressor cache: prefill oltre la capacita' fisica")
            try c.compPoolBatchEnc(kv: fb.compCombKv, sc: fb.compCombSc, out: fb.compPooled,
                                   headDim: h, width: width, ratio: ratio, win: win,
                                   nEmit: nEmit, firstRow: e0 - p0)
            try c.rmsNorm(fb.compPooled, weight: normW, out: fb.compPooled,
                          rows: nEmit, n: h, eps: rmsEps)
            try c.ropeTail(x: fb.compPooled, nTok: nEmit, nHead: 1, headDim: h, nRot: d.nRot,
                           nCtxOrig: rope.nCtxOrig, freqBase: rope.freqBase,
                           freqScale: rope.freqScale, extFactor: rope.extFactor,
                           attnFactor: rope.attnFactor, betaFast: rope.betaFast,
                           betaSlow: rope.betaSlow, pos0: e0 + 1 - ratio, posStep: ratio)
            switch finalize {
            case .fp8:        try c.fp8QuantizeRowsEnc(fb.compPooled, rows: nEmit, headDim: h, nRot: d.nRot)
            case .indexerQat: try c.indexerHadamardFp4Enc(fb.compPooled, rows: nEmit, rowStrideBytes: h * 4)
            }
            try c.blitCopies([(fb.compPooled, 0, comp.cache, comp.count * h * 4, nEmit * h * 4)])
            comp.count += nEmit
        }
        // 5) Rebuild the persistent state (what per-token stores would leave).
        var stateCopies: [(src: GPUTensor, srcOff: Int, dst: GPUTensor, dstOff: Int, bytes: Int)] = []
        if ratio == 4 {
            let g1 = ((posLast + 1) / 4) * 4
            let prevRow = g1 - 4 - p0                // last complete group
            stateCopies.append((fb.compCombKv, prevRow * rowB, comp.stateKv, 0, 4 * rowB))
            stateCopies.append((fb.compCombSc, prevRow * rowB, comp.stateScore, 0, 4 * rowB))
            let partial1 = posLast + 1 - g1
            if partial1 > 0 {                        // trailing partial cur group
                stateCopies.append((fb.compCombKv, (g1 - p0) * rowB, comp.stateKv, 4 * rowB, partial1 * rowB))
                stateCopies.append((fb.compCombSc, (g1 - p0) * rowB, comp.stateScore, 4 * rowB, partial1 * rowB))
            }
        } else {
            let first = posLast - (win - 1)          // last `win` positions
            let s0 = ((first % win) + win) % win
            let len1 = min(win, win - s0)
            stateCopies.append((fb.compCombKv, (first - p0) * rowB, comp.stateKv, s0 * rowB, len1 * rowB))
            stateCopies.append((fb.compCombSc, (first - p0) * rowB, comp.stateScore, s0 * rowB, len1 * rowB))
            if win > len1 {
                stateCopies.append((fb.compCombKv, (first - p0 + len1) * rowB, comp.stateKv, 0, (win - len1) * rowB))
                stateCopies.append((fb.compCombSc, (first - p0 + len1) * rowB, comp.stateScore, 0, (win - len1) * rowB))
            }
        }
        try c.blitCopies(stateCopies)
        // 6) Per-token visible counts (identical to the per-token returns).
        for r in 0..<nq {
            nCompVis[r] = countBefore + ((posFirst + r + 1) / ratio - posFirst / ratio)
        }
    }

    /// Batched compressor kv/score projections: two dense GEMMs over the run's
    /// attn-normed rows (F16, or Q8_0 with the DS4_COMP_Q8 resident requant).
    /// NOTE: the mm kernel stages the activation tile to f16 — same tolerance
    /// class as the other dense GEMMs of this path.
    @_optimize(none)   // stesso bug LICM di Swift 6.3 delle altre funzioni di encode
    private func encodeCompProjBatched(_ c: GraphContext, kv: GPUTensor, gate: GPUTensor,
                                       act: GPUTensor, kvOut: GPUTensor, scOut: GPUTensor,
                                       width: Int, nq: Int, q8: Bool) throws {
        if q8 {
            try c.encodeMMDenseQ8(weight: kv, act: act, actBase: 0, out: kvOut,
                                  inDim: d.nEmbd, outDim: width, nTok: nq)
            try c.encodeMMDenseQ8(weight: gate, act: act, actBase: 0, out: scOut,
                                  inDim: d.nEmbd, outDim: width, nTok: nq)
        } else {
            try c.encodeMMDenseF16(weight: kv, act: act, actBase: 0, out: kvOut,
                                   inDim: d.nEmbd, outDim: width, nTok: nq)
            try c.encodeMMDenseF16(weight: gate, act: act, actBase: 0, out: scOut,
                                   inDim: d.nEmbd, outDim: width, nTok: nq)
        }
    }

    /// Attention-only A1 (lever-1 fallback): per-token pre halves, Q rows blitted.
    // @_optimize(none): il pass LoopInvariantCodeMotion di Swift 6.3 crasha
    // (SIGSEGV in getBorrowIntroducers) ottimizzando questi encoder in -O.
    // Sono orchestrazione di dispatch Metal: il costo CPU è trascurabile.
    @_optimize(none)
    private func encodePerTokenPre(_ c: GraphContext, _ fb: PrefillStage.FlashBatch, _ i: Int,
                                   w: LayerWeights, layerRope: RopeParams, cur: [GPUTensor],
                                   j: Int, jEnd: Int, posBase: Int, nCompVis: inout [Int]) throws {
        let hasIdxState = w.idxKv != nil && w.idxGate != nil    // state-only gate (see encodeRoute)
        let qBytes = d.qDim * 4
        for (r, jj) in (j..<jEnd).enumerated() {
            let pos = posBase + jj
            nCompVis[r] = try c.decodeRoutePre(curHc: cur[jj], w: w, s: scratch, d: d,
                                               rope: layerRope, rawCache: rawCaches[i], pos: pos,
                                               rmsEps: rmsEps, hcEps: hcEps,
                                               comp: compStates[i],
                                               idx: hasIdxState ? indexStates[i] : nil,
                                               indexerScoring: false)  // caller guarantees no scoring
            try c.blitCopies([
                (scratch.q, 0, fb.qMat, r * qBytes, qBytes),
                (scratch.split, 0, fb.split[r], 0, 24 * 4),
            ])
        }
    }

    /// Dense-GEMM A3: batched inverse RoPE, grouped low-rank output, out_b,
    /// HC expand + pre-FFN reduce, batched router logits, per-token finalize
    /// + phase-B snapshots.
    // @_optimize(none): il pass LoopInvariantCodeMotion di Swift 6.3 crasha
    // (SIGSEGV in getBorrowIntroducers) ottimizzando questi encoder in -O.
    // Sono orchestrazione di dispatch Metal: il costo CPU è trascurabile.
    @_optimize(none)
    private func encodeDenseTail(_ c: GraphContext, _ fb: PrefillStage.FlashBatch,
                                 w: LayerWeights, layerRope: RopeParams, stage: PrefillStage,
                                 j: Int, jEnd: Int, posBase: Int, tokens: [Int],
                                 slabDirect: Bool) throws {
        let nq = jEnd - j
        let posFirst = posBase + j
        let hcDim = d.nHC * d.nEmbd
        // LEVA 8 (slabDirect, full-layer chunks): the run's post-attention
        // residual, pre-FFN split and FFN-input rows are written STRAIGHT into
        // the chunk-wide stage slabs at this run's offset — the phase-B
        // snapshot blits (~3.4 MB/token) disappear, and phase B reads the
        // slabs (ids/weights included) without any CPU round-trip.
        let attnOut = slabDirect
            ? stage.attnSlab.subview(byteOffset: j * hcDim * 4,
                                     byteLength: nq * hcDim * 4, count: nq * hcDim)
            : fb.afterAttnMat
        let splitOut = slabDirect
            ? stage.splitSlab.subview(byteOffset: j * 24 * 4,
                                      byteLength: nq * 24 * 4, count: nq * 24)
            : fb.splitF
        let curOut = slabDirect
            ? stage.curSlab.subview(byteOffset: j * d.nEmbd * 4,
                                    byteLength: nq * d.nEmbd * 4, count: nq * d.nEmbd)
            : fb.curMat2
        // ── A3a: batched inverse RoPE + grouped low-rank output + out_b.
        try c.ropeTail(x: fb.heads, nTok: nq, nHead: d.nHead, headDim: d.headDim, nRot: d.nRot,
                       nCtxOrig: layerRope.nCtxOrig, freqBase: layerRope.freqBase,
                       freqScale: layerRope.freqScale, extFactor: layerRope.extFactor,
                       attnFactor: layerRope.attnFactor, betaFast: layerRope.betaFast,
                       betaSlow: layerRope.betaSlow, pos0: posFirst, posStep: 1, inverse: true)
        let groupRowBytes = w.attnOutAQ4 ? (d.attnGroupDim / 256) * 144
                                         : (d.attnGroupDim / 32) * 34
        for g in 0..<d.nOutGroup {
            if w.attnOutAQ4 {
                try c.encodeMMDenseQ4KStrided(weight: w.attnOutA,
                                              weightOffset: g * d.nLoraO * groupRowBytes,
                                              act: fb.heads, actBase: g * d.attnGroupDim * 4,
                                              actRowStride: d.qDim * 4,
                                              out: fb.lowMat, outBase: g * d.nLoraO * 4,
                                              outRowStrideElems: d.attnLowDim,
                                              inDim: d.attnGroupDim, outDim: d.nLoraO, nTok: nq)
            } else {
                try c.encodeMMDenseQ8Strided(weight: w.attnOutA,
                                             weightOffset: g * d.nLoraO * groupRowBytes,
                                             act: fb.heads, actBase: g * d.attnGroupDim * 4,
                                             actRowStride: d.qDim * 4,
                                             out: fb.lowMat, outBase: g * d.nLoraO * 4,
                                             outRowStrideElems: d.attnLowDim,
                                             inDim: d.attnGroupDim, outDim: d.nLoraO, nTok: nq)
            }
        }
        try encodeMMDense(c, weight: w.attnOut, q4: w.attnOutQ4, act: fb.lowMat, out: fb.blockOutMat,
                          inDim: d.attnLowDim, outDim: d.nEmbd, nTok: nq)
        try c.phase("out-proj")                   // DS4_PROFILE_PREFILL boundary
        // ── A3b: batched attention-residual HC expand + pre-FFN HC reduce.
        try c.hcExpand4(blockOut: fb.blockOutMat, residual: fb.hcMat,
                        post: fb.splitA, comb: fb.splitA, blockAdd: nil,
                        out: attnOut, nEmbd: d.nEmbd, nTokens: nq,
                        postByteOffset: 4 * 4, combByteOffset: 8 * 4,
                        splitTokenStride: 24 * 4)
        try encodeHCReduceBatched(c, fb, x: attnOut, mixerFn: w.hcFfnFn, scale: w.ffnScale,
                                  base: w.ffnBase, norm: w.ffnNorm,
                                  split: splitOut, out: curOut, nq: nq)
        try c.phase("hc-ffn")                     // DS4_PROFILE_PREFILL boundary (pre-FFN HC)
        // ── A3c: batched router logits, per-token finalize + snapshots.
        if d.routerF16 {
            try c.encodeMMDenseF16(weight: w.routerW, act: curOut, actBase: 0,
                                   out: fb.logitsMat, inDim: d.nEmbd, outDim: d.nExperts, nTok: nq)
        } else {
            try c.encodeMMDenseQ8(weight: w.routerW, act: curOut, actBase: 0,
                                  out: fb.logitsMat, inDim: d.nEmbd, outDim: d.nExperts, nTok: nq)
        }
        // Probabilità + finalize BATCHATI (un dispatch ciascuno invece di uno
        // per token): stesso corpo per riga ⇒ bit-identico. I layer HASH
        // restano per token — leggono l'id del token per indicizzare tid2eid e
        // batcharli richiederebbe un buffer token per run, con la stessa
        // parità double-buffer della mask; sono 3 layer su 43.
        // (stage.ids/rw sono viste contigue su idsSlab/rwSlab con stride d.k —
        // vedi PrefillStage.slabViews — quindi il batch può indirizzarle a
        // stride senza restaging.)
        let batchedRouter = prefillMicroBatch && d.fusedRouterProbs
            && d.fusedRouterFinalize && w.tid2eid == nil
        if batchedRouter {
            try c.routerProbabilitiesBatch(logits: fb.logitsMat, probabilities: fb.probsMat,
                                           width: d.nExperts, rows: nq)
            try c.routerFinalizeTop6Batch(
                probs: fb.probsMat,
                selected: stage.idsSlab.subview(byteOffset: j * d.k * 4,
                                                byteLength: nq * d.k * 4, count: nq * d.k),
                bias: w.expBias, hashTable: nil, hashRows: 0, tokens: nil,
                weights: stage.rwSlab.subview(byteOffset: j * d.k * 4,
                                              byteLength: nq * d.k * 4, count: nq * d.k),
                nExperts: d.nExperts, nTok: nq,
                probsRow: d.nExperts, selRow: d.k, weightsRow: d.k,
                expertWeightScale: d.expertWeightScale)
        }
        for (r, jj) in (j..<jEnd).enumerated() {
            let logitsRow = fb.logitsMat.subview(byteOffset: r * d.nExperts * 4,
                                                 byteLength: d.nExperts * 4, count: d.nExperts)
            if batchedRouter {
                // già fatto sopra per l'intero run
            } else if d.fusedRouterProbs {
                try c.routerProbabilities(logits: logitsRow, probabilities: scratch.probs,
                                          width: d.nExperts)
            } else {
                try c.unary(logitsRow, op: .softplus, out: scratch.sp, width: d.nExperts)
                try c.unary(scratch.sp, op: .sqrt, out: scratch.probs, width: d.nExperts)
            }
            if !batchedRouter {
                // Finalize writes the per-token stage buffers DIRECTLY (the
                // per-token path blits the same bytes from the shared scratch).
                try c.routerFinalizeTop6(probs: scratch.probs, selected: stage.ids[jj],
                                         bias: w.expBias, hashTable: w.tid2eid,
                                         hashRows: w.tid2eidRows, token: tokens[jj],
                                         weights: d.fusedRouterFinalize ? stage.rw[jj] : nil,
                                         nExperts: d.nExperts,
                                         expertWeightScale: d.expertWeightScale)
                if !d.fusedRouterFinalize {
                    try c.routerWeights(probs: scratch.probs, selected: stage.ids[jj],
                                        weights: stage.rw[jj], nExperts: d.nExperts,
                                        expertWeightScale: d.expertWeightScale)
                }
            }
            if !slabDirect {
                var copies: [(src: GPUTensor, srcOff: Int, dst: GPUTensor, dstOff: Int, bytes: Int)] = [
                    (fb.curMat2, r * d.nEmbd * 4, stage.cur[jj], 0, d.nEmbd * 4),
                    (fb.afterAttnMat, r * hcDim * 4, stage.attn[jj], 0, hcDim * 4),
                    (fb.splitF, r * 24 * 4, stage.split[jj], 0, 24 * 4),
                ]
                if let mm = stage.mm {
                    copies.append((fb.curMat2, r * d.nEmbd * 4, mm.curMat, jj * d.nEmbd * 4, d.nEmbd * 4))
                }
                try c.blitCopies(copies)
            }
        }
        try c.phase("router")                     // DS4_PROFILE_PREFILL boundary
    }

    /// Attention-only A3 (lever-1 fallback): per-token heads blit + unchanged
    /// tail + phase-B snapshot blits.
    // @_optimize(none): il pass LoopInvariantCodeMotion di Swift 6.3 crasha
    // (SIGSEGV in getBorrowIntroducers) ottimizzando questi encoder in -O.
    // Sono orchestrazione di dispatch Metal: il costo CPU è trascurabile.
    @_optimize(none)
    private func encodePerTokenTail(_ c: GraphContext, _ fb: PrefillStage.FlashBatch,
                                    w: LayerWeights, layerRope: RopeParams,
                                    cur: [GPUTensor], stage: PrefillStage,
                                    j: Int, jEnd: Int, posBase: Int, tokens: [Int]) throws {
        let qBytes = d.qDim * 4
        for (r, jj) in (j..<jEnd).enumerated() {
            let pos = posBase + jj
            try c.blitCopies([(fb.heads, r * qBytes, scratch.heads, 0, qBytes)])
            try c.decodeRouteAttnTail(curHc: cur[jj], w: w, s: scratch, d: d, rope: layerRope,
                                      pos: pos, token: tokens[jj], rmsEps: rmsEps, hcEps: hcEps,
                                      attnSplit: fb.split[r])
            // Same phase-B snapshots as the per-token run (see encodeRouteInto
            // call site): FFN inputs + router selection, blitted before the
            // next token's tail overwrites the scratch.
            var copies: [(src: GPUTensor, srcOff: Int, dst: GPUTensor, dstOff: Int, bytes: Int)] = [
                (scratch.cur, 0, stage.cur[jj], 0, d.nEmbd * 4),
                (scratch.afterAttn, 0, stage.attn[jj], 0, d.nHC * d.nEmbd * 4),
                (scratch.split, 0, stage.split[jj], 0, 24 * 4),
                (scratch.selected, 0, stage.ids[jj], 0, d.k * 4),
                (scratch.rw, 0, stage.rw[jj], 0, d.k * 4),
            ]
            if let mm = stage.mm {
                copies.append((scratch.cur, 0, mm.curMat, jj * d.nEmbd * 4, d.nEmbd * 4))
            }
            try c.blitCopies(copies)
        }
    }
}
