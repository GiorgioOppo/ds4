import Foundation
import Metal
import DS4Core

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
                let hiddens = try autoreleasepool {
                    try prefillRange(tokens, start: start, end: end, posBase: startPos)
                }
                lastHC = hiddens.last
                start = end
            }
        } catch {
            // The per-token path (n==1 chunks) commits its routed FFN async: a
            // cancellation/gather error must never escape with a cb in flight
            // over state the caller will tear down (same invariant as forward).
            drainFFN()
            throw error
        }
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
                let hiddens = try autoreleasepool {
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
            drainFFN()
            throw error
        }
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
        let hcDim = d.nHC * d.nEmbd
        var cur: [GPUTensor] = try (0..<n).map { _ in try .zeros(rt, floatCount: hcDim) }
        var other: [GPUTensor] = try (0..<n).map { _ in try .zeros(rt, floatCount: hcDim) }
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
        let stage: PrefillStage? = (expertGather != nil && n > 1)
            ? try PrefillStage(rt, n: n, d: d, mmPath: prefillMM, maxUnion: maxUnionExperts,
                               flashBatch: flashBatch) : nil
        for i in 0..<nLayers {
            // Per-layer pool drain: the layer weights and per-token command
            // buffers are autoreleased ObjC objects — without this they pile up
            // for the whole chunk instead of freeing at each EVICT.
            try autoreleasepool {
                try Task.checkCancellation()
                let w = try layerProvider(i)            // LOAD layer i ONCE for all chunk tokens
                if i + 1 < nLayers { prefetch?(i + 1) }   // read-ahead next layer (overlaps its I/O)
                let layerRope = ropeParams(layer: i)
                if let gather = expertGather, n > 1, let stage {
                    try batchedExpertLayer(i, w: w, layerRope: layerRope, cur: cur, other: other,
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
            }
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
        let stage: PrefillStage? = (specVerifyBatch && n > 1 && remoteExperts == nil
                                    && expertGather != nil && slotCache != nil
                                    && !profileRoute && d.activeExperts >= d.k)
            ? try PrefillStage(rt, n: n, d: d, mmPath: false, maxUnion: d.k) : nil
        do {
            for i in 0..<nLayers {
                try autoreleasepool {
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
                }
            }
        } catch {
            drainFFN()   // stesso invariante di prefill: mai un cb in volo oltre l'errore
            throw error
        }
        drainFFN()
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
            let acquired: (pool: ExpertSlotCache.LayerPool, slots: [Int32])
            do { acquired = try cache.acquire(layer: i, ids: ids) }
            catch { c1.waitCompleted(); throw error }
            let (pool, slots) = acquired
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
            } catch { c1.waitCompleted(); throw error }
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
    private func batchedExpertLayer(_ i: Int, w: LayerWeights, layerRope: RopeParams,
                                    cur: [GPUTensor], other: [GPUTensor], n: Int, posBase: Int,
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
            if jEnd <= j {
                // Per-token path (indexer active, or batching off).
                try autoreleasepool {
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
                try autoreleasepool {
                    try Task.checkCancellation()
                    try encodeFlashRun(fb, i, w: w, layerRope: layerRope, cur: cur, stage: stage,
                                       j: j, jEnd: jEnd, posBase: posBase, tokens: tokens)
                }
            } else {
                try autoreleasepool {
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
            try autoreleasepool {
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
                    // SHARED-expert FFN: batched too when the shared weights
                    // are Q8_0 (gate/up mm -> rows-swiglu at unit weight ->
                    // down mm, one matmul each for the whole group instead of
                    // 3 matvecs per token). DS4_SHARED_Q4 residents keep the
                    // per-token path (the id-kernel with k=1 has no mm twin).
                    let sharedMM = !w.sharedGateQ4 && !w.sharedUpQ4 && !w.sharedDownQ4
                    let actBase = group.tokens.lowerBound * d.nEmbd * 4
                    if sharedMM {
                        try c2.encodeMMDenseQ8(weight: w.sharedGate, act: mm.curMat, actBase: actBase,
                                               out: mm.sGate, inDim: d.nEmbd, outDim: d.sharedFfn, nTok: gTok)
                        try c2.encodeMMDenseQ8(weight: w.sharedUp, act: mm.curMat, actBase: actBase,
                                               out: mm.sUp, inDim: d.nEmbd, outDim: d.sharedFfn, nTok: gTok)
                        try c2.moeSwiGLUWeight(gate: mm.sGate, up: mm.sUp, weights: mm.ones,
                                               mid: mm.sMid, width: d.sharedFfn, rows: gTok,
                                               clampValue: d.swigluClamp)
                        try c2.encodeMMDenseQ8(weight: w.sharedDown, act: mm.sMid, actBase: 0,
                                               out: mm.sOut, inDim: d.sharedFfn, outDim: d.nEmbd, nTok: gTok)
                    }
                    // Per-token tail: blit of the token's shared row + k down
                    // rows into the scratch, then sum6/add/HC expand (identical
                    // dispatches to the matvec path's tail).
                    let tokBytes = d.k * d.nEmbd * 4
                    for (tl, j) in group.tokens.enumerated() {
                        if sharedMM {
                            try c2.blitCopies([
                                (src: mm.sOut, srcOff: tl * d.nEmbd * 4,
                                 dst: scratch.sharedOut, dstOff: 0, bytes: d.nEmbd * 4),
                                (src: mm.down6, srcOff: tl * tokBytes,
                                 dst: scratch.down6, dstOff: 0, bytes: tokBytes),
                            ])
                        } else {
                            try c2.decodeSharedFFN(w: w, s: scratch, d: d, cur: stage.cur[j])
                            try c2.blitCopies([(src: mm.down6, srcOff: tl * tokBytes,
                                                dst: scratch.down6, dstOff: 0, bytes: tokBytes)])
                        }
                        try c2.decodeRoutedTail(s: scratch, d: d, outHc: other[j],
                                                afterAttn: stage.attn[j], split: stage.split[j])
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
                        remapped.withUnsafeBytes {
                            memcpy(stage.ids[j].buffer.contents() + stage.ids[j].byteOffset,
                                   $0.baseAddress!, $0.count)
                        }
                        writeFloats(weights, into: stage.rw[j])
                    }
                    try Task.checkCancellation()
                    t = Date()
                    let c2 = GraphContext(rt); try c2.begin()
                    for j in group.tokens {
                        try c2.decodeExperts(w: w, s: scratch, d: d, gateExp: g, upExp: u, downExp: dn,
                                             ids: stage.ids[j], outHc: other[j], activeK: d.k,
                                             cur: stage.cur[j], afterAttn: stage.attn[j],
                                             split: stage.split[j], rw: stage.rw[j])
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
    private func encodeFlashRun(_ fb: PrefillStage.FlashBatch, _ i: Int, w: LayerWeights,
                                layerRope: RopeParams, cur: [GPUTensor], stage: PrefillStage,
                                j: Int, jEnd: Int, posBase: Int, tokens: [Int]) throws {
        let nq = jEnd - j
        let posFirst = posBase + j
        let rawLo0 = max(0, posFirst + 1 - d.nSWA)
        let nRawSpan = posBase + jEnd - rawLo0
        let denseMM = prefillDenseMM && prefillDenseEligible(w)
        let c = GraphContext(rt); try c.begin()
        var nCompVis = [Int](repeating: 0, count: nq)

        // ── A1: pre-attention halves (split into helper methods — one big
        // function here crashes the release optimizer's LICM pass).
        if denseMM {
            try encodeDensePre(c, fb, i, w: w, layerRope: layerRope, cur: cur,
                               j: j, jEnd: jEnd, posBase: posBase, nCompVis: &nCompVis)
        } else {
            try encodePerTokenPre(c, fb, i, w: w, layerRope: layerRope, cur: cur,
                                  j: j, jEnd: jEnd, posBase: posBase, nCompVis: &nCompVis)
        }

        // ── A2: ONE multi-query FlashAttention over the whole run.
        let nComp = nCompVis[nq - 1]
        // CPU mask fill BEFORE commit: the buffer is only ever read by this
        // run's command buffer (the previous run committed with a blocking
        // wait), so a CPU write here cannot race a GPU reader.
        let maskPtr = (fb.mask.buffer.contents() + fb.mask.byteOffset)
            .bindMemory(to: UInt16.self, capacity: nq * (nRawSpan + nComp))
        GraphContext.fillPrefillAttnMask(maskPtr, nQ: nq, posFirst: posFirst,
                                         rawStart: rawLo0, rawSpan: nRawSpan,
                                         window: d.nSWA, nCompVis: nCompVis, nComp: nComp)
        try c.flashAttnPrefill(q: fb.qMat, kvF32: rawCaches[i], kvF16: fb.kvF16,
                               mask: fb.mask, sinks: w.attnSinks, pad: fb.pad, blk: fb.blk,
                               heads: fb.heads, nHead: d.nHead, nQ: nq,
                               rawSpan: nRawSpan, rawStartRow: rawLo0,
                               comp: compStates[i]?.cache, nComp: nComp)

        // ── A3: post-attention tails.
        if denseMM {
            try encodeDenseTail(c, fb, w: w, layerRope: layerRope, stage: stage,
                                j: j, jEnd: jEnd, posBase: posBase, tokens: tokens)
        } else {
            try encodePerTokenTail(c, fb, w: w, layerRope: layerRope, cur: cur, stage: stage,
                                   j: j, jEnd: jEnd, posBase: posBase, tokens: tokens)
        }
        c.commit()               // one sync for the whole run
        profile.prefillFlashRuns += 1
        if denseMM { profile.prefillDenseRuns += 1 }
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
                                j: Int, jEnd: Int, posBase: Int, nCompVis: inout [Int]) throws {
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
            let wb = comp.width * 4
            for r in 0..<nq {
                nCompVis[r] = try c.runCompressorTail(
                    kvCur: fb.compKvMat.subview(byteOffset: r * wb, byteLength: wb, count: comp.width),
                    scCur: fb.compScMat.subview(byteOffset: r * wb, byteLength: wb, count: comp.width),
                    ape: cape, normW: cnorm, comp: comp, rope: layerRope,
                    pos: posFirst + r, rmsEps: rmsEps, nRot: d.nRot, finalize: .fp8)
            }
        } else if let comp = compStates[i] {
            // Missing compressor weights: historical no-op semantics.
            for r in 0..<nq { nCompVis[r] = comp.count }
        }
        if hasIdxState, let idx = indexStates[i], let ikv = w.idxKv, let igate = w.idxGate,
           let iape = w.idxApe, let inorm = w.idxNorm {
            try encodeCompProjBatched(c, kv: ikv, gate: igate, act: fb.curMat,
                                      kvOut: fb.idxKvMat, scOut: fb.idxScMat,
                                      width: idx.width, nq: nq, q8: w.idxCompQ8)
            let wb = idx.width * 4
            for r in 0..<nq {
                _ = try c.runCompressorTail(
                    kvCur: fb.idxKvMat.subview(byteOffset: r * wb, byteLength: wb, count: idx.width),
                    scCur: fb.idxScMat.subview(byteOffset: r * wb, byteLength: wb, count: idx.width),
                    ape: iape, normW: inorm, comp: idx, rope: layerRope,
                    pos: posFirst + r, rmsEps: rmsEps, nRot: d.nRot, finalize: .indexerQat)
            }
        }
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
        // ── A1d: batched KV path + per-token fp8 ring store.
        try encodeMMDense(c, weight: w.kvW, q4: w.kvQ4, act: fb.curMat, out: fb.kvMat,
                          inDim: d.nEmbd, outDim: d.headDim, nTok: nq)
        try c.rmsNorm(fb.kvMat, weight: w.kvNorm, out: fb.kvMat, rows: nq, n: d.headDim, eps: rmsEps)
        try c.ropeTail(x: fb.kvMat, nTok: nq, nHead: 1, headDim: d.headDim, nRot: d.nRot,
                       nCtxOrig: layerRope.nCtxOrig, freqBase: layerRope.freqBase,
                       freqScale: layerRope.freqScale, extFactor: layerRope.extFactor,
                       attnFactor: layerRope.attnFactor, betaFast: layerRope.betaFast,
                       betaSlow: layerRope.betaSlow, pos0: posFirst, posStep: 1)
        let rawRows = rawCaches[i].count / d.headDim
        for r in 0..<nq {
            let kvRow = fb.kvMat.subview(byteOffset: r * d.headDim * 4,
                                         byteLength: d.headDim * 4, count: d.headDim)
            try c.kvFP8Store(kv: kvRow, rawCache: rawCaches[i], headDim: d.headDim,
                             nRot: d.nRot, rawRow: (posFirst + r) % rawRows)
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
                                 j: Int, jEnd: Int, posBase: Int, tokens: [Int]) throws {
        let nq = jEnd - j
        let posFirst = posBase + j
        let hcDim = d.nHC * d.nEmbd
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
        // ── A3b: batched attention-residual HC expand + pre-FFN HC reduce.
        try c.hcExpand4(blockOut: fb.blockOutMat, residual: fb.hcMat,
                        post: fb.splitA, comb: fb.splitA, blockAdd: nil,
                        out: fb.afterAttnMat, nEmbd: d.nEmbd, nTokens: nq,
                        postByteOffset: 4 * 4, combByteOffset: 8 * 4,
                        splitTokenStride: 24 * 4)
        try encodeHCReduceBatched(c, fb, x: fb.afterAttnMat, mixerFn: w.hcFfnFn, scale: w.ffnScale,
                                  base: w.ffnBase, norm: w.ffnNorm,
                                  split: fb.splitF, out: fb.curMat2, nq: nq)
        // ── A3c: batched router logits, per-token finalize + snapshots.
        if d.routerF16 {
            try c.encodeMMDenseF16(weight: w.routerW, act: fb.curMat2, actBase: 0,
                                   out: fb.logitsMat, inDim: d.nEmbd, outDim: d.nExperts, nTok: nq)
        } else {
            try c.encodeMMDenseQ8(weight: w.routerW, act: fb.curMat2, actBase: 0,
                                  out: fb.logitsMat, inDim: d.nEmbd, outDim: d.nExperts, nTok: nq)
        }
        for (r, jj) in (j..<jEnd).enumerated() {
            let logitsRow = fb.logitsMat.subview(byteOffset: r * d.nExperts * 4,
                                                 byteLength: d.nExperts * 4, count: d.nExperts)
            if d.fusedRouterProbs {
                try c.routerProbabilities(logits: logitsRow, probabilities: scratch.probs,
                                          width: d.nExperts)
            } else {
                try c.unary(logitsRow, op: .softplus, out: scratch.sp, width: d.nExperts)
                try c.unary(scratch.sp, op: .sqrt, out: scratch.probs, width: d.nExperts)
            }
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
