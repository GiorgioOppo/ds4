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
        let stage: PrefillStage? = (expertGather != nil && n > 1)
            ? try PrefillStage(rt, n: n, d: d, mmPath: prefillMM, maxUnion: maxUnionExperts) : nil
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
}
