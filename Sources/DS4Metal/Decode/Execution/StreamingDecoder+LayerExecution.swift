import Foundation
import Metal
import DS4Core

extension StreamingDecoder {
    /// Commit a routed-FFN command buffer. Async by default (DS4_ASYNC_FFN):
    /// the next layer's route commit+wait is on the same in-order queue, so
    /// correctness is by queue order and the CPU encode overlaps this buffer's
    /// GPU execution. DS4_PROFILE_ROUTE keeps the synchronous wait (accurate
    /// per-phase attribution beats the overlap when profiling).
    private func commitFFN(_ c: GraphContext) {
        if asyncFFN && !profileRoute {
            c.commitAsync()
            inflightFFN = c
        } else {
            c.commit()
        }
    }

    /// Join the in-flight routed FFN (end of token, and every error path): the
    /// caller is about to read GPU results CPU-side (output head readback,
    /// readHC, KV export) or to tear down/rebuild state.
    func drainFFN() {
        inflightFFN?.waitCompleted()
        inflightFFN = nil
    }

    /// Speculative look-ahead: prefill layer i+1's slot pool while the GPU
    /// computes layer i (its own gather just finished, so the SSD is idle until
    /// the next layer's demand fill). The id list is resolved on the DECODE
    /// thread (usage prior / tid2eid mmap read — cheap); only the I/O moves to
    /// the background queue. Decode-only: the batched prefill has its own
    /// union pipeline.
    func kickLookahead(after i: Int, token: Int) {
        guard let lookahead, let cache = slotCache, i + 1 < nLayers else { return }
        let next = i + 1
        let ids = lookahead(next, token)
        guard !ids.isEmpty else { return }
        lookaheadQ.async { cache.prefill(layer: next, ids: ids) }
    }

    /// One decode layer for one token: `cur` (HC in) -> `other` (HC out). Writes
    /// KV[i][pos], updates compStates[i]. Shared by `forward` (decode) and the
    /// layer-major `prefill` — identical numerics either way.
    func runLayer(_ i: Int, w: LayerWeights, layerRope: RopeParams,
                          cur: GPUTensor, other: GPUTensor, pos: Int, nKeys: Int, token: Int) throws {
        // Kick the NEXT layer's look-ahead at the START of this one: the fill
        // window becomes the whole layer (route + attention + FFN, ~2x the
        // post-gather window) instead of the few ms before the next acquire.
        // Its I/O shares the SSD with this layer's own gather, but the disk's
        // parallel ceiling is well above the demand queue depth and the demand
        // path preempts on contention for the same layer's lock.
        if remoteExperts == nil { kickLookahead(after: i, token: token) }
        if let remote = remoteExperts {
            // EXPERT PARALLELISM (coordinatore VERTICALE, Fase C): la FFN
            // routed è calcolata DAI WORKER — qui route/attention/selezione
            // come in locale, poi scatter/gather di rete al posto del gather
            // SSD. La FFN condivisa (c1, asincrona) copre la latenza di rete
            // come copriva quella del disco.
            var t = Date()
            try encodeRoute(i, w: w, layerRope: layerRope, curHc: cur, pos: pos, nKeys: nKeys, token: token)
            profile.routeS += Date().timeIntervalSince(t)
            let (ids, rw) = readRouteSelection(layer: i)
            let c1 = GraphContext(rt); try c1.begin()
            try c1.decodeSharedFFN(w: w, s: scratch, d: d)
            c1.commitAsync()
            // Attivazione (s.cur, finale dopo il route committato) → CPU.
            let nE = d.nEmbd
            let curPtr = (scratch.cur.buffer.contents() + scratch.cur.byteOffset)
                .bindMemory(to: Float.self, capacity: nE)
            let activation = Array(UnsafeBufferPointer(start: curPtr, count: nE))
            t = Date()
            let partialSum: [Float]
            do { partialSum = try remote(i, ids, rw, activation) }
            catch { c1.waitCompleted(); throw error }
            guard partialSum.count == nE else {
                c1.waitCompleted()
                throw MetalError.unsupported("remoteExperts: somma parziale di taglia \(partialSum.count) ≠ \(nE)")
            }
            profile.gatherS += Date().timeIntervalSince(t)
            // Upload nel tensore a PARITÀ alternata: il c2 asincrono del layer
            // precedente può ancora leggere il SUO tensore (stesso schema di
            // slotsScratch); i cb più vecchi sono già completati per l'ordine
            // in-order della queue (il route di i-1 è stato atteso).
            let target = (i & 1) == 0 ? remotePartialA : remotePartialB
            _ = partialSum.withUnsafeBytes {
                memcpy(target.buffer.contents() + target.byteOffset, $0.baseAddress!, nE * 4)
            }
            t = Date()
            c1.waitCompleted()   // s.sharedOut pronto (di solito già finito)
            let c2 = GraphContext(rt); try c2.begin()
            try c2.decodeRemoteTail(s: scratch, d: d, partial: target, outHc: other)
            commitFFN(c2)
            profile.expertsS += Date().timeIntervalSince(t)
        } else if let gather = expertGather {
            // Phase 1: route (own cb) -> read the selected ids (top-K reduced).
            var t = Date()
            try encodeRoute(i, w: w, layerRope: layerRope, curHc: cur, pos: pos, nKeys: nKeys, token: token)
            profile.routeS += Date().timeIntervalSince(t)
            let (ids, rw) = readRouteSelection(layer: i)
            let K = ids.count
            if K < d.k {
                writeFloats(rw, into: scratch.rw)
                zeroDown6(from: K)
            }
            // I/O–compute OVERLAP: the shared-expert FFN does not depend on the
            // routing selection, so commit it asynchronously FIRST — the GPU
            // crunches it while the CPU gathers the routed experts from the SSD.
            // On error the in-flight buffer is waited before rethrowing, so a
            // rebuilt turn can never race a stale write into the scratch.
            let c1 = GraphContext(rt); try c1.begin()
            try c1.decodeSharedFFN(w: w, s: scratch, d: d)
            c1.commitAsync()
            // The slot cache is a single size-class (the model-global/first-layer
            // quant). A mixed-precision layer (different expert bytes) can't share
            // the pool, so it falls through to the per-layer-correct gather path.
            let onClass = w.gateQuant == d.gateQuant && w.upQuant == d.upQuant && w.downQuant == d.downQuant
            if let cache = slotCache, onClass {
                // Persistent + changing experts: hits are already resident in the
                // layer's GPU pool (zero copies); only misses are filled from the
                // mmap. The matvec indexes the pool with slot ids.
                t = Date()
                let h0 = cache.hits, m0 = cache.misses, p0 = cache.prefilled
                let acquired: (pool: ExpertSlotCache.LayerPool, slots: [Int32])
                do { acquired = try cache.acquire(layer: i, ids: ids) }
                catch { c1.waitCompleted(); throw error }
                let (pool, slots) = acquired
                profile.gatherS += Date().timeIntervalSince(t)
                // Deltas, not cumulative totals: the cache counts since load,
                // the profile since resetProfile().
                profile.expertHits += cache.hits - h0
                profile.expertMisses += cache.misses - m0
                profile.expertPrefilled += cache.prefilled - p0
                profile.gatherBytes += (cache.misses - m0) * cache.bytesPerExpert
                // Persistent staging (no per-layer alloc), A/B by layer parity:
                // with the async FFN the PREVIOUS layer's command buffer may
                // still be reading its ids buffer while this layer stages its own.
                let slotsBuf = (i & 1) == 0 ? slotsScratch : slotsScratchB
                _ = slots.withUnsafeBytes {
                    memcpy(slotsBuf.buffer.contents(), $0.baseAddress!, $0.count)
                }
                t = Date()
                c1.waitCompleted()   // s.sharedOut ready (usually already done)
                let c2 = GraphContext(rt); try c2.begin()
                try c2.decodeRoutedExperts(w: w, s: scratch, d: d, gateExp: pool.gate,
                                           upExp: pool.up, downExp: pool.down,
                                           ids: slotsBuf, outHc: other, activeK: K,
                                           expertStride: slotCacheStride)
                commitFFN(c2)
                profile.expertsS += Date().timeIntervalSince(t)
            } else {
                // Gather ONLY the selected experts (EXPERT I/O from the mmap), then phase 2.
                t = Date()
                let gathered: (GPUTensor, GPUTensor, GPUTensor)
                do { gathered = try gather(i, ids) }
                catch { c1.waitCompleted(); throw error }
                let (g, u, dn) = gathered
                profile.gatherS += Date().timeIntervalSince(t)
                profile.gatherBytes += g.byteLength + u.byteLength + dn.byteLength
                t = Date()
                c1.waitCompleted()   // s.sharedOut ready (usually already done)
                let c2 = GraphContext(rt); try c2.begin()
                try c2.decodeRoutedExperts(w: w, s: scratch, d: d, gateExp: g, upExp: u, downExp: dn,
                                           ids: idsPacked, outHc: other, activeK: K)
                commitFFN(c2)
                profile.expertsS += Date().timeIntervalSince(t)
            }
        } else {
            let t = Date()
            try encodeRoute(i, w: w, layerRope: layerRope, curHc: cur, pos: pos, nKeys: nKeys, token: token)
            let lc = GraphContext(rt); try lc.begin()
            try lc.decodeExperts(w: w, s: scratch, d: d, gateExp: w.expGate, upExp: w.expUp,
                                 downExp: w.expDown, ids: scratch.selected, outHc: other)
            commitFFN(lc)                    // COMPUTE (cb retains w's buffers until completed)
            profile.layerOtherS += Date().timeIntervalSince(t)
        }
        profile.layers += 1
    }

    /// Encode (and COMMIT) the route for one token on layer `i`. When the NSA
    /// indexer is active (ratio-4 layer with more compressed rows than the top-K),
    /// the command buffer is split at the indexer scores: commit phase 1a, run the
    /// CPU top-K to write the compressed-row mask, then encode the attention —
    /// the C "dense top-k mask" path (indexer_allowed_decode_one). Otherwise a
    /// single command buffer, numerically identical to the pre-indexer code.
    func encodeRoute(_ i: Int, w: LayerWeights, layerRope: RopeParams,
                             curHc: GPUTensor, pos: Int, nKeys: Int, token: Int) throws {
        let idx = indexStates[i]
        // The indexer carries TWO independent weight sets. The compressor pair
        // (idxKv/idxGate) feeds the recurrent STATE update: it runs every token
        // and its cache must stay coherent (KV snapshots export it, and a later
        // activation reads all past rows). attn_q_b/proj are read ONLY by the
        // top-K SCORING. With the lazy staging (DenseStreamer skipIndexerScoring)
        // the scoring pair may not be staged at all, so the two gates SPLIT:
        // the state keeps updating on the compressor pair alone, while the
        // active path additionally requires the scoring pair.
        let hasIdxState = w.idxKv != nil && w.idxGate != nil
        let hasIdxScoring = hasIdxState && w.idxQB != nil && w.idxProj != nil
        let active = hasIdxScoring && indexerActive(i, pos: pos)
        if active, let idx, gpuIndexerTopK {
            // Score -> exact top-K mask -> attention in ONE command buffer. The
            // recurrent compressor advances idx.count synchronously while these
            // dispatches are encoded, so it is already the correct score count.
            let c = GraphContext(rt); if profileRoute { c.phaseTimes = [:] }; try c.begin()
            let nComp = try c.decodeRoutePre(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope,
                                            rawCache: rawCaches[i], pos: pos, rmsEps: rmsEps, hcEps: hcEps,
                                            comp: compStates[i], idx: hasIdxState ? idx : nil,
                                            indexerScoring: true)
            let nRaw = min(nKeys, d.nSWA)
            try c.indexerTopKMask(scores: scratch.idxScores, mask: scratch.mask,
                                  nRaw: nRaw, nComp: nComp, nScores: idx.count,
                                  topK: d.indexerTopK)
            maskDirtyCount = max(maskDirtyCount, nRaw + nComp)
            try c.decodeRouteAttn(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope,
                                  rawCache: rawCaches[i], nKeys: nKeys, pos: pos, token: token,
                                  rmsEps: rmsEps, hcEps: hcEps, nComp: nComp,
                                  comp: compStates[i])
            if profileRoute { try c.phase("router") }
            c.commit()
            if profileRoute { accumulateRoutePhases(c, nil) }
        } else if active, let idx {
            // Indexer layers always split (CPU top-k sits between pre and attn). The
            // phase() boundaries inside decodeRoutePre/Attn are no-ops unless profiling.
            let c1 = GraphContext(rt); if profileRoute { c1.phaseTimes = [:] }; try c1.begin()
            let nComp = try c1.decodeRoutePre(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope,
                                              rawCache: rawCaches[i], pos: pos, rmsEps: rmsEps, hcEps: hcEps,
                                              comp: compStates[i], idx: hasIdxState ? idx : nil,
                                              indexerScoring: true)
            try c1.phase("kv")
            c1.commit()
            applyIndexerMask(nKeys: nKeys, nComp: nComp, nIdxComp: idx.count)
            let c2 = GraphContext(rt); if profileRoute { c2.phaseTimes = [:] }; try c2.begin()
            try c2.decodeRouteAttn(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope, rawCache: rawCaches[i],
                                   nKeys: nKeys, pos: pos, token: token, rmsEps: rmsEps, hcEps: hcEps,
                                   nComp: nComp, comp: compStates[i])
            try c2.phase("router")
            c2.commit()
            if profileRoute { accumulateRoutePhases(c1, c2) }
        } else if profileRoute {
            // Profiling: detailed split. Extra commits inflate ABSOLUTE time.
            clearMaskIfDirty()
            let c = GraphContext(rt); c.phaseTimes = [:]; try c.begin()
            let nComp = try c.decodeRoutePre(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope,
                                             rawCache: rawCaches[i], pos: pos, rmsEps: rmsEps, hcEps: hcEps,
                                             comp: compStates[i], idx: hasIdxState ? idx : nil,
                                             indexerScoring: false)
            try c.phase("kv")
            try c.decodeRouteAttn(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope, rawCache: rawCaches[i],
                                  nKeys: nKeys, pos: pos, token: token, rmsEps: rmsEps, hcEps: hcEps,
                                  nComp: nComp, comp: compStates[i])
            try c.phase("router")
            c.commit()
            accumulateRoutePhases(c, nil)
        } else {
            clearMaskIfDirty()
            let c1 = GraphContext(rt); try c1.begin()
            let nComp = try c1.decodeRoutePre(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope,
                                              rawCache: rawCaches[i], pos: pos, rmsEps: rmsEps, hcEps: hcEps,
                                              comp: compStates[i], idx: hasIdxState ? idx : nil,
                                              indexerScoring: false)
            try c1.decodeRouteAttn(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope, rawCache: rawCaches[i],
                                   nKeys: nKeys, pos: pos, token: token, rmsEps: rmsEps, hcEps: hcEps,
                                   nComp: nComp, comp: compStates[i])
            c1.commit()
        }
    }

    /// Accumulate per-sub-phase route timings (DS4_PROFILE_ROUTE) into the profile.
    private func accumulateRoutePhases(_ a: GraphContext, _ b: GraphContext?) {
        func add(_ pt: [String: Double]) {
            profile.routeCompS += pt["comp", default: 0]
            profile.routeQS += pt["q", default: 0]
            profile.routeKvS += pt["kv", default: 0]
            profile.routeAttnPhaseS += pt["attn", default: 0]
            profile.routeOutProjS += pt["out-proj", default: 0]
            profile.routeHcFfnS += pt["hc-ffn", default: 0]
            profile.routeRouterS += pt["router", default: 0]
        }
        if let pt = a.phaseTimes { add(pt) }
        if let pt = b?.phaseTimes { add(pt) }
    }

    /// Decode sparse threshold: the C Metal decode keeps attention DENSE over all
    /// compressed rows until n_comp exceeds this, because around the ~2K frontier
    /// the sparse path's score/top-k setup dominates the smaller attention scan
    /// (metal_graph_decode_indexer_sparse_threshold, default 1024). It changes
    /// only WHICH implementation consumes the compressed rows — the 512-row
    /// indexer selection (indexerTopK) is a separate, lower bound. Same env
    /// override and allowed values as the C.
    static let indexerSparseThreshold: Int = {
        let allowed: Set<Int> = [64, 128, 256, 512, 1024, 2048, 4096]
        if let s = ProcessInfo.processInfo.environment["DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD"],
           let v = Int(s.trimmingCharacters(in: .whitespaces)), allowed.contains(v) {
            return v
        }
        return 1024
    }()

    /// Static proof that the indexer top-K can NEVER activate in this session:
    /// the densest indexer layers (ratio 4) emit at most maxKeys/4 compressed
    /// rows over the whole context, and activation needs a prospective count
    /// STRICTLY greater than both the sparse threshold and the top-K. When the
    /// bound can't be exceeded, the scoring projections (indexer.attn_q_b +
    /// indexer.proj, ~360 MB/token on Flash) are dead weight in the dense
    /// stream and staging them is skipped at load. Recomputed on every load
    /// from the live maxKeys/threshold, so a larger context or a lower
    /// DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD re-enables the staging.
    static func indexerCanEverActivate(maxKeys: Int, topK: Int) -> Bool {
        let maxProspective = maxKeys / 4
        return maxProspective > indexerSparseThreshold && maxProspective > topK
    }

    /// Will the indexer restrict this token's compressed rows on layer `i`?
    /// (prospective count: the compressor may emit one more row for this token.)
    /// `extraRows` = rows the tokens BEFORE this one in a not-yet-encoded batch
    /// will emit — the batched route phase checks activation prospectively for
    /// the whole run before encoding any of it.
    /// C condition (ds4.c:15246): layer_n_comp > sparse_threshold AND
    /// layer_n_index_comp > DS4_N_INDEXER_TOP_K. On ratio-4 layers the attention
    /// and indexer compressors emit in lockstep, so one prospective count serves
    /// both comparisons.
    func indexerActive(_ i: Int, pos: Int, extraRows: Int = 0) -> Bool {
        guard let idx = indexStates[i] else { return false }
        let prospective = idx.count + extraRows + (((pos + 1) % idx.ratio) == 0 ? 1 : 0)
        return prospective > Self.indexerSparseThreshold && prospective > d.indexerTopK
    }

    /// Encode ONE token's full route (pre + attention) into `c` WITHOUT
    /// committing — the batched phase A packs many tokens per command buffer.
    /// Caller guarantees the indexer is NOT active for (i, pos) and route
    /// profiling is off (both need CPU work mid-route). Same two encodes, same
    /// order as the per-token non-indexer path in encodeRoute.
    func encodeRouteInto(_ c: GraphContext, _ i: Int, w: LayerWeights, layerRope: RopeParams,
                                 curHc: GPUTensor, pos: Int, nKeys: Int, token: Int) throws {
        let hasIdxState = w.idxKv != nil && w.idxGate != nil    // state-only gate (see encodeRoute)
        let nComp = try c.decodeRoutePre(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope,
                                         rawCache: rawCaches[i], pos: pos, rmsEps: rmsEps, hcEps: hcEps,
                                         comp: compStates[i], idx: hasIdxState ? indexStates[i] : nil,
                                         indexerScoring: false)  // caller guarantees no scoring here
        try c.decodeRouteAttn(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope, rawCache: rawCaches[i],
                              nKeys: nKeys, pos: pos, token: token, rmsEps: rmsEps, hcEps: hcEps,
                              nComp: nComp, comp: compStates[i])
    }

    /// CPU top-K over the indexer scores (s.idxScores[0..nIdxComp)) → f16 mask:
    /// raw window rows stay 0; compressed row c gets 0 if selected, -inf if not.
    /// Ties keep the LOWEST row index (the C argmax scan picks the first best).
    /// Selection is heap-based O(n log k), NOT a full sort: it runs per ratio-4
    /// layer per token, and n grows with the context (~nKeys/4).
    private func applyIndexerMask(nKeys: Int, nComp: Int, nIdxComp: Int) {
        let nRaw = nKeys - max(0, nKeys - d.nSWA)
        let scores = scratch.idxScores.buffer.contents()
            .advanced(by: scratch.idxScores.byteOffset).bindMemory(to: Float.self, capacity: nIdxComp)
        let allowed = IndexerSelect.allowedTopK(scores: scores, count: nIdxComp, k: d.indexerTopK)

        let total = nRaw + nComp
        let mask = scratch.mask.buffer.contents().bindMemory(to: UInt16.self, capacity: total)
        let negInf = Half.bits(-Float.infinity)
        for j in 0..<nRaw { mask[j] = 0 }
        for c in 0..<nComp {
            let ok = c < nIdxComp ? allowed[c] : true
            mask[nRaw + c] = ok ? 0 : negInf
        }
        maskDirtyCount = max(maskDirtyCount, total)
    }

    /// Zero the mask region a previous indexer selection dirtied (offsets shift
    /// every token, so a stale -inf would mask the wrong key).
    func clearMaskIfDirty() {
        guard maskDirtyCount > 0 else { return }
        memset(scratch.mask.buffer.contents(), 0, maskDirtyCount * 2)
        maskDirtyCount = 0
    }

}
