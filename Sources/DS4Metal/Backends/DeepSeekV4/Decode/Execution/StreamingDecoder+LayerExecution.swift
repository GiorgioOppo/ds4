import Foundation
import Metal
import DS4Core

extension StreamingDecoder {
    /// Commit a routed-FFN command buffer. Async by default (DS4_ASYNC_FFN):
    /// the next layer's route commit+wait is on the same in-order queue, so
    /// correctness is by queue order and the CPU encode overlaps this buffer's
    /// GPU execution. DS4_PROFILE_ROUTE keeps the synchronous wait (accurate
    /// per-phase attribution beats the overlap when profiling). Internal (non
    /// private): anche la verifica speculativa batchata (+Prefill) committa le
    /// sue FFN routed da qui, così drainFFN traccia SEMPRE l'ultimo cb in volo.
    func commitFFN(_ c: GraphContext) {
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

    /// Establish a hard trial boundary for the in-process machine tuner.
    /// Besides the tracked GPU command buffer, expert look-ahead, mmap hints
    /// and dense-stream reads all run on background queues and may retain the
    /// old model. Draining them makes the following A/B load independent.
    public func quiesceForTeardown() {
        drainFullLayerGather()
        drainFFN()
        lookaheadQ.sync {}
        slotCache?.drainDemandLoads()
        teardownIODrain?()
    }

    /// Build slot-indexed GPU address tables for the indirect masked IQ2 pair.
    /// Addresses point at the exact start of each gate/up slot, so the kernel's
    /// selected ids remain the ordinary cache slot ids and numerics are unchanged.
    func stageSplitAddressTables(_ pool: ExpertSlotCache.LayerPool,
                                 fallbackStride: Int?,
                                 gateAddresses: GPUTensor,
                                 upAddresses: GPUTensor) throws -> Int {
        let slots = pool.owner.count
        guard slots > 0, slots <= 384 else {
            throw MetalError.unsupported("expert split: slot count \(slots) outside 1...384")
        }
        let gateStride: Int
        let upStride: Int
        if let record = pool.expertStride ?? fallbackStride {
            gateStride = record
            upStride = record
        } else {
            guard pool.gate.byteLength % slots == 0, pool.up.byteLength % slots == 0 else {
                throw MetalError.unsupported("expert split: non-integral pool stride")
            }
            gateStride = pool.gate.byteLength / slots
            upStride = pool.up.byteLength / slots
        }
        guard gateAddresses.byteLength >= 384 * MemoryLayout<UInt64>.stride,
              upAddresses.byteLength >= 384 * MemoryLayout<UInt64>.stride else {
            throw MetalError.unsupported("expert split: undersized GPU-address table")
        }
        let gate = gateAddresses.buffer.contents()
            .advanced(by: gateAddresses.byteOffset)
            .bindMemory(to: UInt64.self, capacity: 384)
        let up = upAddresses.buffer.contents()
            .advanced(by: upAddresses.byteOffset)
            .bindMemory(to: UInt64.self, capacity: 384)
        for i in 0..<384 { gate[i] = 0; up[i] = 0 }
        let gateBase = pool.gate.buffer.gpuAddress + UInt64(pool.gate.byteOffset)
        let upBase = pool.up.buffer.gpuAddress + UInt64(pool.up.byteOffset)
        guard gateBase != 0, upBase != 0 else {
            throw MetalError.unsupported("expert split: GPU addresses unavailable")
        }
        for slot in 0..<slots {
            gate[slot] = gateBase + UInt64(slot * gateStride)
            up[slot] = upBase + UInt64(slot * upStride)
        }
        return slots
    }

    /// Snapshot only the already-resident IQ2 gate/up slabs into a dedicated
    /// six-row staging pair. The asynchronous demand fill may write other slots
    /// in the source pool at the same time, but the resident GPU command never
    /// references that pool: this removes the CPU-write/GPU-read resource race
    /// while copying only the useful resident slabs.
    func stageSplitResidentPair(_ pool: ExpertSlotCache.LayerPool,
                                residentSlots: [Int32], activeMask: UInt32,
                                fallbackStride: Int?) throws -> Int {
        let slots = pool.owner.count
        guard slots > 0, slots <= 384, residentSlots.count >= d.k else {
            throw MetalError.unsupported("expert split: invalid resident slot geometry")
        }
        let expertBytes = (d.nEmbd / 256) * MoEQuant.iq2_xxs.blockBytes * d.expertFfn
        guard splitResidentGate.byteLength >= d.k * expertBytes,
              splitResidentUp.byteLength >= d.k * expertBytes else {
            throw MetalError.unsupported("expert split: undersized resident staging pair")
        }

        let gateStride: Int
        let upStride: Int
        if let record = pool.expertStride ?? fallbackStride {
            gateStride = record
            upStride = record
        } else {
            guard pool.gate.byteLength % slots == 0, pool.up.byteLength % slots == 0 else {
                throw MetalError.unsupported("expert split: non-integral resident pool stride")
            }
            gateStride = pool.gate.byteLength / slots
            upStride = pool.up.byteLength / slots
        }

        let gateTable = splitResidentGateAddresses.buffer.contents()
            .advanced(by: splitResidentGateAddresses.byteOffset)
            .bindMemory(to: UInt64.self, capacity: 384)
        let upTable = splitResidentUpAddresses.buffer.contents()
            .advanced(by: splitResidentUpAddresses.byteOffset)
            .bindMemory(to: UInt64.self, capacity: 384)
        for i in 0..<384 { gateTable[i] = 0; upTable[i] = 0 }

        let stagedGateBase = splitResidentGate.buffer.gpuAddress
            + UInt64(splitResidentGate.byteOffset)
        let stagedUpBase = splitResidentUp.buffer.gpuAddress
            + UInt64(splitResidentUp.byteOffset)
        guard stagedGateBase != 0, stagedUpBase != 0 else {
            throw MetalError.unsupported("expert split: resident GPU addresses unavailable")
        }

        let idsPtr = splitSlotsScratch.buffer.contents()
            .advanced(by: splitSlotsScratch.byteOffset)
            .bindMemory(to: Int32.self, capacity: d.k)
        let sourceGateBase = pool.gate.byteOffset
        let sourceUpBase = pool.up.byteOffset
        let destinationGateBase = splitResidentGate.byteOffset
        let destinationUpBase = splitResidentUp.byteOffset
        for j in 0..<d.k {
            idsPtr[j] = Int32(j)
            guard (activeMask & (UInt32(1) << UInt32(j))) != 0 else { continue }
            let slot = Int(residentSlots[j])
            guard slot >= 0, slot < slots else {
                throw MetalError.unsupported("expert split: invalid resident slot \(slot)")
            }
            let gateSource = sourceGateBase + slot * gateStride
            let upSource = sourceUpBase + slot * upStride
            let gateDestination = destinationGateBase + j * expertBytes
            let upDestination = destinationUpBase + j * expertBytes
            guard gateSource + expertBytes <= pool.gate.buffer.length,
                  upSource + expertBytes <= pool.up.buffer.length,
                  gateDestination + expertBytes <= splitResidentGate.buffer.length,
                  upDestination + expertBytes <= splitResidentUp.buffer.length else {
                throw MetalError.unsupported("expert split: resident slab outside its Metal buffer")
            }
            memcpy(splitResidentGate.buffer.contents() + gateDestination,
                   pool.gate.buffer.contents() + gateSource, expertBytes)
            memcpy(splitResidentUp.buffer.contents() + upDestination,
                   pool.up.buffer.contents() + upSource, expertBytes)
            gateTable[j] = stagedGateBase + UInt64(j * expertBytes)
            upTable[j] = stagedUpBase + UInt64(j * expertBytes)
        }
        return d.k
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
            let route = try encodeRoute(i, w: w, layerRope: layerRope, curHc: cur, pos: pos, nKeys: nKeys, token: token)
            // La FFN condivisa non dipende dalla selezione: encode+commit PRIMA
            // di attendere la route, così la GPU concatena route→FFN senza gap
            // (ordine della queue in-order) mentre la CPU aspetta la selezione.
            let c1 = GraphContext(rt)
            do { try c1.begin(); try c1.decodeSharedFFN(w: w, s: scratch, d: d) }
            catch { route.waitCompleted(); throw error }
            c1.commitAsync()
            route.waitCompleted()   // da qui s.selected/s.rw E s.cur sono leggibili CPU-side
            profile.routeS += Date().timeIntervalSince(t)
            let (ids, rw) = readRouteSelection(layer: i)
            // Attivazione (s.cur, finale dopo il route completato) → CPU.
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
            // c2 legge s.sharedOut SOLO su GPU: l'ordine della queue in-order
            // basta (stesso argomento di commitFFN/DS4_ASYNC_FFN) — nessuna
            // attesa CPU su c1 nel percorso felice. Il wait resta sotto
            // profiling (attribuzione storica di expertsS), con
            // DS4_ASYNC_ROUTE=0 e sugli error path.
            if profileRoute || !asyncRoute { c1.waitCompleted() }
            let c2 = GraphContext(rt)
            do { try c2.begin(); try c2.decodeRemoteTail(s: scratch, d: d, partial: target, outHc: other) }
            catch { c1.waitCompleted(); throw error }
            commitFFN(c2)
            profile.expertsS += Date().timeIntervalSince(t)
        } else if let gather = expertGather {
            // Phase 1: route (own cb, in flight) -> shared FFN committed behind
            // it -> JOIN the route -> read the selected ids (top-K reduced).
            var t = Date()
            let route = try encodeRoute(i, w: w, layerRope: layerRope, curHc: cur, pos: pos, nKeys: nKeys, token: token)
            // I/O–compute OVERLAP, senza gap GPU: la FFN condivisa non dipende
            // dalla selezione, quindi è encodata e committata PRIMA del join
            // della route — la GPU concatena route→FFN back-to-back e macina
            // mentre la CPU legge la selezione e gathera gli esperti dall'SSD.
            // On error the in-flight buffers are waited before rethrowing, so a
            // rebuilt turn can never race a stale write into the scratch.
            let c1 = GraphContext(rt)
            do { try c1.begin(); try c1.decodeSharedFFN(w: w, s: scratch, d: d) }
            catch { route.waitCompleted(); throw error }
            c1.commitAsync()
            route.waitCompleted()   // da qui s.selected/s.rw sono leggibili CPU-side
            profile.routeS += Date().timeIntervalSince(t)
            let (ids, rw) = readRouteSelection(layer: i)
            let K = ids.count
            if K < d.k {
                // Route completata ⇒ (queue in-order) anche la FFN routed del
                // layer precedente: queste scritture CPU non possono correre
                // con un lettore GPU in volo (c1 non tocca rw/down6).
                writeFloats(rw, into: scratch.rw)
                zeroDown6(from: K)
            }
            // Legacy caches support only the model-global quant class. The
            // DS4_MULTI_QUANT_CACHE factory builds a layer-aware cache whose
            // pools carry their own bytes and interleaved stride.
            let onClass = w.gateQuant == d.gateQuant && w.upQuant == d.upQuant && w.downQuant == d.downQuant
            if let cache = slotCache, onClass || cache.supports(layer: i) {
                // Persistent + changing experts: hits are already resident in the
                // layer's GPU pool (zero copies); only misses are filled from the
                // mmap. The matvec indexes the pool with slot ids.
                t = Date()
                let h0 = cache.hits, m0 = cache.misses, w0 = cache.warmed, p0 = cache.prefilled
                let hb0 = cache.hitBytes, mb0 = cache.missBytes
                let wb0 = cache.warmedBytes, pb0 = cache.prefilledBytes
                let acquired: (pool: ExpertSlotCache.LayerPool, slots: [Int32])
                var residentCB: GraphContext?
                var splitDeferredMask: UInt32 = 0
                var splitPool: ExpertSlotCache.LayerPool?
                var splitPoolSlotCount = 0
                var splitLease: ExpertSlotCache.PendingAcquire?
                var ordinaryLease: ExpertSlotCache.GPUReadLease?
                let splitEligible = asyncExpertSplit && !profileRoute
                    && K == 6 && d.k == 6 && d.fusedMoE
                    && w.gateQuant == .iq2_xxs && w.upQuant == .iq2_xxs
                    && w.downQuant == .q2_K
                if splitEligible {
                    let pending = cache.acquireAsync(layer: i, ids: ids)
                    splitLease = pending
                    let fullMask = (UInt32(1) << UInt32(K)) - 1
                    let residentMask = pending.residentMask & fullMask
                    splitDeferredMask = fullMask ^ residentMask
                    if residentMask != 0, splitDeferredMask != 0,
                       let residentPool = pending.residentPool {
                        do {
                            let slotCount = try stageSplitResidentPair(
                                residentPool, residentSlots: pending.residentSlots,
                                activeMask: residentMask, fallbackStride: slotCacheStride)
                            let c = GraphContext(rt)
                            try c.begin()
                            try c.decodeRoutedExpertsResidentPair(
                                w: w, s: scratch, d: d,
                                gateExp: splitResidentGate, upExp: splitResidentUp,
                                gateAddresses: splitResidentGateAddresses,
                                upAddresses: splitResidentUpAddresses,
                                ids: splitSlotsScratch, activeMask: residentMask,
                                slotCount: slotCount,
                                expertStride: nil)
                            c.commitAsync()
                            residentCB = c
                            splitPool = residentPool
                            profile.expertSplitLayers += 1
                            profile.expertSplitResidentRows += residentMask.nonzeroBitCount
                        } catch {
                            _ = try? pending.join()
                            pending.release()
                            residentCB?.waitCompleted()
                            c1.waitCompleted()
                            throw error
                        }
                    }
                    do { acquired = try pending.join() }
                    catch {
                        pending.release()
                        residentCB?.waitCompleted()
                        c1.waitCompleted()
                        throw error
                    }
                    // Fork/join boundary: the resident GPU pair and the missing
                    // expert I/O ran concurrently, but the tail consumes all six
                    // `mid6` rows. Do not rely on implicit inter-command-buffer
                    // hazard ordering here; explicitly join both producers before
                    // encoding the shared down-sum/tail. This preserves the full
                    // overlap while making completion deterministic.
                    residentCB?.waitCompleted()
                } else {
                    do {
                        let leased = try cache.acquireLeased(layer: i, ids: ids)
                        acquired = (leased.pool, leased.slots)
                        ordinaryLease = leased.lease
                    }
                    catch { c1.waitCompleted(); throw error }
                }
                let (pool, slots) = acquired
                if let splitPool {
                    guard splitPool.gate.buffer === pool.gate.buffer,
                          splitPool.up.buffer === pool.up.buffer,
                          splitPool.down.buffer === pool.down.buffer else {
                        splitLease?.release()
                        residentCB?.waitCompleted()
                        c1.waitCompleted()
                        throw MetalError.unsupported("expert split: pool changed during demand fill")
                    }
                    do {
                        splitPoolSlotCount = try stageSplitAddressTables(
                            pool, fallbackStride: slotCacheStride,
                            gateAddresses: splitPoolGateAddresses,
                            upAddresses: splitPoolUpAddresses)
                    } catch {
                        splitLease?.release()
                        residentCB?.waitCompleted()
                        c1.waitCompleted()
                        throw error
                    }
                }
                profile.gatherS += Date().timeIntervalSince(t)
                // Deltas, not cumulative totals: the cache counts since load,
                // the profile since resetProfile().
                profile.expertHits += cache.hits - h0
                profile.expertMisses += cache.misses - m0
                profile.expertWarmed += cache.warmed - w0
                profile.expertPrefilled += cache.prefilled - p0
                profile.expertHitBytes += cache.hitBytes - hb0
                profile.expertMissBytes += cache.missBytes - mb0
                profile.expertWarmedBytes += cache.warmedBytes - wb0
                profile.expertPrefilledBytes += cache.prefilledBytes - pb0
                profile.gatherBytes += (cache.missBytes - mb0) + (cache.warmedBytes - wb0)
                // Persistent staging (no per-layer alloc), A/B by layer parity:
                // with the async FFN the PREVIOUS layer's command buffer may
                // still be reading its ids buffer while this layer stages its own.
                let slotsBuf = (i & 1) == 0 ? slotsScratch : slotsScratchB
                _ = slots.withUnsafeBytes {
                    memcpy(slotsBuf.buffer.contents(), $0.baseAddress!, $0.count)
                }
                t = Date()
                // c2 legge s.sharedOut SOLO su GPU (l'add è un dispatch): la
                // queue in-order ordina c1→c2, nessuna attesa CPU nel percorso
                // felice — stesso argomento certificato per DS4_ASYNC_FFN. Il
                // wait resta sotto profiling (attribuzione storica di expertsS),
                // con DS4_ASYNC_ROUTE=0 e sugli error path (c1 non è tracciato
                // da drainFFN).
                if profileRoute || !asyncRoute { c1.waitCompleted() }
                let c2 = GraphContext(rt)
                do {
                    try c2.begin()
                    if residentCB != nil, splitDeferredMask != 0 {
                        try c2.decodeRoutedExpertsSplitTail(
                            w: w, s: scratch, d: d,
                            gateExp: pool.gate, upExp: pool.up, downExp: pool.down,
                            gateAddresses: splitPoolGateAddresses,
                            upAddresses: splitPoolUpAddresses,
                            ids: slotsBuf, deferredMask: splitDeferredMask,
                            slotCount: splitPoolSlotCount, outHc: other,
                            expertStride: pool.expertStride ?? slotCacheStride)
                    } else {
                        try c2.decodeRoutedExperts(w: w, s: scratch, d: d,
                                                   gateExp: pool.gate,
                                                   upExp: pool.up, downExp: pool.down,
                                                   ids: slotsBuf, outHc: other, activeK: K,
                                                   expertStride: pool.expertStride ?? slotCacheStride)
                    }
                } catch {
                    splitLease?.release()
                    ordinaryLease?.release()
                    residentCB?.waitCompleted()
                    c1.waitCompleted()
                    throw error
                }
                if let lease = splitLease {
                    c2.onComplete { lease.release() }
                }
                if let lease = ordinaryLease {
                    c2.onComplete { lease.release() }
                }
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
                let bytes = g.byteLength + u.byteLength + dn.byteLength
                profile.gatherBytes += bytes
                profile.expertBypasses += ids.count
                profile.expertBypassBytes += bytes
                t = Date()
                if profileRoute || !asyncRoute { c1.waitCompleted() }   // vedi sopra
                let c2 = GraphContext(rt)
                do {
                    try c2.begin()
                    try c2.decodeRoutedExperts(w: w, s: scratch, d: d, gateExp: g, upExp: u, downExp: dn,
                                               ids: idsPacked, outHc: other, activeK: K)
                } catch { c1.waitCompleted(); throw error }
                commitFFN(c2)
                profile.expertsS += Date().timeIntervalSince(t)
            }
        } else {
            let t = Date()
            let route = try encodeRoute(i, w: w, layerRope: layerRope, curHc: cur, pos: pos, nKeys: nKeys, token: token)
            // Percorso residente: nessuna lettura CPU della selezione, ma il
            // memset di clearMaskIfDirty del layer successivo non deve correre
            // con l'attention di QUESTO route ancora in volo — join immediato,
            // identico al comportamento storico (commit bloccante).
            route.waitCompleted()
            let lc = GraphContext(rt); try lc.begin()
            try lc.decodeExperts(w: w, s: scratch, d: d, gateExp: w.expGate, upExp: w.expUp,
                                 downExp: w.expDown, ids: scratch.selected, outHc: other)
            commitFFN(lc)                    // COMPUTE (cb retains w's buffers until completed)
            profile.layerOtherS += Date().timeIntervalSince(t)
        }
        profile.layers += 1
    }

    /// Encode (and COMMIT) the route for one token on layer `i`, returning the
    /// route's LAST command-buffer context. On the happy path (DS4_ASYNC_ROUTE,
    /// default ON) the commit is ASYNCHRONOUS and the returned context is in
    /// flight: il chiamante DEVE fare `waitCompleted()` prima di leggere
    /// CPU-side la selezione (s.selected/s.rw) o l'attivazione (s.cur), e prima
    /// di scritture CPU su buffer che la route legge (s.mask) — ma DOPO aver
    /// committato il lavoro GPU che può accodarsi dietro la route sulla queue
    /// in-order (la FFN condivisa), così la GPU non resta mai in bolla tra
    /// route e FFN. Sotto DS4_PROFILE_ROUTE (o DS4_ASYNC_ROUTE=0) ogni commit
    /// resta sincrono: il contesto tornato è già completo e waitCompleted() è
    /// un no-op idempotente.
    ///
    /// When the NSA indexer is active (ratio-4 layer with more compressed rows
    /// than the top-K), the command buffer is split at the indexer scores:
    /// commit phase 1a SYNCHRONOUSLY (the CPU top-K reads the scores), run the
    /// CPU top-K to write the compressed-row mask, then encode the attention —
    /// the C "dense top-k mask" path (indexer_allowed_decode_one); only the
    /// attention half is returned in flight. Otherwise a single command buffer,
    /// numerically identical to the pre-indexer code.
    func encodeRoute(_ i: Int, w: LayerWeights, layerRope: RopeParams,
                             curHc: GPUTensor, pos: Int, nKeys: Int, token: Int) throws -> GraphContext {
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
        if active, let idx, gpuIndexerTopK, d.indexedAttn,
           let compState = compStates[i] {
            // DS4_INDEXED_ATTN: score -> top-K a INDICI (stesso heap del
            // percorso a maschera: set identico) -> sort per id crescente ->
            // attention SOLO su raw SWA + topK righe selezionate. Niente
            // maschera e niente staging F16 dell'intero span: il costo per
            // token resta O(nSWA + topK) al crescere del contesto.
            let c = GraphContext(rt); if profileRoute { c.phaseTimes = [:] }; try c.begin()
            let nComp = try c.decodeRoutePre(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope,
                                             rawCache: rawCaches[i], pos: pos, rmsEps: rmsEps, hcEps: hcEps,
                                             comp: compStates[i], idx: hasIdxState ? idx : nil,
                                             indexerScoring: true)
            try c.indexerTopKIndices(scores: scratch.idxScores, out: scratch.idxTopK,
                                     nScores: idx.count, topK: d.indexerTopK)
            try c.sortTopKAsc(indices: scratch.idxTopK, sorted: scratch.idxTopKSorted,
                              topK: d.indexerTopK)
            try c.decodeRouteAttnIndexed(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope,
                                         rawCache: rawCaches[i], nKeys: nKeys, pos: pos, token: token,
                                         rmsEps: rmsEps, hcEps: hcEps, nComp: nComp,
                                         comp: compState, topk: scratch.idxTopKSorted)
            if profileRoute {
                try c.phase("router")
                c.commit()
                accumulateRoutePhases(c, nil)
            } else if asyncRoute {
                c.commitAsync()
            } else {
                c.commit()
            }
            return c
        }
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
            if profileRoute {
                try c.phase("router")
                c.commit()
                accumulateRoutePhases(c, nil)
            } else if asyncRoute {
                c.commitAsync()
            } else {
                c.commit()
            }
            return c
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
            if profileRoute {
                try c2.phase("router")
                c2.commit()
                accumulateRoutePhases(c1, c2)
            } else if asyncRoute {
                c2.commitAsync()
            } else {
                c2.commit()
            }
            return c2
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
            return c
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
            if asyncRoute { c1.commitAsync() } else { c1.commit() }
            return c1
        }
    }

    /// Accumulate per-sub-phase route timings (DS4_PROFILE_ROUTE, and the
    /// batched prefill's DS4_PROFILE_PREFILL — same phase names) into the
    /// profile. Internal: +Prefill accumulates its batched runs through this.
    func accumulateRoutePhases(_ a: GraphContext, _ b: GraphContext?) {
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
    /// stream. DS4_LAZY_IDX always excludes them from per-token staging; this
    /// proof decides whether the decoder needs a one-shot resident activation
    /// callback at all for the configured maxKeys/threshold.
    static func indexerCanEverActivate(maxKeys: Int, topK: Int) -> Bool {
        indexerScoringNeeded(liveKeys: maxKeys, topK: topK,
                             sparseThreshold: indexerSparseThreshold)
    }

    /// Live equivalent of `indexerCanEverActivate`: ratio-4 layers emit one
    /// index row every four committed keys, and sparse scoring begins only when
    /// that row count is STRICTLY above both bounds. Pure/internal for tests.
    static func indexerScoringNeeded(liveKeys: Int, topK: Int,
                                     sparseThreshold: Int) -> Bool {
        let prospective = max(0, liveKeys) / 4
        return prospective > sparseThreshold && prospective > topK
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

    /// Leva 9 (coda prefill oltre la soglia indexer): la route di UN token
    /// codificata in un ctx CONDIVISO senza commit, replicando ESATTAMENTE i
    /// rami di encodeRoute — indexer attivo indicizzato (DS4_INDEXED_ATTN),
    /// attivo a maschera GPU (mask_one), o inattivo/scorer non staged (il
    /// corpo di encodeRouteInto). Richiede gpuIndexerTopK: il percorso con
    /// top-k CPU ha un tap host tra pre e attention e NON è raggruppabile
    /// (il chiamante gata e resta per-token in quel caso).
    func encodeRouteGroupedInto(_ c: GraphContext, _ i: Int, w: LayerWeights, layerRope: RopeParams,
                                curHc: GPUTensor, pos: Int, nKeys: Int, token: Int) throws {
        let idx = indexStates[i]
        let hasIdxState = w.idxKv != nil && w.idxGate != nil
        let hasIdxScoring = hasIdxState && w.idxQB != nil && w.idxProj != nil
        let active = hasIdxScoring && indexerActive(i, pos: pos)
        guard active, let idx else {
            try encodeRouteInto(c, i, w: w, layerRope: layerRope, curHc: curHc,
                                pos: pos, nKeys: nKeys, token: token)
            return
        }
        guard gpuIndexerTopK else {
            throw MetalError.unsupported(
                "encodeRouteGroupedInto richiede il top-k GPU (DS4_GPU_INDEXER_TOPK)")
        }
        let nComp = try c.decodeRoutePre(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope,
                                         rawCache: rawCaches[i], pos: pos, rmsEps: rmsEps, hcEps: hcEps,
                                         comp: compStates[i], idx: hasIdxState ? idx : nil,
                                         indexerScoring: true)
        if d.indexedAttn, let compState = compStates[i] {
            try c.indexerTopKIndices(scores: scratch.idxScores, out: scratch.idxTopK,
                                     nScores: idx.count, topK: d.indexerTopK)
            try c.sortTopKAsc(indices: scratch.idxTopK, sorted: scratch.idxTopKSorted,
                              topK: d.indexerTopK)
            try c.decodeRouteAttnIndexed(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope,
                                         rawCache: rawCaches[i], nKeys: nKeys, pos: pos, token: token,
                                         rmsEps: rmsEps, hcEps: hcEps, nComp: nComp,
                                         comp: compState, topk: scratch.idxTopKSorted)
        } else {
            let nRaw = min(nKeys, d.nSWA)
            try c.indexerTopKMask(scores: scratch.idxScores, mask: scratch.mask,
                                  nRaw: nRaw, nComp: nComp, nScores: idx.count,
                                  topK: d.indexerTopK)
            maskDirtyCount = max(maskDirtyCount, nRaw + nComp)
            try c.decodeRouteAttn(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope,
                                  rawCache: rawCaches[i], nKeys: nKeys, pos: pos, token: token,
                                  rmsEps: rmsEps, hcEps: hcEps, nComp: nComp,
                                  comp: compStates[i])
        }
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
