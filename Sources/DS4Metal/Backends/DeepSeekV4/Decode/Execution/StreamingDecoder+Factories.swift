import Foundation
import Metal
import DS4Core

extension StreamingDecoder {
    /// Resolve every model-owned runtime field at the GGUF boundary. Callers may
    /// tune execution knobs in `dims`, but routed-expert formats and router
    /// precision must always come from the tensors actually being opened.
    private static func resolvedRuntimeDims(_ requested: DSV4Dims,
                                            model: GGUFModel,
                                            geometry: DSV4RuntimeGeometry,
                                            nLayers: Int) throws -> DSV4Dims {
        try validateRuntimeGeometry(geometry, dims: requested, nLayers: nLayers)
        try GGUFWeights.validateRuntimeLayout(model, geometry: geometry)
        let quant = GGUFWeights.detectMoEQuant(model)
        var resolved = requested
        resolved.gateQuant = quant.gate
        resolved.upQuant = quant.up
        resolved.downQuant = quant.down
        resolved.routerF16 = quant.routerF16
        return resolved
    }

    /// Build a streaming decoder backed by a real GGUF model (the real Stage D
    /// path): each layer is loaded from the mmap on demand.
    public static func fromGGUF(rt: MetalRuntime, model: GGUFModel, dims: DSV4Dims, rope: RopeParams,
                                nLayers: Int, maxKeys: Int, rmsEps: Float = ModelDefaults.rmsEps,
                                hcEps: Float = ModelDefaults.hcEps,
                                geometry: DSV4RuntimeGeometry) throws -> StreamingDecoder {
        let dims = try resolvedRuntimeDims(dims, model: model, geometry: geometry,
                                           nLayers: nLayers)
        let (embed, head) = try GGUFWeights.outputHead(rt, model)
        return try StreamingDecoder(rt: rt, dims: dims, rope: rope, nLayers: nLayers,
                                    layerProvider: { try GGUFWeights.layer(rt, model, $0) },
                                    embedTable: embed, out: head, maxKeys: maxKeys, rmsEps: rmsEps,
                                    hcEps: hcEps, geometry: geometry)
    }

    /// Expert-cache streaming decoder: per layer, only the dense weights are
    /// loaded up front; after routing, ONLY the 6 selected experts are gathered
    /// from the mmap (6/256 ~= 40x less expert IO/RAM). Numerically identical to
    /// the resident path (validated by ExpertCacheLayerTests).
    public static func fromGGUFExpertCached(rt: MetalRuntime, model: GGUFModel, dims: DSV4Dims, rope: RopeParams,
                                            nLayers: Int, maxKeys: Int,
                                            rmsEps: Float = ModelDefaults.rmsEps,
                                            hcEps: Float = ModelDefaults.hcEps,
                                            geometry: DSV4RuntimeGeometry) throws -> StreamingDecoder {
        let dims = try resolvedRuntimeDims(dims, model: model, geometry: geometry,
                                           nLayers: nLayers)
        let (embed, head) = try GGUFWeights.outputHead(rt, model)
        let willNeed = ProcessInfo.processInfo.environment["DS4_WILLNEED_EXPERTS"] != "0"   // default ON; opt-out with =0
        let gather: (Int, [Int32]) throws -> (GPUTensor, GPUTensor, GPUTensor) = { il, ids in
            try GGUFWeights.gatherLayerExperts(rt, model, il, ids: ids, dims: dims, willNeed: willNeed)
        }
        // Memoize the non-routed (dense + NSA compressor) weights: loaded once,
        // resident across tokens (the C --ssd-streaming model). Only the 6 selected
        // experts are gathered per token (gatherExperts memcpy's just those rows from
        // the mmap = ~6/256 of expert IO). This is the fast path: per token ~= a few
        // expert slabs from SSD + GPU compute, instead of re-streaming the whole model.
        let cache = CachedLayerProvider { try GGUFWeights.layer(rt, model, $0, loadExperts: false) }
        return try StreamingDecoder(rt: rt, dims: dims, rope: rope, nLayers: nLayers,
                                    layerProvider: { try cache.get($0) },
                                    embedTable: embed, out: head, maxKeys: maxKeys, rmsEps: rmsEps, hcEps: hcEps,
                                    expertGather: gather, geometry: geometry)
    }

    /// Fastest 16GB path (the C `--ssd-streaming` model): non-routed weights are
    /// NO-COPY mmap views (resident via the OS page cache, single copy, evictable —
    /// no per-token re-copy, no 8GB of dirty buffers that OOM), and only the 6 selected
    /// experts are gathered per token. No memoization needed: the page cache serves
    /// repeated weight reads across tokens. Requires model opened metalMapping:true.
    public static func fromGGUFExpertCachedMapped(rt: MetalRuntime, model: GGUFModel, dims: DSV4Dims, rope: RopeParams,
                                                  nLayers: Int, maxKeys: Int, rmsEps: Float = ModelDefaults.rmsEps, hcEps: Float = ModelDefaults.hcEps,
                                                  cacheSlots: Int? = nil, kvLayers: Range<Int>? = nil,
                                                  geometry: DSV4RuntimeGeometry) throws -> StreamingDecoder {
        let dims = try resolvedRuntimeDims(dims, model: model, geometry: geometry,
                                           nLayers: nLayers)
        LoadProgress.shared.set(0.02, "Apertura pesi…")
        let (embed, headMapped) = try GGUFWeights.outputHeadMapped(rt, model)
        var head = headMapped
        // DS4_MLOCK=1: pin the hot resident buffers (expert pools, resident
        // head, dense-stream staging). Shared MTLBuffers are anonymous memory
        // that macOS COMPRESSES between uses — a buffer touched once per token
        // re-reads at ~2.4 GB/s through the compressor instead of RAM speed
        // (the measured 235 ms output head on a "resident" copy). Best-effort.
        let lockResident = ProcessInfo.processInfo.environment["DS4_MLOCK"] == "1"
        // With DS4_DENSE_STREAM the dense weights no longer occupy RAM (~300 MB
        // of staging instead of ~6 GB), so the OUTPUT HEAD (~560 MB Q8, read in
        // full every token) gets copied RESIDENT: mapped it was re-read through
        // a cold page cache at ~2 GB/s (~260 ms/token measured). The embedding
        // table stays mapped — the decode stages one 8 KB row per token anyway.
        if ProcessInfo.processInfo.environment["DS4_DENSE_STREAM"] == "1" {
            LoadProgress.shared.set(0.04, "Output head residente…")
            head.head = try GGUFWeights.tensor(rt, model, "output.weight")
            if lockResident { head.head.lockResident() }
        }
        let willNeed = ProcessInfo.processInfo.environment["DS4_WILLNEED_EXPERTS"] != "0"   // default ON; opt-out with =0
        // DS4_EXPERT_PREAD=1: expert slabs pread() DIRECT from disk (F_NOCACHE)
        // instead of memcpy'd from the mmap. Zero page-cache footprint for the
        // ~1 GB/token of expert churn, so it stops evicting the DENSE weights
        // (route/attn/embed/head re-fault them otherwise on 16 GB machines).
        // Same bytes, same numerics — only the I/O path changes. A/B per machine.
        let uncachedFD: Int32? =
            ProcessInfo.processInfo.environment["DS4_EXPERT_PREAD"] == "1" ? model.uncachedFD() : nil
        // Routing-frequency stats ("usage imatrix"): always collected (cheap);
        // the service persists them across sessions and they pre-warm the cache.
        let usage = ExpertUsageStats(nLayers: nLayers, nExperts: dims.nExperts)
        // Per-expert slab sizes in the mmap (expert e at absOffset + e*bytes); shared
        // by the slot-cache fill and the read-ahead prefetch.
        let gateBytes = (dims.nEmbd / 256) * dims.gateQuant.blockBytes * dims.expertFfn
        let upBytes = (dims.nEmbd / 256) * dims.upQuant.blockBytes * dims.expertFfn
        let downBytes = (dims.expertFfn / 256) * dims.downQuant.blockBytes * dims.nEmbd
        // DS4_EXPERT_BUNDLE=1: sidecar with each expert's gate|up|down slabs
        // CONTIGUOUS — a miss becomes one ~7 MB sequential burst instead of
        // three scattered ~2 MB reads (measured gather at ~49% of the SSD's
        // parallel ceiling without it). Built once next to the model; any
        // failure falls back to the plain GGUF reads below. Same bytes.
        let bundleEnabled = ProcessInfo.processInfo.environment["DS4_EXPERT_BUNDLE"] == "1"
        let bundle: ExpertBundle? = bundleEnabled
            ? ExpertBundle.openOrBuild(model: model, layers: 0..<nLayers, nExpert: dims.nExperts,
                                       gateBytes: gateBytes, upBytes: upBytes, downBytes: downBytes)
            : nil
        // The bundle STATE must be visible in the engine log at EVERY load —
        // silence ("is it even on?") is the one outcome that cannot be triaged.
        if !bundleEnabled {
            FileHandle.standardError.write(Data("DS4 expbundle: disattivato (DS4_EXPERT_BUNDLE≠1) — gather dal GGUF\n".utf8))
        } else if bundle == nil {
            FileHandle.standardError.write(Data("DS4 expbundle: NON attivo per questo load (motivo nelle righe sopra) — gather dal GGUF\n".utf8))
        }
        let metalIORequested = ProcessInfo.processInfo.environment["DS4_MTLIO"] == "1"
        if metalIORequested {
            if let bundle { _ = bundle.enableMetalIO(device: rt.device) }
            else {
                FileHandle.standardError.write(Data("DS4 expbundle: DS4_MTLIO=1 richiede un expert-bundle valido — uso pread GGUF\n".utf8))
            }
        }
        let gather: (Int, [Int32]) throws -> (GPUTensor, GPUTensor, GPUTensor) = { il, ids in
            if let b = bundle, let packed = b.gatherPacked(rt, layer: il, ids: ids) { return packed }
            return try GGUFWeights.gatherLayerExperts(rt, model, il, ids: ids, dims: dims,
                                                      willNeed: willNeed, uncachedFD: uncachedFD)
        }
        // Persistent + changing experts (cacheSlots param, else env
        // DS4_EXPERT_CACHE_SLOTS; default off): per layer, an N-slot LRU pool
        // keeps hot experts resident in GPU buffers; only misses are memcpy'd
        // from the mmap. The pool is WIRED memory (~6.9 MB/slot on the 2-bit
        // model × nLayers): on tight-RAM machines start small (8) and watch the
        // hit rate in the decode profile / Tuning tab.
        let envSlots = ProcessInfo.processInfo.environment["DS4_EXPERT_CACHE_SLOTS"].flatMap(Int.init)
        let nSlots = cacheSlots ?? envSlots ?? 0
        var cache: ExpertSlotCache? = nil
        /// Stride fra gli slot del pool quando il layout e' interleaved
        /// (record gate|up|down) — nil = layout storico a 3 buffer stretti.
        var slotStride: Int? = nil
        if nSlots > 0 {
            let S = max(8, nSlots)
            // Readahead every missing slab (3 matrices × N ids) BEFORE the
            // copies: the NVMe serves all the regions concurrently. With
            // DS4_EXPERT_PREAD the fill bypasses the page cache, so the
            // madvise hint would be pointless — prefetch disabled (nil).
            var fillPrefetch: ((Int, [Int32]) -> Void)? = nil
            if uncachedFD == nil {
                fillPrefetch = { il, ids in
                    for id in ids {
                        GGUFWeights.adviseExpert(model, "blk.\(il).ffn_gate_exps.weight", id: id, expertBytes: gateBytes)
                        GGUFWeights.adviseExpert(model, "blk.\(il).ffn_up_exps.weight", id: id, expertBytes: upBytes)
                        GGUFWeights.adviseExpert(model, "blk.\(il).ffn_down_exps.weight", id: id, expertBytes: downBytes)
                    }
                }
            }
            // Pool INTERLEAVED (default ON, DS4_POOL_INTERLEAVE=0 per il layout
            // storico a 3 buffer): ogni slot ha gate|up|down CONTIGUI, identico
            // al record del bundle — un miss diventa UNA pread da ~7 MB dritta
            // nello slot (1 syscall invece di 3, I/O piu' grandi a parita' di
            // coda). I kernel non cambiano: gate/up/down sono tre VISTE dello
            // stesso buffer e lo stride fra esperti (nb02) e' il record.
            let interleave = ProcessInfo.processInfo.environment["DS4_POOL_INTERLEAVE"] != "0"
            let recordBytes = gateBytes + upBytes + downBytes
            typealias Pool = (gate: GPUTensor, up: GPUTensor, down: GPUTensor)
            let makePool: (Int) throws -> Pool
            if interleave {
                makePool = { slots in
                    let buf = try GPUTensor.zerosBytes(rt, byteLength: slots * recordBytes)
                    if lockResident { buf.lockResident() }   // pin ONCE: covers all three views
                    let up = GPUTensor(buffer: buf.buffer, byteLength: slots * recordBytes - gateBytes,
                                       count: slots * recordBytes - gateBytes, byteOffset: gateBytes)
                    let down = GPUTensor(buffer: buf.buffer,
                                         byteLength: slots * recordBytes - gateBytes - upBytes,
                                         count: slots * recordBytes - gateBytes - upBytes,
                                         byteOffset: gateBytes + upBytes)
                    return (gate: buf, up: up, down: down)
                }
            } else {
                makePool = { slots in
                    let p = (gate: try GPUTensor.zerosBytes(rt, byteLength: slots * gateBytes),
                             up: try GPUTensor.zerosBytes(rt, byteLength: slots * upBytes),
                             down: try GPUTensor.zerosBytes(rt, byteLength: slots * downBytes))
                    if lockResident {
                        // DS4_MLOCK: a hit must cost zero I/O — pin the pool so the
                        // compressor can't steal cold slots between reuses.
                        p.gate.lockResident(); p.up.lockResident(); p.down.lockResident()
                    }
                    return p
                }
            }
            slotStride = interleave ? recordBytes : nil
            let fillOne: (Int, Int32, ExpertSlotCache.LayerPool, Int) throws -> Void = { il, id, pool, slot in
                // Sidecar bundle first: layout del record == layout dello slot
                // interleaved -> UNA pread; col layout storico restano i 3
                // pread adiacenti (comunque un burst sequenziale).
                if let b = bundle {
                    if interleave, b.copyExpertInterleaved(layer: il, id: id, dst: pool.gate,
                                                           slot: slot, stride: recordBytes) {
                        return
                    }
                    if !interleave, b.copyExpert(layer: il, id: id, gateDst: pool.gate,
                                                 upDst: pool.up, downDst: pool.down, slot: slot) {
                        return
                    }
                }
                // The 3 slabs (gate/up/down) of a missing expert are read
                // CONCURRENTLY: with fillAll's parallelism across misses this
                // raises the NVMe queue depth from ~misses to ~3×misses. It
                // matters most under DS4_DENSE_STREAM, where the gather shares
                // the disk with the dense reads and depth is what keeps it fed.
                // nonisolated(unsafe): i 3 job scrivono slab DISGIUNTI dello slot,
                // model e' letto e basta, l'errore e' protetto dal lock.
                nonisolated(unsafe) let jobs: [(name: String, bytes: Int, dst: GPUTensor)] = [
                    ("blk.\(il).ffn_gate_exps.weight", gateBytes, pool.gate),
                    ("blk.\(il).ffn_up_exps.weight", upBytes, pool.up),
                    ("blk.\(il).ffn_down_exps.weight", downBytes, pool.down)]
                let lock = NSLock()
                nonisolated(unsafe) var firstError: Error? = nil
                nonisolated(unsafe) let modelRef = model
                DispatchQueue.concurrentPerform(iterations: jobs.count) { j in
                    do {
                        try GGUFWeights.copyExpert(modelRef, jobs[j].name, id: id, expertBytes: jobs[j].bytes,
                                                   into: jobs[j].dst, slot: slot, uncachedFD: uncachedFD,
                                                   slotStride: interleave ? recordBytes : nil)
                    } catch {
                        lock.lock()
                        if firstError == nil { firstError = error }
                        lock.unlock()
                    }
                }
                if let e = firstError { throw e }
            }
            let fillBatch: ((Int, [(id: Int32, slot: Int)], ExpertSlotCache.LayerPool) throws -> Void)? =
                metalIORequested && bundle != nil ? { il, pairs, pool in
                    if bundle!.copyExpertsMetalIO(layer: il, pairs: pairs,
                                                   gateDst: pool.gate, upDst: pool.up, downDst: pool.down,
                                                   slotStride: interleave ? recordBytes : nil) {
                        return
                    }
                    // Backend unavailable/failed: preserve the exact historical
                    // fallback for every slot (bundle pread, then GGUF pread).
                    for pair in pairs { try fillOne(il, pair.id, pool, pair.slot) }
                } : nil
            cache = ExpertSlotCache(slotsPerLayer: S, bytesPerExpert: recordBytes, makePool: makePool,
                                    fill: fillOne, fillBatch: fillBatch, prefetch: fillPrefetch,
               warm: { il in   // acquire trims to the pool's size; the range filter makes a
                               // corrupt profile degrade to "entry ignored", never a pool
                               // whose creation throws forever (copyExpert bounds-check)
                usage.top(layer: il, n: 128).filter { $0 >= 0 && $0 < Int32(dims.nExperts) }
            },
               slotsFor: { il in
                // Usage-driven allocation: same total wired budget (S × routed
                // layers) but more slots where the routing concentrates, fewer
                // where it's flat. Recomputed at pool creation — i.e. at load
                // and after every invalidate() (agent switch), when the usage
                // prior has changed. Falls back to the uniform S until there's
                // enough history to trust. Opt-out: DS4_EXPERT_CACHE_UNIFORM=1.
                if ProcessInfo.processInfo.environment["DS4_EXPERT_CACHE_UNIFORM"] == "1" { return S }
                return usage.slotAllocation(base: S)?[il] ?? S
            })
        }
        // Expert look-ahead (kickLookahead): EXACT tid2eid ids for the hash
        // layers — their selection depends only on the token id, so their
        // expert I/O can always run under the previous layer's compute (the
        // C engine's begin_selected_load trick) — and usage-prior top-N for
        // the other layers (speculative: a wrong guess wastes idle-window
        // bandwidth only; opt-in with DS4_EXPERT_LOOKAHEAD=N, try 6..12).
        // Ids resolve on the decode thread; mixed-precision layers (outside
        // the slot cache's size class) are excluded.
        var offClass = Set<Int>()
        for il in 0..<nLayers {
            let pfx = "blk.\(il)."
            guard model.findTensor(pfx + "ffn_gate_exps.weight") != nil else {
                offClass.insert(il); continue
            }
            func q(_ n: String) -> MoEQuant? {
                model.findTensor(pfx + n).flatMap { MoEQuant.from(ggufType: $0.type) }
            }
            if q("ffn_gate_exps.weight") != dims.gateQuant || q("ffn_up_exps.weight") != dims.upQuant
                || q("ffn_down_exps.weight") != dims.downQuant {
                offClass.insert(il)
            }
        }
        let lookN = ProcessInfo.processInfo.environment["DS4_EXPERT_LOOKAHEAD"].flatMap(Int.init) ?? 0
        let lookahead: ((Int, Int) -> [Int32])? = cache == nil ? nil : { il, token in
            if offClass.contains(il) { return [] }
            if token >= 0, let exact = GGUFWeights.hashSelectedIds(model, il, token: token) {
                return exact.filter { $0 >= 0 && $0 < Int32(dims.nExperts) }
            }
            guard lookN > 0 else { return [] }
            return usage.top(layer: il, n: lookN).filter { $0 >= 0 && $0 < Int32(dims.nExperts) }
        }
        // Read-ahead: overlap the NEXT layer's SSD I/O with the current layer's
        // compute. DEFAULT OFF: on the I/O-bound streaming path speculative reads
        // can STEAL SSD bandwidth from the real gather (worse when the usage prior is
        // cold) — it must be measured per machine. Opt in with DS4_PREFETCH=1 (then
        // it prefetches the always-needed non-routed weights). DS4_PREFETCH_EXPERTS>0
        // additionally prefetches that many usage-prior experts (speculative; off by
        // default). Hint-only on the read-only mapping — cannot affect numerics.
        let prefetchOn = ProcessInfo.processInfo.environment["DS4_PREFETCH"] == "1"
        let prefetchExperts = ProcessInfo.processInfo.environment["DS4_PREFETCH_EXPERTS"].flatMap(Int.init) ?? 0
        let denseStream = ProcessInfo.processInfo.environment["DS4_DENSE_STREAM"] == "1"
        let lazyIndexerScoring = ProcessInfo.processInfo.environment["DS4_LAZY_IDX"] != "0"
        let denseNames: [String] = {
            var names = ["hc_attn_fn.weight", "attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight",
                         "attn_output_a.weight", "attn_output_b.weight", "hc_ffn_fn.weight",
                         "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
                         "ffn_gate_inp.weight", "indexer_compressor_kv.weight",
                         "indexer_compressor_gate.weight", "attn_compressor_kv.weight",
                         "attn_compressor_gate.weight"]
            // Prefetch is also a per-token I/O plan. In lazy mode the scoring
            // projections must stay completely cold until live activation.
            if !denseStream || !lazyIndexerScoring {
                names.append(contentsOf: ["indexer.attn_q_b.weight", "indexer.proj.weight"])
            }
            return names
        }()
        let expertTensors = [("ffn_gate_exps.weight", gateBytes), ("ffn_up_exps.weight", upBytes),
                             ("ffn_down_exps.weight", downBytes)]
        let mapBaseAddr = Int(bitPattern: model.mapBase)   // Sendable: cross into the bg queue as an int
        let prefetchQ = DispatchQueue(label: "ds4.prefetch", qos: .utility)
        let prefetch: ((Int) -> Void)? = prefetchOn ? { il in
            var ranges: [(offset: UInt64, bytes: UInt64)] = []
            let p = "blk.\(il)."
            for s in denseNames {
                if let t = model.findTensor(p + s) { ranges.append((offset: t.absOffset, bytes: t.bytes)) }
            }
            if prefetchExperts > 0 {
                let hot = usage.top(layer: il, n: prefetchExperts)   // decode thread (same as record)
                for (name, ebytes) in expertTensors {
                    if let t = model.findTensor(p + name) {
                        for e in hot { ranges.append((offset: t.absOffset + UInt64(e) * UInt64(ebytes),
                                                      bytes: UInt64(ebytes))) }
                    }
                }
            }
            let snapshot = ranges   // immutable copy for the @Sendable background block
            prefetchQ.async {
                if let base = UnsafeRawPointer(bitPattern: mapBaseAddr) {
                    GGUFModel.prefetch(base: base, ranges: snapshot)
                }
            }
        } : nil
        // Dense-weight residency. Default: per-layer dense weights are NO-COPY mmap
        // views (evictable). On a machine where the 70GB model can't fit, the 71GB
        // expert stream churns the page cache and EVICTS the ~5GB of hot dense
        // weights (q_b/output_a/…, read every token) → route/attn re-faults them
        // from SSD every token (the "compute" that doesn't warm up). DS4_RESIDENT_DENSE=1
        // copies them into resident (wired) Metal buffers ONCE (memoized), so they
        // stay put and the matvec is RAM-bound. Costs ~5GB wired — worth it when it
        // fits, frees route/attn; on very tight RAM it can pressure the expert cache.
        let residentDense = ProcessInfo.processInfo.environment["DS4_RESIDENT_DENSE"] == "1"
        // DS4_DENSE_STREAM=1: the dense weights don't try to be resident AT ALL —
        // they are pread(F_NOCACHE) into a 2-slot staging ring, one layer AHEAD,
        // so the SSD read of layer i+1 overlaps the GPU compute of layer i (the
        // dense access pattern is perfectly sequential, no speculation needed).
        // ~300 MB of staging instead of ~6 GB resident; frees the page cache for
        // embed/head and the RAM for the expert cache. Takes precedence over
        // DS4_RESIDENT_DENSE. Same bytes → identical numerics.
        let denseProvider: (Int) throws -> LayerWeights
        var activateIndexerScoring: (() throws -> Void)?
        if denseStream {
            // DS4_DENSE_Q4=1 (requires the stream): the two giant plain-matvec
            // projections (q_b, output_b — Q8, 71 of ~145 MB/layer) are
            // requantized to Q4_K at load and kept RESIDENT: half their bytes,
            // read at RAM speed, and ~3 GB/token OFF the SSD stream. LOSSY on
            // those two tensors (Q8→Q4 requant) — opt-in, A/B the output.
            let q4Dense = ProcessInfo.processInfo.environment["DS4_DENSE_Q4"] == "1"
            // DS4_LAZY_IDX (default ON): scoring projections never enter the
            // per-token stream. If this context can reach sparse top-K, a
            // decoder callback loads them once into resident buffers exactly at
            // the live frontier. "0" restores historical always-stream I/O.
            let indexerCanActivate = indexerCanEverActivate(maxKeys: maxKeys,
                                                            topK: dims.indexerTopK)
            // DS4_RESIDENT_COMP (default ON): the four NSA compressor
            // projections stop streaming and live in ~0.6 GB of resident RAM —
            // they're read EVERY token on 41 of 43 layers, the single densest
            // repeat-read in the stream. Same bytes → identical numerics.
            // "0" restores full streaming (tight-RAM fallback / A/B).
            let residentComp = ProcessInfo.processInfo.environment["DS4_RESIDENT_COMP"] != "0"
            let streamer = try DenseStreamer(rt: rt, model: model, layers: kvLayers ?? 0..<nLayers,
                                             lockResident: lockResident, q4Dense: q4Dense,
                                             lazyIndexerScoring: lazyIndexerScoring,
                                             residentComp: residentComp)
            denseProvider = { try streamer.weights($0) }
            if lazyIndexerScoring, indexerCanActivate {
                activateIndexerScoring = { try streamer.activateIndexerScoring() }
                FileHandle.standardError.write(Data(
                    "DS4 lazy-idx: attivazione live abilitata; scorer esclusi dallo stream fino alla soglia\n".utf8))
            } else {
                activateIndexerScoring = nil
                let message = lazyIndexerScoring
                    ? "DS4 lazy-idx: il contesto non può raggiungere la soglia; scorer mai caricati\n"
                    : "DS4 lazy-idx: disattivato (DS4_LAZY_IDX=0); scorer nello stream per-token storico\n"
                FileHandle.standardError.write(Data(message.utf8))
            }
        } else if residentDense {
            let denseCache = CachedLayerProvider { try GGUFWeights.layer(rt, model, $0, loadExperts: false) }
            denseProvider = { try denseCache.get($0) }
            activateIndexerScoring = nil
        } else {
            denseProvider = { try GGUFWeights.layerMappedDense(rt, model, $0) }
            activateIndexerScoring = nil
        }
        LoadProgress.shared.set(0.95, "Allocazione KV e scratch…")
        let dec = try StreamingDecoder(rt: rt, dims: dims, rope: rope, nLayers: nLayers,
                                       layerProvider: denseProvider,
                                       embedTable: embed, out: head, maxKeys: maxKeys, rmsEps: rmsEps, hcEps: hcEps,
                                       expertGather: gather, slotCache: cache, usage: usage,
                                       prefetch: prefetch, lookahead: lookahead, kvLayers: kvLayers,
                                       activateIndexerScoring: activateIndexerScoring,
                                       slotCacheStride: slotStride, geometry: geometry)
        LoadProgress.shared.set(1.0, "Pronto")
        return dec
    }

    /// Mapped-experts streaming decoder: per layer the dense weights are copied,
    /// but the routed experts are NO-COPY mmap views over the FULL expert tensors
    /// (all 256). The single-cb decode path runs mul_mv_id with the real selected
    /// ids; the OS page cache caches touched experts across tokens — no per-token
    /// re-gather. Requires model opened with metalMapping:true.
    public static func fromGGUFMappedExperts(rt: MetalRuntime, model: GGUFModel, dims: DSV4Dims, rope: RopeParams,
                                             nLayers: Int, maxKeys: Int,
                                             rmsEps: Float = ModelDefaults.rmsEps,
                                             hcEps: Float = ModelDefaults.hcEps,
                                             geometry: DSV4RuntimeGeometry) throws -> StreamingDecoder {
        let dims = try resolvedRuntimeDims(dims, model: model, geometry: geometry,
                                           nLayers: nLayers)
        let (embed, head) = try GGUFWeights.outputHead(rt, model)
        // Memoize per-layer weights: dense (incl. NSA compressor) are COPIED resident
        // and reused across tokens; experts are no-copy mmap. Without this the ~8GB of
        // non-routed weights were re-copied from the mmap EVERY token (minutes/token on
        // 16GB). This is the C `--ssd-streaming` model: non-routed resident, experts paged.
        let cache = CachedLayerProvider { try GGUFWeights.layerMappedExperts(rt, model, $0) }
        return try StreamingDecoder(rt: rt, dims: dims, rope: rope, nLayers: nLayers,
                                    layerProvider: { try cache.get($0) },
                                    embedTable: embed, out: head, maxKeys: maxKeys, rmsEps: rmsEps, hcEps: hcEps,
                                    expertGather: nil, geometry: geometry)   // single-cb decodeLayer with real ids
    }

    // MARK: Stack di gather per lo SHARD VERTICALE (expert parallelism, D2)

    /// Assembla per un consumatore ESTERNO (l'expert shard del worker
    /// verticale, modulo DS4Engine) la stessa terna del motore locale:
    /// gather con bundle sidecar, slot-cache LRU con pool interleaved e fill
    /// a pread concorrenti, pre-warm e allocazione per-layer dalla usage
    /// imatrix. DUPLICA volutamente l'assemblaggio della factory locale
    /// invece di rifattorizzarla (zero rischio di regressione sul percorso
    /// caldo); stessa semantica degli stessi env: DS4_EXPERT_BUNDLE,
    /// DS4_EXPERT_PREAD, DS4_WILLNEED_EXPERTS, DS4_POOL_INTERLEAVE,
    /// DS4_EXPERT_CACHE_UNIFORM (MLOCK via `lockResident`).
    public static func makeExpertGatherStack(rt: MetalRuntime, model: GGUFModel, dims: DSV4Dims,
                                             nLayers: Int, slots: Int, usage: ExpertUsageStats,
                                             lockResident: Bool)
        -> (gather: (Int, [Int32]) throws -> (GPUTensor, GPUTensor, GPUTensor),
            cache: ExpertSlotCache?, stride: Int?) {
        let env = ProcessInfo.processInfo.environment
        let willNeed = env["DS4_WILLNEED_EXPERTS"] != "0"
        let uncachedFD = env["DS4_EXPERT_PREAD"] == "1" ? model.uncachedFD() : nil
        let gateBytes = (dims.nEmbd / 256) * dims.gateQuant.blockBytes * dims.expertFfn
        let upBytes = (dims.nEmbd / 256) * dims.upQuant.blockBytes * dims.expertFfn
        let downBytes = (dims.expertFfn / 256) * dims.downQuant.blockBytes * dims.nEmbd
        let bundleEnabled = env["DS4_EXPERT_BUNDLE"] == "1"
        let bundle: ExpertBundle? = bundleEnabled
            ? ExpertBundle.openOrBuild(model: model, layers: 0..<nLayers, nExpert: dims.nExperts,
                                       gateBytes: gateBytes, upBytes: upBytes, downBytes: downBytes)
            : nil
        if env["DS4_MTLIO"] == "1" {
            if let bundle { _ = bundle.enableMetalIO(device: rt.device) }
            else {
                FileHandle.standardError.write(Data("DS4 expbundle: DS4_MTLIO=1 richiede un expert-bundle valido — uso pread GGUF\n".utf8))
            }
        }
        let gather: (Int, [Int32]) throws -> (GPUTensor, GPUTensor, GPUTensor) = { il, ids in
            if let b = bundle, let packed = b.gatherPacked(rt, layer: il, ids: ids) { return packed }
            return try GGUFWeights.gatherLayerExperts(rt, model, il, ids: ids, dims: dims,
                                                      willNeed: willNeed, uncachedFD: uncachedFD)
        }
        guard slots > 0 else { return (gather, nil, nil) }
        let S = max(8, slots)
        var fillPrefetch: ((Int, [Int32]) -> Void)? = nil
        if uncachedFD == nil {
            fillPrefetch = { il, ids in
                for id in ids {
                    GGUFWeights.adviseExpert(model, "blk.\(il).ffn_gate_exps.weight", id: id, expertBytes: gateBytes)
                    GGUFWeights.adviseExpert(model, "blk.\(il).ffn_up_exps.weight", id: id, expertBytes: upBytes)
                    GGUFWeights.adviseExpert(model, "blk.\(il).ffn_down_exps.weight", id: id, expertBytes: downBytes)
                }
            }
        }
        let interleave = env["DS4_POOL_INTERLEAVE"] != "0"
        let recordBytes = gateBytes + upBytes + downBytes
        typealias Pool = (gate: GPUTensor, up: GPUTensor, down: GPUTensor)
        let makePool: (Int) throws -> Pool
        if interleave {
            makePool = { nSlots in
                let buf = try GPUTensor.zerosBytes(rt, byteLength: nSlots * recordBytes)
                if lockResident { buf.lockResident() }
                let up = GPUTensor(buffer: buf.buffer, byteLength: nSlots * recordBytes - gateBytes,
                                   count: nSlots * recordBytes - gateBytes, byteOffset: gateBytes)
                let down = GPUTensor(buffer: buf.buffer,
                                     byteLength: nSlots * recordBytes - gateBytes - upBytes,
                                     count: nSlots * recordBytes - gateBytes - upBytes,
                                     byteOffset: gateBytes + upBytes)
                return (gate: buf, up: up, down: down)
            }
        } else {
            makePool = { nSlots in
                let p = (gate: try GPUTensor.zerosBytes(rt, byteLength: nSlots * gateBytes),
                         up: try GPUTensor.zerosBytes(rt, byteLength: nSlots * upBytes),
                         down: try GPUTensor.zerosBytes(rt, byteLength: nSlots * downBytes))
                if lockResident { p.gate.lockResident(); p.up.lockResident(); p.down.lockResident() }
                return p
            }
        }
        let fillOne: (Int, Int32, ExpertSlotCache.LayerPool, Int) throws -> Void = { il, id, pool, slot in
            if let b = bundle {
                if interleave, b.copyExpertInterleaved(layer: il, id: id, dst: pool.gate,
                                                       slot: slot, stride: recordBytes) {
                    return
                }
                if !interleave, b.copyExpert(layer: il, id: id, gateDst: pool.gate,
                                             upDst: pool.up, downDst: pool.down, slot: slot) {
                    return
                }
            }
            nonisolated(unsafe) let jobs: [(name: String, bytes: Int, dst: GPUTensor)] = [
                ("blk.\(il).ffn_gate_exps.weight", gateBytes, pool.gate),
                ("blk.\(il).ffn_up_exps.weight", upBytes, pool.up),
                ("blk.\(il).ffn_down_exps.weight", downBytes, pool.down)]
            let lock = NSLock()
            nonisolated(unsafe) var firstError: Error? = nil
            nonisolated(unsafe) let modelRef = model
            DispatchQueue.concurrentPerform(iterations: jobs.count) { j in
                do {
                    try GGUFWeights.copyExpert(modelRef, jobs[j].name, id: id, expertBytes: jobs[j].bytes,
                                               into: jobs[j].dst, slot: slot, uncachedFD: uncachedFD,
                                               slotStride: interleave ? recordBytes : nil)
                } catch {
                    lock.lock()
                    if firstError == nil { firstError = error }
                    lock.unlock()
                }
            }
            if let e = firstError { throw e }
        }
        let fillBatch: ((Int, [(id: Int32, slot: Int)], ExpertSlotCache.LayerPool) throws -> Void)? =
            env["DS4_MTLIO"] == "1" && bundle != nil ? { il, pairs, pool in
                if bundle!.copyExpertsMetalIO(layer: il, pairs: pairs,
                                               gateDst: pool.gate, upDst: pool.up, downDst: pool.down,
                                               slotStride: interleave ? recordBytes : nil) {
                    return
                }
                for pair in pairs { try fillOne(il, pair.id, pool, pair.slot) }
            } : nil
        let cache = ExpertSlotCache(slotsPerLayer: S, bytesPerExpert: recordBytes, makePool: makePool,
                                    fill: fillOne, fillBatch: fillBatch, prefetch: fillPrefetch,
           warm: { il in
            usage.top(layer: il, n: 128).filter { $0 >= 0 && $0 < Int32(dims.nExperts) }
        },
           slotsFor: { il in
            if ProcessInfo.processInfo.environment["DS4_EXPERT_CACHE_UNIFORM"] == "1" { return S }
            return usage.slotAllocation(base: S)?[il] ?? S
        })
        return (gather, cache, interleave ? recordBytes : nil)
    }
}
