import Foundation
import Metal
import DS4Core

// Stage D: per-layer SSD streaming decode. Instead of holding all N layers'
// weights resident (164GB), each layer is loaded on demand (copied from the GGUF
// mmap into GPUTensors), computed, then evicted. Because a layer's GPU buffers
// must stay alive until the GPU finishes reading them, each layer runs in its OWN
// command buffer (commit + wait, then evict) — the "split command buffer"
// streaming pattern from the C engine. Working set = one layer's weights
// (~few GB) instead of the whole model, so the real model runs on 16GB.
//
// Resident across layers (small): output-head weights, embedding table, per-layer
// KV caches, HC ping-pong buffers, scratch. layerProvider(i) supplies layer i's
// weights (real path: GGUFWeights.layer(rt, model, i)); the returned LayerWeights
// is dropped after the layer commits, freeing its Metal buffers (eviction).

/// Per-phase wall-clock accumulator for the decode forward pass. Each phase is
/// timed around a committed (and waited) command buffer / a CPU gather, so the
/// numbers reflect real elapsed time and answer "I/O vs compute". Times are
/// totals over all forward() calls; `report()` averages per token.
public struct DecodeProfile: Sendable {
    public var embedS = 0.0       // token embedding
    public var routeS = 0.0       // attention + router (compute)
    // DS4_PROFILE_ROUTE: route/attn split into 5 timed sub-phases (ratios meaningful).
    public var routeCompS = 0.0       // hc-pre + NSA compressor (attn + indexer)
    public var routeQS = 0.0          // Q projection (q_a, q_b)
    public var routeKvS = 0.0         // KV projection + indexer scoring
    public var routeAttnPhaseS = 0.0  // flash-attn
    public var routeOutS = 0.0        // output proj + HC-reduce + router
    public var gatherS = 0.0      // gather the 6 selected experts from the mmap (EXPERT I/O)
    public var expertsS = 0.0     // shared FFN + routed MoE matvec (compute)
    public var layerOtherS = 0.0  // non-split decode path (resident experts)
    public var headS = 0.0        // output head
    public var forwards = 0       // number of forward() calls (= tokens)
    public var layers = 0         // total per-layer iterations
    public var expertHits = 0     // expert slot-cache hits (persistent experts)
    public var expertMisses = 0   // expert slot-cache misses (changed experts)
    public var expertPrefilled = 0  // slabs filled by the look-ahead (I/O hidden under compute)
    public var gatherBytes = 0    // expert bytes copied from the mmap (EXPERT I/O volume)

    public init() {}

    /// Engine-side seconds accounted by the per-phase counters (what report()
    /// calls "totale"). Wall-clock minus this = time spent OUTSIDE the engine
    /// (sampler, streaming, UI) — the GUI logs that split per turn.
    public var totalS: Double { embedS + routeS + gatherS + expertsS + layerOtherS + headS }

    public func report(title: String = "Profilo decode") -> String {
        guard forwards > 0 else { return "\(title): nessun forward registrato." }
        let f = Double(forwards)
        let total = embedS + routeS + gatherS + expertsS + layerOtherS + headS
        func ms(_ s: Double) -> String { String(format: "%6.1f", s / f * 1000) }
        func pct(_ s: Double) -> String { String(format: "%2.0f%%", total > 0 ? s / total * 100 : 0) }
        let tps = total > 0 ? f / total : 0
        var cacheLine = ""
        if expertHits + expertMisses > 0 {
            let rate = Double(expertHits) / Double(expertHits + expertMisses) * 100
            let ahead = expertPrefilled > 0 ? " — \(expertPrefilled) slab da look-ahead" : ""
            cacheLine = "\n  cache expert \(expertHits) hit / \(expertMisses) miss  (\(String(format: "%.0f", rate))% hit)\(ahead)"
        }
        // Effective gather bandwidth: how fast the expert slabs actually leave the
        // SSD/page cache. Compare against the raw sequential bandwidth of the disk
        // to see the streaming headroom (bytes/gatherS; page-cache hits inflate it).
        if gatherBytes > 0 && gatherS > 0 {
            let mbTok = Double(gatherBytes) / f / 1_048_576
            let gbs = Double(gatherBytes) / gatherS / 1e9
            cacheLine += "\n  gather IO    \(String(format: "%6.1f", mbTok)) MB/token — banda effettiva \(String(format: "%.2f", gbs)) GB/s"
        }
        var routeSplit = ""   // DS4_PROFILE_ROUTE: ratios meaningful, absolutes inflated (extra commits)
        if routeCompS + routeQS + routeKvS + routeAttnPhaseS + routeOutS > 0 {
            routeSplit = "\n     ├ comp \(ms(routeCompS)) ms/token (hc-pre + NSA compressor)"
                       + "\n     ├ q    \(ms(routeQS)) ms/token (q_a + q_b proj)"
                       + "\n     ├ kv   \(ms(routeKvS)) ms/token (kv proj + indexer)"
                       + "\n     ├ attn \(ms(routeAttnPhaseS)) ms/token (flash-attn)"
                       + "\n     └ out  \(ms(routeOutS)) ms/token (output proj + HC + router)"
        }
        return """
        \(title) — \(forwards) token, \(layers) iterazioni-layer
          embed        \(ms(embedS)) ms/token  (\(pct(embedS)))
          route/attn   \(ms(routeS)) ms/token  (\(pct(routeS)))   compute\(routeSplit)
          gather IO    \(ms(gatherS)) ms/token  (\(pct(gatherS)))   <- streaming esperti (SSD/page cache)
          experts      \(ms(expertsS)) ms/token  (\(pct(expertsS)))   compute
          layer (alt)  \(ms(layerOtherS)) ms/token  (\(pct(layerOtherS)))
          output head  \(ms(headS)) ms/token  (\(pct(headS)))\(cacheLine)
          ----------------------------------------
          totale       \(ms(total)) ms/token  (~\(String(format: "%.2f", tps)) tok/s)
        """
    }
}

public final class StreamingDecoder {
    let rt: MetalRuntime
    let d: DSV4Dims
    let rope: RopeParams
    let nLayers: Int
    let layerProvider: (Int) throws -> LayerWeights
    let embedTable: GPUTensor
    let out: OutputHeadWeights
    let rmsEps: Float, hcEps: Float

    /// Per-phase decode timing (opt-in: read after a run, reset between runs).
    public var profile = DecodeProfile()
    public func resetProfile() { profile = DecodeProfile() }
    /// DS4_PROFILE_ROUTE=1: split route/attn into pre (Q/KV proj + compressor) and
    /// attn (attention + out proj + HC + router), each its own command buffer + timed.
    /// Adds a commit/wait per layer (absolute numbers inflate); read the RATIO.
    let profileRoute = ProcessInfo.processInfo.environment["DS4_PROFILE_ROUTE"] == "1"
    /// Batched prefill phase B: encode ALL of a group's token-FFNs into ONE
    /// command buffer (serial encoder ⇒ same dispatch order and visibility as
    /// N separate buffers) instead of one commit+wait per token — the dominant
    /// fixed cost at 512-token chunks (43 layers × 512 sync round-trips).
    /// DS4_PREFILL_FFN_BATCH=0 restores the per-token path (A/B parity check).
    let prefillFFNBatch = ProcessInfo.processInfo.environment["DS4_PREFILL_FFN_BATCH"] != "0"
    /// Batched prefill phase A: encode up to DS4_PREFILL_ROUTE_BATCH consecutive
    /// tokens' routes into ONE command buffer — per-token scratch snapshots are
    /// blit-copied GPU-side between tokens, and the CPU reads ALL the selections
    /// after a single wait, instead of one commit+wait per token per layer.
    /// Attention stays token-sequential INSIDE the buffer (serial encoder ⇒ same
    /// dispatch order ⇒ identical numerics). Default 32: cuts the route syncs
    /// 32× while keeping each buffer's GPU run bounded. 0/1 = off (parity);
    /// layers with the indexer ACTIVE always fall back to per-token (a CPU
    /// top-k sits between the two halves of their route).
    let prefillRouteBatch: Int = {
        let v = ProcessInfo.processInfo.environment["DS4_PREFILL_ROUTE_BATCH"].flatMap(Int.init) ?? 32
        return max(1, v)
    }()
    /// DS4_PREFILL_MM=1 (OPT-IN): the group's routed FFN runs through the
    /// mul_mm_id matrix-matrix kernels (expert weights read once per tile for
    /// ALL the group's tokens) instead of one matvec chain per token. Same
    /// math, different accumulation order (simdgroup MMA, f16 mid) — outputs
    /// are close but NOT bit-identical to the matvec path, hence opt-in until
    /// validated A/B on-device. iq2_xxs gate/up + q2_K down only (the Flash
    /// shape); groups with reduced selections (DS4_ACTIVE_EXPERTS < k) fall
    /// back to the matvec path (map0 requires k distinct ids per token).
    let prefillMM = ProcessInfo.processInfo.environment["DS4_PREFILL_MM"] == "1"

    /// Expert-cache hook: given (layer index, the 6 selected ids), gather and pack
    /// ONLY those experts' gate/up/down. When set, forward() splits each layer at
    /// the router and loads 6/256 experts on demand instead of the full set.
    let expertGather: ((Int, [Int32]) throws -> (GPUTensor, GPUTensor, GPUTensor))?
    /// Optional LRU slot-cache ("persistent + changing experts"): when set, the
    /// DECODE path serves hits from resident GPU pools (zero copies) and gathers
    /// only the misses; the matvec runs on the pool with slot-index ids.
    public let slotCache: ExpertSlotCache?
    /// Stride in byte fra gli slot del pool quando il layout e' INTERLEAVED
    /// (record gate|up|down contigui, come nel bundle): passato come nb02 ai
    /// dispatch MoE del percorso slot-cache. nil = layout storico (3 buffer
    /// stretti, stride = dimensione del singolo slab).
    let slotCacheStride: Int?
    /// Routing-frequency statistics (the "usage imatrix"): fed by every route,
    /// persisted by the service, and used to pre-warm the slot cache.
    public let usage: ExpertUsageStats?
    /// Optional read-ahead hook. `prefetch(i)` resolves layer i's mmap byte ranges
    /// (cheap, on the decode thread) and madvises them WILLNEED on a background
    /// queue, so the next layer's SSD I/O overlaps the current layer's compute.
    /// nil = off (resident paths don't need it). Cannot affect numerics.
    let prefetch: ((Int) -> Void)?
    /// Expert look-ahead (the C engine's overlap, adapted to the slot cache):
    /// `lookahead(layer, token)` returns the expert ids to PREFILL into that
    /// layer's pool — EXACT for the hash-routed layers (tid2eid row is known
    /// from the token id alone), usage-prior top-N for the others (speculative,
    /// DS4_EXPERT_LOOKAHEAD). runLayer(i) kicks prefill(i+1) on `lookaheadQ`
    /// right after its own gather, so the next layer's expert I/O runs in the
    /// SSD-idle window while the GPU computes layer i. Cannot affect numerics:
    /// the pool holds the same bytes either way; a wrong guess is just an
    /// unused slot. nil = off.
    let lookahead: ((_ layer: Int, _ token: Int) -> [Int32])?
    /// Serial queue for the speculative prefills (one layer ahead at a time —
    /// a backlog would just re-touch already-resident ids and skip).
    private let lookaheadQ = DispatchQueue(label: "ds4.expert-lookahead", qos: .userInitiated)

    let scratch: DecodeScratch
    let rawCaches: [GPUTensor]
    let compStates: [CompressorState?]   // NSA compressor state per compressed layer (nil on layers 0,1)
    /// NSA indexer compressor state (DSA): ratio-4 layers only. Beyond
    /// `d.indexerTopK` compressed rows, attention is restricted to the top-K
    /// most relevant for the current query (C: indexer_allowed_decode_one).
    let indexStates: [CompressorState?]
    /// Halves of s.mask dirtied by the last indexer selection (0 = clean).
    private var maskDirtyCount = 0
    /// Layers with real KV allocation (full model: 0..<nLayers; distributed slice: its range).
    let kvRange: Range<Int>
    /// KV capacity in tokens (raw rows per layer).
    let maxKeys: Int
    let hcA, hcB, embd: GPUTensor
    let flat, pre, owts, otmp, oembd, onormed, logits: GPUTensor
    let idsPacked: GPUTensor   // [0,1,...,k-1] for the packed-experts matvec
    /// Persistent staging for the decode slot-cache ids: rewritten every layer
    /// instead of allocating a fresh 24-byte MTLBuffer per layer per token
    /// (~43 allocations/token). Safe to reuse even with the ASYNC routed FFN:
    /// the memcpy happens after this layer's route commit+wait, which
    /// queue-orders after (= joins) every previous FFN command buffer.
    let slotsScratch: GPUTensor
    /// Second staging buffer, alternated by layer parity. DEFENSIVE: today at
    /// most ONE routed-FFN cb is ever in flight (see slotsScratch), so a single
    /// buffer would suffice — the parity keeps the invariant local instead of
    /// depending on the route-wait ordering, and costs 24 bytes.
    let slotsScratchB: GPUTensor
    /// The last layer's routed-FFN command buffer, committed WITHOUT a CPU wait
    /// (DS4_ASYNC_FFN, default ON): the next layer's route commit+wait lands on
    /// the same in-order queue, so the GPU stays fed while the CPU encodes —
    /// the per-layer bubble (encode time x 43) disappears. Explicitly waited at
    /// end of token (before the output head / readHC) and on every error path.
    private var inflightFFN: GraphContext?
    let asyncFFN = ProcessInfo.processInfo.environment["DS4_ASYNC_FFN"] != "0"
    /// One embedding-table ROW (F16, nEmbd × 2 B), CPU-staged per token.
    /// Binding the full multi-hundred-MB no-copy table to a command buffer
    /// makes Metal wire the WHOLE mapping every token — on tight-RAM machines
    /// where the expert churn evicts it, that re-faults it from SSD each time
    /// (the ~500 ms "embed" phase). Staging the row costs an ~8 KB memcpy and
    /// keeps the table out of the GPU residency set. Identical bytes → same
    /// numerics.
    let embedRowStage: GPUTensor

    public init(rt: MetalRuntime, dims: DSV4Dims, rope: RopeParams, nLayers: Int,
                layerProvider: @escaping (Int) throws -> LayerWeights,
                embedTable: GPUTensor, out: OutputHeadWeights, maxKeys: Int,
                rmsEps: Float = ModelDefaults.rmsEps, hcEps: Float = ModelDefaults.hcEps,
                expertGather: ((Int, [Int32]) throws -> (GPUTensor, GPUTensor, GPUTensor))? = nil,
                slotCache: ExpertSlotCache? = nil,
                usage: ExpertUsageStats? = nil,
                prefetch: ((Int) -> Void)? = nil,
                lookahead: ((_ layer: Int, _ token: Int) -> [Int32])? = nil,
                kvLayers: Range<Int>? = nil,
                slotCacheStride: Int? = nil) throws {
        // Il kernel del router (kernel_dsv4_router_finalize_one) ha 256 esperti
        // e scala expert-weight 1.5 CABLATI (bitonic a 256 thread fissi): con
        // una shape diversa — es. la "pro" a 384 esperti / scala 2.5 — gli
        // esperti oltre il 255 non verrebbero MAI selezionati e i pesi di
        // route sarebbero scalati male: output plausibile ma sistematicamente
        // sbagliato. Fallire forte al load finché il kernel non è
        // parametrizzato (audit kernel 2026-07-04, finding CERTO).
        guard dims.nExperts == 256 else {
            throw MetalError.unsupported(
                "router: il kernel supporta SOLO 256 esperti (shape con \(dims.nExperts))")
        }
        self.rt = rt; self.d = dims; self.rope = rope; self.nLayers = nLayers
        self.layerProvider = layerProvider; self.embedTable = embedTable; self.out = out
        self.rmsEps = rmsEps; self.hcEps = hcEps; self.expertGather = expertGather
        self.slotCache = slotCache
        self.slotCacheStride = slotCacheStride
        self.usage = usage
        self.prefetch = prefetch
        self.lookahead = lookahead
        let hcDim = dims.nHC * dims.nEmbd
        // Distributed slice: allocate KV/compressor state ONLY for `kvLayers`
        // (a worker never runs the other layers — dummy 1-float buffers there).
        // nil = full model (the single-machine default).
        let kvRange = kvLayers ?? 0..<nLayers
        self.kvRange = kvRange
        self.maxKeys = maxKeys
        // NSA compressor state per compressed layer (ratio!=0); comp rows accumulate
        // ~1 per `ratio` tokens, so the attention KV scratch must hold maxKeys raw rows
        // + up to maxKeys/4 compressed rows (ratio-4 is the densest).
        let maxComp = maxKeys / 4 + 8
        compStates = try (0..<nLayers).map { il -> CompressorState? in
            guard kvRange.contains(il) else { return nil }
            let ratio = DSV4Shape.compressRatio(layer: il)
            guard ratio != 0 else { return nil }
            return try CompressorState(rt, ratio: ratio, headDim: dims.headDim, maxComp: maxKeys / ratio + 8)
        }
        // NSA indexer compressor (DSA): ratio-4 layers only (head_dim 128).
        indexStates = try (0..<nLayers).map { il -> CompressorState? in
            guard kvRange.contains(il), DSV4Shape.compressRatio(layer: il) == 4 else { return nil }
            return try CompressorState(rt, ratio: 4, headDim: dims.nIndexerHeadDim, maxComp: maxKeys / 4 + 8)
        }
        scratch = try DecodeScratch(rt, dims, maxKeys: maxKeys + maxComp)
        idsPacked = try GPUTensor.bytes(rt, Array(0..<Int32(dims.k)).withUnsafeBytes { Array($0) }, elementCount: dims.k)
        slotsScratch = try GPUTensor.zerosBytes(rt, byteLength: dims.k * 4)
        slotsScratchB = try GPUTensor.zerosBytes(rt, byteLength: dims.k * 4)
        embedRowStage = try GPUTensor.zerosBytes(rt, byteLength: max(2, embedTable.byteLength / max(1, dims.vocab)))
        // Raw-KV cache rows: the full context (default) or a ring-buffer of nSWA when
        // DS4_RAW_RING=1. Attention only ever reads the last nSWA raw rows (NSA), so
        // the older rows need not stay resident — the ring cuts the raw-KV RAM from
        // O(contextSize) to a constant. The write slot, attention staging and
        // export/import all key off `rawCache.count/headDim`, so the full cache is a
        // no-wrap special case (behaviour identical). Opt-in: validate the parity
        // tests (StreamingDecoder/GraphAttn/KV-snapshot) before making it the default.
        let ringOn = getenv("DS4_RAW_RING").map { String(cString: $0) == "1" } ?? false   // live (set from the UI toggle)
        let rawRows = ringOn ? min(dims.nSWA, maxKeys) : maxKeys
        // lazyZeros (no memset): the raw cache is sized for the FULL context but only
        // rows 0..<pos are ever written, and attention reads only written rows (the
        // NSA window is the last nSWA, always recently written). With eager .zeros the
        // memset commits maxKeys*headDim*4*nLayer of PHYSICAL RAM at load (≈2 GB/layer
        // at a 1M ctx!), which on the SSD-streaming path evicts the page cache the
        // experts/dense weights live in → more re-faults → slower decode. Zero-fill-on-
        // demand makes the physical footprint scale with the tokens ACTUALLY generated,
        // so a large default context is free until the conversation grows. (Unwritten
        // rows are never read; if they were, fresh shared pages read back as zero.)
        rawCaches = try (0..<nLayers).map { il in
            kvRange.contains(il) ? try GPUTensor.lazyZeros(rt, floatCount: rawRows * dims.headDim)
                                 : try GPUTensor.zeros(rt, floatCount: 1)
        }
        hcA = try .zeros(rt, floatCount: hcDim); hcB = try .zeros(rt, floatCount: hcDim)
        embd = try .zeros(rt, floatCount: dims.nEmbd)
        flat = try .zeros(rt, floatCount: hcDim); pre = try .zeros(rt, floatCount: dims.nHC)
        owts = try .zeros(rt, floatCount: dims.nHC); otmp = try .zeros(rt, floatCount: dims.nHC)
        oembd = try .zeros(rt, floatCount: dims.nEmbd); onormed = try .zeros(rt, floatCount: dims.nEmbd)
        logits = try .zeros(rt, floatCount: dims.vocab)
    }

    public func forward(token: Int, pos: Int, nKeys: Int) throws -> [Float] {
        // Fresh sequence: reset the recurrent compressor state (score=-inf, count=0).
        if pos == 0 { for c in compStates { try c?.reset(rt) }; for c in indexStates { try c?.reset(rt) } }
        // Layer 0's expert I/O can start NOW: the token id is known before any
        // GPU work (hash layer -> exact ids), so its fill overlaps embed+route(0).
        kickLookahead(after: -1, token: token)
        try embedToken(token, into: hcA)
        var cur = hcA, other = hcB
        do {
            for i in 0..<nLayers {
                // Per-layer pool drain, like the prefill loops: the command buffers/
                // encoders are autoreleased ObjC objects — without this a LONG
                // generation (hundreds of tokens x ~3 cb/layer) accumulates them
                // for the whole turn instead of freeing at each layer.
                try autoreleasepool {
                    let w = try layerProvider(i)        // LOAD layer i (dense; experts on demand if cached)
                    if i + 1 < nLayers { prefetch?(i + 1) }   // read-ahead next layer (overlaps its I/O)
                    try runLayer(i, w: w, layerRope: DSV4Shape.ropeParams(layer: i),
                                 cur: cur, other: other, pos: pos, nKeys: nKeys, token: token)
                    swap(&cur, &other)
                    // w (and any gathered experts) drop here -> Metal buffers freed (EVICT)
                }
            }
        } catch {
            drainFFN()   // never leave the routed FFN in flight on a torn-down token
            throw error
        }
        // Join the last layer's async FFN: the output head's own commit+wait
        // would cover the GPU ordering, but exportKV/readHC and error paths
        // must find NOTHING in flight — one explicit drain keeps the invariant.
        drainFFN()
        profile.forwards += 1
        return try outputHead(cur)
    }

    // MARK: - Distributed slice execution (pipeline parallelism)
    //
    // These let a node run only PART of the model: the coordinator owns the
    // embedding + output head, each worker owns a contiguous layer range and runs
    // it over an incoming HC state. The HC state (nHC*nEmbd floats) is what crosses
    // the wire between nodes. They reuse embedToken/runLayer/outputHead, so a slice
    // [start,end] is numerically identical to the same layers inside forward().

    /// HC state width that crosses the wire (nHC * nEmbd floats).
    public var hcStateCount: Int { d.nHC * d.nEmbd }

    /// Coordinator: embed `token` into the HC state (the start of the pipeline).
    public func embed(token: Int, pos: Int) throws -> [Float] {
        try embedToken(token, into: hcA)
        return readHC(hcA)
    }

    /// Worker: run layers `start...end` over an incoming HC state at absolute `pos`,
    /// returning the produced HC state to forward to the next slice. Resets only this
    /// slice's recurrent compressor state on a fresh sequence (pos == 0).
    public func forwardSlice(hc hcIn: [Float], pos: Int, nKeys: Int, start: Int, end: Int,
                             token: Int = -1) throws -> [Float] {
        precondition(start >= 0 && end < nLayers && start <= end, "invalid layer slice \(start)...\(end)")
        if pos == 0 { for i in start...end { try compStates[i]?.reset(rt); try indexStates[i]?.reset(rt) } }
        writeFloats(hcIn, into: hcA)
        kickLookahead(after: start - 1, token: token)
        var cur = hcA, other = hcB
        do {
            for i in start...end {
                try autoreleasepool {   // per-layer drain (see forward)
                    let w = try layerProvider(i)
                    if i + 1 <= end { prefetch?(i + 1) }
                    try runLayer(i, w: w, layerRope: DSV4Shape.ropeParams(layer: i),
                                 cur: cur, other: other, pos: pos, nKeys: nKeys, token: token)
                    swap(&cur, &other)
                }
            }
        } catch {
            drainFFN()
            throw error
        }
        drainFFN()   // readHC reads `cur` CPU-side: the async FFN must be complete
        profile.forwards += 1
        return readHC(cur)
    }

    /// Worker, chunked prefill: run layers `start...end` over `hcs.count` consecutive
    /// tokens' HC states starting at absolute `posBase`. Token-outer (numerically
    /// identical to consecutive forwardSlice calls); amortizes the NETWORK round
    /// trip over the chunk — one WORK/RESULT per chunk instead of per token.
    public func forwardSliceBatch(hcs: [[Float]], posBase: Int, start: Int, end: Int,
                                  tokens: [Int]? = nil) throws -> [[Float]] {
        var out: [[Float]] = []
        out.reserveCapacity(hcs.count)
        for (i, hc) in hcs.enumerated() {
            let pos = posBase + i
            out.append(try forwardSlice(hc: hc, pos: pos, nKeys: pos + 1, start: start, end: end,
                                        token: tokens.map { $0[i] } ?? -1))
        }
        return out
    }

    /// Coordinator/last node: run the output head over the final HC state → logits.
    public func head(hc hcIn: [Float]) throws -> [Float] {
        writeFloats(hcIn, into: hcA)
        return try outputHead(hcA)
    }

    /// Read the HC state (nHC*nEmbd floats) out of a GPU buffer.
    private func readHC(_ t: GPUTensor) -> [Float] {
        let n = d.nHC * d.nEmbd
        let p = t.buffer.contents().advanced(by: t.byteOffset).bindMemory(to: Float.self, capacity: n)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }

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
                lastHC = try autoreleasepool {
                    try prefillRange(tokens, start: start, end: end, posBase: startPos)
                }
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

    /// Reusable per-token staging for the batched prefill: ONE set per chunk,
    /// rewritten at every layer (layer i's phase B completes before layer i+1's
    /// phase A touches them) — instead of 3·n fresh Metal buffers per LAYER
    /// (43 × 512 × 3 ≈ 66k allocations per chunk).
    private struct PrefillStage {
        let cur: [GPUTensor]     // n × nEmbd        (attn-normed FFN input)
        let attn: [GPUTensor]    // n × nHC·nEmbd    (post-attention residual)
        let split: [GPUTensor]   // n × 24           (HC split)
        let ids: [GPUTensor]     // n × k Int32      (remapped ids, padded to k)
        let rw: [GPUTensor]      // n × k Float      (route weights, 0-padded)
        /// Extra buffers for the mul_mm_id path (DS4_PREFILL_MM), rewritten per
        /// group: token-major activation matrix, group-local ids/weights, the
        /// expert-major map (htpe/hids) and the mid/down6 outputs.
        struct MMBuffers {
            let curMat: GPUTensor    // n × nEmbd f32 (chunk-global rows)
            let idsMat: GPUTensor    // n × k Int32   (group-local rows)
            let wMat: GPUTensor      // n × k f32     (group-local rows)
            let htpe: GPUTensor      // maxUnion u32
            let hids: GPUTensor      // maxUnion × n Int32
            let mid: GPUTensor       // n × k × expertFfn f16
            let down6: GPUTensor     // n × k × nEmbd f32
            // Batched SHARED-expert FFN (Q8_0 path only): token-major gate/up/
            // mid intermediates and the per-token shared output rows.
            let sGate: GPUTensor     // n × sharedFfn f32
            let sUp: GPUTensor       // n × sharedFfn f32
            let sMid: GPUTensor      // n × sharedFfn f32
            let sOut: GPUTensor      // n × nEmbd f32
            let ones: GPUTensor      // n × f32 = 1 (unit route weights for the rows-swiglu)
        }
        let mm: MMBuffers?
        init(_ rt: MetalRuntime, n: Int, d: DSV4Dims, mmPath: Bool, maxUnion: Int) throws {
            cur = try (0..<n).map { _ in try .zeros(rt, floatCount: d.nEmbd) }
            attn = try (0..<n).map { _ in try .zeros(rt, floatCount: d.nHC * d.nEmbd) }
            split = try (0..<n).map { _ in try .zeros(rt, floatCount: 24) }
            ids = try (0..<n).map { _ in try .zerosBytes(rt, byteLength: d.k * 4) }
            rw = try (0..<n).map { _ in try .zeros(rt, floatCount: d.k) }
            if mmPath {
                let onesBuf = try GPUTensor.zeros(rt, floatCount: n)
                let op = (onesBuf.buffer.contents() + onesBuf.byteOffset)
                    .bindMemory(to: Float.self, capacity: n)
                for i in 0..<n { op[i] = 1 }
                mm = MMBuffers(
                    curMat: try .zeros(rt, floatCount: n * d.nEmbd),
                    idsMat: try .zerosBytes(rt, byteLength: n * d.k * 4),
                    wMat: try .zeros(rt, floatCount: n * d.k),
                    htpe: try .zerosBytes(rt, byteLength: max(1, maxUnion) * 4),
                    hids: try .zerosBytes(rt, byteLength: max(1, maxUnion) * n * 4),
                    mid: try .zerosBytes(rt, byteLength: n * d.k * d.expertFfn * 2),
                    down6: try .zeros(rt, floatCount: n * d.k * d.nEmbd),
                    sGate: try .zeros(rt, floatCount: n * d.sharedFfn),
                    sUp: try .zeros(rt, floatCount: n * d.sharedFfn),
                    sMid: try .zeros(rt, floatCount: n * d.sharedFfn),
                    sOut: try .zeros(rt, floatCount: n * d.nEmbd),
                    ones: onesBuf)
            } else {
                mm = nil
            }
        }
    }

    /// Process one prompt chunk [start, end) layer-major at absolute positions
    /// posBase+start … . Weights for each layer are loaded once and applied to all
    /// the chunk's tokens (in order). On the expert-gather path the routed-FFN
    /// phase is BATCHED: each unique expert is gathered once per group instead of
    /// 6 per token. Returns the chunk's last token's final HC state.
    private func prefillRange(_ tokens: [Int], start: Int, end: Int, posBase: Int) throws -> GPUTensor {
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
                let layerRope = DSV4Shape.ropeParams(layer: i)
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
        // Free the chunk's activation buffers now (2·n HC tensors); only the last
        // HC state survives into the next chunk / output head.
        let last = cur[n - 1]
        cur.removeAll(keepingCapacity: false)
        other.removeAll(keepingCapacity: false)
        return last
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
        var idsT: [[Int32]] = [], rwT: [[Float]] = []
        idsT.reserveCapacity(n); rwT.reserveCapacity(n)
        var j = 0
        while j < n {
            // Extent of the batchable run starting at j: consecutive tokens for
            // which the indexer stays INACTIVE — its compressed-row count grows
            // deterministically with pos, so activation is checked prospectively
            // (extraRows) for the whole run before encoding any of it.
            var jEnd = j
            if prefillRouteBatch > 1 && !profileRoute {
                var extraRows = 0
                while jEnd < n && (jEnd - j) < prefillRouteBatch {
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
                    try encodeRoute(i, w: w, layerRope: layerRope, curHc: cur[j], pos: pos, nKeys: pos + 1,
                                    token: tokens[j])
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

    /// Embed a whole prefill chunk in ONE command buffer: the tokens' table rows
    /// are CPU-staged into a transient n-row table, then n embedTokenHC encodes
    /// run back-to-back (the encoder is serial, so the shared `embd` intermediate
    /// is safe). Numerically identical to n embedToken calls — it only removes
    /// the n-1 per-token commit+wait round-trips (512 GPU syncs per chunk).
    private func embedTokensBatch(_ toks: [Int], into hcs: [GPUTensor]) throws {
        guard toks.count > 1 else {
            if let t = toks.first { try embedToken(t, into: hcs[0]) }
            return
        }
        let t = Date()
        let rowBytes = embedRowStage.byteLength
        let stage = try GPUTensor.zerosBytes(rt, byteLength: toks.count * rowBytes)
        for (j, token) in toks.enumerated() {
            precondition(token >= 0 && token < d.vocab, "embedTokensBatch: token \(token) out of vocab")
            memcpy(stage.buffer.contents() + j * rowBytes,
                   embedTable.buffer.contents() + embedTable.byteOffset + token * rowBytes,
                   rowBytes)
        }
        let ec = GraphContext(rt)
        try ec.begin()
        for j in 0..<toks.count {
            try ec.embedTokenHC(table: stage, token: j, embd: embd, hc: hcs[j],
                                nEmbd: d.nEmbd, nVocab: toks.count, nHC: d.nHC)
        }
        ec.commit()
        profile.embedS += Date().timeIntervalSince(t)
    }

    /// Embed one token into the HC state buffer `hc` (own command buffer).
    /// The token's table row is CPU-staged into `embedRowStage` and the kernel
    /// runs on that (token 0 of a 1-row table) — see embedRowStage for why.
    private func embedToken(_ token: Int, into hc: GPUTensor) throws {
        let t = Date()
        let rowBytes = embedRowStage.byteLength
        precondition(token >= 0 && token < d.vocab, "embedToken: token \(token) out of vocab")
        memcpy(embedRowStage.buffer.contents(),
               embedTable.buffer.contents() + embedTable.byteOffset + token * rowBytes,
               rowBytes)
        let ec = GraphContext(rt)
        try ec.begin()
        try ec.embedTokenHC(table: embedRowStage, token: 0, embd: embd, hc: hc,
                            nEmbd: d.nEmbd, nVocab: 1, nHC: d.nHC)
        ec.commit()
        profile.embedS += Date().timeIntervalSince(t)
    }

    /// Read back the router's selection after a committed decodeRoute, applying
    /// the activeExperts top-K reduction (route weights renormalized to the
    /// original total). Returns the final (ids, weights), both of count K ≤ d.k.
    /// Also feeds the usage statistics ("usage imatrix") for `layer`.
    private func readRouteSelection(layer: Int) -> (ids: [Int32], rw: [Float]) {
        selection(sel: scratch.selected, weights: scratch.rw, layer: layer)
    }

    /// Core of readRouteSelection, reading from ARBITRARY buffers — the batched
    /// route phase snapshots each token's selection into per-token buffers and
    /// reads them all back after one commit.
    private func selection(sel: GPUTensor, weights: GPUTensor, layer: Int) -> (ids: [Int32], rw: [Float]) {
        let selPtr = (sel.buffer.contents() + sel.byteOffset).bindMemory(to: Int32.self, capacity: d.k)
        var ids = Array(UnsafeBufferPointer(start: selPtr, count: d.k))
        let wptr = (weights.buffer.contents() + weights.byteOffset).bindMemory(to: Float.self, capacity: d.k)
        var rw = Array(UnsafeBufferPointer(start: wptr, count: d.k))
        let K = max(1, min(d.activeExperts, d.k))
        if K < d.k {
            let keep = (0..<d.k).sorted { rw[$0] > rw[$1] }.prefix(K)
            let origSum = rw.reduce(0, +)
            let keptSum = keep.reduce(Float(0)) { $0 + rw[$1] }
            let scale = keptSum > 0 ? origSum / keptSum : 1
            ids = keep.map { ids[$0] }
            rw = keep.map { rw[$0] * scale }
        }
        usage?.record(layer: layer, ids: ids)
        return (ids, rw)
    }

    /// CPU-write `a` into the head of a shared GPU buffer (safe between commits).
    private func writeFloats(_ a: [Float], into t: GPUTensor) {
        a.withUnsafeBytes {
            _ = memcpy(t.buffer.contents().advanced(by: t.byteOffset), $0.baseAddress!, $0.count)
        }
    }

    /// CPU-copy `count` floats between shared GPU buffers (after a commit).
    private func copyFloats(from src: GPUTensor, to dst: GPUTensor, count: Int) {
        memcpy(dst.buffer.contents().advanced(by: dst.byteOffset),
               src.buffer.contents().advanced(by: src.byteOffset), count * 4)
    }

    /// Zero s.down6 rows K..d.k-1 so the fixed moeSum6 adds zeros for unused slots.
    private func zeroDown6(from K: Int) {
        guard K < d.k else { return }
        let dptr = scratch.down6.buffer.contents().bindMemory(to: Float.self, capacity: d.k * d.nEmbd)
        for r in K..<d.k { for c in 0..<d.nEmbd { dptr[r * d.nEmbd + c] = 0 } }
    }

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
    private func drainFFN() {
        inflightFFN?.waitCompleted()
        inflightFFN = nil
    }

    /// Speculative look-ahead: prefill layer i+1's slot pool while the GPU
    /// computes layer i (its own gather just finished, so the SSD is idle until
    /// the next layer's demand fill). The id list is resolved on the DECODE
    /// thread (usage prior / tid2eid mmap read — cheap); only the I/O moves to
    /// the background queue. Decode-only: the batched prefill has its own
    /// union pipeline.
    private func kickLookahead(after i: Int, token: Int) {
        guard let lookahead, let cache = slotCache, i + 1 < nLayers else { return }
        let next = i + 1
        let ids = lookahead(next, token)
        guard !ids.isEmpty else { return }
        lookaheadQ.async { cache.prefill(layer: next, ids: ids) }
    }

    /// One decode layer for one token: `cur` (HC in) -> `other` (HC out). Writes
    /// KV[i][pos], updates compStates[i]. Shared by `forward` (decode) and the
    /// layer-major `prefill` — identical numerics either way.
    private func runLayer(_ i: Int, w: LayerWeights, layerRope: RopeParams,
                          cur: GPUTensor, other: GPUTensor, pos: Int, nKeys: Int, token: Int) throws {
        // Kick the NEXT layer's look-ahead at the START of this one: the fill
        // window becomes the whole layer (route + attention + FFN, ~2x the
        // post-gather window) instead of the few ms before the next acquire.
        // Its I/O shares the SSD with this layer's own gather, but the disk's
        // parallel ceiling is well above the demand queue depth and the demand
        // path preempts on contention for the same layer's lock.
        kickLookahead(after: i, token: token)
        if let gather = expertGather {
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
    private func encodeRoute(_ i: Int, w: LayerWeights, layerRope: RopeParams,
                             curHc: GPUTensor, pos: Int, nKeys: Int, token: Int) throws {
        let idx = indexStates[i]
        let hasIdxWeights = w.idxKv != nil && w.idxQB != nil && w.idxProj != nil
        let active = hasIdxWeights && indexerActive(i, pos: pos)
        if active, let idx {
            // Indexer layers always split (CPU top-k sits between pre and attn). The
            // phase() boundaries inside decodeRoutePre/Attn are no-ops unless profiling.
            let c1 = GraphContext(rt); if profileRoute { c1.phaseTimes = [:] }; try c1.begin()
            let nComp = try c1.decodeRoutePre(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope,
                                              rawCache: rawCaches[i], pos: pos, rmsEps: rmsEps, hcEps: hcEps,
                                              comp: compStates[i], idx: hasIdxWeights ? idx : nil,
                                              indexerScoring: true)
            try c1.phase("kv")
            c1.commit()
            applyIndexerMask(nKeys: nKeys, nComp: nComp, nIdxComp: idx.count)
            let c2 = GraphContext(rt); if profileRoute { c2.phaseTimes = [:] }; try c2.begin()
            try c2.decodeRouteAttn(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope, rawCache: rawCaches[i],
                                   nKeys: nKeys, pos: pos, token: token, rmsEps: rmsEps, hcEps: hcEps,
                                   nComp: nComp, comp: compStates[i])
            try c2.phase("out")
            c2.commit()
            if profileRoute { accumulateRoutePhases(c1, c2) }
        } else if profileRoute {
            // Profiling: split route/attn into 5 timed sub-phases (comp, q, kv, attn,
            // out). The extra commits inflate the ABSOLUTE time — read the RATIOS.
            clearMaskIfDirty()
            let c = GraphContext(rt); c.phaseTimes = [:]; try c.begin()
            let nComp = try c.decodeRoutePre(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope,
                                             rawCache: rawCaches[i], pos: pos, rmsEps: rmsEps, hcEps: hcEps,
                                             comp: compStates[i], idx: hasIdxWeights ? idx : nil,
                                             indexerScoring: false)
            try c.phase("kv")
            try c.decodeRouteAttn(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope, rawCache: rawCaches[i],
                                  nKeys: nKeys, pos: pos, token: token, rmsEps: rmsEps, hcEps: hcEps,
                                  nComp: nComp, comp: compStates[i])
            try c.phase("out")
            c.commit()
            accumulateRoutePhases(c, nil)
        } else {
            clearMaskIfDirty()
            let c1 = GraphContext(rt); try c1.begin()
            let nComp = try c1.decodeRoutePre(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope,
                                              rawCache: rawCaches[i], pos: pos, rmsEps: rmsEps, hcEps: hcEps,
                                              comp: compStates[i], idx: hasIdxWeights ? idx : nil,
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
            profile.routeOutS += pt["out", default: 0]
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

    /// Will the indexer restrict this token's compressed rows on layer `i`?
    /// (prospective count: the compressor may emit one more row for this token.)
    /// `extraRows` = rows the tokens BEFORE this one in a not-yet-encoded batch
    /// will emit — the batched route phase checks activation prospectively for
    /// the whole run before encoding any of it.
    /// C condition (ds4.c:15246): layer_n_comp > sparse_threshold AND
    /// layer_n_index_comp > DS4_N_INDEXER_TOP_K. On ratio-4 layers the attention
    /// and indexer compressors emit in lockstep, so one prospective count serves
    /// both comparisons.
    private func indexerActive(_ i: Int, pos: Int, extraRows: Int = 0) -> Bool {
        guard let idx = indexStates[i] else { return false }
        let prospective = idx.count + extraRows + (((pos + 1) % idx.ratio) == 0 ? 1 : 0)
        return prospective > Self.indexerSparseThreshold && prospective > d.indexerTopK
    }

    /// Encode ONE token's full route (pre + attention) into `c` WITHOUT
    /// committing — the batched phase A packs many tokens per command buffer.
    /// Caller guarantees the indexer is NOT active for (i, pos) and route
    /// profiling is off (both need CPU work mid-route). Same two encodes, same
    /// order as the per-token non-indexer path in encodeRoute.
    private func encodeRouteInto(_ c: GraphContext, _ i: Int, w: LayerWeights, layerRope: RopeParams,
                                 curHc: GPUTensor, pos: Int, nKeys: Int, token: Int) throws {
        let hasIdxWeights = w.idxKv != nil && w.idxQB != nil && w.idxProj != nil
        let nComp = try c.decodeRoutePre(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope,
                                         rawCache: rawCaches[i], pos: pos, rmsEps: rmsEps, hcEps: hcEps,
                                         comp: compStates[i], idx: hasIdxWeights ? indexStates[i] : nil,
                                         indexerScoring: false)
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
    private func clearMaskIfDirty() {
        guard maskDirtyCount > 0 else { return }
        memset(scratch.mask.buffer.contents(), 0, maskDirtyCount * 2)
        maskDirtyCount = 0
    }

    /// Output head for one token's final HC state -> logits[vocab].
    private func outputHead(_ cur: GPUTensor) throws -> [Float] {
        let hcDim = d.nHC * d.nEmbd
        let t = Date()
        let oc = GraphContext(rt)
        try oc.begin()
        try oc.rmsNorm(cur, weight: nil, out: flat, rows: 1, n: hcDim, eps: rmsEps)
        try oc.matmulF16(weight: out.hcFn, x: flat, out: pre, inDim: hcDim, outDim: d.nHC)
        try oc.outputHCWeights(pre: pre, scaleScalar: out.hcScaleScalar, base: out.hcBase,
                               weights: owts, tmp: otmp, nHC: d.nHC, eps: hcEps)
        try oc.hcWeightedSum(x: cur, weights: owts, out: oembd, nEmbd: d.nEmbd, nHC: d.nHC, nTokens: 1)
        try oc.rmsNorm(oembd, weight: out.norm, out: onormed, rows: 1, n: d.nEmbd, eps: rmsEps)
        try oc.matmulQ8_0(weight: out.head, x: onormed, out: logits, inDim: d.nEmbd, outDim: d.vocab)
        oc.commit()
        profile.headS += Date().timeIntervalSince(t)
        return logits.floatArray(d.vocab)
    }

    /// Convenience: streaming generate (same loop as DSV4Decoder.generate).
    public func generate(prompt: [Int], maxNew: Int, sampling: DSV4Decoder.Sampling = .init(), eos: Int? = nil) throws -> [Int] {
        precondition(!prompt.isEmpty)
        var rng = sampling.seed
        var pos = 0
        var last: [Float] = []
        for tok in prompt { last = try forward(token: tok, pos: pos, nKeys: pos + 1); pos += 1 }
        var gen: [Int] = []
        for _ in 0..<maxNew {
            let next = Sampler.sample(last, temperature: sampling.temperature, topK: sampling.topK,
                                      topP: sampling.topP, minP: sampling.minP, rng: &rng)
            if let e = eos, next == e { break }
            gen.append(next)
            last = try forward(token: next, pos: pos, nKeys: pos + 1); pos += 1
        }
        return gen
    }

    /// Build a streaming decoder backed by a real GGUF model (the real Stage D
    /// path): each layer is loaded from the mmap on demand.
    public static func fromGGUF(rt: MetalRuntime, model: GGUFModel, dims: DSV4Dims, rope: RopeParams,
                                nLayers: Int, maxKeys: Int, rmsEps: Float = ModelDefaults.rmsEps, hcEps: Float = ModelDefaults.hcEps) throws -> StreamingDecoder {
        let (embed, head) = try GGUFWeights.outputHead(rt, model)
        return try StreamingDecoder(rt: rt, dims: dims, rope: rope, nLayers: nLayers,
                                    layerProvider: { try GGUFWeights.layer(rt, model, $0) },
                                    embedTable: embed, out: head, maxKeys: maxKeys, rmsEps: rmsEps, hcEps: hcEps)
    }

    /// Expert-cache streaming decoder: per layer, only the dense weights are
    /// loaded up front; after routing, ONLY the 6 selected experts are gathered
    /// from the mmap (6/256 ~= 40x less expert IO/RAM). Numerically identical to
    /// the resident path (validated by ExpertCacheLayerTests).
    public static func fromGGUFExpertCached(rt: MetalRuntime, model: GGUFModel, dims: DSV4Dims, rope: RopeParams,
                                            nLayers: Int, maxKeys: Int, rmsEps: Float = ModelDefaults.rmsEps, hcEps: Float = ModelDefaults.hcEps) throws -> StreamingDecoder {
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
                                    expertGather: gather)
    }

    /// Fastest 16GB path (the C `--ssd-streaming` model): non-routed weights are
    /// NO-COPY mmap views (resident via the OS page cache, single copy, evictable —
    /// no per-token re-copy, no 8GB of dirty buffers that OOM), and only the 6 selected
    /// experts are gathered per token. No memoization needed: the page cache serves
    /// repeated weight reads across tokens. Requires model opened metalMapping:true.
    public static func fromGGUFExpertCachedMapped(rt: MetalRuntime, model: GGUFModel, dims: DSV4Dims, rope: RopeParams,
                                                  nLayers: Int, maxKeys: Int, rmsEps: Float = ModelDefaults.rmsEps, hcEps: Float = ModelDefaults.hcEps,
                                                  cacheSlots: Int? = nil, kvLayers: Range<Int>? = nil) throws -> StreamingDecoder {
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
        let usage = ExpertUsageStats(nLayers: nLayers)
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
            cache = ExpertSlotCache(slotsPerLayer: S, bytesPerExpert: recordBytes, makePool: makePool,
                                    fill: { il, id, pool, slot in
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
            }, prefetch: fillPrefetch,
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
        let denseNames = ["hc_attn_fn.weight", "attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight",
                          "attn_output_a.weight", "attn_output_b.weight", "hc_ffn_fn.weight",
                          "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
                          "ffn_gate_inp.weight", "indexer.attn_q_b.weight", "indexer.proj.weight",
                          "indexer_compressor_kv.weight", "indexer_compressor_gate.weight",
                          "attn_compressor_kv.weight", "attn_compressor_gate.weight"]
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
        let denseStream = ProcessInfo.processInfo.environment["DS4_DENSE_STREAM"] == "1"
        let denseProvider: (Int) throws -> LayerWeights
        if denseStream {
            // DS4_DENSE_Q4=1 (requires the stream): the two giant plain-matvec
            // projections (q_b, output_b — Q8, 71 of ~145 MB/layer) are
            // requantized to Q4_K at load and kept RESIDENT: half their bytes,
            // read at RAM speed, and ~3 GB/token OFF the SSD stream. LOSSY on
            // those two tensors (Q8→Q4 requant) — opt-in, A/B the output.
            let q4Dense = ProcessInfo.processInfo.environment["DS4_DENSE_Q4"] == "1"
            let streamer = try DenseStreamer(rt: rt, model: model, layers: kvLayers ?? 0..<nLayers,
                                             lockResident: lockResident, q4Dense: q4Dense)
            denseProvider = { try streamer.weights($0) }
        } else if residentDense {
            let denseCache = CachedLayerProvider { try GGUFWeights.layer(rt, model, $0, loadExperts: false) }
            denseProvider = { try denseCache.get($0) }
        } else {
            denseProvider = { try GGUFWeights.layerMappedDense(rt, model, $0) }
        }
        LoadProgress.shared.set(0.95, "Allocazione KV e scratch…")
        let dec = try StreamingDecoder(rt: rt, dims: dims, rope: rope, nLayers: nLayers,
                                       layerProvider: denseProvider,
                                       embedTable: embed, out: head, maxKeys: maxKeys, rmsEps: rmsEps, hcEps: hcEps,
                                       expertGather: gather, slotCache: cache, usage: usage,
                                       prefetch: prefetch, lookahead: lookahead, kvLayers: kvLayers,
                                       slotCacheStride: slotStride)
        LoadProgress.shared.set(1.0, "Pronto")
        return dec
    }

    /// Mapped-experts streaming decoder: per layer the dense weights are copied,
    /// but the routed experts are NO-COPY mmap views over the FULL expert tensors
    /// (all 256). The single-cb decode path runs mul_mv_id with the real selected
    /// ids; the OS page cache caches touched experts across tokens — no per-token
    /// re-gather. Requires model opened with metalMapping:true.
    public static func fromGGUFMappedExperts(rt: MetalRuntime, model: GGUFModel, dims: DSV4Dims, rope: RopeParams,
                                             nLayers: Int, maxKeys: Int, rmsEps: Float = ModelDefaults.rmsEps, hcEps: Float = ModelDefaults.hcEps) throws -> StreamingDecoder {
        let (embed, head) = try GGUFWeights.outputHead(rt, model)
        // Memoize per-layer weights: dense (incl. NSA compressor) are COPIED resident
        // and reused across tokens; experts are no-copy mmap. Without this the ~8GB of
        // non-routed weights were re-copied from the mmap EVERY token (minutes/token on
        // 16GB). This is the C `--ssd-streaming` model: non-routed resident, experts paged.
        let cache = CachedLayerProvider { try GGUFWeights.layerMappedExperts(rt, model, $0) }
        return try StreamingDecoder(rt: rt, dims: dims, rope: rope, nLayers: nLayers,
                                    layerProvider: { try cache.get($0) },
                                    embedTable: embed, out: head, maxKeys: maxKeys, rmsEps: rmsEps, hcEps: hcEps,
                                    expertGather: nil)   // single-cb decodeLayer with real ids
    }
}

/// Loads each layer's weights once and reuses them across tokens (weights are
/// read-only during decode). Keeps non-routed weights resident instead of
/// re-streaming them from the mmap every token.
final class CachedLayerProvider {
    private let make: (Int) throws -> LayerWeights
    private var cache: [Int: LayerWeights] = [:]
    init(_ make: @escaping (Int) throws -> LayerWeights) { self.make = make }
    func get(_ il: Int) throws -> LayerWeights {
        if let w = cache[il] { return w }
        let w = try make(il); cache[il] = w; return w
    }
}

/// Background expert-gather for the batched prefill pipeline: runs one union's
/// gather on a background queue so the SSD I/O of group g+1 overlaps the GPU
/// FFNs of group g. @unchecked Sendable: the gather closure only reads the
/// read-only GGUF mmap and creates FRESH Metal buffers (MTLDevice is
/// thread-safe); the result is handed back through the semaphore, which gives
/// the consumer a happens-before edge on everything the worker wrote.
private final class PrefillGather: @unchecked Sendable {
    typealias Tensors = (GPUTensor, GPUTensor, GPUTensor)
    private let layer: Int
    private let gather: (Int, [Int32]) throws -> Tensors

    init(layer: Int, gather: @escaping (Int, [Int32]) throws -> Tensors) {
        self.layer = layer
        self.gather = gather
    }

    final class Pending: @unchecked Sendable {
        fileprivate var result: Result<Tensors, Error>?
        fileprivate let sem = DispatchSemaphore(value: 0)
        /// Block until the gather completes; returns the tensors or rethrows.
        /// Consume ONCE: call either wait() or join(), never both.
        func wait() throws -> Tensors {
            sem.wait()
            return try result!.get()
        }
        /// Block until the gather completes, discarding the outcome (the
        /// error/cancellation path — the pipeline must never leave a worker
        /// touching the mmap after the caller unwinds).
        func join() { sem.wait() }
    }

    func start(_ union: [Int32]) -> Pending {
        let p = Pending()
        DispatchQueue.global(qos: .userInitiated).async {
            p.result = Result { try self.gather(self.layer, union) }
            p.sem.signal()
        }
        return p
    }
}
