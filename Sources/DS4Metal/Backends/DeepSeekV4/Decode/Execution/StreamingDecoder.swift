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

public final class StreamingDecoder {
    let rt: MetalRuntime
    /// `var` SOLO per `setActiveExperts` (draft self-speculative): mutata
    /// esclusivamente tra un forward e l'altro, sul thread del decode.
    var d: DSV4Dims
    let rope: RopeParams
    /// Validated, instance-owned architecture geometry. Every public GGUF
    /// factory requires it; `nil` is possible only through the internal direct
    /// initializer used by small synthetic tests with non-production shapes.
    let geometry: DSV4RuntimeGeometry?
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
    /// Keep NSA indexer score selection on the GPU. `=0` restores the historical
    /// CPU heap/readback path for parity diagnostics.
    let gpuIndexerTopK = ProcessInfo.processInfo.environment["DS4_GPU_INDEXER_TOPK"] != "0"
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
    /// 32× while keeping each buffer's GPU run bounded. Read at each prefill
    /// layer so the in-app benchmark can tune it without reloading the model.
    /// 0/1 = off (parity);
    /// layers with the indexer ACTIVE always fall back to per-token (a CPU
    /// top-k sits between the two halves of their route).
    var prefillRouteBatch: Int {
        let v = ProcessInfo.processInfo.environment["DS4_PREFILL_ROUTE_BATCH"].flatMap(Int.init) ?? 32
        return max(1, v)
    }
    /// DS4_PREFILL_BATCH_ATTN (default ON): the batched prefill route phase
    /// runs ONE multi-query FlashAttention (the C prefill's nqptg=8 MMA
    /// kernel) per run of route-batch tokens, instead of one single-query vec
    /// dispatch per token. Visibility is identical to the per-token path
    /// (causal + SWA window + per-token comp rows, enforced by a per-query
    /// mask); only the accumulation order differs (simdgroup MMA blocks), so
    /// outputs are close but not bit-identical. `=0` restores the per-token
    /// attention for A/B parity. Runs that would overflow the raw ring
    /// (DS4_RAW_RING) fall back automatically.
    let prefillBatchAttn = ProcessInfo.processInfo.environment["DS4_PREFILL_BATCH_ATTN"] != "0"
    /// DS4_PROFILE_PREFILL=1: time the BATCHED prefill run's sub-phases
    /// (dense path only) with commit boundaries, accumulated under the same
    /// route-split labels the decode profiler prints (comp/q/kv/attn/out-proj/
    /// hc-ffn/router). Forces synchronous commits — absolute numbers inflate,
    /// read the RATIOS (same caveat as DS4_PROFILE_ROUTE).
    let profilePrefill = ProcessInfo.processInfo.environment["DS4_PROFILE_PREFILL"] == "1"
    /// DS4_PREFILL_DENSE_MM (default ON, requires the batched-attention path):
    /// inside a batched route run, EVERY dense projection (q_a, q_b, kv, the
    /// grouped low-rank output, output_b, router) and both HC reduces run as
    /// matrix-matrix kernels over the whole run — weights read once per run
    /// instead of once per token (the C engine's layer-major prefill shape).
    /// Per-token work shrinks to the recurrent compressor, the KV row store
    /// and the router finalize. Same math, MMA accumulation order — outputs
    /// close but not bit-identical; `=0` restores per-token dense projections
    /// for A/B. Layers with Q4-requantized dense weights (DS4_DENSE_Q4 /
    /// DS4_QKV_Q4) fall back automatically.
    let prefillDenseMM = ProcessInfo.processInfo.environment["DS4_PREFILL_DENSE_MM"] != "0"
    /// DS4_PREFILL_FULL_LAYER (default ON): when the chunk has at least
    /// DS4_PREFILL_FULL_LAYER_MIN tokens (512), prefill phase B streams the
    /// WHOLE routed layer once (all experts, global ids — no unions, no
    /// remap) and double-buffers the NEXT layer's slab during this layer's
    /// FFNs — the C engine's prefill shape (its 3.38 GiB "prefill expert
    /// reserve"). Bytes/token = layerBytes/chunkTokens (26 MB/token at 2.7k
    /// measured on ds4) vs the union path's ~155 MB/token; sequential reads
    /// instead of per-union re-reads. Below the threshold the union path is
    /// cheaper (reads only the selected experts) and remains in charge.
    /// `=0` restores the union grouping everywhere.
    let prefillFullLayer = ProcessInfo.processInfo.environment["DS4_PREFILL_FULL_LAYER"] != "0"
    /// DS4_PREFILL_RESIDENT_IDS (default ON, leva 8): in full-layer phase B
    /// the router selections/weights are consumed straight from the GPU
    /// slabs (no CPU readback, no restaging, no snapshot blits); usage stats
    /// are recorded from the slab after the commit. `=0` restores the CPU
    /// round-trip for A/B (bit-identical expected: same kernels, same bytes).
    let prefillResidentIds = ProcessInfo.processInfo.environment["DS4_PREFILL_RESIDENT_IDS"] != "0"
    /// Minimum chunk tokens for the full-layer path (crossover ≈ 500 on the
    /// 2-bit Flash: below it the union of selected experts is fewer bytes).
    let prefillFullLayerMin = max(8, ProcessInfo.processInfo.environment["DS4_PREFILL_FULL_LAYER_MIN"].flatMap(Int.init) ?? 512)
    /// In-flight NEXT-layer full-slab gather (full-layer prefill pipeline):
    /// kicked when layer i's phase B starts computing, consumed by layer
    /// i+1's phase B. Never crosses a chunk (the last layer kicks nothing);
    /// error paths must drainFullLayerGather() before unwinding.
    var fullLayerPending: (layer: Int, pending: PrefillGather.Pending)?

    /// Join the in-flight full-layer slab gather (error/teardown paths).
    func drainFullLayerGather() {
        fullLayerPending?.pending.join()
        fullLayerPending = nil
    }

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
    /// Exact per-step resident allocation of the dense staging ring. Zero when
    /// dense streaming is disabled.
    public let denseStagingBytesPerAheadSlot: Int
    /// True when at least one routed layer of this load can fill experts from
    /// the GGUF F_NOCACHE descriptor, where DS4_PREAD_SPLIT is effective.
    public let usesDirectExpertPread: Bool
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
    /// One-shot materializer for the NSA indexer scoring projections. With
    /// DS4_LAZY_IDX the dense streamer leaves these weights out of every
    /// per-token read and installs resident copies only when the LIVE context
    /// first reaches the sparse-indexer frontier. Nil on non-streaming paths,
    /// when the configured context can never reach the frontier, and after a
    /// successful activation.
    var activateIndexerScoring: (() throws -> Void)?
    /// Factory-owned queues/read-ahead objects that can outlive the last
    /// synchronous forward unless explicitly joined before an in-process
    /// benchmark reload.
    let teardownIODrain: (() -> Void)?
    /// Serial queue for the speculative prefills (one layer ahead at a time —
    /// a backlog would just re-touch already-resident ids and skip).
    let lookaheadQ = DispatchQueue(label: "ds4.expert-lookahead", qos: .userInitiated)

    /// Scratch capacity is independent from the configured KV capacity.  A
    /// large context must not eagerly commit its raw+compressed F16 attention
    /// staging on the first token: grow only when live rows require it.
    var scratch: DecodeScratch
    let scratchMaximumRows: Int
    let rawCaches: [GPUTensor]
    let compStates: [CompressorState?]   // NSA compressor state per compressed layer (nil on layers 0,1)
    /// NSA indexer compressor state (DSA): ratio-4 layers only. Beyond
    /// `d.indexerTopK` compressed rows, attention is restricted to the top-K
    /// most relevant for the current query (C: indexer_allowed_decode_one).
    let indexStates: [CompressorState?]
    /// Halves of s.mask dirtied by the last indexer selection (0 = clean).
    var maskDirtyCount = 0
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
    /// EXPERT PARALLELISM (coordinatore verticale): quando impostata, la FFN
    /// routed è calcolata dai worker remoti — (layer, id selezionati, pesi di
    /// route, attivazione nEmbd) → somma pesata nEmbd. Chiamata SINCRONA dal
    /// decode thread (la latenza di rete è coperta dalla FFN condivisa
    /// asincrona, come lo era il gather SSD). nil = percorso locale.
    public var remoteExperts: (@Sendable (Int, [Int32], [Float], [Float]) throws -> [Float])?
    /// Tensori di consegna della somma remota, a PARITÀ di layer alternata:
    /// il c2 asincrono del layer precedente può ancora leggere il suo mentre
    /// la CPU scrive quello del layer corrente (stesso schema di slotsScratch).
    let remotePartialA: GPUTensor
    let remotePartialB: GPUTensor
    /// The last layer's routed-FFN command buffer, committed WITHOUT a CPU wait
    /// (DS4_ASYNC_FFN, default ON): the next layer's route commit+wait lands on
    /// the same in-order queue, so the GPU stays fed while the CPU encodes —
    /// the per-layer bubble (encode time x 43) disappears. Explicitly waited at
    /// end of token (before the output head / readHC) and on every error path.
    var inflightFFN: GraphContext?
    let asyncFFN = ProcessInfo.processInfo.environment["DS4_ASYNC_FFN"] != "0"
    /// DS4_ASYNC_ROUTE (default ON): the decode route commits WITHOUT a CPU
    /// wait and the shared FFN is committed right behind it, so the GPU chains
    /// route→sharedFFN with no encode gap while the CPU waits for the
    /// selection; the CPU join on the shared FFN before encoding the routed
    /// FFN is also skipped (in-order queue + hazard tracking give the same
    /// ordering — the DS4_ASYNC_FFN argument). `=0` restores the historical
    /// fully-synchronous route path for A/B and debugging. Token-identical
    /// either way; DS4_PROFILE_ROUTE forces the synchronous path regardless
    /// (accurate per-phase attribution).
    let asyncRoute = ProcessInfo.processInfo.environment["DS4_ASYNC_ROUTE"] != "0"
    /// DS4_SPEC_VERIFY_BATCH (default ON): la verifica speculativa incoda la
    /// route/attention dell'INTERA finestra in un solo command buffer per layer
    /// (fase A del prefill batchato) e serve le FFN routed dalla slot-cache,
    /// invece del giro per-token completo. Stessi dispatch, stesso ordine per
    /// token: numerica identica. `=0` ripristina il percorso per-token storico
    /// di specVerifyStep per A/B.
    let specVerifyBatch = ProcessInfo.processInfo.environment["DS4_SPEC_VERIFY_BATCH"] != "0"
    /// One embedding-table ROW (F16, nEmbd × 2 B), CPU-staged per token.
    /// Binding the full multi-hundred-MB no-copy table to a command buffer
    /// makes Metal wire the WHOLE mapping every token — on tight-RAM machines
    /// where the expert churn evicts it, that re-faults it from SSD each time
    /// (the ~500 ms "embed" phase). Staging the row costs an ~8 KB memcpy and
    /// keeps the table out of the GPU residency set. Identical bytes → same
    /// numerics.
    let embedRowStage: GPUTensor

    /// Live attention rows for the densest possible layer: the raw NSA span is
    /// bounded by SWA, while a ratio-4 compressor can contribute floor(n/4)
    /// rows.  Eight rows match the compressor-state allocation headroom and
    /// cover the prospective emission at a chunk/token boundary.
    ///
    /// Internal and pure on purpose: DS4Metal tests exercise context boundary
    /// arithmetic without requiring a Metal device.
    static func scratchRowsNeeded(nKeys: Int, nSWA: Int) -> Int {
        let used = max(0, nKeys)
        let raw = min(used, max(0, nSWA))
        return max(1, raw + used / 4 + 8)
    }

    /// Geometric high-water growth with a hard session cap. A nearby boundary
    /// doubles the current allocation; a larger jump (for example one prefill
    /// chunk) goes straight to its required high-water mark instead of walking
    /// through intermediate allocations. Starting from at least one row avoids
    /// a zero-capacity fixed point; `cap / 2` keeps doubling overflow-safe.
    /// Internal and pure for `@testable` boundary tests.
    static func grownScratchCapacity(current: Int, required: Int, maximum: Int) -> Int {
        let cap = max(1, maximum)
        let target = min(max(1, required), cap)
        let capacity = min(max(1, current), cap)
        guard target > capacity else { return capacity }
        let doubled = capacity > cap / 2 ? cap : capacity * 2
        return min(cap, max(target, doubled))
    }

    /// Per-layer architecture data. Keeping these lookups on the decoder
    /// instance prevents a Pro session from falling back to Flash compression
    /// or RoPE constants while preserving source compatibility for synthetic
    /// unit-test decoders.
    func compressRatio(layer: Int) -> Int {
        geometry?.compressRatio(layer: layer) ?? DSV4Shape.compressRatio(layer: layer)
    }

    func ropeParams(layer: Int) -> RopeParams {
        geometry?.ropeParams(layer: layer) ?? DSV4Shape.ropeParams(layer: layer)
    }

    /// Prepare every resource whose cost follows the live context rather than
    /// the configured maximum. Scratch grows geometrically; indexer scoring is
    /// materialized once at its real activation frontier. Replacement/loading
    /// happens only after joining an async routed FFN that may retain buffers.
    func prepareLiveContext(nKeys: Int) throws {
        precondition(nKeys >= 0 && nKeys <= maxKeys,
                     "live context: nKeys \(nKeys) fuori da 0...\(maxKeys)")
        let required = Self.scratchRowsNeeded(nKeys: nKeys, nSWA: d.nSWA)
        if required > scratch.attentionRows {
            let next = Self.grownScratchCapacity(current: scratch.attentionRows,
                                                 required: required,
                                                 maximum: scratchMaximumRows)
            precondition(next >= required, "attention scratch: capacity cap insufficiente")
            drainFFN()
            let replacement = try DecodeScratch(rt, d, maxKeys: next)
            scratch = replacement
            // A dirty count belongs to the discarded mask; the replacement
            // starts zeroed and must not use offsets from the old capacity.
            maskDirtyCount = 0
        }
        if let activateIndexerScoring,
           Self.indexerScoringNeeded(liveKeys: nKeys, topK: d.indexerTopK,
                                     sparseThreshold: Self.indexerSparseThreshold) {
            drainFFN()
            try activateIndexerScoring()
            self.activateIndexerScoring = nil
        }
    }

    /// Pure architecture contract used by the GGUF factories before any large
    /// tensor is loaded. Keep every shape-defining `DSV4Dims` field here: a
    /// partial check can let a Pro model silently inherit a Flash allocation or
    /// router constant even when the headline dimensions happen to match.
    static func runtimeGeometryMismatches(_ geometry: DSV4RuntimeGeometry,
                                          dims: DSV4Dims,
                                          nLayers: Int) -> [String] {
        let expected = geometry.dims
        var fields: [String] = []
        if geometry.nLayers != nLayers { fields.append("nLayers") }
        if expected.nEmbd != dims.nEmbd { fields.append("nEmbd") }
        if expected.nHC != dims.nHC { fields.append("nHC") }
        if expected.headDim != dims.headDim { fields.append("headDim") }
        if expected.nHead != dims.nHead { fields.append("nHead") }
        if expected.qRank != dims.qRank { fields.append("qRank") }
        if expected.qDim != dims.qDim { fields.append("qDim") }
        if expected.sharedFfn != dims.sharedFfn { fields.append("sharedFfn") }
        if expected.nExperts != dims.nExperts { fields.append("nExperts") }
        if expected.expertFfn != dims.expertFfn { fields.append("expertFfn") }
        if expected.k != dims.k { fields.append("k") }
        if expected.nRot != dims.nRot { fields.append("nRot") }
        if expected.vocab != dims.vocab { fields.append("vocab") }
        if expected.nOutGroup != dims.nOutGroup { fields.append("nOutGroup") }
        if expected.nLoraO != dims.nLoraO { fields.append("nLoraO") }
        if expected.swigluClamp != dims.swigluClamp { fields.append("swigluClamp") }
        if expected.nSWA != dims.nSWA { fields.append("nSWA") }
        if expected.nIndexerHead != dims.nIndexerHead { fields.append("nIndexerHead") }
        if expected.nIndexerHeadDim != dims.nIndexerHeadDim { fields.append("nIndexerHeadDim") }
        if expected.indexerTopK != dims.indexerTopK { fields.append("indexerTopK") }
        if expected.expertWeightScale != dims.expertWeightScale { fields.append("expertWeightScale") }
        if expected.nHashLayers != dims.nHashLayers { fields.append("nHashLayers") }
        if expected.sinkhornIterations != dims.sinkhornIterations { fields.append("sinkhornIterations") }
        return fields
    }

    static func validateRuntimeGeometry(_ geometry: DSV4RuntimeGeometry,
                                        dims: DSV4Dims,
                                        nLayers: Int) throws {
        let mismatches = runtimeGeometryMismatches(geometry, dims: dims, nLayers: nLayers)
        guard mismatches.isEmpty else {
            throw MetalError.unsupported(
                "runtime geometry does not match decoder dimensions: "
                    + mismatches.joined(separator: ", "))
        }
    }

    /// Internal on purpose. Production callers must use a public GGUF factory,
    /// all of which require validated geometry. The optional geometry exists
    /// only so @testable synthetic decoders can exercise intentionally tiny,
    /// non-production dimensions.
    init(rt: MetalRuntime, dims: DSV4Dims, rope: RopeParams, nLayers: Int,
         layerProvider: @escaping (Int) throws -> LayerWeights,
         embedTable: GPUTensor, out: OutputHeadWeights, maxKeys: Int,
         rmsEps: Float = ModelDefaults.rmsEps, hcEps: Float = ModelDefaults.hcEps,
         expertGather: ((Int, [Int32]) throws -> (GPUTensor, GPUTensor, GPUTensor))? = nil,
         slotCache: ExpertSlotCache? = nil,
         usage: ExpertUsageStats? = nil,
         prefetch: ((Int) -> Void)? = nil,
         lookahead: ((_ layer: Int, _ token: Int) -> [Int32])? = nil,
         kvLayers: Range<Int>? = nil,
         activateIndexerScoring: (() throws -> Void)? = nil,
         teardownIODrain: (() -> Void)? = nil,
         denseStagingBytesPerAheadSlot: Int = 0,
         usesDirectExpertPread: Bool = false,
         slotCacheStride: Int? = nil,
         geometry: DSV4RuntimeGeometry? = nil) throws {
        if let geometry {
            try Self.validateRuntimeGeometry(geometry, dims: dims, nLayers: nLayers)
        }
        // DS4_Q8_NSG: riletto QUI (non a ogni dispatch) così un reload del
        // modello — es. l'auto-tune delle Settings — può fare lo sweep senza
        // riavviare il processo.
        GraphContext.refreshQ8NSG()
        self.rt = rt; self.d = dims; self.rope = rope
        self.geometry = geometry; self.nLayers = nLayers
        self.layerProvider = layerProvider; self.embedTable = embedTable; self.out = out
        self.rmsEps = rmsEps; self.hcEps = hcEps; self.expertGather = expertGather
        self.slotCache = slotCache
        self.denseStagingBytesPerAheadSlot = max(0, denseStagingBytesPerAheadSlot)
        self.usesDirectExpertPread = usesDirectExpertPread
        self.slotCacheStride = slotCacheStride
        self.usage = usage
        self.prefetch = prefetch
        self.lookahead = lookahead
        self.activateIndexerScoring = activateIndexerScoring
        self.teardownIODrain = teardownIODrain
        let hcDim = dims.nHC * dims.nEmbd
        // Distributed slice: allocate KV/compressor state ONLY for `kvLayers`
        // (a worker never runs the other layers — dummy 1-float buffers there).
        // nil = full model (the single-machine default).
        let kvRange = kvLayers ?? 0..<nLayers
        self.kvRange = kvRange
        self.maxKeys = maxKeys
        // NSA compressor state per compressed layer (ratio!=0). The states retain
        // their full lazy context capacity; only the transient attention scratch
        // below starts small and grows with the live high-water mark.
        compStates = try (0..<nLayers).map { il -> CompressorState? in
            guard kvRange.contains(il) else { return nil }
            let ratio = geometry?.compressRatio(layer: il) ?? DSV4Shape.compressRatio(layer: il)
            guard ratio != 0 else { return nil }
            return try CompressorState(rt, ratio: ratio, headDim: dims.headDim, maxComp: maxKeys / ratio + 8)
        }
        // NSA indexer compressor (DSA): ratio-4 layers only (head_dim 128).
        indexStates = try (0..<nLayers).map { il -> CompressorState? in
            let ratio = geometry?.compressRatio(layer: il) ?? DSV4Shape.compressRatio(layer: il)
            guard kvRange.contains(il), ratio == 4 else { return nil }
            return try CompressorState(rt, ratio: 4, headDim: dims.nIndexerHeadDim, maxComp: maxKeys / 4 + 8)
        }
        scratchMaximumRows = Self.scratchRowsNeeded(nKeys: maxKeys, nSWA: dims.nSWA)
        let initialKeys = maxKeys > 0 ? 1 : 0
        let initialScratchRows = Self.scratchRowsNeeded(nKeys: initialKeys, nSWA: dims.nSWA)
        scratch = try DecodeScratch(rt, dims, maxKeys: initialScratchRows)
        idsPacked = try GPUTensor.bytes(rt, Array(0..<Int32(dims.k)).withUnsafeBytes { Array($0) }, elementCount: dims.k)
        slotsScratch = try GPUTensor.zerosBytes(rt, byteLength: dims.k * 4)
        slotsScratchB = try GPUTensor.zerosBytes(rt, byteLength: dims.k * 4)
        remotePartialA = try GPUTensor.zeros(rt, floatCount: dims.nEmbd)
        remotePartialB = try GPUTensor.zeros(rt, floatCount: dims.nEmbd)
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
        // Compile the wrap-aware copy before generation reaches the first wrap;
        // otherwise lazy PSO creation would create a one-token latency spike at
        // position nSWA. The pipeline is cached by MetalRuntime afterwards.
        if ringOn { _ = try rt.pipeline("kernel_dsv4_raw_ring_cpy_f32_f16") }
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
        // The ring is tiny (~10.75 MiB Flash, ~15.25 MiB Pro) and touched on
        // every layer/token. Honor the existing best-effort mlock knob so macOS
        // does not compress these hottest KV pages between uses.
        if ringOn, ProcessInfo.processInfo.environment["DS4_MLOCK"] == "1" {
            for il in kvRange { _ = rawCaches[il].lockResident() }
        }
        hcA = try .zeros(rt, floatCount: hcDim); hcB = try .zeros(rt, floatCount: hcDim)
        embd = try .zeros(rt, floatCount: dims.nEmbd)
        flat = try .zeros(rt, floatCount: hcDim); pre = try .zeros(rt, floatCount: dims.nHC)
        owts = try .zeros(rt, floatCount: dims.nHC); otmp = try .zeros(rt, floatCount: dims.nHC)
        oembd = try .zeros(rt, floatCount: dims.nEmbd); onormed = try .zeros(rt, floatCount: dims.nEmbd)
        logits = try .zeros(rt, floatCount: dims.vocab)
    }
}
