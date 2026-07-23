import Foundation
import Metal
import DS4Core

extension StreamingDecoder {
    /// Reusable per-token staging for the batched prefill: ONE set per chunk,
    /// rewritten at every layer (layer i's phase B completes before layer i+1's
    /// phase A touches them) — instead of 3·n fresh Metal buffers per LAYER
    /// (43 × 512 × 3 ≈ 66k allocations per chunk).
    struct PrefillStage {
        let cur: [GPUTensor]     // n × nEmbd        (attn-normed FFN input)
        let attn: [GPUTensor]    // n × nHC·nEmbd    (post-attention residual)
        let split: [GPUTensor]   // n × 24           (HC split)
        let ids: [GPUTensor]     // n × k Int32      (remapped ids, padded to k)
        let rw: [GPUTensor]      // n × k Float      (route weights, 0-padded)
        /// Whole-slab views of attn/split (the per-token entries above are row
        /// views into these): the batched phase-B tail reads residuals and HC
        /// splits as matrices (hcExpand4 with nTokens = run length).
        let attnSlab: GPUTensor
        let splitSlab: GPUTensor
        let curSlab: GPUTensor
        /// LEVA 8: router selections/weights stay GPU-resident — these slabs
        /// have EXACTLY the [n × k] token-major layout mul_mm_id's map0 and
        /// pair kernels consume, so the full-layer phase B binds them directly
        /// (no CPU readback, no restaging).
        let idsSlab: GPUTensor
        let rwSlab: GPUTensor
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
            let routedMat: GPUTensor // n × nEmbd f32 (batched sum6 -> +shared, in place)
        }
        let mm: MMBuffers?
        /// Buffers for the batched multi-query prefill attention
        /// (DS4_PREFILL_BATCH_ATTN): ONE FlashAttention dispatch per run of
        /// route-batch tokens instead of one vec dispatch per token. Sized once
        /// per chunk for the route-batch capacity and the chunk-end KV span.
        struct FlashBatch {
            let nq: Int          // run capacity (route batch)
            let maxKv: Int       // raw-span + comp capacity, pad margin included
            let qMat: GPUTensor  // nq × nHead·512 f32 (row-major query rows)
            let heads: GPUTensor // nq × nHead·512 f32 (attention output rows)
            let mask: GPUTensor  // nq × maxKv f16 (CPU-filled per run)
            /// Second mask for the ASYNC phase-A pipeline: run r+1's CPU mask
            /// fill must not race run r's in-flight GPU reads — runs alternate
            /// between the two buffers by parity. Everything else in the batch
            /// is GPU-written, so the in-order queue serializes it.
            let maskB: GPUTensor
            let kvF16: GPUTensor // maxKv × 512 f16 (staged raw span + comp rows)
            let pad: GPUTensor   // final partial 64-block K/V/mask padding
            let blk: GPUTensor   // mask block map (skip fully-masked tiles)
            let splitA: GPUTensor    // nq × 24 f32 slab (attention HC split)
            let split: [GPUTensor]   // row views into splitA (per-token use)
            /// Dense-GEMM staging (DS4_PREFILL_DENSE_MM): token-major activation
            /// matrices so every dense projection of the route reads its weights
            /// ONCE per run instead of once per token.
            let hcMat: GPUTensor      // nq × nHC·nEmbd (packed input HC states)
            let flatMat: GPUTensor    // nq × nHC·nEmbd (HC rms-norm scratch)
            let mixMat: GPUTensor     // nq × 24 (HC mixer output)
            let embdMat: GPUTensor    // nq × nEmbd (HC collapse scratch)
            let curMat: GPUTensor     // nq × nEmbd (attn-normed rows)
            let qrMat: GPUTensor      // nq × qRank
            let qrNormMat: GPUTensor  // nq × qRank
            let kvMat: GPUTensor      // nq × headDim (latent rows pre-store)
            let lowMat: GPUTensor     // nq × attnLowDim (grouped low-rank out)
            let blockOutMat: GPUTensor // nq × nEmbd (output projection)
            let afterAttnMat: GPUTensor // nq × nHC·nEmbd (post-attn residual)
            let splitF: GPUTensor     // nq × 24 slab (pre-FFN HC split)
            let curMat2: GPUTensor    // nq × nEmbd (FFN input rows)
            let logitsMat: GPUTensor  // nq × nExperts (router logits)
            /// Batched NSA compressor projections: kv/score rows for the whole
            /// run (attention compressor: width ≤ 2·headDim; indexer
            /// compressor: width = 2·nIndexerHeadDim). The recurrent state
            /// update reads each token's row.
            let compKvMat: GPUTensor  // nq × 2·headDim
            let compScMat: GPUTensor  // nq × 2·headDim
            let idxKvMat: GPUTensor   // nq × 2·nIndexerHeadDim
            let idxScMat: GPUTensor   // nq × 2·nIndexerHeadDim
            /// LEVA 7 — batched compressor pool: position-ordered combined
            /// window (state head + run projections, APE'd scores) and the
            /// pooled emission rows. Sized for the widest case (ratio-128
            /// head of 127 rows, attention-compressor width 2·headDim).
            let compCombKv: GPUTensor  // (128 + nq) × 2·headDim
            let compCombSc: GPUTensor  // (128 + nq) × 2·headDim
            let compPooled: GPUTensor  // (nq/4 + 2) × headDim
            /// LEVA 9 v2 — run indexer-attivi (DS4_INDEXED_ATTN): query e pesi
            /// dell'indexer per il run (GEMM), righe di score per query e
            /// indici top-K per query (grezzi dallo heap + ordinati per id).
            let idxQMat: GPUTensor      // nq × nIndexerHead·nIndexerHeadDim
            let idxWMat: GPUTensor      // nq × nIndexerHead
            let idxScoresMat: GPUTensor // nq × maxKv f32 (score per query)
            let idxTopKMat: GPUTensor   // nq × indexerTopK int32
            let idxTopKSortMat: GPUTensor // nq × indexerTopK int32 (id crescenti)
            let idxSortScratch: GPUTensor // 2 × nq × maxKv int32 (ping-pong argsort)

            init(_ rt: MetalRuntime, nq: Int, maxKv: Int, d: DSV4Dims) throws {
                let sb = GraphContext.flashPrefillScratchBytes(nHead: d.nHead, nQ: nq, maxKv: maxKv)
                self.nq = nq
                self.maxKv = maxKv
                qMat = try .zerosBytes(rt, byteLength: sb.q)
                heads = try .zerosBytes(rt, byteLength: sb.heads)
                mask = try .zerosBytes(rt, byteLength: sb.mask)
                maskB = try .zerosBytes(rt, byteLength: sb.mask)
                kvF16 = try .zerosBytes(rt, byteLength: sb.kvF16)
                pad = try .zerosBytes(rt, byteLength: sb.pad)
                blk = try .zerosBytes(rt, byteLength: sb.blk)
                let splitSlab = try GPUTensor.zeros(rt, floatCount: nq * 24)
                splitA = splitSlab
                split = (0..<nq).map {
                    splitSlab.subview(byteOffset: $0 * 24 * 4, byteLength: 24 * 4, count: 24)
                }
                let hcDim = d.nHC * d.nEmbd
                hcMat = try .zeros(rt, floatCount: nq * hcDim)
                flatMat = try .zeros(rt, floatCount: nq * hcDim)
                mixMat = try .zeros(rt, floatCount: nq * 24)
                embdMat = try .zeros(rt, floatCount: nq * d.nEmbd)
                curMat = try .zeros(rt, floatCount: nq * d.nEmbd)
                qrMat = try .zeros(rt, floatCount: nq * d.qRank)
                qrNormMat = try .zeros(rt, floatCount: nq * d.qRank)
                kvMat = try .zeros(rt, floatCount: nq * d.headDim)
                lowMat = try .zeros(rt, floatCount: nq * d.attnLowDim)
                blockOutMat = try .zeros(rt, floatCount: nq * d.nEmbd)
                afterAttnMat = try .zeros(rt, floatCount: nq * hcDim)
                splitF = try .zeros(rt, floatCount: nq * 24)
                curMat2 = try .zeros(rt, floatCount: nq * d.nEmbd)
                logitsMat = try .zeros(rt, floatCount: nq * d.nExperts)
                compKvMat = try .zeros(rt, floatCount: nq * 2 * d.headDim)
                compScMat = try .zeros(rt, floatCount: nq * 2 * d.headDim)
                idxKvMat = try .zeros(rt, floatCount: nq * 2 * d.nIndexerHeadDim)
                idxScMat = try .zeros(rt, floatCount: nq * 2 * d.nIndexerHeadDim)
                compCombKv = try .zeros(rt, floatCount: (128 + nq) * 2 * d.headDim)
                compCombSc = try .zeros(rt, floatCount: (128 + nq) * 2 * d.headDim)
                compPooled = try .zeros(rt, floatCount: (nq / 4 + 2) * d.headDim)
                idxQMat = try .zeros(rt, floatCount: nq * d.nIndexerHead * d.nIndexerHeadDim)
                idxWMat = try .zeros(rt, floatCount: nq * d.nIndexerHead)
                idxScoresMat = try .zeros(rt, floatCount: nq * maxKv)
                idxTopKMat = try .zerosBytes(rt, byteLength: nq * max(1, d.indexerTopK) * 4)
                idxTopKSortMat = try .zerosBytes(rt, byteLength: nq * max(1, d.indexerTopK) * 4)
                idxSortScratch = try .zerosBytes(rt, byteLength: 2 * nq * maxKv * 4)
            }
        }
        let flash: FlashBatch?

        /// Allocate one zeroed Metal slab and expose `n` fixed-size logical
        /// rows as GPUTensor views. Before this helper PrefillStage allocated
        /// five MTLBuffers per prompt token (2,560 buffers at chunk=512), which
        /// made buffer creation and Objective-C lifetime management measurable
        /// prefill work. Views keep the same hazard-tracked shared buffer while
        /// preserving every call site's existing GPUTensor API.
        private static func rowViews(_ rt: MetalRuntime, n: Int,
                                     rowBytes: Int, rowCount: Int) throws -> [GPUTensor] {
            try slabViews(rt, n: n, rowBytes: rowBytes, rowCount: rowCount).views
        }

        static func slabViews(_ rt: MetalRuntime, n: Int, rowBytes: Int,
                              rowCount: Int) throws -> (slab: GPUTensor, views: [GPUTensor]) {
            let slab = try GPUTensor.zerosBytes(rt, byteLength: n * rowBytes)
            let views = (0..<n).map {
                slab.subview(byteOffset: $0 * rowBytes, byteLength: rowBytes,
                             count: rowCount)
            }
            return (slab, views)
        }

        init(_ rt: MetalRuntime, n: Int, d: DSV4Dims, mmPath: Bool, maxUnion: Int,
             flashBatch: (nq: Int, maxKv: Int)? = nil) throws {
            flash = try flashBatch.flatMap { fb in
                fb.nq >= 2 ? try FlashBatch(rt, nq: fb.nq, maxKv: fb.maxKv, d: d) : nil
            }
            let curPair = try Self.slabViews(rt, n: n, rowBytes: d.nEmbd * 4,
                                             rowCount: d.nEmbd)
            curSlab = curPair.slab; cur = curPair.views
            let attnPair = try Self.slabViews(rt, n: n, rowBytes: d.nHC * d.nEmbd * 4,
                                              rowCount: d.nHC * d.nEmbd)
            attnSlab = attnPair.slab; attn = attnPair.views
            let splitPair = try Self.slabViews(rt, n: n, rowBytes: 24 * 4, rowCount: 24)
            splitSlab = splitPair.slab; split = splitPair.views
            let idsPair = try Self.slabViews(rt, n: n, rowBytes: d.k * 4, rowCount: d.k)
            idsSlab = idsPair.slab; ids = idsPair.views
            let rwPair = try Self.slabViews(rt, n: n, rowBytes: d.k * 4, rowCount: d.k)
            rwSlab = rwPair.slab; rw = rwPair.views
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
                    ones: onesBuf,
                    routedMat: try .zeros(rt, floatCount: n * d.nEmbd))
            } else {
                mm = nil
            }
        }
    }
}
