import Foundation
import Metal

// Stage 3 of the compressed-attention port (docs/COMPRESSED-ATTENTION-PORT.md).
//
// Encode-form (GraphContext / GPUTensor) dispatchers for the KV-compression +
// sparse-indexer kernels. The existing MetalKVCompress/MetalIndexerPool/
// MetalSparseSelect wrappers are array-in/array-out *test* helpers (commit+wait
// per call); the decode graph needs encode-form ops that bind resident GPUTensors
// into the running command buffer. These mirror the C engine's
// ds4_gpu_compressor_update_tensor / ds4_gpu_indexer_score_one_tensor /
// ds4_gpu_attention_indexed_mixed_batch_heads_tensor (ds4_metal.m), with the exact
// arg structs from metal/dsv4_kv.metal + metal/dsv4_misc.metal.
//
// NOT BUILT/VERIFIED here (no Swift/Metal/model on Linux). Items marked TODO(verify)
// need tracing/checking on macOS against the C golden vectors. Where a composite
// nests further C encode helpers I could not fully capture, a `// TODO(port)` cites
// the ds4_metal.m line to finish on-device.

extension GPUTensor {
    /// A sub-view of `floatCount` F32 elements starting at float index `rowFloat`,
    /// sharing this tensor's buffer (no copy). Used to address one cache row.
    func rowView(_ rowFloat: Int, _ floatCount: Int) -> GPUTensor {
        GPUTensor(buffer: buffer, byteLength: floatCount * 4, count: floatCount,
                  byteOffset: byteOffset + rowFloat * 4)
    }
}

extension GraphContext {
    // MARK: - Compressor frontier store (kernel_dsv4_compressor_store_one)

    /// state_kv[dst]=kv[g]; state_score[dst]=score[g]+ape[(pos%ratio)*width+g]
    /// dst_row = ratio==4 ? ratio + pos%ratio : pos%ratio. ape is F16 (ape_type=1).
    public func compressorStoreOne(kv: GPUTensor, score: GPUTensor, ape: GPUTensor,
                                   stateKv: GPUTensor, stateScore: GPUTensor,
                                   width: Int, ratio: Int, pos: Int) throws {
        var args = [UInt8](repeating: 0, count: 16)
        func u32(_ o: Int, _ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { args[o+k] = $0[k] } } }
        u32(0, UInt32(width)); u32(4, UInt32(ratio)); u32(8, UInt32(pos)); u32(12, 1) // ape_type=1 (F16)
        let pso = try rt.pipeline("kernel_dsv4_compressor_store_one")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: 16, index: 0) }
        e.setBuffer(kv.buffer, offset: kv.byteOffset, index: 1)
        e.setBuffer(score.buffer, offset: score.byteOffset, index: 2)
        e.setBuffer(ape.buffer, offset: ape.byteOffset, index: 3)
        e.setBuffer(stateKv.buffer, offset: stateKv.byteOffset, index: 4)
        e.setBuffer(stateScore.buffer, offset: stateScore.byteOffset, index: 5)
        e.dispatchThreadgroups(MTLSize(width: (width + 255) / 256, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    // MARK: - Softmax pool (kernel_dsv4_softmax_pool)

    /// Pool `rows` contiguous state rows (each `width` floats) into one `width`-wide
    /// row via per-dimension softmax over scores. Used directly for ratio!=4
    /// (width==headDim). ratio==4's two-lane pool needs a concat first (see
    /// compressorPool). Mirrors MetalRuntime.softmaxPoolArgs.
    public func softmaxPoolContiguous(out: GPUTensor, stateKv: GPUTensor, stateScore: GPUTensor,
                                      rows: Int, width: Int) throws {
        let args = MetalRuntime.softmaxPoolArgs(nRows: rows, width: width)
        let pso = try rt.pipeline("kernel_dsv4_softmax_pool")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(stateKv.buffer, offset: stateKv.byteOffset, index: 1)
        e.setBuffer(stateScore.buffer, offset: stateScore.byteOffset, index: 2)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 3)
        e.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: min(256, max(1, width)), height: 1, depth: 1))
    }

    // MARK: - ratio-4 recurrent state lane shift (kernel_dsv4_ratio4_shift_f32)

    /// After an emit on a ratio==4 layer: state[0:4*width] <- state[4*width:8*width]
    /// (lane rotation). ds4.c encode_compressor_shift_ratio4.
    public func ratio4Shift(stateKv: GPUTensor, stateScore: GPUTensor, width: Int) throws {
        let n = 4 * width
        var w = UInt32(width)
        let pso = try rt.pipeline("kernel_dsv4_ratio4_shift_f32")
        let e = encoder
        e.setComputePipelineState(pso)
        e.setBytes(&w, length: 4, index: 0)
        e.setBuffer(stateKv.buffer, offset: stateKv.byteOffset, index: 1)
        e.setBuffer(stateScore.buffer, offset: stateScore.byteOffset, index: 2)
        e.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    // MARK: - emit-time quantization

    /// E4M3FN quantize the NoPE part (first headDim-nRot dims) of one cache row.
    /// kernel_dsv4_fp8_kv_quantize_f32. Used on attention compressed rows (headDim=512).
    public func fp8QuantizeRow(_ row: GPUTensor, headDim: Int, nRot: Int) throws {
        if nRot == headDim { return }
        let args = MetalRuntime.fp8KVQuantizeArgs(headDim: headDim, nTok: 1, nRot: nRot)
        let pso = try rt.pipeline("kernel_dsv4_fp8_kv_quantize_f32")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(row.buffer, offset: row.byteOffset, index: 1)
        e.setBuffer(row.buffer, offset: row.byteOffset, index: 2)
        e.setThreadgroupMemoryLength(64 * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    /// QAT (128-pt Hadamard + E2M1/FP4) on `nRows` indexer rows (128 wide).
    /// kernel_dsv4_indexer_hadamard_fp4_f32. Used on indexer compressed rows and the
    /// indexer query (per head).
    public func indexerQAT(_ rows: GPUTensor, nRows: Int) throws {
        let headDim = 128
        var args = [UInt8](repeating: 0, count: 16)
        func u32(_ o: Int, _ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { args[o+k] = $0[k] } } }
        func u64(_ o: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { args[o+k] = $0[k] } } }
        u32(0, UInt32(nRows)); u32(4, UInt32(headDim)); u64(8, UInt64(headDim * 4))
        let pso = try rt.pipeline("kernel_dsv4_indexer_hadamard_fp4_f32")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: 16, index: 0) }
        e.setBuffer(rows.buffer, offset: rows.byteOffset, index: 1)
        e.setThreadgroupMemoryLength(256 * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: nRows, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    // MARK: - indexer scoring (kernel_dsv4_indexer_score_one_direct, n_head=64,head_dim=128)

    /// scores[c] = Σ_h ReLU(dot(indexComp[c], q[h])) * weights[h] * scale, for c<nComp.
    /// Direct decode kernel. ds4_metal.m ds4_gpu_indexer_score_one_tensor.
    public func indexerScoreOneDirect(scores: GPUTensor, q: GPUTensor, weights: GPUTensor,
                                      indexComp: GPUTensor, nComp: Int, scale: Float) throws {
        let nHead = DSV4Shape.nIndexerHead, headDim = DSV4Shape.nIndexerHeadDim
        var args = [UInt8](repeating: 0, count: 72)
        func u32(_ o: Int, _ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { args[o+k] = $0[k] } } }
        func u64(_ o: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { args[o+k] = $0[k] } } }
        func f32(_ o: Int, _ v: Float) { withUnsafeBytes(of: v.bitPattern.littleEndian) { for k in 0..<4 { args[o+k] = $0[k] } } }
        u32(0, UInt32(nComp)); u32(4, 1); u32(8, UInt32(nHead)); u32(12, UInt32(headDim))
        u32(16, 0); u32(20, 4)                                    // pos0=0, ratio=4
        u64(24, UInt64(nHead * headDim * 4))                      // q_token_stride
        u64(32, UInt64(headDim * 4))                              // q_head_stride
        u64(40, UInt64(nHead * 4))                                // weights_token_stride
        u64(48, UInt64(headDim * 4))                              // index_row_stride
        u64(56, UInt64(nComp * 4))                                // score_token_stride
        f32(64, scale)
        let pso = try rt.pipeline("kernel_dsv4_indexer_score_one_direct")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: 72, index: 0) }
        e.setBuffer(q.buffer, offset: q.byteOffset, index: 1)
        e.setBuffer(weights.buffer, offset: weights.byteOffset, index: 2)
        e.setBuffer(indexComp.buffer, offset: indexComp.byteOffset, index: 3)
        e.setBuffer(scores.buffer, offset: scores.byteOffset, index: 4)
        e.setThreadgroupMemoryLength((128 + 4) * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: nComp, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
    }

    // MARK: - top-k selection

    /// Select the `topK` highest-scoring indexer rows into `selected` (Int32 indices).
    /// Single-pass bitonic argsort (valid only when nComp <= threadgroup limit,
    /// ~256/1024). For nComp beyond one threadgroup the C engine
    /// (ds4_gpu_indexer_topk_tensor @11431) runs multi-pass argsort + merge
    /// (kernel_argsort_merge_f32_i32_desc) — required for real top-512 over thousands
    /// of rows.
    /// TODO(port): wire the multi-pass merge for nComp > maxThreads. ds4_metal.m:11431.
    public func indexerTopKSinglePass(selected: GPUTensor, scores: GPUTensor,
                                      nComp: Int, topK: Int) throws {
        let nth = min(nComp.nextPow2, 1024)
        var args = [UInt8](repeating: 0, count: 56)   // ds4_gpu_kargs_argsort
        func i32(_ o: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { args[o+k] = $0[k] } } }
        func u64(_ o: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { args[o+k] = $0[k] } } }
        i32(0, Int32(nComp)); i32(4, 1); i32(8, 1); i32(12, 1)            // ne00..ne03
        u64(16, 4); u64(24, UInt64(nComp * 4)); u64(32, UInt64(nComp * 4)); u64(40, UInt64(nComp * 4)) // nb00..nb03
        i32(48, Int32(topK)); i32(52, 1)                                  // ne0(=top_k slots), ne1
        // NOTE: arg layout per ds4_gpu_kargs_argsort may differ (ne0..ne3 + top_k);
        // verify field order/offsets against the struct in ds4_metal.m. TODO(verify).
        let pso = try rt.pipeline("kernel_argsort_f32_i32_desc")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: 56, index: 0) }
        e.setBuffer(scores.buffer, offset: scores.byteOffset, index: 1)
        e.setBuffer(selected.buffer, offset: selected.byteOffset, index: 2)
        e.setThreadgroupMemoryLength(((nth * 4) + 15) & ~15, index: 0)
        e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }

    // MARK: - indexed mixed attention (kernel_dsv4_indexed_mixed_attention_heads8_rb16)

    /// Decode (1-token) sparse attention over the SWA raw window + the `topK`
    /// selected compressed rows. headDim must be 512. Sinks = attn_sinks GPUTensor
    /// (F32 [nHead]). ds4_metal.m ds4_gpu_attention_indexed_mixed_batch_heads_tensor.
    /// Fast decode skips the chronological topk sort (matches non-quality mode).
    public func indexedMixedAttentionDecode(heads: GPUTensor, sinks: GPUTensor, q: GPUTensor,
                                            rawKv: GPUTensor, compKv: GPUTensor, topk: GPUTensor,
                                            nHead: Int, nRaw: Int, rawCap: Int, rawStart: Int,
                                            nComp: Int, topK: Int, window: Int, ratio: Int,
                                            pos0: Int) throws {
        let headDim = 512
        let rowBytes = UInt64(headDim * 4)
        var args = [UInt8](repeating: 0, count: 112)
        func u32(_ o: Int, _ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { args[o+k] = $0[k] } } }
        func u64(_ o: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { args[o+k] = $0[k] } } }
        func f32(_ o: Int, _ v: Float) { withUnsafeBytes(of: v.bitPattern.littleEndian) { for k in 0..<4 { args[o+k] = $0[k] } } }
        u32(0, 1); u32(4, UInt32(nHead)); u32(8, UInt32(nRaw)); u32(12, UInt32(rawCap))
        u32(16, UInt32(rawStart)); u32(20, UInt32(nComp)); u32(24, UInt32(topK)); u32(28, UInt32(pos0))
        u32(32, UInt32(window)); u32(36, UInt32(ratio)); u32(40, 0); u32(44, 0)   // comp_kv_f16=0, pad0
        u64(48, UInt64(nHead) * rowBytes)   // q_token_stride
        u64(56, rowBytes)                   // q_head_stride
        u64(64, rowBytes)                   // raw_row_stride
        u64(72, rowBytes)                   // comp_row_stride (F32 cache)
        u64(80, UInt64(topK * 4))           // topk_token_stride
        u64(88, UInt64(nHead) * rowBytes)   // dst_token_stride
        u64(96, rowBytes)                   // dst_head_stride
        f32(104, 1.0 / Float(headDim).squareRoot())
        let pso = try rt.pipeline("kernel_dsv4_indexed_mixed_attention_heads8_rb16")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: 112, index: 0) }
        e.setBuffer(q.buffer, offset: q.byteOffset, index: 1)
        e.setBuffer(rawKv.buffer, offset: rawKv.byteOffset, index: 2)
        e.setBuffer(compKv.buffer, offset: compKv.byteOffset, index: 3)
        e.setBuffer(topk.buffer, offset: topk.byteOffset, index: 4)
        e.setBuffer(sinks.buffer, offset: sinks.byteOffset, index: 5)
        e.setBuffer(heads.buffer, offset: heads.byteOffset, index: 6)
        e.setThreadgroupMemoryLength(16 * 128 * 4 * 2, index: 0)   // rb16 staging (16 rows x 128 x 4 x f16)
        e.dispatchThreadgroups(MTLSize(width: 1, height: (nHead + 7) / 8, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
    }
}

private extension Int {
    /// Smallest power of two >= self (>=1).
    var nextPow2: Int {
        if self <= 1 { return 1 }
        var v = 1
        while v < self { v <<= 1 }
        return v
    }
}

// MARK: - High-level orchestration

extension GraphContext {
    /// One compressor lane update (attention lane: headDim=512, or indexer lane:
    /// headDim=128). Projects `x`=attn_norm, appends to the recurrent state, and on
    /// an emit boundary pools+norms+RoPEs one compressed row into `cache[compRow]`,
    /// applies `quantize`, and (ratio==4) rotates the lanes. Mirrors
    /// ds4_gpu_compressor_update_tensor (ds4_metal.m:14689).
    ///
    /// `quantize` is the emit-time round-trip: fp8QuantizeRow for the attention lane,
    /// indexerQAT for the indexer lane.
    public func compressorUpdate(x: GPUTensor, w: CompressorWeights, stateKv: GPUTensor,
                                 stateScore: GPUTensor, cache: GPUTensor, kvCur: GPUTensor,
                                 scCur: GPUTensor, nEmbd: Int, headDim: Int, width: Int,
                                 ratio: Int, pos: Int, compRow: Int, rope: RopeParams, nRot: Int,
                                 rmsEps: Float, quantize: (GPUTensor) throws -> Void) throws {
        // 1. paired F16 projections (reference path = two F16 matvecs).
        try matmulF16(weight: w.kv, x: x, out: kvCur, inDim: nEmbd, outDim: width)
        try matmulF16(weight: w.gate, x: x, out: scCur, inDim: nEmbd, outDim: width)
        // 2-3. APE add + store into the recurrent state row.
        try compressorStoreOne(kv: kvCur, score: scCur, ape: w.ape, stateKv: stateKv,
                               stateScore: stateScore, width: width, ratio: ratio, pos: pos)
        guard DSV4Shape.emits(pos: pos, ratio: ratio) else { return }
        // 5. pool the window into cache[compRow].
        let row = cache.rowView(compRow * headDim, headDim)
        if ratio != 4 {
            // width == headDim here; pool `ratio` contiguous rows.
            try softmaxPoolContiguous(out: row, stateKv: stateKv, stateScore: stateScore,
                                      rows: ratio, width: headDim)
        } else {
            // TODO(port): ratio==4 two-lane pool. C concats low-half(rows0..3) +
            // high-half(rows4..7) into a packed [8 x headDim] buffer then softmax-pools
            // 8 rows (ds4_gpu_encode_compressor_pool, ds4_metal.m:13801). Needs an
            // encode-form concat_f32_dim1 (+ packed scratch in CompressedLayerState).
            try softmaxPoolContiguous(out: row, stateKv: stateKv, stateScore: stateScore,
                                      rows: ratio, width: headDim)   // PLACEHOLDER (lane 0 only)
        }
        // 6. RMS-norm with weight.
        try rmsNorm(row, weight: w.norm, out: row, rows: 1, n: headDim, eps: rmsEps)
        // 7. RoPE at the window-start position.
        let cpos = DSV4Shape.compRopePos(pos: pos, ratio: ratio)
        try ropeTail(x: row, nTok: 1, nHead: 1, headDim: headDim, nRot: nRot, nCtxOrig: rope.nCtxOrig,
                     freqBase: rope.freqBase, freqScale: rope.freqScale, extFactor: rope.extFactor,
                     attnFactor: rope.attnFactor, betaFast: rope.betaFast, betaSlow: rope.betaSlow,
                     pos0: cpos, posStep: 1)
        // 8. emit-time quantization round-trip (part of the forward math).
        try quantize(row)
        // 10. ratio==4 lane rotation.
        if ratio == 4 { try ratio4Shift(stateKv: stateKv, stateScore: stateScore, width: width) }
    }

    /// Compressed-layer attention (ratio 4/128): runs the attention compressor lane,
    /// the indexer lane + top-512 selection (ratio==4 past threshold), and the right
    /// attention variant, writing nHead*headDim into `s.heads`. Replaces the dense
    /// flashAttnCore for compressed layers. Mirrors the `if compressed` block of
    /// metal_graph_encode_decode_layer (ds4.c:14122-14459).
    ///
    /// Preconditions: `s.cur` = attn_norm, `s.qrNorm` = q-LoRA norm, `s.q` = RoPE'd
    /// query, and the per-token raw KV already stored into `rawCache[...]`.
    ///
    /// PARTIAL (scaffold): see TODO(port) markers — the SWA circular raw cache +
    /// raw_start, the non-indexed raw+all-compressed attention, the ratio-4 two-lane
    /// pool, and the multi-pass top-512 all need finishing/verifying on macOS.
    public func decodeCompressedAttention(s: DecodeScratch, w: LayerWeights, d: DSV4Dims,
                                          state: CompressedLayerState, rawCache: GPUTensor,
                                          nKeys: Int, pos: Int, rope: RopeParams, rmsEps: Float) throws {
        guard let comp = w.comp else { return }
        let ratio = state.ratio, nRot = d.nRot, headDim = state.headDim

        // --- Attention compressor lane (headDim=512) ---
        try compressorUpdate(x: s.cur, w: comp, stateKv: state.stateKv, stateScore: state.stateScore,
                             cache: state.attnCompCache, kvCur: state.compKvCur, scCur: state.compScCur,
                             nEmbd: d.nEmbd, headDim: headDim, width: state.compWidth, ratio: ratio,
                             pos: pos, compRow: state.nComp, rope: rope, nRot: nRot, rmsEps: rmsEps) { row in
            try self.fp8QuantizeRow(row, headDim: headDim, nRot: nRot)
        }
        if DSV4Shape.emits(pos: pos, ratio: ratio) { state.nComp += 1 }

        // --- Indexer lane + selection (ratio==4) ---
        var selected: GPUTensor? = nil
        var nSel = 0
        if ratio == 4, let index = w.index,
           let iStateKv = state.indexStateKv, let iStateScore = state.indexStateScore,
           let iCache = state.indexCompCache, let iQ = state.indexerQ,
           let iW = state.indexerWeights, let iScores = state.indexerScores, let iSel = state.compSelected {
            try compressorUpdate(x: s.cur, w: index.comp, stateKv: iStateKv, stateScore: iStateScore,
                                 cache: iCache, kvCur: state.compKvCur, scCur: state.compScCur,
                                 nEmbd: d.nEmbd, headDim: state.indexHeadDim, width: state.indexWidth,
                                 ratio: ratio, pos: pos, compRow: state.nIndexComp, rope: rope,
                                 nRot: nRot, rmsEps: rmsEps) { row in
                try self.indexerQAT(row, nRows: 1)
            }
            if DSV4Shape.emits(pos: pos, ratio: ratio) { state.nIndexComp += 1 }

            // Gate: only run the sparse selection once there are enough compressed rows.
            if state.nComp > 0 && state.nIndexComp > DSV4Shape.nIndexerTopK {
                // indexer query: qr_norm -> [nIndexerHead*128] (qB is F16 or Q8_0).
                // TODO(verify): if index.qB is Q8_0 use matmulQ8_0 here instead.
                try matmulF16(weight: index.qB, x: s.qrNorm, out: iQ,
                              inDim: d.qRank, outDim: DSV4Shape.indexerQDim)
                try ropeTail(x: iQ, nTok: 1, nHead: DSV4Shape.nIndexerHead, headDim: DSV4Shape.nIndexerHeadDim,
                             nRot: nRot, nCtxOrig: rope.nCtxOrig, freqBase: rope.freqBase, freqScale: rope.freqScale,
                             extFactor: rope.extFactor, attnFactor: rope.attnFactor, betaFast: rope.betaFast,
                             betaSlow: rope.betaSlow, pos0: pos, posStep: 1)
                try indexerQAT(iQ, nRows: DSV4Shape.nIndexerHead)
                try matmulF16(weight: index.proj, x: s.cur, out: iW, inDim: d.nEmbd, outDim: DSV4Shape.nIndexerHead)
                let scale = 1.0 / Float(DSV4Shape.nIndexerHeadDim * DSV4Shape.nIndexerHead).squareRoot()
                try indexerScoreOneDirect(scores: iScores, q: iQ, weights: iW, indexComp: iCache,
                                          nComp: state.nIndexComp, scale: scale)
                // TODO(port): multi-pass top-512 for nIndexComp beyond one threadgroup.
                try indexerTopKSinglePass(selected: iSel, scores: iScores, nComp: state.nIndexComp,
                                          topK: DSV4Shape.nIndexerTopK)
                selected = iSel
                nSel = min(DSV4Shape.nIndexerTopK, state.nIndexComp)
            }
        }

        // --- Attention ---
        // TODO(port): compressed layers cap the raw cache at nSWA=128 (circular) with
        // raw_start = raw_start_for_span(pos). The StreamingDecoder still passes a full
        // raw cache, so rawCap/rawStart/nRaw below are PLACEHOLDERS and the result
        // double-counts recent tokens (raw AND compressed). Wiring the SWA circular
        // raw cache in StreamingDecoder is the key remaining numeric task.
        let rawCap = max(1, nKeys)
        let nRaw = nKeys
        let rawStart = 0
        if let sel = selected, nSel > 0 {
            try indexedMixedAttentionDecode(heads: s.heads, sinks: w.attnSinks, q: s.q, rawKv: rawCache,
                                            compKv: state.attnCompCache, topk: sel, nHead: d.nHead,
                                            nRaw: nRaw, rawCap: rawCap, rawStart: rawStart, nComp: state.nComp,
                                            topK: nSel, window: DSV4Shape.nSWA, ratio: ratio, pos0: pos)
        } else {
            // TODO(port): non-indexed raw + all-compressed attention for ratio==128 and
            // ratio==4 pre-threshold (ds4_gpu_attention_decode_heads_tensor,
            // ds4_metal.m:18240). Falls back to raw-only flash attention for now —
            // WRONG for the compressed history but keeps the graph runnable.
            try flashAttnCore(q: s.q, kvF32: rawCache, kvF16: s.kvF16, mask: s.mask, sinks: w.attnSinks,
                              pad: s.pad, tmp: s.tmp, heads: s.heads, nHead: d.nHead, nKeys: nKeys, hasSinks: true)
        }
    }
}

