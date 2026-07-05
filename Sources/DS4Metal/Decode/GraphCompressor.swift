import Foundation
import Metal

// NSA attention-compressor integration (decode). Per compressed layer (ratio!=0),
// every token: project attn_norm -> kv_cur/sc_cur (F16 matvec), store into the
// recurrent state (+ APE); every `ratio` tokens emit one pooled+normed+roped+fp8
// compressed KV row into comp_cache. Attention then runs over raw SWA rows + ALL
// emitted compressed rows. Faithful to compressor_decode_one (ds4.c:8478) +
// ds4_gpu_compressor_update_tensor (ds4_metal.m:14689). All sub-ops were validated
// vs CPU (MetalCompressorTests); this wires them into the decode command buffer.

/// Persistent per-layer compressor state (lives across tokens for the whole
/// generation). Allocated only for compressed layers (ratio 4 or 128).
public final class CompressorState {
    public let ratio: Int
    public let headDim: Int          // 512 (attention compressor)
    public let width: Int            // coff*headDim (coff = ratio==4 ? 2 : 1)
    public let maxComp: Int          // capacity of comp_cache (rows)
    public let stateKv: GPUTensor    // [coff*ratio x width] f32
    public let stateScore: GPUTensor // [coff*ratio x width] f32, init -1e30
    public let cache: GPUTensor      // [maxComp x headDim] f32 (the emitted rows)
    public let kvCur: GPUTensor      // [width] projection scratch
    public let scCur: GPUTensor      // [width] projection scratch
    public let rowScratch: GPUTensor // [headDim] emitted-row scratch
    public let packedKv: GPUTensor   // [8 x headDim] ratio-4 pool gather scratch
    public let packedScore: GPUTensor
    // Unfused-pool scratch (decode emits pool ONE row): the window transposed to
    // [headDim x poolRows] plus the per-dim softmax — the same graph sequence
    // ds4_metal.m keeps for n_comp == 1 (the fused kernel reduces in a different
    // order; see compressorPoolEnc).
    let poolKvT: GPUTensor           // transposed kv, then the kv*softmax product
    let poolScoreT: GPUTensor        // transposed score
    let poolSoftmax: GPUTensor       // per-dim softmax of poolScoreT
    var poolRows: Int { ratio == 4 ? 8 : ratio }
    public var count: Int = 0        // n_comp emitted so far

    public init(_ rt: MetalRuntime, ratio: Int, headDim: Int, maxComp: Int) throws {
        self.ratio = ratio; self.headDim = headDim
        let coff = ratio == 4 ? 2 : 1
        self.width = coff * headDim
        self.maxComp = maxComp
        let rows = coff * ratio
        stateKv = try .zeros(rt, floatCount: rows * width)
        stateScore = try .floats(rt, [Float](repeating: -1e30, count: rows * width))
        let poolN = (ratio == 4 ? 8 : ratio) * headDim
        poolKvT = try .zeros(rt, floatCount: poolN)
        poolScoreT = try .zeros(rt, floatCount: poolN)
        poolSoftmax = try .zeros(rt, floatCount: poolN)
        // Sized to the full context (maxComp = maxKeys/ratio) but allocated
        // zero-fill-on-demand: only the rows actually emitted ([0..count], the only
        // region attention/indexer ever read) cost physical RAM. So a 1M-context
        // model no longer commits ~13 GB of compressor caches up front. reset()
        // doesn't touch cache (rows are overwritten as re-emitted), so skipping the
        // zero is safe — unwritten rows are never read.
        cache = try .lazyZeros(rt, floatCount: maxComp * headDim)
        kvCur = try .zeros(rt, floatCount: width)
        scCur = try .zeros(rt, floatCount: width)
        rowScratch = try .zeros(rt, floatCount: headDim)
        packedKv = try .zeros(rt, floatCount: 8 * headDim)
        packedScore = try .zeros(rt, floatCount: 8 * headDim)
    }

    /// Reset for a fresh sequence (pos 0): score=-1e30, count=0.
    public func reset(_ rt: MetalRuntime) throws {
        let coff = ratio == 4 ? 2 : 1
        let rows = coff * ratio
        try stateScore.fill(rt, value: -1e30, floatCount: rows * width)
        try stateKv.fill(rt, value: 0, floatCount: rows * width)
        count = 0
    }
}

extension GPUTensor {
    /// Overwrite the first `floatCount` floats with `value` (CPU upload).
    func fill(_ rt: MetalRuntime, value: Float, floatCount: Int) throws {
        let p = buffer.contents().advanced(by: byteOffset).bindMemory(to: Float.self, capacity: floatCount)
        for i in 0..<floatCount { p[i] = value }
    }
    /// A sub-view starting at row `row` of width `cols` floats (same backing buffer).
    func rowView(row: Int, cols: Int) -> GPUTensor {
        GPUTensor(buffer: buffer, byteLength: cols * 4, count: cols, byteOffset: byteOffset + row * cols * 4)
    }
}

extension GraphContext {
    /// Store kv_cur/sc_cur into the recurrent state at the current window row, adding
    /// the APE positional bias to the score. kernel_dsv4_compressor_store_one.
    func compressorStoreOneEnc(kvCur: GPUTensor, scCur: GPUTensor, ape: GPUTensor, apeType: Int,
                               stateKv: GPUTensor, stateScore: GPUTensor, width: Int, ratio: Int, pos: Int) throws {
        var args = [UInt8](repeating: 0, count: 16)
        func u32(_ off: Int, _ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { args[off+k] = $0[k] } } }
        u32(0, UInt32(width)); u32(4, UInt32(ratio)); u32(8, UInt32(pos)); u32(12, UInt32(apeType))
        let pso = try rt.pipeline("kernel_dsv4_compressor_store_one")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: 16, index: 0) }
        e.setBuffer(kvCur.buffer, offset: kvCur.byteOffset, index: 1)
        e.setBuffer(scCur.buffer, offset: scCur.byteOffset, index: 2)
        e.setBuffer(ape.buffer, offset: ape.byteOffset, index: 3)
        e.setBuffer(stateKv.buffer, offset: stateKv.byteOffset, index: 4)
        e.setBuffer(stateScore.buffer, offset: stateScore.byteOffset, index: 5)
        e.dispatchThreadgroups(MTLSize(width: (width + 255) / 256, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Unfused one-comp pool building blocks — encode-forms of the exact helpers
    /// the C decode uses (ds4_gpu_encode_dsv4_softmax_pool_one_comp_ggml):
    /// transpose copy, per-row softmax, elementwise mul, per-row sum. Same
    /// kernels, same args, same nth/grid math — bit-identical reduction order.

    /// kernel_cpy_f32_f32 with a strided 3d source (ds4_gpu_encode_cpy_f32_f32_
    /// 3d_src_strided): dst[i1*dstRowStride + i0] = src[i0*srcColStride + i1*srcRowStride].
    private func cpy3dSrcStridedEnc(src: GPUTensor, srcOff: Int, dst: GPUTensor, dstOff: Int,
                                    cols: Int, rows: Int,
                                    srcColStride: Int, srcRowStride: Int) throws {
        var b = [UInt8](repeating: 0, count: 136)
        func i64(_ off: Int, _ v: Int64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        let srcPlane = UInt64(rows * srcRowStride), dstRow = UInt64(cols) * 4
        let dstPlane = UInt64(rows) * dstRow
        i64(0, Int64(cols)); i64(8, Int64(cols)); i64(16, Int64(rows)); i64(24, 1); i64(32, 1)
        u64(40, UInt64(srcColStride)); u64(48, UInt64(srcRowStride)); u64(56, srcPlane); u64(64, srcPlane)
        i64(72, Int64(cols)); i64(80, Int64(rows)); i64(88, 1); i64(96, 1)
        u64(104, 4); u64(112, dstRow); u64(120, dstPlane); u64(128, dstPlane)
        let pso = try rt.pipeline("kernel_cpy_f32_f32")
        var nth = 32; let maxT = pso.maxTotalThreadsPerThreadgroup
        while nth < cols && nth < maxT { nth *= 2 }
        if nth > maxT { nth = maxT }; if nth > cols { nth = cols }; if nth == 0 { nth = 1 }
        let colGroups = (cols + nth - 1) / nth
        let e = encoder
        e.setComputePipelineState(pso)
        b.withUnsafeBytes { e.setBytes($0.baseAddress!, length: b.count, index: 0) }
        e.setBuffer(src.buffer, offset: srcOff, index: 1)
        e.setBuffer(dst.buffer, offset: dstOff, index: 2)
        e.dispatchThreadgroups(MTLSize(width: colGroups * rows, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }

    /// Per-row softmax over contiguous [rows x width] f32 (ds4_gpu_encode_softmax_
    /// f32_contiguous: scale 1, no bias/mask, _f32_4 variant when width % 4 == 0).
    private func softmaxRowsEnc(src: GPUTensor, srcOff: Int, dst: GPUTensor, dstOff: Int,
                                width: Int, rows: Int) throws {
        let args = MetalRuntime.softmaxArgs(width: width, rows: rows, planes: 1)
        let use4 = width % 4 == 0
        let pso = try rt.pipeline(use4 ? "kernel_soft_max_f32_4" : "kernel_soft_max_f32")
        var nth = 32
        let cap = use4 ? width / 4 : width
        while nth < cap && nth * rows < 256 { nth *= 2 }
        let maxT = pso.maxTotalThreadsPerThreadgroup
        if nth > maxT { nth = maxT }; if nth == 0 { nth = 1 }
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(src.buffer, offset: srcOff, index: 1)
        e.setBuffer(src.buffer, offset: srcOff, index: 2)
        e.setBuffer(src.buffer, offset: srcOff, index: 3)
        e.setBuffer(dst.buffer, offset: dstOff, index: 4)
        e.setThreadgroupMemoryLength(32 * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }

    /// Elementwise a*b over contiguous [rows x cols] f32 (ds4_gpu_encode_bin_f32_
    /// rows with g_mul_pipeline + make_bin_contiguous_3d_args).
    private func mulRowsEnc(a: GPUTensor, aOff: Int, b bT: GPUTensor, bOff: Int,
                            out: GPUTensor, outOff: Int, cols: Int, rows: Int) throws {
        let args = MetalRuntime.binArgs(width: cols, rows: rows, rhsWidth: cols)
        let pso = try rt.binPipeline(op: .mul)
        var nthMax = pso.maxTotalThreadsPerThreadgroup
        if nthMax > 256 { nthMax = 256 }
        var nth = 1
        while 2 * nth < cols && nth < nthMax { nth *= 2 }
        if nth == 0 { nth = 1 }
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(a.buffer, offset: aOff, index: 1)
        e.setBuffer(bT.buffer, offset: bOff, index: 2)
        e.setBuffer(out.buffer, offset: outOff, index: 3)
        e.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }

    /// Per-row sum of contiguous [rows x width] f32 into `rows` floats
    /// (ds4_gpu_encode_sum_rows_f32).
    private func sumRowsEnc(src: GPUTensor, srcOff: Int, dst: GPUTensor, dstOff: Int,
                            width: Int, rows: Int) throws {
        let args = MetalRuntime.sumRowsArgs(width: width, rows: rows)
        let pso = try rt.sumRowsPipeline(op: 10)   // 10 = sum
        var nth = 32; let maxT = pso.maxTotalThreadsPerThreadgroup
        while nth < width && nth < maxT { nth *= 2 }
        if nth > maxT { nth = maxT }; if nth > width { nth = width }; if nth == 0 { nth = 1 }
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(src.buffer, offset: srcOff, index: 1)
        e.setBuffer(dst.buffer, offset: dstOff, index: 2)
        e.setThreadgroupMemoryLength(32 * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }

    /// Copy `cols` floats per row for `rows` rows from src (stride srcRowStride floats,
    /// starting srcRow) into dst (packed, stride cols). kernel_concat is overkill; use a
    /// straight per-row gather via kernel_cpy_f32_f32 row by row.
    private func gatherRowsEnc(src: GPUTensor, srcByteOffset: Int, srcRowStride: Int,
                               dst: GPUTensor, dstByteOffset: Int, rows: Int, cols: Int) throws {
        let pso = try rt.pipeline("kernel_cpy_f32_f32")
        let e = encoder
        for r in 0..<rows {
            let args = MetalRuntime.cpyArgs(n: cols, srcElem: 4, dstElem: 4)
            var nth = 32; let maxT = pso.maxTotalThreadsPerThreadgroup
            while nth < cols && nth < maxT { nth *= 2 }; if nth > maxT { nth = maxT }; if nth > cols { nth = cols }; if nth == 0 { nth = 1 }
            e.setComputePipelineState(pso)
            args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
            e.setBuffer(src.buffer, offset: srcByteOffset + r * srcRowStride * 4, index: 1)
            e.setBuffer(dst.buffer, offset: dstByteOffset + r * cols * 4, index: 2)
            e.dispatchThreadgroups(MTLSize(width: (cols + nth - 1) / nth, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        }
    }

    /// Compressor pool into `out` (headDim). ratio-128: pool the `ratio` single-lane
    /// rows directly. ratio-4: gather two lanes (prev rows0..3 cols0..headDim ; cur
    /// rows4..7 cols headDim..2headDim) into packed 8 x headDim then pool.
    ///
    /// Decode emits ONE row (n_comp == 1), so this stays the UNFUSED graph
    /// sequence — transpose, per-dim softmax, mul, sum_rows — exactly like
    /// ds4_gpu_encode_dsv4_softmax_pool_one_comp_ggml. The fused
    /// kernel_dsv4_softmax_pool is mathematically equivalent but reduces in a
    /// different order; that alone creates ~1e-6 compressor differences and
    /// later FP8/routing flips (ds4_metal.m:14773 keeps the same boundary).
    func compressorPoolEnc(_ comp: CompressorState, out: GPUTensor) throws {
        let h = comp.headDim, ratio = comp.ratio, width = comp.width
        let kvSrc: GPUTensor, scSrc: GPUTensor
        let nRows = comp.poolRows, rowStride: Int
        if ratio != 4 {
            kvSrc = comp.stateKv; scSrc = comp.stateScore; rowStride = width
        } else {
            // prev lane: rows 0..3, cols 0..h  (offset 0, row stride width)
            try gatherRowsEnc(src: comp.stateKv, srcByteOffset: comp.stateKv.byteOffset, srcRowStride: width,
                              dst: comp.packedKv, dstByteOffset: comp.packedKv.byteOffset, rows: 4, cols: h)
            try gatherRowsEnc(src: comp.stateScore, srcByteOffset: comp.stateScore.byteOffset, srcRowStride: width,
                              dst: comp.packedScore, dstByteOffset: comp.packedScore.byteOffset, rows: 4, cols: h)
            // cur lane: rows 4..7, cols h..2h  (offset 4*width + h, row stride width)
            let curOff = (4 * width + h) * 4
            try gatherRowsEnc(src: comp.stateKv, srcByteOffset: comp.stateKv.byteOffset + curOff, srcRowStride: width,
                              dst: comp.packedKv, dstByteOffset: comp.packedKv.byteOffset + 4 * h * 4, rows: 4, cols: h)
            try gatherRowsEnc(src: comp.stateScore, srcByteOffset: comp.stateScore.byteOffset + curOff, srcRowStride: width,
                              dst: comp.packedScore, dstByteOffset: comp.packedScore.byteOffset + 4 * h * 4, rows: 4, cols: h)
            kvSrc = comp.packedKv; scSrc = comp.packedScore; rowStride = h
        }
        // kvT[g][r] = kv[r][g]; scoreT[g][r] = score[r][g]  (contiguous [h x nRows])
        try cpy3dSrcStridedEnc(src: kvSrc, srcOff: kvSrc.byteOffset,
                               dst: comp.poolKvT, dstOff: comp.poolKvT.byteOffset,
                               cols: nRows, rows: h, srcColStride: rowStride * 4, srcRowStride: 4)
        try cpy3dSrcStridedEnc(src: scSrc, srcOff: scSrc.byteOffset,
                               dst: comp.poolScoreT, dstOff: comp.poolScoreT.byteOffset,
                               cols: nRows, rows: h, srcColStride: rowStride * 4, srcRowStride: 4)
        // softmax per dim, product, per-dim sum -> out[g]
        try softmaxRowsEnc(src: comp.poolScoreT, srcOff: comp.poolScoreT.byteOffset,
                           dst: comp.poolSoftmax, dstOff: comp.poolSoftmax.byteOffset,
                           width: nRows, rows: h)
        try mulRowsEnc(a: comp.poolKvT, aOff: comp.poolKvT.byteOffset,
                       b: comp.poolSoftmax, bOff: comp.poolSoftmax.byteOffset,
                       out: comp.poolKvT, outOff: comp.poolKvT.byteOffset, cols: nRows, rows: h)
        try sumRowsEnc(src: comp.poolKvT, srcOff: comp.poolKvT.byteOffset,
                       dst: out, dstOff: out.byteOffset, width: nRows, rows: h)
    }

    /// ratio-4 state shift (prev<-cur). kernel_dsv4_ratio4_shift_f32.
    func ratio4ShiftEnc(stateKv: GPUTensor, stateScore: GPUTensor, width: Int) throws {
        var w = UInt32(width)
        let pso = try rt.pipeline("kernel_dsv4_ratio4_shift_f32")
        let e = encoder
        e.setComputePipelineState(pso)
        e.setBytes(&w, length: 4, index: 0)
        e.setBuffer(stateKv.buffer, offset: stateKv.byteOffset, index: 1)
        e.setBuffer(stateScore.buffer, offset: stateScore.byteOffset, index: 2)
        e.dispatchThreadgroups(MTLSize(width: (4 * width + 255) / 256, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// In-place E4M3FN fp8 quantization of one row (non-rope part). kernel_dsv4_fp8_kv_quantize_f32.
    func fp8QuantizeRowEnc(_ row: GPUTensor, headDim: Int, nRot: Int) throws {
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

    /// How an emitted compressed row is finalized (after pool+norm+rope): the
    /// ATTENTION compressor fp8-rounds the non-rope part; the INDEXER compressor
    /// applies the 128-wide Hadamard + FP4 activation simulation (C:
    /// dsv4_indexer_qat_row_inplace).
    enum CompressorFinalize { case fp8, indexerQat }

    /// Run the compressor for one token on a compressed layer. `attnNorm` is the
    /// rms-normed pre-attention input (s.cur). Updates comp.state + on emit appends one
    /// row to comp.cache and bumps comp.count. Returns n_comp visible to THIS token's
    /// attention (= count after a possible emit).
    func runCompressor(attnNorm: GPUTensor, w: LayerWeights, comp: CompressorState,
                       rope: RopeParams, pos: Int, rmsEps: Float, nRot: Int) throws -> Int {
        guard let compKv = w.compKv, let compGate = w.compGate, let ape = w.compApe, let normW = w.compNorm else {
            return comp.count
        }
        return try runCompressor(x: attnNorm, kv: compKv, gate: compGate, ape: ape, normW: normW,
                                 comp: comp, rope: rope, pos: pos, rmsEps: rmsEps, nRot: nRot,
                                 finalize: .fp8)
    }

    /// Generalized compressor step (shared by the attention and INDEXER
    /// compressors — same recurrence in the C `compressor_decode_one`, different
    /// weights/width/finalize).
    func runCompressor(x: GPUTensor, kv: GPUTensor, gate: GPUTensor, ape: GPUTensor, normW: GPUTensor,
                       comp: CompressorState, rope: RopeParams, pos: Int, rmsEps: Float, nRot: Int,
                       finalize: CompressorFinalize) throws -> Int {
        let h = comp.headDim, ratio = comp.ratio, width = comp.width
        // 1) project x -> kv_cur / sc_cur (F16 matvec).
        try matmulF16(weight: kv, x: x, out: comp.kvCur, inDim: x.count, outDim: width)
        try matmulF16(weight: gate, x: x, out: comp.scCur, inDim: x.count, outDim: width)
        // 2) store into recurrent state (+ APE, F16 -> ape_type 1).
        try compressorStoreOneEnc(kvCur: comp.kvCur, scCur: comp.scCur, ape: ape, apeType: 1,
                                  stateKv: comp.stateKv, stateScore: comp.stateScore, width: width, ratio: ratio, pos: pos)
        let emit = ((pos + 1) % ratio) == 0
        if !emit { return comp.count }
        // 3) emit: pool -> rmsNorm -> rope(comp_pos) -> finalize -> write cache[count].
        try compressorPoolEnc(comp, out: comp.rowScratch)
        try rmsNorm(comp.rowScratch, weight: normW, out: comp.rowScratch, rows: 1, n: h, eps: rmsEps)
        let compPos = pos + 1 - ratio
        try ropeTail(x: comp.rowScratch, nTok: 1, nHead: 1, headDim: h, nRot: nRot, nCtxOrig: rope.nCtxOrig,
                     freqBase: rope.freqBase, freqScale: rope.freqScale, extFactor: rope.extFactor,
                     attnFactor: rope.attnFactor, betaFast: rope.betaFast, betaSlow: rope.betaSlow, pos0: compPos, posStep: 1)
        switch finalize {
        case .fp8:        try fp8QuantizeRowEnc(comp.rowScratch, headDim: h, nRot: nRot)
        case .indexerQat: try indexerHadamardFp4Enc(comp.rowScratch, rows: 1, rowStrideBytes: h * 4)
        }
        // copy rowScratch -> cache[count]
        let dstRow = comp.cache.rowView(row: comp.count, cols: h)
        try gatherRowsEnc(src: comp.rowScratch, srcByteOffset: comp.rowScratch.byteOffset, srcRowStride: h,
                          dst: dstRow, dstByteOffset: dstRow.byteOffset, rows: 1, cols: h)
        if ratio == 4 { try ratio4ShiftEnc(stateKv: comp.stateKv, stateScore: comp.stateScore, width: width) }
        comp.count += 1
        return comp.count
    }

    /// 128-wide Hadamard rotation + FP4 activation simulation, in place on `rows`
    /// rows (kernel_dsv4_indexer_hadamard_fp4_f32; C dsv4_indexer_qat_rows_inplace).
    func indexerHadamardFp4Enc(_ x: GPUTensor, rows: Int, rowStrideBytes: Int) throws {
        var args = [UInt8](repeating: 0, count: 16)
        args.withUnsafeMutableBytes { p in
            p.storeBytes(of: UInt32(rows), toByteOffset: 0, as: UInt32.self)
            p.storeBytes(of: UInt32(128), toByteOffset: 4, as: UInt32.self)
            p.storeBytes(of: UInt64(rowStrideBytes), toByteOffset: 8, as: UInt64.self)
        }
        let pso = try rt.pipeline("kernel_dsv4_indexer_hadamard_fp4_f32")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(x.buffer, offset: x.byteOffset, index: 1)
        e.setThreadgroupMemoryLength(256 * 4, index: 0)    // vals[128] + absbuf[128]
        e.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Indexer relevance scores for the current token over `nComp` indexer rows
    /// (kernel_dsv4_indexer_score_one_direct): scores[c] = scale·Σ_h max(q_h·k_c,0)·w_h.
    func indexerScoresEnc(q: GPUTensor, weights: GPUTensor, indexComp: GPUTensor,
                          scores: GPUTensor, nComp: Int, nHead: Int, headDim: Int, scale: Float) throws {
        var args = [UInt8](repeating: 0, count: 72)
        args.withUnsafeMutableBytes { p in
            p.storeBytes(of: UInt32(nComp), toByteOffset: 0, as: UInt32.self)    // n_comp
            p.storeBytes(of: UInt32(1), toByteOffset: 4, as: UInt32.self)        // n_tokens
            p.storeBytes(of: UInt32(nHead), toByteOffset: 8, as: UInt32.self)
            p.storeBytes(of: UInt32(headDim), toByteOffset: 12, as: UInt32.self)
            p.storeBytes(of: UInt32(0), toByteOffset: 16, as: UInt32.self)       // pos0 (unused)
            p.storeBytes(of: UInt32(4), toByteOffset: 20, as: UInt32.self)       // ratio (unused)
            p.storeBytes(of: UInt64(nHead * headDim * 4), toByteOffset: 24, as: UInt64.self) // q_token_stride
            p.storeBytes(of: UInt64(headDim * 4), toByteOffset: 32, as: UInt64.self)         // q_head_stride
            p.storeBytes(of: UInt64(nHead * 4), toByteOffset: 40, as: UInt64.self)           // weights_token_stride
            p.storeBytes(of: UInt64(headDim * 4), toByteOffset: 48, as: UInt64.self)         // index_row_stride
            p.storeBytes(of: UInt64(nComp * 4), toByteOffset: 56, as: UInt64.self)           // score_token_stride
            p.storeBytes(of: scale, toByteOffset: 64, as: Float.self)
        }
        let pso = try rt.pipeline("kernel_dsv4_indexer_score_one_direct")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(q.buffer, offset: q.byteOffset, index: 1)
        e.setBuffer(weights.buffer, offset: weights.byteOffset, index: 2)
        e.setBuffer(indexComp.buffer, offset: indexComp.byteOffset, index: 3)
        e.setBuffer(scores.buffer, offset: scores.byteOffset, index: 4)
        e.setThreadgroupMemoryLength((128 + 4) * 4, index: 0)   // ktg[128] + psum[4]
        e.dispatchThreadgroups(MTLSize(width: nComp, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
}
