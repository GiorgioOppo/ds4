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
    public var count: Int = 0        // n_comp emitted so far

    public init(_ rt: MetalRuntime, ratio: Int, headDim: Int, maxComp: Int) throws {
        self.ratio = ratio; self.headDim = headDim
        let coff = ratio == 4 ? 2 : 1
        self.width = coff * headDim
        self.maxComp = maxComp
        let rows = coff * ratio
        stateKv = try .zeros(rt, floatCount: rows * width)
        stateScore = try .floats(rt, [Float](repeating: -1e30, count: rows * width))
        cache = try .zeros(rt, floatCount: maxComp * headDim)
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

    /// Per-dimension softmax pool of `nRows` rows (stride `rowStride` floats) into one
    /// `headDim` row. kernel_dsv4_softmax_pool with per-dim score stride.
    private func softmaxPoolPerDimEnc(kv: GPUTensor, kvOff: Int, score: GPUTensor, scOff: Int,
                                      out: GPUTensor, nRows: Int, headDim: Int, rowStride: Int) throws {
        let args = MetalRuntime.softmaxPoolPerDimArgs(nRows: nRows, headDim: headDim, rowStride: rowStride)
        let pso = try rt.pipeline("kernel_dsv4_softmax_pool")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(kv.buffer, offset: kvOff, index: 1)
        e.setBuffer(score.buffer, offset: scOff, index: 2)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 3)
        e.dispatchThreadgroups(MTLSize(width: (headDim + 255) / 256, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: min(256, headDim), height: 1, depth: 1))
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
    func compressorPoolEnc(_ comp: CompressorState, out: GPUTensor) throws {
        let h = comp.headDim, ratio = comp.ratio, width = comp.width
        if ratio != 4 {
            try softmaxPoolPerDimEnc(kv: comp.stateKv, kvOff: comp.stateKv.byteOffset,
                                     score: comp.stateScore, scOff: comp.stateScore.byteOffset,
                                     out: out, nRows: ratio, headDim: h, rowStride: width)
            return
        }
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
        try softmaxPoolPerDimEnc(kv: comp.packedKv, kvOff: comp.packedKv.byteOffset,
                                 score: comp.packedScore, scOff: comp.packedScore.byteOffset,
                                 out: out, nRows: 8, headDim: h, rowStride: h)
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

    /// Run the compressor for one token on a compressed layer. `attnNorm` is the
    /// rms-normed pre-attention input (s.cur). Updates comp.state + on emit appends one
    /// row to comp.cache and bumps comp.count. Returns n_comp visible to THIS token's
    /// attention (= count after a possible emit).
    func runCompressor(attnNorm: GPUTensor, w: LayerWeights, comp: CompressorState,
                       rope: RopeParams, pos: Int, rmsEps: Float, nRot: Int) throws -> Int {
        guard let compKv = w.compKv, let compGate = w.compGate, let ape = w.compApe, let normW = w.compNorm else {
            return comp.count
        }
        let h = comp.headDim, ratio = comp.ratio, width = comp.width
        // 1) project attn_norm -> kv_cur / sc_cur (F16 matvec).
        try matmulF16(weight: compKv, x: attnNorm, out: comp.kvCur, inDim: attnNorm.count, outDim: width)
        try matmulF16(weight: compGate, x: attnNorm, out: comp.scCur, inDim: attnNorm.count, outDim: width)
        // 2) store into recurrent state (+ APE, F16 -> ape_type 1).
        try compressorStoreOneEnc(kvCur: comp.kvCur, scCur: comp.scCur, ape: ape, apeType: 1,
                                  stateKv: comp.stateKv, stateScore: comp.stateScore, width: width, ratio: ratio, pos: pos)
        let emit = ((pos + 1) % ratio) == 0
        if !emit { return comp.count }
        // 3) emit: pool -> rmsNorm(compNorm) -> rope(comp_pos) -> fp8 -> write cache[count].
        try compressorPoolEnc(comp, out: comp.rowScratch)
        try rmsNorm(comp.rowScratch, weight: normW, out: comp.rowScratch, rows: 1, n: h, eps: rmsEps)
        let compPos = pos + 1 - ratio
        try ropeTail(x: comp.rowScratch, nTok: 1, nHead: 1, headDim: h, nRot: nRot, nCtxOrig: rope.nCtxOrig,
                     freqBase: rope.freqBase, freqScale: rope.freqScale, extFactor: rope.extFactor,
                     attnFactor: rope.attnFactor, betaFast: rope.betaFast, betaSlow: rope.betaSlow, pos0: compPos, posStep: 1)
        try fp8QuantizeRowEnc(comp.rowScratch, headDim: h, nRot: nRot)
        // copy rowScratch -> cache[count]
        let dstRow = comp.cache.rowView(row: comp.count, cols: h)
        try gatherRowsEnc(src: comp.rowScratch, srcByteOffset: comp.rowScratch.byteOffset, srcRowStride: h,
                          dst: dstRow, dstByteOffset: dstRow.byteOffset, rows: 1, cols: h)
        if ratio == 4 { try ratio4ShiftEnc(stateKv: comp.stateKv, stateScore: comp.stateScore, width: width) }
        comp.count += 1
        return comp.count
    }
}
