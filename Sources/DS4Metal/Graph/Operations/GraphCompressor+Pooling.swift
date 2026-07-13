import Foundation
import Metal

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
    func gatherRowsEnc(src: GPUTensor, srcByteOffset: Int, srcRowStride: Int,
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
}
