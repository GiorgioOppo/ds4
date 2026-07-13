import Foundation
import Metal

// NSA attention compressor pooling — the softmax-weighted pool over the rolling
// compression window that produces one compressed KV row every `ratio` tokens.
// Faithful to compressor_pool_decode_state (ds4.c:8423) + ds4_gpu_encode_compressor_pool
// (ds4_metal.m:13801), dispatching the real kernel_dsv4_softmax_pool with PER-DIMENSION
// scores (each output dim j has its own softmax over the window rows).
//
// State layout (per compressed layer): coff = (ratio==4 ? 2 : 1); width = coff*headDim;
// rows = coff*ratio. ratio-128 is single-lane (width=headDim); ratio-4 is two-lane
// (width=2*headDim): the pool combines the "prev" lane (rows 0..3, cols 0..headDim) and
// the "cur" lane (rows 4..7, cols headDim..2*headDim) into 8 contributions per output dim.
extension MetalRuntime {
    /// Softmax-pool `nRows` rows into one `headDim` row, with a PER-DIMENSION score:
    /// out[j] = sum_r softmax_r(score[r][j]) * kv[r][j]. `rowStride` is the element
    /// stride (in floats) between consecutive rows in kv/score (headDim for packed,
    /// width for raw state). The kernel reads score at id*nb11 so nb11=sizeof(float).
    public func softmaxPoolPerDim(kv: [Float], score: [Float], nRows: Int, headDim: Int,
                                  rowStride: Int) throws -> [Float] {
        precondition(kv.count >= (nRows - 1) * rowStride + headDim)
        precondition(score.count >= (nRows - 1) * rowStride + headDim)
        let args = Self.softmaxPoolPerDimArgs(nRows: nRows, headDim: headDim, rowStride: rowStride)
        guard let kvbuf = device.makeBuffer(bytes: kv, length: kv.count * 4, options: .storageModeShared),
              let scbuf = device.makeBuffer(bytes: score, length: score.count * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: headDim * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_softmax_pool")
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw MetalError.bufferAlloc }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(kvbuf, offset: 0, index: 1)
        enc.setBuffer(scbuf, offset: 0, index: 2)
        enc.setBuffer(dbuf, offset: 0, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: (headDim + 255) / 256, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: min(256, max(1, headDim)), height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let p = dbuf.contents().bindMemory(to: Float.self, capacity: headDim)
        return Array(UnsafeBufferPointer(start: p, count: headDim))
    }

    /// Full compressor pool: produces one pooled headDim row from the recurrent state.
    /// ratio-128: pool the `ratio` single-lane rows directly. ratio-4: gather the two
    /// lanes into a packed 8 x headDim buffer (prev rows[0..3] cols[0..headDim], cur
    /// rows[4..7] cols[headDim..2headDim]) then pool — mirroring encode_compressor_pool.
    public func compressorPool(stateKv: [Float], stateScore: [Float], headDim: Int, ratio: Int) throws -> [Float] {
        let coff = ratio == 4 ? 2 : 1
        let width = coff * headDim
        let rows = coff * ratio
        precondition(stateKv.count >= rows * width && stateScore.count >= rows * width)
        if ratio != 4 {
            return try softmaxPoolPerDim(kv: stateKv, score: stateScore, nRows: ratio, headDim: headDim, rowStride: width)
        }
        // Gather two lanes -> packed [8 x headDim] (prev then cur).
        var pkv = [Float](repeating: 0, count: 8 * headDim)
        var psc = [Float](repeating: 0, count: 8 * headDim)
        for r in 0..<4 {
            for j in 0..<headDim {
                pkv[r * headDim + j] = stateKv[r * width + j]               // prev lane: rows 0..3, cols 0..headDim
                psc[r * headDim + j] = stateScore[r * width + j]
                pkv[(4 + r) * headDim + j] = stateKv[(4 + r) * width + headDim + j]   // cur lane: rows 4..7, cols headDim..
                psc[(4 + r) * headDim + j] = stateScore[(4 + r) * width + headDim + j]
            }
        }
        return try softmaxPoolPerDim(kv: pkv, score: psc, nRows: 8, headDim: headDim, rowStride: headDim)
    }

    /// 104-byte ds4_metal_args_dsv4_softmax_pool with PER-DIMENSION strides.
    static func softmaxPoolPerDimArgs(nRows: Int, headDim: Int, rowStride: Int) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 104)
        func i64(_ off: Int, _ v: Int64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        let rs = UInt64(rowStride) * 4
        i64(0, Int64(nRows)); i64(8, Int64(headDim)); i64(16, 1)     // ne00(rows), ne01(headDim), ne02(n_comp)
        u64(24, rs); u64(32, 4); u64(40, 0)                          // nb00(kv row), nb01(kv per-dim), nb02
        u64(48, rs); u64(56, 4); u64(64, 0)                          // nb10(score row), nb11(score per-dim), nb12
        i64(72, Int64(headDim)); i64(80, 1)                          // ne0(headDim out), ne1
        u64(88, 4); u64(96, UInt64(headDim) * 4)                     // nb0(out per-dim), nb1
        return b
    }
}
