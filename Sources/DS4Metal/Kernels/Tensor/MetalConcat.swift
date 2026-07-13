import Foundation
import Metal

// Phase 9 (utility): concatenate two row tensors along the row dimension (dim 1).
// Faithful port of ds4_gpu_encode_concat_f32_dim1 dispatching the real
// metal/concat.metal kernel_concat (ds4_gpu_concat_args 152 bytes).

extension MetalRuntime {
    /// Stack `a` (rows0 x cols) on top of `b` (rows1 x cols) -> (rows0+rows1) x cols.
    public func concatRows(_ a: [Float], rows0: Int, _ b: [Float], rows1: Int, cols: Int) throws -> [Float] {
        precondition(a.count >= rows0 * cols && b.count >= rows1 * cols)
        let rows = rows0 + rows1
        let args = Self.concatArgs(cols: cols, rows0: rows0, rows1: rows1)
        guard let abuf = device.makeBuffer(bytes: a, length: rows0 * cols * 4, options: .storageModeShared),
              let bbuf = device.makeBuffer(bytes: b, length: rows1 * cols * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: rows * cols * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_concat")
        var nth = cols < 1024 ? cols : 1024
        let maxT = pso.maxTotalThreadsPerThreadgroup
        if nth > maxT { nth = maxT }
        if nth == 0 { nth = 1 }

        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(abuf, offset: 0, index: 1)
        enc.setBuffer(bbuf, offset: 0, index: 2)
        enc.setBuffer(dbuf, offset: 0, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = dbuf.contents().bindMemory(to: Float.self, capacity: rows * cols)
        return Array(UnsafeBufferPointer(start: p, count: rows * cols))
    }

    /// 152-byte ds4_gpu_concat_args (dim 1, row-major equal-cols).
    static func concatArgs(cols: Int, rows0: Int, rows1: Int) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 152)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        let rowStride = UInt64(cols) * 4
        let p0 = UInt64(rows0) * rowStride, p1 = UInt64(rows1) * rowStride
        let rows = rows0 + rows1
        let dstPlane = UInt64(rows) * rowStride
        i32(0, Int32(cols)); i32(4, Int32(rows0)); i32(8, 1); i32(12, 1)      // ne00..ne03
        u64(16, 4); u64(24, rowStride); u64(32, p0); u64(40, p0)              // nb00..nb03
        i32(48, Int32(cols)); i32(52, Int32(rows1)); i32(56, 1); i32(60, 1)   // ne10..ne13
        u64(64, 4); u64(72, rowStride); u64(80, p1); u64(88, p1)              // nb10..nb13
        i32(96, Int32(cols)); i32(100, Int32(rows)); i32(104, 1); i32(108, 1) // ne0..ne3
        u64(112, 4); u64(120, rowStride); u64(128, dstPlane); u64(136, dstPlane) // nb0..nb3
        i32(144, 1)                                                           // dim = 1
        return b
    }
}
