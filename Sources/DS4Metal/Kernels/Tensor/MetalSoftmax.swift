import Foundation
import Metal

// Phase 9 (utility): row softmax. Faithful port of the plain-softmax dispatch in
// ds4_metal.m (no mask, no sink, scale=1) using the real metal/softmax.metal
// kernels kernel_soft_max_f32 / kernel_soft_max_f32_4. The fused
// dsv4_softmax_pool path is a separate kernel.

extension MetalRuntime {
    /// Softmax over each row of `width` floats; `rows` rows, `planes` planes
    /// (rows contiguous, then planes). scale=1, no mask. Returns same shape.
    public func softmax(_ x: [Float], width: Int, rows: Int = 1, planes: Int = 1) throws -> [Float] {
        precondition(x.count >= width * rows * planes)
        let args = Self.softmaxArgs(width: width, rows: rows, planes: planes)
        let total = width * rows * planes
        guard let sbuf = device.makeBuffer(bytes: x, length: total * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: total * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let use4 = (width % 4 == 0)
        let pso = try pipeline(use4 ? "kernel_soft_max_f32_4" : "kernel_soft_max_f32")
        var nth = 32
        let limit = use4 ? width / 4 : width
        while nth < limit && nth * rows * planes < 256 { nth *= 2 }
        let maxT = pso.maxTotalThreadsPerThreadgroup
        if nth > maxT { nth = maxT }
        if nth == 0 { nth = 1 }

        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(sbuf, offset: 0, index: 1)
        enc.setBuffer(sbuf, offset: 0, index: 2) // src1 == src0 -> no mask
        enc.setBuffer(sbuf, offset: 0, index: 3) // src2 == src0 -> no sink
        enc.setBuffer(dbuf, offset: 0, index: 4)
        enc.setThreadgroupMemoryLength(32 * 4, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: planes, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = dbuf.contents().bindMemory(to: Float.self, capacity: total)
        return Array(UnsafeBufferPointer(start: p, count: total))
    }

    /// 128-byte ds4_gpu_softmax_args (matches ds4_metal_args_soft_max layout).
    static func softmaxArgs(width: Int, rows: Int, planes: Int) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 128)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func f32(_ off: Int, _ v: Float) { withUnsafeBytes(of: v.bitPattern.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        let rowBytes = UInt64(width) * 4
        let planeBytes = UInt64(rows) * rowBytes
        i32(0, Int32(width)); i32(4, Int32(rows)); i32(8, Int32(planes))    // ne00, ne01, ne02
        u64(16, rowBytes); u64(24, planeBytes); u64(32, UInt64(planes) * planeBytes) // nb01, nb02, nb03
        i32(40, Int32(width)); i32(44, Int32(rows)); i32(48, Int32(planes)) // ne11, ne12, ne13
        u64(56, rowBytes); u64(64, planeBytes); u64(72, UInt64(planes) * planeBytes) // nb11, nb12, nb13
        u64(80, rowBytes); u64(88, planeBytes); u64(96, UInt64(planes) * planeBytes) // nb1, nb2, nb3
        f32(104, 1.0); f32(108, 0.0); f32(112, 0.0); f32(116, 0.0); i32(120, 1)      // scale, max_bias, m0, m1, n_head_log2
        return b
    }
}
