import Foundation
import Metal

// Phase 9: typed copy/convert at graph boundaries (F32<->F16). Faithful port of
// ds4_gpu_encode_cpy_*_1d dispatching the unchanged metal/cpy.metal kernels
// (kernel_cpy_f32_f16 / kernel_cpy_f16_f32), ds4_gpu_cpy_args (136 bytes, 1D case).

extension MetalRuntime {
    public func cpyF32toF16(_ x: [Float]) throws -> [UInt16] {
        let n = x.count
        let out = try cpy1D(kernel: "kernel_cpy_f32_f16", srcElem: 4, dstElem: 2,
                            srcBytes: x.withUnsafeBytes { Array($0) }, n: n, outElemBytes: 2)
        return out.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
    }

    public func cpyF16toF32(_ x: [UInt16]) throws -> [Float] {
        let n = x.count
        let out = try cpy1D(kernel: "kernel_cpy_f16_f32", srcElem: 2, dstElem: 4,
                            srcBytes: x.withUnsafeBytes { Array($0) }, n: n, outElemBytes: 4)
        return out.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    public func cpyF32toF32(_ x: [Float]) throws -> [Float] {
        let n = x.count
        let out = try cpy1D(kernel: "kernel_cpy_f32_f32", srcElem: 4, dstElem: 4,
                            srcBytes: x.withUnsafeBytes { Array($0) }, n: n, outElemBytes: 4)
        return out.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func cpy1D(kernel: String, srcElem: UInt64, dstElem: UInt64,
                       srcBytes: [UInt8], n: Int, outElemBytes: Int) throws -> [UInt8] {
        let args = Self.cpyArgs(n: n, srcElem: srcElem, dstElem: dstElem)
        guard let sbuf = device.makeBuffer(bytes: srcBytes, length: srcBytes.count, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: n * outElemBytes, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline(kernel)
        var nth = 32
        let maxT = pso.maxTotalThreadsPerThreadgroup
        while nth < n && nth < maxT { nth *= 2 }
        if nth > maxT { nth = maxT }
        if nth > n { nth = n }
        if nth == 0 { nth = 1 }
        let groups = (n + nth - 1) / nth

        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(sbuf, offset: 0, index: 1)
        enc.setBuffer(dbuf, offset: 0, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = dbuf.contents().bindMemory(to: UInt8.self, capacity: n * outElemBytes)
        return Array(UnsafeBufferPointer(start: p, count: n * outElemBytes))
    }

    /// 136-byte ds4_gpu_cpy_args, 1D (make_cpy_1d_args).
    static func cpyArgs(n: Int, srcElem: UInt64, dstElem: UInt64) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 136)
        func i64(_ off: Int, _ v: Int64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        let N = Int64(n)
        let srcRow = UInt64(n) * srcElem, dstRow = UInt64(n) * dstElem
        i64(0, N); i64(8, N); i64(16, 1); i64(24, 1); i64(32, 1)          // nk0, ne00, ne01, ne02, ne03
        u64(40, srcElem); u64(48, srcRow); u64(56, srcRow); u64(64, srcRow) // nb00..nb03
        i64(72, N); i64(80, 1); i64(88, 1); i64(96, 1)                    // ne0, ne1, ne2, ne3
        u64(104, dstElem); u64(112, dstRow); u64(120, dstRow); u64(128, dstRow) // nb0..nb3
        return b
    }
}
