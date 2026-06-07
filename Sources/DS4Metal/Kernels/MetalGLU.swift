import Foundation
import Metal

// Phase 9: SwiGLU activation for the FFN inner state. Faithful port of
// ds4_gpu_encode_swiglu_flat dispatching the unchanged metal/glu.metal
// kernel_swiglu_f32: out[i] = silu(min(gate,limit)) * clamp(up,-limit,limit) * alpha
// (clamp only when limit > 1e-6).

extension MetalRuntime {
    /// Single-row SwiGLU over `gate` and `up` (both length n) -> length n.
    public func swiglu(gate: [Float], up: [Float], alpha: Float = 1, limit: Float = 0) throws -> [Float] {
        let n = gate.count
        precondition(up.count >= n)
        let args = Self.gluArgs(n: n, alpha: alpha, limit: limit)

        guard let gbuf = device.makeBuffer(bytes: gate, length: n * 4, options: .storageModeShared),
              let ubuf = device.makeBuffer(bytes: up, length: n * 4, options: .storageModeShared),
              let obuf = device.makeBuffer(length: n * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_swiglu_f32")
        var nth = pso.maxTotalThreadsPerThreadgroup
        let want = n > 1 ? n / 2 : 1
        if nth > want { nth = want }
        if nth == 0 { nth = 1 }

        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(gbuf, offset: 0, index: 1)
        enc.setBuffer(ubuf, offset: 0, index: 2)
        enc.setBuffer(obuf, offset: 0, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = obuf.contents().bindMemory(to: Float.self, capacity: n)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }

    /// 64-byte ds4_metal_args_glu (single-row flat case).
    static func gluArgs(n: Int, alpha: Float, limit: Float) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 64)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func f32(_ off: Int, _ v: Float) { withUnsafeBytes(of: v.bitPattern.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        let rb = UInt64(n) * 4
        i32(0, Int32(n)); u64(8, rb)        // ne00, nb01
        i32(16, Int32(n)); u64(24, rb)      // ne10, nb11
        i32(32, Int32(n)); u64(40, rb)      // ne0, nb1
        i32(48, 0); i32(52, 0)              // i00, i10
        f32(56, alpha); f32(60, limit)      // alpha, limit
        return b
    }
}
