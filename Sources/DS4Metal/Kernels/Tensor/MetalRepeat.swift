import Foundation
import Metal

// Phase 9 (hyper-connections): expand a token embedding into the HC channel
// dimension. Faithful port of ds4_gpu_encode_repeat_hc_embedding dispatching the
// real metal/repeat.metal kernel_repeat_f32 (ds4_gpu_repeat_args 96 bytes).

extension MetalRuntime {
    /// Repeat each token's `nEmbd`-vector across `nHC` HC channels:
    /// out[token][hc][:] = src[token][:]. src is nTokens*nEmbd, out is
    /// nTokens*nHC*nEmbd.
    public func repeatHC(src: [Float], nTokens: Int, nEmbd: Int, nHC: Int) throws -> [Float] {
        precondition(src.count >= nTokens * nEmbd)
        let args = Self.repeatArgs(nEmbd: nEmbd, nTokens: nTokens, nHC: nHC)
        guard let sbuf = device.makeBuffer(bytes: src, length: nTokens * nEmbd * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: nTokens * nHC * nEmbd * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_repeat_f32")
        var nth = nEmbd
        let maxT = pso.maxTotalThreadsPerThreadgroup
        if nth > maxT { nth = maxT }
        if nth == 0 { nth = 1 }

        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(sbuf, offset: 0, index: 1)
        enc.setBuffer(dbuf, offset: 0, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: nHC, height: nTokens, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = dbuf.contents().bindMemory(to: Float.self, capacity: nTokens * nHC * nEmbd)
        return Array(UnsafeBufferPointer(start: p, count: nTokens * nHC * nEmbd))
    }

    /// 96-byte ds4_gpu_repeat_args (matches ds4_gpu_encode_repeat_hc_embedding).
    static func repeatArgs(nEmbd: Int, nTokens: Int, nHC: Int) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 96)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        let embdBytes = UInt64(nEmbd) * 4
        i32(0, Int32(nEmbd)); i32(4, 1); i32(8, Int32(nTokens)); i32(12, 1)   // ne00, ne01, ne02, ne03
        u64(16, 4); u64(24, embdBytes); u64(32, embdBytes); u64(40, UInt64(nTokens) * embdBytes) // nb00..nb03
        i32(48, Int32(nEmbd)); i32(52, Int32(nHC)); i32(56, Int32(nTokens)); i32(60, 1) // ne0, ne1, ne2, ne3
        u64(64, 4); u64(72, embdBytes); u64(80, UInt64(nHC) * embdBytes); u64(88, UInt64(nTokens) * UInt64(nHC) * embdBytes) // nb0..nb3
        return b
    }
}
