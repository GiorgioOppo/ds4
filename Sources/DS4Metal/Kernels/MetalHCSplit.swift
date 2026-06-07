import Foundation
import Metal

// Phase 9 / Stage A6: HC mixer split with Sinkhorn-normalized combination matrix.
// Faithful port of ds4_gpu_hc_split_sinkhorn_tensor dispatching the real
// metal/dsv4_hc.metal kernel_dsv4_hc_split_sinkhorn (HC=4 path). Produces, per
// token row, the 24-value split: 4 pre weights (sigmoid), 4 post gates
// (2*sigmoid), and a 4x4 Sinkhorn-normalized combination matrix.

extension MetalRuntime {
    /// HC=4 mixer split. `mix` is nRows x 24 (per-row mixer projection); `scale`
    /// is [pre,post,comb]; `base` is the shared 24-value base bias. Returns
    /// nRows x 24: [0:4] pre, [4:8] post, [8:24] comb (row-major [dst_hc][src_hc]).
    public func hcSplitSinkhorn(mix: [Float], scale: [Float], base: [Float],
                                nRows: Int, sinkhornIters: Int, eps: Float) throws -> [Float] {
        let nHc = 4, mixHc = 2 * nHc + nHc * nHc // 24
        precondition(mix.count >= nRows * mixHc && base.count >= mixHc && scale.count >= 3)
        let args = Self.hcSplitArgs(nHc: nHc, sinkhornIters: sinkhornIters, nRows: nRows, mixHc: mixHc, eps: eps)
        guard let mbuf = device.makeBuffer(bytes: mix, length: nRows * mixHc * 4, options: .storageModeShared),
              let scbuf = device.makeBuffer(bytes: scale, length: 3 * 4, options: .storageModeShared),
              let bbuf = device.makeBuffer(bytes: base, length: mixHc * 4, options: .storageModeShared),
              let obuf = device.makeBuffer(length: nRows * mixHc * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_hc_split_sinkhorn")
        let nth = min(256, max(1, nRows))
        let nTg = (nRows + nth - 1) / nth
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw MetalError.bufferAlloc }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(mbuf, offset: 0, index: 1)
        enc.setBuffer(scbuf, offset: 0, index: 2)
        enc.setBuffer(bbuf, offset: 0, index: 3)
        enc.setBuffer(obuf, offset: 0, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: nTg, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let p = obuf.contents().bindMemory(to: Float.self, capacity: nRows * mixHc)
        return Array(UnsafeBufferPointer(start: p, count: nRows * mixHc))
    }

    /// 48-byte ds4_gpu_hc_split_args.
    static func hcSplitArgs(nHc: Int, sinkhornIters: Int, nRows: Int, mixHc: Int, eps: Float) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 48)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func i64(_ off: Int, _ v: Int64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func f32(_ off: Int, _ v: Float) { withUnsafeBytes(of: v.bitPattern.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        i32(0, Int32(nHc)); i32(4, Int32(sinkhornIters))
        i64(8, Int64(nRows)); i64(16, Int64(mixHc))
        u64(24, UInt64(mixHc) * 4); u64(32, UInt64(mixHc) * 4)  // nb01, nb1
        f32(40, eps)
        return b
    }
}
