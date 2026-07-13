import Foundation
import Metal

// Phase 9 (hyper-connections): the heart of the DS4 4-stream residual design.
// Faithful ports dispatching the real metal/dsv4_hc.metal kernels:
//   - kernel_dsv4_hc_expand4: per-layer HC update (block + residual streams
//     mixed through post gates and the HC-to-HC combination matrix).
//   - kernel_dsv4_hc_weighted_sum: final collapse of the n_hc streams to one.

extension MetalRuntime {
    /// Per-layer HC expansion (n_hc == 4). For each (token,d):
    ///   block_v = blockOut[t][d] (+ blockAdd[t][d] if given)
    ///   out[t][k][d] = block_v*post[t][k] + sum_j comb[t][k][j]*residual[t][j][d]
    /// Shapes: blockOut nTokens*nEmbd, residual nTokens*4*nEmbd, post nTokens*4,
    /// comb nTokens*4*4 (layout [t][j][k], k innermost), out nTokens*4*nEmbd.
    public func hcExpand4(blockOut: [Float], residual: [Float], post: [Float], comb: [Float],
                          nEmbd: Int, nTokens: Int, blockAdd: [Float]? = nil) throws -> [Float] {
        let nHC = 4
        precondition(blockOut.count >= nTokens * nEmbd)
        precondition(residual.count >= nTokens * nHC * nEmbd)
        precondition(post.count >= nTokens * nHC)
        precondition(comb.count >= nTokens * nHC * nHC)
        let hasAdd = blockAdd != nil
        let addArr = blockAdd ?? blockOut
        let args = Self.hcExpandArgs(nEmbd: nEmbd, nHC: nHC, nTokens: nTokens, hasAdd: hasAdd)

        guard let bbuf = device.makeBuffer(bytes: blockOut, length: nTokens * nEmbd * 4, options: .storageModeShared),
              let rbuf = device.makeBuffer(bytes: residual, length: nTokens * nHC * nEmbd * 4, options: .storageModeShared),
              let pbuf = device.makeBuffer(bytes: post, length: nTokens * nHC * 4, options: .storageModeShared),
              let cbuf = device.makeBuffer(bytes: comb, length: nTokens * nHC * nHC * 4, options: .storageModeShared),
              let abuf = device.makeBuffer(bytes: addArr, length: addArr.count * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: nTokens * nHC * nEmbd * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_hc_expand4")
        let nElem = nEmbd * nTokens
        let nth = min(256, max(1, nElem))
        let nTg = (nElem + nth - 1) / nth

        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(bbuf, offset: 0, index: 1)
        enc.setBuffer(rbuf, offset: 0, index: 2)
        enc.setBuffer(pbuf, offset: 0, index: 3)
        enc.setBuffer(cbuf, offset: 0, index: 4)
        enc.setBuffer(abuf, offset: 0, index: 5)
        enc.setBuffer(dbuf, offset: 0, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: nTg, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = dbuf.contents().bindMemory(to: Float.self, capacity: nTokens * nHC * nEmbd)
        return Array(UnsafeBufferPointer(start: p, count: nTokens * nHC * nEmbd))
    }

    /// Final HC collapse: out[t][d] = sum_h x[t][h][d] * weights[t][h].
    public func hcWeightedSum(x: [Float], weights: [Float], nEmbd: Int, nHC: Int, nTokens: Int) throws -> [Float] {
        precondition(x.count >= nTokens * nHC * nEmbd)
        precondition(weights.count >= nTokens * nHC)
        let args = Self.hcWeightedSumArgs(nEmbd: nEmbd, nHC: nHC, nTokens: nTokens)
        guard let xbuf = device.makeBuffer(bytes: x, length: nTokens * nHC * nEmbd * 4, options: .storageModeShared),
              let wbuf = device.makeBuffer(bytes: weights, length: nTokens * nHC * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: nTokens * nEmbd * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_hc_weighted_sum")
        let nElem = nEmbd * nTokens
        let nth = min(256, max(1, nElem))
        let nTg = (nElem + nth - 1) / nth

        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(xbuf, offset: 0, index: 1)
        enc.setBuffer(wbuf, offset: 0, index: 2)
        enc.setBuffer(dbuf, offset: 0, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: nTg, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = dbuf.contents().bindMemory(to: Float.self, capacity: nTokens * nEmbd)
        return Array(UnsafeBufferPointer(start: p, count: nTokens * nEmbd))
    }

    /// 80-byte ds4_gpu_hc_weighted_sum_args.
    static func hcWeightedSumArgs(nEmbd: Int, nHC: Int, nTokens: Int) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 80)
        func i64(_ off: Int, _ v: Int64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        let e = UInt64(nEmbd) * 4
        i64(0, Int64(nEmbd)); i64(8, Int64(nHC)); i64(16, Int64(nTokens))  // n_embd, n_hc, n_tokens
        u64(24, 4); u64(32, e); u64(40, UInt64(nHC) * e)                   // nb_x0, nb_x1, nb_x2
        u64(48, 4); u64(56, UInt64(nHC) * 4)                              // nb_w0, nb_w1
        u64(64, 4); u64(72, e)                                            // nb0, nb1
        return b
    }

    /// 152-byte ds4_gpu_hc_expand_args.
    static func hcExpandArgs(nEmbd: Int, nHC: Int, nTokens: Int, hasAdd: Bool) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 152)
        func i64(_ off: Int, _ v: Int64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        let e = UInt64(nEmbd) * 4
        let hc = UInt64(nHC) * 4
        i64(0, Int64(nEmbd)); i64(8, Int64(nHC)); i64(16, Int64(nTokens)) // n_embd, n_hc, n_tokens
        u64(24, 4); u64(32, e)                                            // nb_block0, nb_block1
        u64(40, 4); u64(48, e)                                            // nb_add0, nb_add1
        u64(56, 4); u64(64, e); u64(72, UInt64(nHC) * e)                  // nb_res0, nb_res1, nb_res2
        u64(80, 4); u64(88, hc)                                           // nb_post0, nb_post1
        u64(96, 4); u64(104, hc); u64(112, UInt64(nHC) * hc)             // nb_comb0, nb_comb1, nb_comb2
        u64(120, 4); u64(128, e); u64(136, UInt64(nHC) * e)              // nb0, nb1, nb2
        i32(144, hasAdd ? 1 : 0)                                          // has_add
        return b
    }
}
