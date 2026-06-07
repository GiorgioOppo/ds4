import Foundation
import Metal

// Phase 9 / Stage A5: MoE router selection + weight normalization. Faithful ports
// dispatching the real metal/dsv4_misc.metal kernels kernel_dsv4_router_finalize_one
// (bitonic top-6 over 256 experts by prob[+bias]) and kernel_dsv4_router_weights_one
// (normalize the 6 selected probs and scale by 1.5). The sqrt(softplus(logit))
// probability step is the separate unary kernel.

extension MetalRuntime {
    /// Select the 6 highest experts (by prob, or prob+bias) from `probs` (256).
    /// Returns the 6 selected expert indices (descending order).
    public func routerFinalizeTop6(probs: [Float], bias: [Float]? = nil) throws -> [Int32] {
        precondition(probs.count >= 256)
        let hasBias = bias != nil
        var args = [UInt8](repeating: 0, count: 20)
        func u32(_ off: Int, _ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { args[off+k] = $0[k] } } }
        u32(0, hasBias ? 1 : 0); u32(4, 0); u32(8, 0); u32(12, 0); u32(16, 1) // has_bias, hash_mode, use_token_buffer, token, hash_rows

        let biasArr = bias ?? [Float](repeating: 0, count: 256)
        let dummyHash: [Int32] = [0]
        let dummyTok: [Int32] = [0]
        guard let pbuf = device.makeBuffer(bytes: probs, length: 256 * 4, options: .storageModeShared),
              let bbuf = device.makeBuffer(bytes: biasArr, length: biasArr.count * 4, options: .storageModeShared),
              let hbuf = device.makeBuffer(bytes: dummyHash, length: 4, options: .storageModeShared),
              let tbuf = device.makeBuffer(bytes: dummyTok, length: 4, options: .storageModeShared),
              let sbuf = device.makeBuffer(length: 6 * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_router_finalize_one")
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw MetalError.bufferAlloc }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: 20, index: 0) }
        enc.setBuffer(pbuf, offset: 0, index: 1)
        enc.setBuffer(bbuf, offset: 0, index: 2)
        enc.setBuffer(hbuf, offset: 0, index: 3)
        enc.setBuffer(tbuf, offset: 0, index: 4)
        enc.setBuffer(sbuf, offset: 0, index: 5)
        enc.setThreadgroupMemoryLength(256 * 4 + 256 * 4, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let p = sbuf.contents().bindMemory(to: Int32.self, capacity: 6)
        return Array(UnsafeBufferPointer(start: p, count: 6))
    }

    /// Normalize the 6 selected expert probabilities and scale by 1.5:
    /// w[i] = probs[selected[i]] / sum(probs[selected]) * 1.5.
    public func routerWeights(probs: [Float], selected: [Int32]) throws -> [Float] {
        precondition(selected.count >= 6)
        guard let pbuf = device.makeBuffer(bytes: probs, length: probs.count * 4, options: .storageModeShared),
              let sbuf = device.makeBuffer(bytes: selected, length: 6 * 4, options: .storageModeShared),
              let wbuf = device.makeBuffer(length: 6 * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_router_weights_one")
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw MetalError.bufferAlloc }
        enc.setComputePipelineState(pso)
        enc.setBuffer(pbuf, offset: 0, index: 0)
        enc.setBuffer(sbuf, offset: 0, index: 1)
        enc.setBuffer(wbuf, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: 6, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 6, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let p = wbuf.contents().bindMemory(to: Float.self, capacity: 6)
        return Array(UnsafeBufferPointer(start: p, count: 6))
    }
}
