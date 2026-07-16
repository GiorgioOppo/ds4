import Foundation
import Metal

// Phase 9 / Stage A5: MoE router selection + weight normalization. Faithful ports
// dispatching the real metal/dsv4_misc.metal kernels kernel_dsv4_router_finalize_one
// (bitonic top-6 over 256/384 experts by prob[+bias]) and kernel_dsv4_router_weights_one
// (normalize the 6 selected probs and apply the configured expert scale). The sqrt(softplus(logit))
// probability step is the separate unary kernel.

extension MetalRuntime {
    /// Select the 6 highest experts (by prob, or prob+bias) from `probs`.
    /// Returns the 6 selected expert indices (descending order).
    public func routerFinalizeTop6(probs: [Float], bias: [Float]? = nil,
                                   nExperts: Int) throws -> [Int32] {
        guard nExperts == 256 || nExperts == 384 else {
            throw MetalError.unsupported("router expert count \(nExperts); expected 256 or 384")
        }
        precondition(probs.count >= nExperts)
        if let bias { precondition(bias.count >= nExperts) }
        let hasBias = bias != nil
        let sortWidth = nExperts == 256 ? 256 : 512
        var args = [UInt8](repeating: 0, count: 32)
        func u32(_ off: Int, _ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { args[off+k] = $0[k] } } }
        u32(0, hasBias ? 1 : 0); u32(4, 0); u32(8, 0); u32(12, 0); u32(16, 1)
        u32(20, 0); u32(24, UInt32(nExperts)); u32(28, Float(1.5).bitPattern)

        let biasArr = bias ?? [Float](repeating: 0, count: nExperts)
        let dummyHash: [Int32] = [0]
        let dummyTok: [Int32] = [0]
        guard let pbuf = device.makeBuffer(bytes: probs, length: nExperts * 4, options: .storageModeShared),
              let bbuf = device.makeBuffer(bytes: biasArr, length: biasArr.count * 4, options: .storageModeShared),
              let hbuf = device.makeBuffer(bytes: dummyHash, length: 4, options: .storageModeShared),
              let tbuf = device.makeBuffer(bytes: dummyTok, length: 4, options: .storageModeShared),
              let sbuf = device.makeBuffer(length: 6 * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_router_finalize_one")
        guard sortWidth <= pso.maxTotalThreadsPerThreadgroup else {
            throw MetalError.unsupported("router requires \(sortWidth) threads, GPU supports \(pso.maxTotalThreadsPerThreadgroup)")
        }
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw MetalError.bufferAlloc }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: 32, index: 0) }
        enc.setBuffer(pbuf, offset: 0, index: 1)
        enc.setBuffer(bbuf, offset: 0, index: 2)
        enc.setBuffer(hbuf, offset: 0, index: 3)
        enc.setBuffer(tbuf, offset: 0, index: 4)
        enc.setBuffer(sbuf, offset: 0, index: 5)
        enc.setThreadgroupMemoryLength(sortWidth * 4 + sortWidth * 4, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: sortWidth, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let p = sbuf.contents().bindMemory(to: Int32.self, capacity: 6)
        return Array(UnsafeBufferPointer(start: p, count: 6))
    }

    /// Normalize the 6 selected expert probabilities and apply the configured scale.
    public func routerWeights(probs: [Float], selected: [Int32], nExperts: Int,
                              expertWeightScale: Float) throws -> [Float] {
        guard nExperts == 256 || nExperts == 384 else {
            throw MetalError.unsupported("router expert count \(nExperts); expected 256 or 384")
        }
        precondition(expertWeightScale.isFinite && expertWeightScale >= 0)
        precondition(probs.count >= nExperts)
        precondition(selected.count >= 6)
        precondition(selected.prefix(6).allSatisfy { $0 >= 0 && Int($0) < nExperts })
        var args = [UInt8](repeating: 0, count: 8)
        func u32(_ off: Int, _ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { args[off+k] = $0[k] } } }
        u32(0, UInt32(nExperts)); u32(4, expertWeightScale.bitPattern)
        guard let pbuf = device.makeBuffer(bytes: probs, length: probs.count * 4, options: .storageModeShared),
              let sbuf = device.makeBuffer(bytes: selected, length: 6 * 4, options: .storageModeShared),
              let wbuf = device.makeBuffer(length: 6 * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_router_weights_one")
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw MetalError.bufferAlloc }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: 8, index: 0) }
        enc.setBuffer(pbuf, offset: 0, index: 1)
        enc.setBuffer(sbuf, offset: 0, index: 2)
        enc.setBuffer(wbuf, offset: 0, index: 3)
        enc.dispatchThreads(MTLSize(width: 6, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 6, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let p = wbuf.contents().bindMemory(to: Float.self, capacity: 6)
        return Array(UnsafeBufferPointer(start: p, count: 6))
    }

    /// Backwards-compatible Flash overload. New model-aware callers should pass
    /// the architecture values explicitly.
    public func routerFinalizeTop6(probs: [Float], bias: [Float]? = nil) throws -> [Int32] {
        try routerFinalizeTop6(probs: probs, bias: bias, nExperts: 256)
    }

    /// Backwards-compatible Flash overload.
    public func routerWeights(probs: [Float], selected: [Int32]) throws -> [Float] {
        try routerWeights(probs: probs, selected: selected,
                          nExperts: 256, expertWeightScale: 1.5)
    }
}
