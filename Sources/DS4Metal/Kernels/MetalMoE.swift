import Foundation
import Metal

// Phase 9 (MoE, step 1): expert-routed decode matvec. Dispatches the real
// metal/moe.metal kernel_mul_mv_id_q8_0_f32 (the id wrapper around the Q8_0
// matvec), validating the expert-routing mechanism (ds4_metal_args_mul_mv_id +
// ids buffer). The harder K-quant/IQ2 dequant and the fused pair/group/slots
// MoE kernels are separate follow-up steps.

extension MetalRuntime {
    /// Routed MoE matvec for one token: for each of `expertIds` (K selected
    /// experts), compute expert[id] (Q8_0) * activation -> a row of K x outDim.
    /// `experts` is nExperts rows of outDim Q8_0-quantized in-dim vectors.
    public func moeMatvecQ8_0(experts: [UInt8], expertIds: [Int32], activation: [Float],
                              nExperts: Int, inDim: Int, outDim: Int) throws -> [Float] {
        precondition(inDim % 32 == 0)
        let rowBytes = (inDim / 32) * 34
        let expertBytes = rowBytes * outDim
        precondition(experts.count >= expertBytes * nExperts)
        precondition(activation.count >= inDim)
        let k = expertIds.count

        let nsg: Int16 = 4
        let nr0 = 2
        let args = Self.mulMVIdArgs(nei0: k, nei1: 1, nbi1: UInt64(k * 4),
                                    ne00: inDim, ne01: outDim,
                                    nb00: 34, nb01: UInt64(rowBytes), nb02: UInt64(expertBytes),
                                    ne10: inDim, nb10: 4, nb11: UInt64(inDim * 4), nb12: UInt64(inDim * 4),
                                    ne0: outDim, nb1: UInt64(outDim * 4), nr0: Int32(nr0))

        guard let ebuf = device.makeBuffer(bytes: experts, length: expertBytes * nExperts, options: .storageModeShared),
              let xbuf = device.makeBuffer(bytes: activation, length: inDim * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: k * outDim * 4, options: .storageModeShared),
              let idbuf = device.makeBuffer(bytes: expertIds, length: k * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }

        let pso = try mulMVPipeline("kernel_mul_mv_id_q8_0_f32", nsg: nsg)
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(ebuf, offset: 0, index: 1)
        enc.setBuffer(xbuf, offset: 0, index: 2)
        enc.setBuffer(dbuf, offset: 0, index: 3)
        enc.setBuffer(idbuf, offset: 0, index: 4)
        enc.setThreadgroupMemoryLength(32 * 2 * MemoryLayout<Float>.size, index: 0)
        let groups = MTLSize(width: (outDim + nr0 - 1) / nr0, height: 1, depth: k)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: 32, height: Int(nsg), depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = dbuf.contents().bindMemory(to: Float.self, capacity: k * outDim)
        return Array(UnsafeBufferPointer(start: p, count: k * outDim))
    }

    /// Routed MoE matvec for one token over Q4_K experts (the dominant routed
    /// quant). Dispatches kernel_mul_mv_id_q4_K_f32. Each block_q4_K is 144 bytes
    /// per 256 elements; the q4_K impl packs NSG simdgroups x nr0 rows per
    /// threadgroup, so grid.x = ceil(outDim/(NSG*nr0)).
    public func moeMatvecQ4_K(experts: [UInt8], expertIds: [Int32], activation: [Float],
                              nExperts: Int, inDim: Int, outDim: Int) throws -> [Float] {
        precondition(inDim % 256 == 0, "Q4_K in_dim must be a multiple of 256")
        let rowBytes = (inDim / 256) * 144
        let expertBytes = rowBytes * outDim
        precondition(experts.count >= expertBytes * nExperts)
        precondition(activation.count >= inDim)
        let k = expertIds.count
        let nsg = 4, nr0 = 2

        let args = Self.mulMVIdArgs(nei0: k, nei1: 1, nbi1: UInt64(k * 4),
                                    ne00: inDim, ne01: outDim,
                                    nb00: 144, nb01: UInt64(rowBytes), nb02: UInt64(expertBytes),
                                    ne10: inDim, nb10: 4, nb11: UInt64(inDim * 4), nb12: UInt64(inDim * 4),
                                    ne0: outDim, nb1: UInt64(outDim * 4), nr0: Int32(nr0))

        guard let ebuf = device.makeBuffer(bytes: experts, length: expertBytes * nExperts, options: .storageModeShared),
              let xbuf = device.makeBuffer(bytes: activation, length: inDim * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: k * outDim * 4, options: .storageModeShared),
              let idbuf = device.makeBuffer(bytes: expertIds, length: k * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try mulMVPipeline("kernel_mul_mv_id_q4_K_f32", nsg: Int16(nsg))
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(ebuf, offset: 0, index: 1)
        enc.setBuffer(xbuf, offset: 0, index: 2)
        enc.setBuffer(dbuf, offset: 0, index: 3)
        enc.setBuffer(idbuf, offset: 0, index: 4)
        enc.setThreadgroupMemoryLength(256, index: 0) // unused by q4_K impl but bound
        let groups = MTLSize(width: (outDim + nsg * nr0 - 1) / (nsg * nr0), height: 1, depth: k)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: 32, height: nsg, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = dbuf.contents().bindMemory(to: Float.self, capacity: k * outDim)
        return Array(UnsafeBufferPointer(start: p, count: k * outDim))
    }

    /// Routed MoE matvec for one token over Q2_K experts. Dispatches
    /// kernel_mul_mv_id_q2_K_f32. block_q2_K is 84 bytes per 256 elements
    /// (scales[16], qs[64], d half, dmin half). nr0 = N_R0_Q2_K = 4.
    public func moeMatvecQ2_K(experts: [UInt8], expertIds: [Int32], activation: [Float],
                              nExperts: Int, inDim: Int, outDim: Int) throws -> [Float] {
        precondition(inDim % 256 == 0, "Q2_K in_dim must be a multiple of 256")
        let rowBytes = (inDim / 256) * 84
        let expertBytes = rowBytes * outDim
        precondition(experts.count >= expertBytes * nExperts)
        let k = expertIds.count
        let nsg = 4, nr0 = 4

        let args = Self.mulMVIdArgs(nei0: k, nei1: 1, nbi1: UInt64(k * 4),
                                    ne00: inDim, ne01: outDim,
                                    nb00: 84, nb01: UInt64(rowBytes), nb02: UInt64(expertBytes),
                                    ne10: inDim, nb10: 4, nb11: UInt64(inDim * 4), nb12: UInt64(inDim * 4),
                                    ne0: outDim, nb1: UInt64(outDim * 4), nr0: Int32(nr0))

        guard let ebuf = device.makeBuffer(bytes: experts, length: expertBytes * nExperts, options: .storageModeShared),
              let xbuf = device.makeBuffer(bytes: activation, length: inDim * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: k * outDim * 4, options: .storageModeShared),
              let idbuf = device.makeBuffer(bytes: expertIds, length: k * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try mulMVPipeline("kernel_mul_mv_id_q2_K_f32", nsg: Int16(nsg))
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(ebuf, offset: 0, index: 1)
        enc.setBuffer(xbuf, offset: 0, index: 2)
        enc.setBuffer(dbuf, offset: 0, index: 3)
        enc.setBuffer(idbuf, offset: 0, index: 4)
        enc.setThreadgroupMemoryLength(256, index: 0)
        let groups = MTLSize(width: (outDim + nsg * nr0 - 1) / (nsg * nr0), height: 1, depth: k)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: 32, height: nsg, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = dbuf.contents().bindMemory(to: Float.self, capacity: k * outDim)
        return Array(UnsafeBufferPointer(start: p, count: k * outDim))
    }

    /// Standalone IQ2_XXS routed matvec (real kernel_mul_mv_id_iq2_xxs_f32) for
    /// validation. Mirrors moeMatvecQ2_K but with the 66-byte block_iq2_xxs.
    public func moeMatvecIQ2XXS(experts: [UInt8], expertIds: [Int32], activation: [Float],
                                nExperts: Int, inDim: Int, outDim: Int) throws -> [Float] {
        precondition(inDim % 256 == 0, "IQ2_XXS in_dim must be a multiple of 256")
        let rowBytes = (inDim / 256) * 66
        let expertBytes = rowBytes * outDim
        precondition(experts.count >= expertBytes * nExperts)
        let k = expertIds.count
        let nsg = 4, nr0 = 4

        let args = Self.mulMVIdArgs(nei0: k, nei1: 1, nbi1: UInt64(k * 4),
                                    ne00: inDim, ne01: outDim,
                                    nb00: 66, nb01: UInt64(rowBytes), nb02: UInt64(expertBytes),
                                    ne10: inDim, nb10: 4, nb11: UInt64(inDim * 4), nb12: UInt64(inDim * 4),
                                    ne0: outDim, nb1: UInt64(outDim * 4), nr0: Int32(nr0))

        guard let ebuf = device.makeBuffer(bytes: experts, length: expertBytes * nExperts, options: .storageModeShared),
              let xbuf = device.makeBuffer(bytes: activation, length: inDim * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: k * outDim * 4, options: .storageModeShared),
              let idbuf = device.makeBuffer(bytes: expertIds, length: k * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try mulMVPipeline("kernel_mul_mv_id_iq2_xxs_f32", nsg: Int16(nsg))
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(ebuf, offset: 0, index: 1)
        enc.setBuffer(xbuf, offset: 0, index: 2)
        enc.setBuffer(dbuf, offset: 0, index: 3)
        enc.setBuffer(idbuf, offset: 0, index: 4)
        enc.setThreadgroupMemoryLength(256 * 8 + 128, index: 0)
        let groups = MTLSize(width: (outDim + nsg * nr0 - 1) / (nsg * nr0), height: 1, depth: k)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: 32, height: nsg, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = dbuf.contents().bindMemory(to: Float.self, capacity: k * outDim)
        return Array(UnsafeBufferPointer(start: p, count: k * outDim))
    }

    /// 120-byte ds4_metal_args_mul_mv_id with explicit offsets.
    static func mulMVIdArgs(nei0: Int, nei1: Int, nbi1: UInt64, ne00: Int, ne01: Int,
                            nb00: UInt64, nb01: UInt64, nb02: UInt64,
                            ne10: Int, nb10: UInt64, nb11: UInt64, nb12: UInt64,
                            ne0: Int, nb1: UInt64, nr0: Int32) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 120)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        i32(0, Int32(nei0)); i32(4, Int32(nei1)); u64(8, nbi1)        // nei0, nei1, nbi1
        i32(16, Int32(ne00)); i32(20, Int32(ne01)); i32(24, 1)        // ne00, ne01, ne02
        u64(32, nb00); u64(40, nb01); u64(48, nb02)                   // nb00, nb01, nb02
        i32(56, Int32(ne10)); i32(60, 1); i32(64, 1); i32(68, 1)      // ne10, ne11, ne12, ne13
        u64(72, nb10); u64(80, nb11); u64(88, nb12)                   // nb10, nb11, nb12
        i32(96, Int32(ne0)); i32(100, 1); u64(104, nb1); i32(112, nr0) // ne0, ne1, nb1, nr0
        return b
    }
}
