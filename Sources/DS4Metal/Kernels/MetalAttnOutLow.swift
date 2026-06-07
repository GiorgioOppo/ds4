import Foundation
import Metal

// Stage E (fidelity): grouped low-rank attention output projection. Faithful port
// of ds4_gpu_attention_output_low_q8 dispatching the real metal/moe.metal kernel
// kernel_dsv4_attn_out_low_q8_0_f32 (a mul_mv_id-style grouped Q8 matvec with
// IDENTITY group mapping — no ids buffer). For each of nGroups groups, projects
// the group's groupDim-slice of `heads` by that group's Q8 weight (rank x groupDim)
// -> low[g][rank]. low_dim = nGroups*rank (= n_out_group*n_lora_o).

extension MetalRuntime {
    /// `outputA` = nGroups groups, each `rank` rows of `groupDim` Q8_0 values.
    /// `heads` = nGroups*groupDim F32 (the attention output, grouped). Returns
    /// nGroups*rank F32. groupDim must be a multiple of 32.
    public func attnOutLowQ8(outputA: [UInt8], heads: [Float], nGroups: Int, groupDim: Int, rank: Int) throws -> [Float] {
        precondition(groupDim % 32 == 0)
        let rowABytes = (groupDim / 32) * 34
        precondition(outputA.count >= nGroups * rank * rowABytes)
        precondition(heads.count >= nGroups * groupDim)
        let nsg: Int16 = 4, nr0 = 2
        let args = Self.attnLowArgs(nGroups: nGroups, groupDim: groupDim, rank: rank, rowABytes: rowABytes)

        guard let abuf = device.makeBuffer(bytes: outputA, length: nGroups * rank * rowABytes, options: .storageModeShared),
              let hbuf = device.makeBuffer(bytes: heads, length: nGroups * groupDim * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: nGroups * rank * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try mulMVPipeline("kernel_dsv4_attn_out_low_q8_0_f32", nsg: nsg)
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw MetalError.bufferAlloc }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(abuf, offset: 0, index: 1)
        enc.setBuffer(hbuf, offset: 0, index: 2)
        enc.setBuffer(dbuf, offset: 0, index: 3)
        enc.setThreadgroupMemoryLength(32 * 2 * 4, index: 0)
        let rowGroups = (rank + nr0 - 1) / nr0
        enc.dispatchThreadgroups(MTLSize(width: rowGroups, height: 1, depth: nGroups),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: Int(nsg), depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let p = dbuf.contents().bindMemory(to: Float.self, capacity: nGroups * rank)
        return Array(UnsafeBufferPointer(start: p, count: nGroups * rank))
    }

    /// 120-byte ds4_gpu_mul_mv_id_args for attn_out_low (nei0=nGroups, ne11=nGroups
    /// for per-group activation, nbi1=0 / no ids).
    static func attnLowArgs(nGroups: Int, groupDim: Int, rank: Int, rowABytes: Int) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 120)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        i32(0, Int32(nGroups)); i32(4, 1); u64(8, 0)                        // nei0, nei1, nbi1
        i32(16, Int32(groupDim)); i32(20, Int32(rank)); i32(24, Int32(nGroups)) // ne00, ne01, ne02
        u64(32, 34); u64(40, UInt64(rowABytes)); u64(48, UInt64(rank) * UInt64(rowABytes)) // nb00, nb01, nb02
        i32(56, Int32(groupDim)); i32(60, Int32(nGroups)); i32(64, 1); i32(68, 1) // ne10, ne11, ne12, ne13
        u64(72, 4); u64(80, UInt64(groupDim) * 4); u64(88, UInt64(nGroups) * UInt64(groupDim) * 4) // nb10, nb11, nb12
        i32(96, Int32(rank)); i32(100, Int32(nGroups)); u64(104, UInt64(rank) * 4); i32(112, 2) // ne0, ne1, nb1, nr0
        return b
    }
}
