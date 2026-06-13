import Foundation
import Metal

// Stage-A wrapper for kernel_dsv4_indexer_score_one_direct: the decode-only
// ratio-4 indexer relevance score (one threadgroup per compressed row, the row
// staged once in threadgroup memory, 64 heads walked in four-head groups).
//   score[row] = scale * Σ_head max(q[head]·k_row, 0) * weights[head]
// Implements the API referenced by MetalIndexerScoreTests.
extension MetalRuntime {
    public func indexerScoreOne(q: [Float], weights: [Float], indexComp: [Float],
                                nComp: Int, nHead: Int, headDim: Int,
                                scale: Float) throws -> [Float] {
        precondition(nHead == 64 && headDim == 128,
                     "kernel_dsv4_indexer_score_one_direct is specialized for 64 heads × 128 dim")
        precondition(q.count >= nHead * headDim)
        precondition(weights.count >= nHead)
        precondition(indexComp.count >= nComp * headDim)

        // ds4_metal_args_dsv4_indexer_scores_fused: 6×u32, 5×u64, f32 (72 B padded).
        var args = [UInt8](repeating: 0, count: 72)
        func u32(_ off: Int, _ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { args[off + k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { args[off + k] = $0[k] } } }
        func f32(_ off: Int, _ v: Float) { withUnsafeBytes(of: v.bitPattern.littleEndian) { for k in 0..<4 { args[off + k] = $0[k] } } }
        u32(0, UInt32(nComp))                   // n_comp
        u32(4, 1)                               // n_tokens (single-token variant)
        u32(8, UInt32(nHead))                   // n_head
        u32(12, UInt32(headDim))                // head_dim
        u32(16, 0)                              // pos0 (unused here)
        u32(20, 4)                              // ratio (unused here)
        u64(24, UInt64(nHead * headDim * 4))    // q_token_stride
        u64(32, UInt64(headDim * 4))            // q_head_stride
        u64(40, UInt64(nHead * 4))              // weights_token_stride
        u64(48, UInt64(headDim * 4))            // index_row_stride
        u64(56, UInt64(nComp * 4))              // score_token_stride
        f32(64, scale)

        guard let qb = device.makeBuffer(bytes: q, length: q.count * 4, options: .storageModeShared),
              let wb = device.makeBuffer(bytes: weights, length: weights.count * 4, options: .storageModeShared),
              let kb = device.makeBuffer(bytes: indexComp, length: indexComp.count * 4, options: .storageModeShared),
              let ob = device.makeBuffer(length: nComp * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_indexer_score_one_direct")
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(qb, offset: 0, index: 1)
        enc.setBuffer(wb, offset: 0, index: 2)
        enc.setBuffer(kb, offset: 0, index: 3)
        enc.setBuffer(ob, offset: 0, index: 4)
        enc.setThreadgroupMemoryLength((128 + 4) * 4, index: 0)   // ktg[128] + psum[4]
        enc.dispatchThreadgroups(MTLSize(width: nComp, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = ob.contents().bindMemory(to: Float.self, capacity: nComp)
        return Array(UnsafeBufferPointer(start: p, count: nComp))
    }
}
