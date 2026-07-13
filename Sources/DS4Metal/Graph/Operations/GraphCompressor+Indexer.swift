import Foundation
import Metal

extension GraphContext {
    /// 128-wide Hadamard rotation + FP4 activation simulation, in place on `rows`
    /// rows (kernel_dsv4_indexer_hadamard_fp4_f32; C dsv4_indexer_qat_rows_inplace).
    func indexerHadamardFp4Enc(_ x: GPUTensor, rows: Int, rowStrideBytes: Int) throws {
        var args = [UInt8](repeating: 0, count: 16)
        args.withUnsafeMutableBytes { p in
            p.storeBytes(of: UInt32(rows), toByteOffset: 0, as: UInt32.self)
            p.storeBytes(of: UInt32(128), toByteOffset: 4, as: UInt32.self)
            p.storeBytes(of: UInt64(rowStrideBytes), toByteOffset: 8, as: UInt64.self)
        }
        let pso = try rt.pipeline("kernel_dsv4_indexer_hadamard_fp4_f32")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(x.buffer, offset: x.byteOffset, index: 1)
        e.setThreadgroupMemoryLength(256 * 4, index: 0)    // vals[128] + absbuf[128]
        e.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Indexer relevance scores for the current token over `nComp` indexer rows
    /// (kernel_dsv4_indexer_score_one_direct): scores[c] = scale·Σ_h max(q_h·k_c,0)·w_h.
    func indexerScoresEnc(q: GPUTensor, weights: GPUTensor, indexComp: GPUTensor,
                          scores: GPUTensor, nComp: Int, nHead: Int, headDim: Int, scale: Float) throws {
        var args = [UInt8](repeating: 0, count: 72)
        args.withUnsafeMutableBytes { p in
            p.storeBytes(of: UInt32(nComp), toByteOffset: 0, as: UInt32.self)    // n_comp
            p.storeBytes(of: UInt32(1), toByteOffset: 4, as: UInt32.self)        // n_tokens
            p.storeBytes(of: UInt32(nHead), toByteOffset: 8, as: UInt32.self)
            p.storeBytes(of: UInt32(headDim), toByteOffset: 12, as: UInt32.self)
            p.storeBytes(of: UInt32(0), toByteOffset: 16, as: UInt32.self)       // pos0 (unused)
            p.storeBytes(of: UInt32(4), toByteOffset: 20, as: UInt32.self)       // ratio (unused)
            p.storeBytes(of: UInt64(nHead * headDim * 4), toByteOffset: 24, as: UInt64.self) // q_token_stride
            p.storeBytes(of: UInt64(headDim * 4), toByteOffset: 32, as: UInt64.self)         // q_head_stride
            p.storeBytes(of: UInt64(nHead * 4), toByteOffset: 40, as: UInt64.self)           // weights_token_stride
            p.storeBytes(of: UInt64(headDim * 4), toByteOffset: 48, as: UInt64.self)         // index_row_stride
            p.storeBytes(of: UInt64(nComp * 4), toByteOffset: 56, as: UInt64.self)           // score_token_stride
            p.storeBytes(of: scale, toByteOffset: 64, as: Float.self)
        }
        let pso = try rt.pipeline("kernel_dsv4_indexer_score_one_direct")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(q.buffer, offset: q.byteOffset, index: 1)
        e.setBuffer(weights.buffer, offset: weights.byteOffset, index: 2)
        e.setBuffer(indexComp.buffer, offset: indexComp.byteOffset, index: 3)
        e.setBuffer(scores.buffer, offset: scores.byteOffset, index: 4)
        e.setThreadgroupMemoryLength((128 + 4) * 4, index: 0)   // ktg[128] + psum[4]
        e.dispatchThreadgroups(MTLSize(width: nComp, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
}
