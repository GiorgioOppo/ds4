import Foundation
import Metal

// Phase 9 / Stage A5: compressed-row top-k mask + ordering kernels for sparse
// attention. Faithful ports dispatching the real metal/dsv4_misc.metal kernels
// kernel_dsv4_topk_mask, kernel_dsv4_topk_mask_scatter, kernel_dsv4_sort_i32_rows_asc.
// All share ds4_metal_args_dsv4_topk_mask (64 bytes). Strides chosen row-major
// (mask = nTokens x nComp; topk = nTokens x topK); the kernels are generic over
// strides so this validates the dispatch; Stage B wires the exact graph layout.

extension MetalRuntime {
    /// Build a dense mask (nTokens x nComp) where selected compressed rows are 0
    /// and all others -inf: init -inf (topk_mask) then scatter 0 at `topk` indices.
    public func topkMaskAndScatter(topk: [Int32], nTokens: Int, topK: Int, nComp: Int) throws -> [Float] {
        precondition(topk.count >= nTokens * topK)
        guard let dst = device.makeBuffer(length: nTokens * nComp * 4, options: .storageModeShared),
              let tkbuf = device.makeBuffer(bytes: topk, length: nTokens * topK * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        // 1) topk_mask: fill -inf. uses ne0=nComp, ne1=nTokens, nb0=4, nb1=nComp*4.
        let maskArgs = Self.topkMaskArgs(ne00: 0, ne01: 0, nb00: 0, nb01: 0,
                                         ne0: nComp, ne1: nTokens, nb0: 4, nb1: UInt64(nComp) * 4)
        try runTopkMask("kernel_dsv4_topk_mask", args: maskArgs, total: nComp * nTokens,
                        bufs: [tkbuf, dst])
        // 2) scatter: set 0 at selected. ne00=topK, ne01=nTokens, nb00=4, nb01=topK*4,
        //    ne0=nComp, nb0=4, nb1=nComp*4.
        let scArgs = Self.topkMaskArgs(ne00: topK, ne01: nTokens, nb00: 4, nb01: UInt64(topK) * 4,
                                       ne0: nComp, ne1: nTokens, nb0: 4, nb1: UInt64(nComp) * 4)
        try runTopkMask("kernel_dsv4_topk_mask_scatter", args: scArgs, total: topK * nTokens,
                        bufs: [tkbuf, dst])
        let p = dst.contents().bindMemory(to: Float.self, capacity: nTokens * nComp)
        return Array(UnsafeBufferPointer(start: p, count: nTokens * nComp))
    }

    /// Sort each token's `topK` selected row indices ascending (topK power of two).
    public func sortI32RowsAsc(_ src: [Int32], nTokens: Int, topK: Int) throws -> [Int32] {
        precondition(src.count >= nTokens * topK)
        precondition(topK & (topK - 1) == 0, "topK must be a power of two for bitonic sort")
        let args = Self.topkMaskArgs(ne00: topK, ne01: nTokens, nb00: 4, nb01: UInt64(topK) * 4,
                                     ne0: 0, ne1: 0, nb0: 0, nb1: 0)
        guard let sbuf = device.makeBuffer(bytes: src, length: nTokens * topK * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: nTokens * topK * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_sort_i32_rows_asc")
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw MetalError.bufferAlloc }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(sbuf, offset: 0, index: 1)
        enc.setBuffer(dbuf, offset: 0, index: 2)
        enc.setThreadgroupMemoryLength(topK * 4, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: nTokens, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: topK, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let p = dbuf.contents().bindMemory(to: Int32.self, capacity: nTokens * topK)
        return Array(UnsafeBufferPointer(start: p, count: nTokens * topK))
    }

    private func runTopkMask(_ kernel: String, args: [UInt8], total: Int, bufs: [MTLBuffer]) throws {
        let pso = try pipeline(kernel)
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw MetalError.bufferAlloc }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(bufs[0], offset: 0, index: 1)
        enc.setBuffer(bufs[1], offset: 0, index: 2)
        let tg = min(256, max(1, total))
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    }

    /// 64-byte ds4_metal_args_dsv4_topk_mask.
    static func topkMaskArgs(ne00: Int, ne01: Int, nb00: UInt64, nb01: UInt64,
                             ne0: Int, ne1: Int, nb0: UInt64, nb1: UInt64) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 64)
        func i64(_ off: Int, _ v: Int64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        i64(0, Int64(ne00)); i64(8, Int64(ne01)); u64(16, nb00); u64(24, nb01)
        i64(32, Int64(ne0)); i64(40, Int64(ne1)); u64(48, nb0); u64(56, nb1)
        return b
    }
}
