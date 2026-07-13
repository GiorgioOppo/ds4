import Foundation
import Metal

// Phase 9 (router/indexer): descending top-k via bitonic argsort. Faithful port
// of the single-pass branch (npr<=1) of the indexer/router top-k dispatch in
// ds4_metal.m using the real metal/argsort.metal kernel_argsort_f32_i32_desc.
// The multi-block merge path (n > threadgroup size) is a separate follow-up.

extension MetalRuntime {
    /// Return indices of the `topK` largest of `scores` (length `n`), descending.
    /// Single-pass only: requires n <= pipeline.maxTotalThreadsPerThreadgroup.
    public func argsortTopKDesc(_ scores: [Float], n: Int, topK: Int) throws -> [Int32] {
        precondition(scores.count >= n)
        precondition(topK <= n)
        let pso = try pipeline("kernel_argsort_f32_i32_desc")
        var maxT = pso.maxTotalThreadsPerThreadgroup
        if maxT == 0 { maxT = 256 }
        precondition(n <= maxT, "single-pass argsort requires n <= \(maxT)")
        var nth = 1
        while nth < n && 2 * nth <= maxT { nth *= 2 }
        let blockTopK = min(topK, nth)
        let workWidth = topK // npr == 1

        let args = Self.argsortArgs(n: n, rows: 1, ne0: workWidth, topK: blockTopK)
        guard let sbuf = device.makeBuffer(bytes: scores, length: n * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: workWidth * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let smem = ((nth * 4) + 15) & ~15

        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(sbuf, offset: 0, index: 1)
        enc.setBuffer(dbuf, offset: 0, index: 2)
        enc.setThreadgroupMemoryLength(smem, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = dbuf.contents().bindMemory(to: Int32.self, capacity: workWidth)
        return Array(UnsafeBufferPointer(start: p, count: workWidth))
    }

    /// 72-byte ds4_gpu_kargs_argsort.
    static func argsortArgs(n: Int, rows: Int, ne0: Int, topK: Int) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 72)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        let rowBytes = UInt64(n) * 4
        i32(0, Int32(n)); i32(4, Int32(rows)); i32(8, 1); i32(12, 1)            // ne00..ne03
        u64(16, 4); u64(24, rowBytes); u64(32, UInt64(rows) * rowBytes); u64(40, UInt64(rows) * rowBytes) // nb00..nb03
        i32(48, Int32(ne0)); i32(52, Int32(rows)); i32(56, 1); i32(60, 1)       // ne0..ne3
        i32(64, Int32(topK))                                                    // top_k
        return b
    }
}
