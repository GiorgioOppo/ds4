import Foundation
import Metal

// Phase 9 (utility): scatter rows into the KV cache by token position. Faithful
// port of ds4_gpu_encode_set_rows_f32_i32 dispatching the real
// metal/set_rows.metal kernel_set_rows_f32_i32 (ds4_gpu_set_rows_args 88 bytes).

extension MetalRuntime {
    /// Scatter `nRows` rows of `width` floats (`src`) into a `dstRows`-row buffer:
    /// dst[rowIndices[i]] = src[i]. Returns the full dst (zero-initialized).
    public func setRows(src: [Float], rowIndices: [Int32], dstRows: Int, width: Int) throws -> [Float] {
        let nRows = rowIndices.count
        precondition(src.count >= nRows * width)
        let args = Self.setRowsArgs(width: width, nRows: nRows, dstRows: dstRows)
        guard let sbuf = device.makeBuffer(bytes: src, length: nRows * width * 4, options: .storageModeShared),
              let ibuf = device.makeBuffer(bytes: rowIndices, length: nRows * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: dstRows * width * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        // zero dst
        memset(dbuf.contents(), 0, dstRows * width * 4)

        let pso = try pipeline("kernel_set_rows_f32_i32")
        let maxT = pso.maxTotalThreadsPerThreadgroup
        var nth = 32
        let nk0 = max(width, 1)
        while nth < nk0 && nth < maxT { nth *= 2 }
        var nrptg = 1
        if nth > nk0 {
            nrptg = (nth + nk0 - 1) / nk0
            nth = nk0
            if nrptg * nth > maxT { nrptg -= 1 }
        }
        if nth > nk0 { nth = nk0 }
        if nth == 0 { nth = 1 }
        if nrptg == 0 { nrptg = 1 }

        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(sbuf, offset: 0, index: 1)
        enc.setBuffer(ibuf, offset: 0, index: 2)
        enc.setBuffer(dbuf, offset: 0, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: (nRows + nrptg - 1) / nrptg, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: nrptg, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = dbuf.contents().bindMemory(to: Float.self, capacity: dstRows * width)
        return Array(UnsafeBufferPointer(start: p, count: dstRows * width))
    }

    /// 88-byte ds4_gpu_set_rows_args (single-plane KV scatter; ne11/ne12 = 1).
    static func setRowsArgs(width: Int, nRows: Int, dstRows: Int) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 88)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        let rowBytes = UInt64(width) * 4
        let rowsBytes = UInt64(nRows) * 4
        i32(0, Int32(width)); i32(4, Int32(nRows))                            // nk0, ne01
        u64(8, rowBytes); u64(16, UInt64(nRows) * rowBytes); u64(24, UInt64(nRows) * rowBytes) // nb01, nb02, nb03
        i32(32, 1); i32(36, 1)                                                // ne11, ne12
        u64(40, 4); u64(48, rowsBytes); u64(56, rowsBytes)                    // nb10, nb11, nb12
        u64(64, rowBytes); u64(72, UInt64(dstRows) * rowBytes); u64(80, UInt64(dstRows) * rowBytes) // nb1, nb2, nb3
        return b
    }
}
