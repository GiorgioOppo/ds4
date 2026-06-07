import Foundation
import Metal

// Phase 9 (utility): per-row reduction used by the compressor-pooling graph
// boundary. Faithful port of ds4_gpu_encode_sum_rows_f32 dispatching the real
// metal/sum_rows.metal kernel_sum_rows_f32_f32 with FC_sum_rows_op (function
// constant 1400) = OP_SUM_ROWS_NUM_SUM_ROWS (10) for sum, or 11 for mean.

extension MetalRuntime {
    /// Build (and cache) the sum_rows pipeline with the FC_sum_rows_op constant.
    public func sumRowsPipeline(op: Int16) throws -> MTLComputePipelineState {
        let key = "kernel_sum_rows_f32_f32_op=\(op)"
        if let p = mulMVPipelineCache[key] { return p }
        let constants = MTLFunctionConstantValues()
        var o = op
        constants.setConstantValue(&o, type: .short, index: 1400)
        let fn = try library.makeFunction(name: "kernel_sum_rows_f32_f32", constantValues: constants)
        let pso = try device.makeComputePipelineState(function: fn)
        mulMVPipelineCache[key] = pso
        return pso
    }

    /// Reduce each of `rows` rows of `width` floats to one float (sum, or mean if
    /// `mean` is true). Input is `rows*width` row-major floats; output is `rows`.
    public func sumRows(_ x: [Float], width: Int, rows: Int, mean: Bool = false) throws -> [Float] {
        precondition(x.count >= width * rows)
        let args = Self.sumRowsArgs(width: width, rows: rows)
        guard let sbuf = device.makeBuffer(bytes: x, length: width * rows * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: rows * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try sumRowsPipeline(op: mean ? 11 : 10)
        var nth = 32
        let maxT = pso.maxTotalThreadsPerThreadgroup
        while nth < width && nth < maxT { nth *= 2 }
        if nth > maxT { nth = maxT }
        if nth > width { nth = width }
        if nth == 0 { nth = 1 }

        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(sbuf, offset: 0, index: 1)
        enc.setBuffer(dbuf, offset: 0, index: 2)
        enc.setThreadgroupMemoryLength(32 * 4, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = dbuf.contents().bindMemory(to: Float.self, capacity: rows)
        return Array(UnsafeBufferPointer(start: p, count: rows))
    }

    /// 128-byte ds4_metal_args_sum_rows (matches ds4_gpu_encode_sum_rows_f32).
    static func sumRowsArgs(width: Int, rows: Int) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 128)
        func i64(_ off: Int, _ v: Int64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        let W = Int64(width), R = Int64(rows)
        let srcRow = UInt64(width) * 4
        i64(0, W); i64(8, R); i64(16, 1); i64(24, 1)                     // ne00, ne01, ne02, ne03
        u64(32, 4); u64(40, srcRow); u64(48, UInt64(rows) * srcRow); u64(56, UInt64(rows) * srcRow) // nb00..nb03
        i64(64, 1); i64(72, R); i64(80, 1); i64(88, 1)                   // ne0, ne1, ne2, ne3
        u64(96, 4); u64(104, 4); u64(112, UInt64(rows) * 4); u64(120, UInt64(rows) * 4) // nb0..nb3
        return b
    }
}
