import Foundation
import Metal

// Phase 9 (utility): elementwise binary ops (add/sub/mul/div) over equal-shape
// row tensors. Faithful port of ds4_gpu_encode_bin_f32_rows dispatching the real
// metal/bin.metal kernel_bin_fuse_f32_f32_f32 with FC_bin_op (1300), FC_bin_f
// (1301=1 single rhs), FC_bin_rb (1302=false), FC_bin_cb (1303=false).

extension MetalRuntime {
    public enum BinOp: Int16 { case add = 0, sub = 1, mul = 2, div = 3 }

    public func binPipeline(op: BinOp, columnBroadcast: Bool = false) throws -> MTLComputePipelineState {
        let key = "kernel_bin_fuse_f32_f32_f32_op=\(op.rawValue)_cb=\(columnBroadcast)"
        if let p = mulMVPipelineCache[key] { return p }
        let constants = MTLFunctionConstantValues()
        var o = op.rawValue, f: Int16 = 1, rb = false, cb = columnBroadcast
        constants.setConstantValue(&o, type: .short, index: 1300)
        constants.setConstantValue(&f, type: .short, index: 1301)
        constants.setConstantValue(&rb, type: .bool, index: 1302)
        constants.setConstantValue(&cb, type: .bool, index: 1303)
        let fn = try library.makeFunction(name: "kernel_bin_fuse_f32_f32_f32", constantValues: constants)
        let pso = try device.makeComputePipelineState(function: fn)
        mulMVPipelineCache[key] = pso
        return pso
    }

    /// Elementwise a `op` b for equal-shape (`width` x `rows`) tensors.
    public func binary(_ a: [Float], _ b: [Float], op: BinOp, width: Int, rows: Int = 1) throws -> [Float] {
        precondition(a.count >= width * rows && b.count >= width * rows)
        let total = width * rows
        let args = Self.binArgs(width: width, rows: rows, rhsWidth: width)
        guard let abuf = device.makeBuffer(bytes: a, length: total * 4, options: .storageModeShared),
              let bbuf = device.makeBuffer(bytes: b, length: total * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: total * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try binPipeline(op: op)
        var nthMax = pso.maxTotalThreadsPerThreadgroup
        if nthMax > 256 { nthMax = 256 }
        var nth = 1
        while 2 * nth < width && nth < nthMax { nth *= 2 }
        if nth == 0 { nth = 1 }

        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(abuf, offset: 0, index: 1)
        enc.setBuffer(bbuf, offset: 0, index: 2)
        enc.setBuffer(dbuf, offset: 0, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = dbuf.contents().bindMemory(to: Float.self, capacity: total)
        return Array(UnsafeBufferPointer(start: p, count: total))
    }

    /// 216-byte ds4_gpu_bin_args (matches ds4_gpu_make_bin_same_rows_args).
    static func binArgs(width: Int, rows: Int, rhsWidth: Int) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 216)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        let rowBytes = UInt64(width) * 4
        let plane = UInt64(rows) * rowBytes
        i32(0, Int32(width)); i32(4, Int32(rows)); i32(8, 1); i32(12, 1)        // ne00..ne03
        u64(16, 4); u64(24, rowBytes); u64(32, plane); u64(40, plane)           // nb00..nb03
        i32(48, Int32(rhsWidth)); i32(52, Int32(rows)); i32(56, 1); i32(60, 1)  // ne10..ne13
        u64(64, 4); u64(72, rowBytes); u64(80, plane); u64(88, plane)           // nb10..nb13
        i32(96, Int32(width)); i32(100, Int32(rows)); i32(104, 1); i32(108, 1)  // ne0..ne3
        u64(112, 4); u64(120, rowBytes); u64(128, plane); u64(136, plane)       // nb0..nb3
        u64(144, 0)                                                             // offs
        for j in 0..<8 { u64(152 + j * 8, 0) }                                  // o1[8]
        return b
    }
}
