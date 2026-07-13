import Foundation
import Metal

// Phase 9 (utility): elementwise unary activations. Faithful port of
// ds4_gpu_encode_unary_f32_rows dispatching the real metal/unary.metal
// kernel_unary_f32_f32 with FC_unary_op (function constant 1200) and
// FC_unary_cnt (1201 = false, the row-major non-contiguous path DS4 uses).

extension MetalRuntime {
    public enum UnaryOp: Int16 {
        case scale = 10, fill = 11, clamp = 12, sqr = 13, sqrt = 14
        case tanh = 100, relu = 101, sigmoid = 102, gelu = 103, geluErf = 104
        case geluQuick = 105, silu = 106, exp = 114, softplus = 115
    }

    public func unaryPipeline(op: UnaryOp) throws -> MTLComputePipelineState {
        let key = "kernel_unary_f32_f32_op=\(op.rawValue)"
        if let p = mulMVPipelineCache[key] { return p }
        let constants = MTLFunctionConstantValues()
        var o = op.rawValue
        var cnt = false
        constants.setConstantValue(&o, type: .short, index: 1200)
        constants.setConstantValue(&cnt, type: .bool, index: 1201)
        let fn = try library.makeFunction(name: "kernel_unary_f32_f32", constantValues: constants)
        let pso = try device.makeComputePipelineState(function: fn)
        mulMVPipelineCache[key] = pso
        return pso
    }

    /// Apply `op` elementwise to `x` (width per row, `rows` rows). For clamp,
    /// pass min/max; for scale, pass scale/bias.
    public func unary(_ x: [Float], op: UnaryOp, width: Int, rows: Int = 1,
                      minV: Float = 0, maxV: Float = 0,
                      scale: Float = 0, bias: Float = 0, val: Float = 0) throws -> [Float] {
        precondition(x.count >= width * rows)
        let args = Self.unaryArgs(width: width, rows: rows, minV: minV, maxV: maxV,
                                  scale: scale, bias: bias, val: val)
        guard let sbuf = device.makeBuffer(bytes: x, length: width * rows * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: width * rows * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try unaryPipeline(op: op)
        var nthMax = pso.maxTotalThreadsPerThreadgroup
        if nthMax > 256 { nthMax = 256 }
        var nth = width
        if nth > nthMax { nth = nthMax }
        if nth == 0 { nth = 1 }
        let nk0 = (width + nth - 1) / nth

        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(sbuf, offset: 0, index: 1)
        enc.setBuffer(dbuf, offset: 0, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: nk0 * rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = dbuf.contents().bindMemory(to: Float.self, capacity: width * rows)
        return Array(UnsafeBufferPointer(start: p, count: width * rows))
    }

    /// 120-byte ds4_gpu_unary_args (matches ds4_gpu_make_unary_rows_args, c4=0).
    static func unaryArgs(width: Int, rows: Int, minV: Float, maxV: Float,
                          scale: Float, bias: Float, val: Float) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 120)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func f32(_ off: Int, _ v: Float) { withUnsafeBytes(of: v.bitPattern.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        let rowBytes = UInt64(width) * 4
        i32(0, Int32(width)); i32(4, Int32(rows)); i32(8, 1); i32(12, 1)       // ne00..ne03
        u64(16, 4); u64(24, rowBytes); u64(32, rowBytes); u64(40, rowBytes)    // nb00..nb03
        i32(48, Int32(width)); i32(52, Int32(rows)); i32(56, 1); i32(60, 1)    // ne0..ne3
        u64(64, 4); u64(72, rowBytes); u64(80, rowBytes); u64(88, rowBytes)    // nb0..nb3
        f32(96, 0); f32(100, scale); f32(104, bias); f32(108, val); f32(112, minV); f32(116, maxV) // slope,scale,bias,val,min,max
        return b
    }
}
