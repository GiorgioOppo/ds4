import Foundation
import Metal

// Phase 9 / Stage A2: MoE FFN-tail fused activation kernels. Faithful ports of
// ds4_gpu_encode_moe_swiglu_weight and ds4_gpu_encode_moe_sum6 dispatching the
// real metal/moe.metal kernels kernel_dsv4_moe_swiglu_weight[_f16] and
// kernel_dsv4_moe_sum6_f32.

extension MetalRuntime {
    /// Routed-MoE activation per row: mid = silu(clamp(gate)) * clamp(up) * route_weight.
    /// gate/up are `rows` x `width` F32; `weights` is one route weight per row.
    /// clamp disabled when clampValue <= 1e-6. Returns mid (`rows` x `width` F32).
    public func moeSwiGLUWeight(gate: [Float], up: [Float], weights: [Float],
                                width: Int, rows: Int, clampValue: Float = 0) throws -> [Float] {
        precondition(gate.count >= rows * width && up.count >= rows * width)
        precondition(weights.count >= rows)
        let args = Self.moeSwiGLUWeightArgs(width: width, rows: rows, clampValue: clampValue, midF16: false)
        guard let gbuf = device.makeBuffer(bytes: gate, length: rows * width * 4, options: .storageModeShared),
              let ubuf = device.makeBuffer(bytes: up, length: rows * width * 4, options: .storageModeShared),
              let mbuf = device.makeBuffer(length: rows * width * 4, options: .storageModeShared),
              let wbuf = device.makeBuffer(bytes: weights, length: rows * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_moe_swiglu_weight")
        var nth = pso.maxTotalThreadsPerThreadgroup
        if nth > 256 { nth = 256 }
        if nth > width { nth = width }
        if nth == 0 { nth = 1 }

        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(gbuf, offset: 0, index: 1)
        enc.setBuffer(ubuf, offset: 0, index: 2)
        enc.setBuffer(mbuf, offset: 0, index: 3)
        enc.setBuffer(wbuf, offset: 0, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = mbuf.contents().bindMemory(to: Float.self, capacity: rows * width)
        return Array(UnsafeBufferPointer(start: p, count: rows * width))
    }

    /// Sum the 6 routed-expert output rows per token: dst[t][c] = sum_{e<6} src[t][e][c].
    /// `experts` is `tokens` x 6 x `width` F32 (6 expert rows contiguous per token).
    public func moeSum6(experts: [Float], width: Int, tokens: Int) throws -> [Float] {
        precondition(experts.count >= tokens * 6 * width)
        let args = Self.moeSum6Args(width: width, tokens: tokens)
        guard let sbuf = device.makeBuffer(bytes: experts, length: tokens * 6 * width * 4, options: .storageModeShared),
              let dbuf = device.makeBuffer(length: tokens * width * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_moe_sum6_f32")
        var nth = pso.maxTotalThreadsPerThreadgroup
        if nth > 256 { nth = 256 }
        if nth > width { nth = width }
        if nth == 0 { nth = 1 }

        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(sbuf, offset: 0, index: 1)
        enc.setBuffer(dbuf, offset: 0, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: tokens, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = dbuf.contents().bindMemory(to: Float.self, capacity: tokens * width)
        return Array(UnsafeBufferPointer(start: p, count: tokens * width))
    }

    /// 48-byte ds4_gpu_dsv4_moe_swiglu_weight_args.
    static func moeSwiGLUWeightArgs(width: Int, rows: Int, clampValue: Float, midF16: Bool) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 48)
        func u32(_ off: Int, _ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func f32(_ off: Int, _ v: Float) { withUnsafeBytes(of: v.bitPattern.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        let rowBytes = UInt64(width) * 4
        let midBytes = UInt64(width) * (midF16 ? 2 : 4)
        u32(0, UInt32(width)); u32(4, UInt32(rows))
        u64(8, rowBytes); u64(16, rowBytes); u64(24, midBytes); u64(32, 4) // gate, up, mid, weight strides
        u32(40, 0); f32(44, clampValue)                                    // write_clamped, clamp_value
        return b
    }

    /// 24-byte ds4_gpu_dsv4_moe_sum6_args.
    static func moeSum6Args(width: Int, tokens: Int) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 24)
        func u32(_ off: Int, _ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        let rowBytes = UInt64(width) * 4
        u32(0, UInt32(width)); u32(4, UInt32(tokens))
        u64(8, 6 * rowBytes); u64(16, rowBytes)                            // src_token_stride, dst_token_stride
        return b
    }
}
