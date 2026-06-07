import Foundation
import Metal

// Phase 9: DeepSeek V4 partial RoPE ("tail" rotation with YaRN). Faithful port
// of ds4_gpu_encode_rope_tail_inplace + make_rope_tail_args, dispatching the
// unchanged metal/dsv4_rope.metal kernel_dsv4_rope_tail_f32 (mode 0, non-neox,
// pow path — DS4_METAL_ROPE_EXP2_LOG2 off by default). The first (head_dim -
// n_rot) dims are copied; the last n_rot dims are rotated in pairs.

extension MetalRuntime {
    /// In-place RoPE over a [nTok][nHead][headDim] float tensor. Positions are
    /// pos0 + t*posStep. Returns the transformed tensor.
    public func ropeTail(_ x: [Float], nTok: Int, nHead: Int, headDim: Int, nRot: Int,
                         nCtxOrig: Int, inverse: Bool,
                         freqBase: Float, freqScale: Float, extFactor: Float,
                         attnFactor: Float, betaFast: Float, betaSlow: Float,
                         pos0: Int, posStep: Int) throws -> [Float] {
        precondition(x.count >= nTok * nHead * headDim)
        let total = nTok * nHead * headDim
        let byteLen = total * MemoryLayout<Float>.size

        var positions = [Int32](repeating: 0, count: nTok)
        for t in 0..<nTok { positions[t] = Int32(pos0 + t * posStep) }

        let args = Self.ropeArgs(nTok: nTok, nHead: nHead, headDim: headDim, nRot: nRot,
                                 nCtxOrig: nCtxOrig, inverse: inverse, freqBase: freqBase,
                                 freqScale: freqScale, extFactor: extFactor, attnFactor: attnFactor,
                                 betaFast: betaFast, betaSlow: betaSlow)

        guard let xbuf = device.makeBuffer(bytes: x, length: byteLen, options: .storageModeShared),
              let posbuf = device.makeBuffer(bytes: positions, length: nTok * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }

        let pso = try pipeline("kernel_dsv4_rope_tail_f32")
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(xbuf, offset: 0, index: 1)   // src0 (in-place)
        enc.setBuffer(posbuf, offset: 0, index: 2) // src1 = positions
        enc.setBuffer(xbuf, offset: 0, index: 3)   // src2 (unused; src2 flag false)
        enc.setBuffer(xbuf, offset: 0, index: 4)   // dst (in-place)
        let nth = min(headDim, 256)
        enc.dispatchThreadgroups(MTLSize(width: nHead, height: nTok, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = xbuf.contents().bindMemory(to: Float.self, capacity: total)
        return Array(UnsafeBufferPointer(start: p, count: total))
    }

    /// Build the 144-byte ds4_metal_args_dsv4_rope_tail buffer (make_rope_tail_args).
    static func ropeArgs(nTok: Int, nHead: Int, headDim: Int, nRot: Int, nCtxOrig: Int,
                         inverse: Bool, freqBase: Float, freqScale: Float, extFactor: Float,
                         attnFactor: Float, betaFast: Float, betaSlow: Float) -> [UInt8] {
        let rowBytes = UInt64(headDim) * 4
        let tokBytes = UInt64(nHead) * rowBytes
        var b = [UInt8](repeating: 0, count: 144)
        func i64(_ off: Int, _ v: Int64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func f32(_ off: Int, _ v: Float) { withUnsafeBytes(of: v.bitPattern.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        i64(0, Int64(headDim)); i64(8, Int64(nHead)); i64(16, Int64(nTok)); i64(24, 1)   // ne00..ne03
        u64(32, 4); u64(40, rowBytes); u64(48, tokBytes); u64(56, UInt64(nTok) * tokBytes) // nb00..nb03
        u64(64, 4); u64(72, rowBytes); u64(80, tokBytes); u64(88, UInt64(nTok) * tokBytes) // nb0..nb3
        i32(96, Int32(nRot)); i32(100, 0); i32(104, Int32(nCtxOrig)); i32(108, inverse ? 1 : 0)
        f32(112, freqBase); f32(116, freqScale); f32(120, extFactor); f32(124, attnFactor)
        f32(128, betaFast); f32(132, betaSlow)
        b[136] = 0  // src2 = false
        return b
    }
}
