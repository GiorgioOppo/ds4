import Foundation
import Metal

// DeepSeek V4 partial RoPE ("tail" rotation with YaRN). The generic path is the
// reference implementation; the pair path skips the unchanged prefix, and the
// affine path also reconstructs positions on-GPU. Both optimized kernels keep
// the reference arithmetic order and have an exact fallback for unsupported
// shapes. DS4_ROPE_PAIR=1 enables the specialization for A/B measurements;
// DS4_ROPE_AFFINE=0 keeps the pair kernel but restores the position binding.

/// Explicit selection is primarily for numerical A/B tests. Production callers
/// use `.automatic`, controlled once per process by DS4_ROPE_PAIR/AFFINE.
public enum RoPEKernelMode: Sendable, Equatable {
    case automatic
    case baseline
    case pair
    case affine
}

extension MetalRuntime {
    /// In-place RoPE over a [nTok][nHead][headDim] float tensor. Positions are
    /// pos0 + t*posStep. Returns the transformed tensor.
    public func ropeTail(_ x: [Float], nTok: Int, nHead: Int, headDim: Int, nRot: Int,
                         nCtxOrig: Int, inverse: Bool,
                         freqBase: Float, freqScale: Float, extFactor: Float,
                         attnFactor: Float, betaFast: Float, betaSlow: Float,
                         pos0: Int, posStep: Int,
                         kernelMode: RoPEKernelMode = .automatic) throws -> [Float] {
        precondition(x.count >= nTok * nHead * headDim)
        let total = nTok * nHead * headDim
        let byteLen = total * MemoryLayout<Float>.size

        let args = Self.ropeArgs(nTok: nTok, nHead: nHead, headDim: headDim, nRot: nRot,
                                 nCtxOrig: nCtxOrig, inverse: inverse, freqBase: freqBase,
                                 freqScale: freqScale, extFactor: extFactor, attnFactor: attnFactor,
                                 betaFast: betaFast, betaSlow: betaSlow)

        var selected = Self.resolveRoPEKernelMode(kernelMode, nTok: nTok,
                                                  headDim: headDim, nRot: nRot)

        guard let xbuf = device.makeBuffer(bytes: x, length: byteLen,
                                           options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }

        let pso: MTLComputePipelineState
        if selected != .baseline,
           let optimized = try? pipeline(Self.ropePipelineName(selected)) {
            pso = optimized
        } else {
            // Old/custom metal directories may not yet contain the optional
            // specialization. Keep them correct instead of failing at runtime.
            selected = .baseline
            pso = try pipeline(Self.ropePipelineName(.baseline))
        }
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        if selected == .affine {
            let affineArgs = Self.ropeAffinePairArgs(
                nHead: nHead, headDim: headDim, nRot: nRot,
                nCtxOrig: nCtxOrig, inverse: inverse, pos0: pos0,
                posStep: posStep, freqBase: freqBase, freqScale: freqScale,
                extFactor: extFactor, attnFactor: attnFactor,
                betaFast: betaFast, betaSlow: betaSlow)
            affineArgs.withUnsafeBytes {
                enc.setBytes($0.baseAddress!, length: affineArgs.count, index: 0)
            }
            enc.setBuffer(xbuf, offset: 0, index: 1)
        } else {
            var positions = [Int32](repeating: 0, count: nTok)
            for t in 0..<nTok {
                positions[t] = Int32(truncatingIfNeeded: pos0 + t * posStep)
            }
            guard let posbuf = device.makeBuffer(bytes: positions, length: nTok * 4,
                                                 options: .storageModeShared) else {
                throw MetalError.bufferAlloc
            }
            args.withUnsafeBytes {
                enc.setBytes($0.baseAddress!, length: args.count, index: 0)
            }
            enc.setBuffer(xbuf, offset: 0, index: 1)
            enc.setBuffer(posbuf, offset: 0, index: 2)
            enc.setBuffer(xbuf, offset: 0, index: 3) // src2 unused, but ABI-bound
        }
        enc.setBuffer(xbuf, offset: 0, index: 4)
        let nth = selected == .baseline ? min(headDim, 256) : min(max(nRot, 1), 256)
        enc.dispatchThreadgroups(MTLSize(width: nHead, height: nTok, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = xbuf.contents().bindMemory(to: Float.self, capacity: total)
        return Array(UnsafeBufferPointer(start: p, count: total))
    }

    /// Snapshot once per process: demo A/B runs are separate processes and the
    /// decode hot path must not materialize ProcessInfo.environment per layer.
    // Opt-in after the order-balanced M1 Pro end-to-end A/B measured exact
    // logits but no decode gain. Other Apple GPU families can still promote it
    // through their own process-level benchmark.
    private static let automaticRoPEEnabled =
        ProcessInfo.processInfo.environment["DS4_ROPE_PAIR"] == "1"
    private static let automaticAffineRoPEEnabled =
        ProcessInfo.processInfo.environment["DS4_ROPE_AFFINE"] != "0"

    static func resolveRoPEKernelMode(_ requested: RoPEKernelMode, nTok: Int,
                                      headDim: Int, nRot: Int) -> RoPEKernelMode {
        // The 32-aligned prefix preserves the generic kernel's SIMD lane mapping
        // and therefore its fast-math results. Any unusual shape stays exact by
        // returning to the reference kernel.
        let eligible = nTok > 0 && nRot > 0 && nRot <= headDim &&
            nRot % 2 == 0 && (headDim - nRot) % 32 == 0
        guard eligible else { return .baseline }

        switch requested {
        case .baseline:
            return .baseline
        case .pair:
            return .pair
        case .affine:
            return .affine
        case .automatic:
            guard automaticRoPEEnabled else { return .baseline }
            // The branch's measured-safe schedule uses affine positions for
            // single-token decode. Prefill uses the pair kernel until its own
            // affine A/B data justifies changing the default.
            return nTok == 1 && automaticAffineRoPEEnabled ? .affine : .pair
        }
    }

    static func ropePipelineName(_ mode: RoPEKernelMode) -> String {
        switch mode {
        case .affine:
            return "kernel_dsv4_rope_tail_f32_inplace_pair_affine"
        case .pair:
            return "kernel_dsv4_rope_tail_f32_inplace_pair"
        case .automatic, .baseline:
            return "kernel_dsv4_rope_tail_f32"
        }
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

    /// Build the 64-byte ds4_metal_args_dsv4_rope_affine_pair ABI.
    static func ropeAffinePairArgs(nHead: Int, headDim: Int, nRot: Int,
                                   nCtxOrig: Int, inverse: Bool,
                                   pos0: Int, posStep: Int,
                                   freqBase: Float, freqScale: Float,
                                   extFactor: Float, attnFactor: Float,
                                   betaFast: Float, betaSlow: Float) -> [UInt8] {
        let rowBytes = UInt64(headDim) * 4
        var b = [UInt8](repeating: 0, count: 64)
        func u64(_ off: Int, _ v: UInt64) {
            withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } }
        }
        func i32(_ off: Int, _ v: Int32) {
            withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } }
        }
        func u32(_ off: Int, _ v: UInt32) {
            withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } }
        }
        func f32(_ off: Int, _ v: Float) {
            withUnsafeBytes(of: v.bitPattern.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } }
        }
        u64(0, rowBytes)
        u64(8, UInt64(nHead) * rowBytes)
        i32(16, Int32(headDim)); i32(20, Int32(nRot))
        i32(24, Int32(nCtxOrig)); i32(28, inverse ? 1 : 0)
        u32(32, UInt32(truncatingIfNeeded: pos0))
        u32(36, UInt32(truncatingIfNeeded: posStep))
        f32(40, freqBase); f32(44, freqScale)
        f32(48, extFactor); f32(52, attnFactor)
        f32(56, betaFast); f32(60, betaSlow)
        return b
    }
}
