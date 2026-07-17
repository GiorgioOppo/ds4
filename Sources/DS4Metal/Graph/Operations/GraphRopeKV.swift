import Foundation
import Metal

extension GraphContext {
    /// Encode-form partial RoPE (tail rotation, in-place) over [nTok][nHead][headDim].
    public func ropeTail(x: GPUTensor, nTok: Int, nHead: Int, headDim: Int, nRot: Int, nCtxOrig: Int,
                         freqBase: Float, freqScale: Float, extFactor: Float, attnFactor: Float,
                         betaFast: Float, betaSlow: Float, pos0: Int, posStep: Int,
                         inverse: Bool = false,
                         kernelMode: RoPEKernelMode = .automatic) throws {
        let args = MetalRuntime.ropeArgs(nTok: nTok, nHead: nHead, headDim: headDim, nRot: nRot,
                                         nCtxOrig: nCtxOrig, inverse: inverse, freqBase: freqBase,
                                         freqScale: freqScale, extFactor: extFactor, attnFactor: attnFactor,
                                         betaFast: betaFast, betaSlow: betaSlow)
        var selected = MetalRuntime.resolveRoPEKernelMode(
            kernelMode, nTok: nTok, headDim: headDim, nRot: nRot)
        let pso: MTLComputePipelineState
        if selected != .baseline,
           let optimized = try? rt.pipeline(MetalRuntime.ropePipelineName(selected)) {
            pso = optimized
        } else {
            selected = .baseline
            pso = try rt.pipeline(MetalRuntime.ropePipelineName(.baseline))
        }
        let nth = selected == .baseline ? min(headDim, 256) : min(max(nRot, 1), 256)
        let e = encoder
        e.setComputePipelineState(pso)
        if selected == .affine {
            let affineArgs = MetalRuntime.ropeAffinePairArgs(
                nHead: nHead, headDim: headDim, nRot: nRot,
                nCtxOrig: nCtxOrig, inverse: inverse, pos0: pos0,
                posStep: posStep, freqBase: freqBase, freqScale: freqScale,
                extFactor: extFactor, attnFactor: attnFactor,
                betaFast: betaFast, betaSlow: betaSlow)
            affineArgs.withUnsafeBytes {
                e.setBytes($0.baseAddress!, length: affineArgs.count, index: 0)
            }
            e.setBuffer(x.buffer, offset: x.byteOffset, index: 1)
        } else {
            var positions = [Int32](repeating: 0, count: nTok)
            for t in 0..<nTok {
                positions[t] = Int32(truncatingIfNeeded: pos0 + t * posStep)
            }
            args.withUnsafeBytes {
                e.setBytes($0.baseAddress!, length: args.count, index: 0)
            }
            e.setBuffer(x.buffer, offset: x.byteOffset, index: 1)
            // Positions inline (setBytes) instead of a fresh MTLBuffer: RoPE
            // runs ~150 times/token in decode. Long prefill batches stay on a
            // transient buffer because setBytes is capped at 4 KB.
            if nTok * 4 <= 4096 {
                positions.withUnsafeBytes {
                    e.setBytes($0.baseAddress!, length: nTok * 4, index: 2)
                }
            } else {
                guard let posbuf = rt.device.makeBuffer(bytes: positions,
                                                        length: nTok * 4,
                                                        options: .storageModeShared) else {
                    throw MetalError.bufferAlloc
                }
                e.setBuffer(posbuf, offset: 0, index: 2)
            }
            e.setBuffer(x.buffer, offset: x.byteOffset, index: 3)
        }
        e.setBuffer(x.buffer, offset: x.byteOffset, index: 4)
        e.dispatchThreadgroups(MTLSize(width: nHead, height: nTok, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }

    /// Encode-form fused KV FP8 store: FP8 (E4M3) the non-RoPE part of `kv` in
    /// place and write the row (F16-rounded) into rawCache[rawRow]. 12-byte args.
    public func kvFP8Store(kv: GPUTensor, rawCache: GPUTensor, headDim: Int, nRot: Int, rawRow: Int) throws {
        var args = [UInt8](repeating: 0, count: 12)
        withUnsafeBytes(of: Int32(headDim).littleEndian) { for k in 0..<4 { args[k] = $0[k] } }
        withUnsafeBytes(of: Int32(nRot).littleEndian) { for k in 0..<4 { args[4+k] = $0[k] } }
        withUnsafeBytes(of: Int32(rawRow).littleEndian) { for k in 0..<4 { args[8+k] = $0[k] } }
        let pso = try rt.pipeline("kernel_dsv4_kv_fp8_store_f32")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: 12, index: 0) }
        e.setBuffer(kv.buffer, offset: kv.byteOffset, index: 1)
        e.setBuffer(rawCache.buffer, offset: rawCache.byteOffset, index: 2)
        e.setThreadgroupMemoryLength(64 * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }
}
