import Foundation
import Metal

extension GraphContext {
    /// Encode-form routed MoE matvec over typed experts (Q4_K/Q2_K/IQ2_XXS).
    /// K = ids.count slots. perExpertAct=false: all experts use the same
    /// `activation` row (gate/up); true: expert slot e uses activation row e (down).
    /// `out` is K x outDim. inDim multiple of 256. All three quants share the same
    /// `kernel_mul_mv_id<...>` wrapper (same 120-byte args, same dispatch) — only
    /// the block byte-size (nb00) and nr0 differ.
    public func moeMatvecID(_ quant: MoEQuant, experts: GPUTensor, ids: GPUTensor, activation: GPUTensor,
                            out: GPUTensor, k: Int, inDim: Int, outDim: Int, perExpertAct: Bool,
                            expertStride: Int? = nil) throws {
        precondition(inDim % 256 == 0)
        let nsg = Int(GraphContext.moeNSG), nr0 = quant.nr0   // DS4_MOE_NSG (default 4)
        let blockBytes = quant.blockBytes
        let rowBytes = (inDim / 256) * blockBytes
        // expertStride: byte fra un esperto e il successivo nel buffer — di
        // default il packing stretto (rowBytes*outDim); il pool INTERLEAVED
        // della slot-cache passa la dimensione del record gate|up|down.
        let expertBytes = expertStride ?? (rowBytes * outDim)
        let ne11 = perExpertAct ? k : 1
        let args = Self.mulMVIdArgsFull(nei0: k, nei1: 1, nbi1: UInt64(k * 4), ne00: inDim, ne01: outDim,
                                        nb00: UInt64(blockBytes), nb01: UInt64(rowBytes), nb02: UInt64(expertBytes),
                                        ne10: inDim, ne11: ne11, nb10: 4, nb11: UInt64(inDim * 4),
                                        nb12: UInt64(inDim * 4), ne0: outDim, nb1: UInt64(outDim * 4), nr0: Int32(nr0))
        let pso = try rt.mulMVPipeline(quant.kernel, nsg: Int16(nsg))
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(experts.buffer, offset: experts.byteOffset, index: 1)   // byteOffset != 0 for mmap views
        e.setBuffer(activation.buffer, offset: activation.byteOffset, index: 2)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 3)
        e.setBuffer(ids.buffer, offset: ids.byteOffset, index: 4)
        e.setThreadgroupMemoryLength(quant.threadgroupBytes, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (outDim + nsg * nr0 - 1) / (nsg * nr0), height: 1, depth: k),
                               threadsPerThreadgroup: MTLSize(width: 32, height: nsg, depth: 1))
    }

    /// Back-compat wrapper: Q4_K routed MoE matvec.
    public func moeMatvecQ4K(experts: GPUTensor, ids: GPUTensor, activation: GPUTensor, out: GPUTensor,
                             k: Int, inDim: Int, outDim: Int, perExpertAct: Bool) throws {
        try moeMatvecID(.q4_K, experts: experts, ids: ids, activation: activation, out: out,
                        k: k, inDim: inDim, outDim: outDim, perExpertAct: perExpertAct)
    }

    /// FUSED routed gate+up matvec + SwiGLU·route-weight (1 dispatch instead of 3):
    /// mid[e] = silu(clamp(gate_e·x)) · clamp(up_e·x) · w[e] for each selected expert.
    /// Mirrors the C engine's release path (kernel_mul_mv_id_<q>_pair_swiglu_f32).
    /// `gateScratch`/`upScratch` receive diagnostic raw projections (row collisions
    /// are harmless — nothing reads them in release). Only iq2_xxs/q4_K exist.
    public func moePairSwiGLU(_ quant: MoEQuant, gateExp: GPUTensor, upExp: GPUTensor,
                              ids: GPUTensor, activation: GPUTensor, weights: GPUTensor,
                              gateScratch: GPUTensor, upScratch: GPUTensor, mid: GPUTensor,
                              k: Int, inDim: Int, outDim: Int, clamp: Float,
                              expertStride: Int? = nil) throws {
        let kernel: String
        switch quant {
        case .iq2_xxs: kernel = "kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32"
        case .q4_K:    kernel = "kernel_mul_mv_id_q4_K_pair_swiglu_f32"
        case .q2_K:    throw MetalError.missingKernel("no q2_K pair_swiglu kernel")
        }
        let nsg = Int(GraphContext.moeNSG), nr0 = quant.nr0   // DS4_MOE_NSG (default 4)
        let rowBytes = (inDim / 256) * quant.blockBytes
        let args = Self.mulMVIdArgsFull(nei0: k, nei1: 1, nbi1: UInt64(k * 4), ne00: inDim, ne01: outDim,
                                        nb00: UInt64(quant.blockBytes), nb01: UInt64(rowBytes),
                                        nb02: UInt64(expertStride ?? (rowBytes * outDim)), ne10: inDim, ne11: 1,
                                        nb10: 4, nb11: UInt64(inDim * 4), nb12: UInt64(inDim * 4),
                                        ne0: outDim, nb1: UInt64(outDim * 4), nr0: Int32(nr0))
        let act = MetalRuntime.moeSwiGLUWeightArgs(width: outDim, rows: k, clampValue: clamp, midF16: false)
        let pso = try rt.mulMVPipeline(kernel, nsg: Int16(nsg))
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        act.withUnsafeBytes { e.setBytes($0.baseAddress!, length: act.count, index: 1) }
        e.setBuffer(gateExp.buffer, offset: gateExp.byteOffset, index: 2)
        e.setBuffer(upExp.buffer, offset: upExp.byteOffset, index: 3)
        e.setBuffer(activation.buffer, offset: activation.byteOffset, index: 4)
        e.setBuffer(gateScratch.buffer, offset: gateScratch.byteOffset, index: 5)
        e.setBuffer(upScratch.buffer, offset: upScratch.byteOffset, index: 6)
        e.setBuffer(mid.buffer, offset: mid.byteOffset, index: 7)
        e.setBuffer(ids.buffer, offset: ids.byteOffset, index: 8)
        e.setBuffer(weights.buffer, offset: weights.byteOffset, index: 9)
        e.setThreadgroupMemoryLength(quant.threadgroupBytes, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (outDim + nsg * nr0 - 1) / (nsg * nr0), height: 1, depth: k),
                               threadsPerThreadgroup: MTLSize(width: 32, height: nsg, depth: 1))
    }

    /// FUSED routed down-projection + sum over the 6 selected experts (1 dispatch
    /// instead of 2): out[outDim] = Σ_e down_e · mid[e]. The kernel hardcodes 6
    /// expert slots, so it requires k == 6. Only q2_K/q4_K exist.
    public func moeDownSum6(_ quant: MoEQuant, experts: GPUTensor, ids: GPUTensor,
                            mid: GPUTensor, out: GPUTensor, inDim: Int, outDim: Int,
                            expertStride: Int? = nil) throws {
        let kernel: String
        switch quant {
        case .q2_K:    kernel = "kernel_mul_mv_id_q2_K_sum6_f32"
        case .q4_K:    kernel = "kernel_mul_mv_id_q4_K_sum6_f32"
        case .iq2_xxs: throw MetalError.missingKernel("no iq2_xxs down_sum6 kernel")
        }
        let nsg = Int(GraphContext.moeNSG), nr0 = quant.nr0   // DS4_MOE_NSG (default 4)
        let rowBytes = (inDim / 256) * quant.blockBytes
        let args = Self.mulMVIdArgsFull(nei0: 6, nei1: 1, nbi1: 6 * 4, ne00: inDim, ne01: outDim,
                                        nb00: UInt64(quant.blockBytes), nb01: UInt64(rowBytes),
                                        nb02: UInt64(expertStride ?? (rowBytes * outDim)), ne10: inDim, ne11: 6,
                                        nb10: 4, nb11: UInt64(inDim * 4), nb12: UInt64(6 * inDim * 4),
                                        ne0: outDim, nb1: UInt64(outDim * 4), nr0: Int32(nr0))
        let pso = try rt.mulMVPipeline(kernel, nsg: Int16(nsg))
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(experts.buffer, offset: experts.byteOffset, index: 1)
        e.setBuffer(mid.buffer, offset: mid.byteOffset, index: 2)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 3)
        e.setBuffer(ids.buffer, offset: ids.byteOffset, index: 4)
        e.setThreadgroupMemoryLength(quant.threadgroupBytes, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (outDim + nsg * nr0 - 1) / (nsg * nr0), height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: nsg, depth: 1))
    }

    /// Encode-form per-expert SwiGLU+route-weight: mid[e]=silu(gate[e])*up[e]*w[e].
    public func moeSwiGLUWeight(gate: GPUTensor, up: GPUTensor, weights: GPUTensor, mid: GPUTensor,
                                width: Int, rows: Int, clampValue: Float = 0) throws {
        let args = MetalRuntime.moeSwiGLUWeightArgs(width: width, rows: rows, clampValue: clampValue, midF16: false)
        let pso = try rt.pipeline("kernel_dsv4_moe_swiglu_weight")
        var nth = pso.maxTotalThreadsPerThreadgroup; if nth > 256 { nth = 256 }; if nth > width { nth = width }; if nth == 0 { nth = 1 }
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(gate.buffer, offset: gate.byteOffset, index: 1)
        e.setBuffer(up.buffer, offset: up.byteOffset, index: 2)
        e.setBuffer(mid.buffer, offset: mid.byteOffset, index: 3)
        e.setBuffer(weights.buffer, offset: weights.byteOffset, index: 4)
        e.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }

    /// Encode-form sum of 6 routed-expert output rows: out[c] = sum_{e<6} experts[e][c].
    public func moeSum6(experts: GPUTensor, out: GPUTensor, width: Int, tokens: Int = 1) throws {
        let args = MetalRuntime.moeSum6Args(width: width, tokens: tokens)
        let pso = try rt.pipeline("kernel_dsv4_moe_sum6_f32")
        var nth = pso.maxTotalThreadsPerThreadgroup; if nth > 256 { nth = 256 }; if nth > width { nth = width }; if nth == 0 { nth = 1 }
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(experts.buffer, offset: experts.byteOffset, index: 1)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 2)
        e.dispatchThreadgroups(MTLSize(width: tokens, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }

    /// 120-byte ds4_metal_args_mul_mv_id with explicit ne11/nb11 (per-expert act).
    static func mulMVIdArgsFull(nei0: Int, nei1: Int, nbi1: UInt64, ne00: Int, ne01: Int,
                                nb00: UInt64, nb01: UInt64, nb02: UInt64, ne10: Int, ne11: Int,
                                nb10: UInt64, nb11: UInt64, nb12: UInt64, ne0: Int, nb1: UInt64, nr0: Int32) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 120)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        i32(0, Int32(nei0)); i32(4, Int32(nei1)); u64(8, nbi1)
        i32(16, Int32(ne00)); i32(20, Int32(ne01)); i32(24, 1)
        u64(32, nb00); u64(40, nb01); u64(48, nb02)
        i32(56, Int32(ne10)); i32(60, Int32(ne11)); i32(64, 1); i32(68, 1)
        u64(72, nb10); u64(80, nb11); u64(88, nb12)
        i32(96, Int32(ne0)); i32(100, 1); u64(104, nb1); i32(112, nr0)
        return b
    }
}

