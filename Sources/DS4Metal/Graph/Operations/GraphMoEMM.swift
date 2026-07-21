import Foundation
import Metal

// Batched routed-FFN for the PREFILL (DS4_PREFILL_MM): instead of one matvec
// per token (the packed expert weights re-read from device memory once per
// token), the group's tokens run through the mul_mm_id matrix-matrix kernels —
// expert weights are read once per 64x32 tile for ALL the tokens routed to
// that expert. Three dispatches per layer group:
//
//   1. kernel_mul_mm_id_map0_ne20_6      token-major ids -> expert-major map
//                                        (htpe = tokens per expert, hids =
//                                        expert-major list of token*6+slot)
//   2. kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16
//                                        gate+up mm + swiglu + route weight
//                                        fused -> mid[token][slot][ffn] (f16)
//   3. kernel_mul_mm_id_q2_K_f16         down projection over the mid rows
//                                        -> down6[token][slot][nEmbd] (f32)
//
// then per token: moeSum6 + shared add + HC expand (the existing tail).
// NUMERICS: same math as the matvec path but different accumulation order
// (simdgroup 8x8 MMA) and f16 mid — outputs are NOT bit-identical to the
// matvec path, which is why DS4_PREFILL_MM is opt-in until validated A/B.
//
// Requirements enforced by the caller: gate/up iq2_xxs + down q2_K (the Flash
// shape), every token with exactly k DISTINCT selections (map0 encodes the
// slot as sum((ids[slot]==expert)*(slot+1)) — a duplicate expert id within one
// token corrupts the map), nEmbd % 256 == 0 and expertFfn % 256 == 0.
extension GraphContext {
    /// 48-byte ds4_metal_args_mul_mm_id_map0.
    static func mmIdMap0Args(nExperts: Int, nTok: Int, kPerTok: Int) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 48)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        i32(0, Int32(nExperts))                       // ne02 (experts in the union)
        i32(4, 0); i32(8, 0)                          // ne10/ne11 (unused by map0)
        u64(16, 0); u64(24, 0)                        // nb11/nb12 (unused by map0)
        i32(32, Int32(nTok))                          // ne21 (tokens; also the hids row stride)
        i32(36, Int32(kPerTok))                       // ne20 (selections per token)
        u64(40, UInt64(kPerTok * 4))                  // nb21 (ids row stride, bytes)
        return b
    }

    /// 96-byte ds4_metal_args_mul_mm_id.
    /// `actNb11`/`actNb12` are the src1 strides consumed via i11 = (id%ne20)%ne11
    /// and i12 = id/ne20: pass ne11=1, nb12=tokenStride for a token-major
    /// activation (gate/up), or ne11=k, nb11=slotStride, nb12=tokenStride for
    /// the per-(token,slot) mid rows (down).
    static func mmIdArgs(inDim: Int, rowBytes: Int, expertBytes: Int, nExperts: Int,
                         actNe11: Int, actNb10: Int, actNb11: Int, actNb12: Int,
                         kPerTok: Int, nTok: Int, outDim: Int, outNe1: Int) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 96)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func i16(_ off: Int, _ v: Int16) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<2 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        i32(0, Int32(inDim))                          // ne00
        i32(4, Int32(nExperts))                       // ne02
        u64(8, UInt64(rowBytes))                      // nb01
        u64(16, UInt64(expertBytes))                  // nb02
        u64(24, UInt64(expertBytes))                  // nb03 (i13 = 0)
        i32(32, Int32(actNe11))                       // ne11
        u64(40, UInt64(actNb10))                      // nb10 (element stride)
        u64(48, UInt64(actNb11))                      // nb11 (slot stride)
        u64(56, UInt64(actNb12))                      // nb12 (token stride)
        u64(64, 0)                                    // nb13 (i13 = 0)
        i32(72, Int32(kPerTok))                       // ne20
        i32(76, Int32(nTok))                          // ne21
        i32(80, Int32(outDim))                        // ne0
        i32(84, Int32(outNe1))                        // ne1 (slots per token in dst)
        i16(88, 1); i16(90, 1)                        // r2, r3
        return b
    }

    /// Build the expert-major map for a token group: ids [nTok x k] Int32
    /// (union-relative), htpe [nExperts] u32, hids [nExperts x nTok] i32.
    /// ONE threadgroup, one thread per expert.
    public func encodeMoEMap0(ids: GPUTensor, htpe: GPUTensor, hids: GPUTensor,
                              nTok: Int, kPerTok: Int, nExperts: Int) throws {
        precondition(kPerTok == 6, "map0 is instantiated for ne20=6")
        let args = Self.mmIdMap0Args(nExperts: nExperts, nTok: nTok, kPerTok: kPerTok)
        let pso = try rt.pipeline("kernel_mul_mm_id_map0_ne20_6")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(ids.buffer, offset: ids.byteOffset, index: 1)
        e.setBuffer(htpe.buffer, offset: htpe.byteOffset, index: 2)
        e.setBuffer(hids.buffer, offset: hids.byteOffset, index: 3)
        // Staging: ntg tokens per sweep, kPerTok u16 each.
        e.setThreadgroupMemoryLength(max(64, nExperts * kPerTok * 2), index: 0)
        e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nExperts, height: 1, depth: 1))
    }

    /// Fused gate+up mm + swiglu + route weight over the group's union of
    /// iq2_xxs experts -> mid [nTok x k x ffnDim] f16 (already weighted).
    /// `act` is the token-major activation matrix [nTok x inDim] f32 starting
    /// at act.byteOffset + actBase.
    public func encodeMMIdPairSwiGLUIQ2(gate: GPUTensor, up: GPUTensor,
                                        act: GPUTensor, actBase: Int,
                                        htpe: GPUTensor, hids: GPUTensor,
                                        mid: GPUTensor, weights: GPUTensor,
                                        nTok: Int, kPerTok: Int, nExperts: Int,
                                        inDim: Int, ffnDim: Int, clamp: Float) throws {
        precondition(inDim % 256 == 0 && inDim % 32 == 0)
        let rowBytes = (inDim / 256) * 66                       // block_iq2_xxs
        let expertBytes = rowBytes * ffnDim
        let args = Self.mmIdArgs(inDim: inDim, rowBytes: rowBytes, expertBytes: expertBytes,
                                 nExperts: nExperts,
                                 actNe11: 1, actNb10: 4, actNb11: 0, actNb12: inDim * 4,
                                 kPerTok: kPerTok, nTok: nTok, outDim: ffnDim, outNe1: kPerTok)
        // act args: only mid_row_stride (f16), weight_stride and clamp_value
        // are read by the mm kernel.
        let act48 = MetalRuntime.moeSwiGLUWeightArgs(width: ffnDim, rows: kPerTok,
                                                     clampValue: clamp, midF16: true)
        let pso = try rt.pipeline("kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        act48.withUnsafeBytes { e.setBytes($0.baseAddress!, length: act48.count, index: 1) }
        e.setBuffer(gate.buffer, offset: gate.byteOffset, index: 2)
        e.setBuffer(up.buffer, offset: up.byteOffset, index: 3)
        e.setBuffer(act.buffer, offset: act.byteOffset + actBase, index: 4)
        e.setBuffer(htpe.buffer, offset: htpe.byteOffset, index: 5)
        e.setBuffer(hids.buffer, offset: hids.byteOffset, index: 6)
        e.setBuffer(mid.buffer, offset: mid.byteOffset, index: 7)
        e.setBuffer(weights.buffer, offset: weights.byteOffset, index: 8)
        // Final staging needs temp_gate + temp_up = 2 * 64*32 floats = 16 KB
        // (the k-loop itself uses 6144) — allocate the max.
        e.setThreadgroupMemoryLength(16384, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (nTok + 31) / 32, height: (ffnDim + 63) / 64, depth: nExperts),
                               threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Dense Q8_0 matmul over a token-major activation matrix (the batched
    /// SHARED-expert FFN of the mm prefill path): out[nTok x outDim] f32 =
    /// act[nTok x inDim] x W(Q8_0)[outDim x inDim]. Same kernel_mul_mm_q8_0_f32
    /// the stage-A wrapper validates; weight byteOffset honored (mmap views).
    public func encodeMMDenseQ8(weight: GPUTensor, act: GPUTensor, actBase: Int,
                                out: GPUTensor, inDim: Int, outDim: Int, nTok: Int) throws {
        precondition(inDim % 32 == 0)
        let rowBytes = (inDim / 32) * 34
        let args = MetalRuntime.mulMMArgs(inDim: inDim, outDim: outDim, nTok: nTok,
                                          rowBytes: UInt64(rowBytes))
        let bcOut = (outDim % 64) != 0 || (nTok % 32) != 0
        let pso = try rt.mulMMPipeline("kernel_mul_mm_q8_0_f32", bcInp: false, bcOut: bcOut)
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(weight.buffer, offset: weight.byteOffset, index: 1)
        e.setBuffer(act.buffer, offset: act.byteOffset + actBase, index: 2)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 3)
        e.setThreadgroupMemoryLength(bcOut ? 8192 : 6144, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (nTok + 31) / 32, height: (outDim + 63) / 64, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Dense F16 matmul over a token-major activation matrix (the batched
    /// prefill's HC mixer and the IQ2 model's F16 router): out[nTok x outDim]
    /// f32 = act[nTok x inDim] f32 x W(F16)[outDim x inDim]. NOTE: the mm
    /// kernel stages the activation tile to f16 in threadgroup memory, so
    /// outputs are close but not bit-identical to the matvec path.
    public func encodeMMDenseF16(weight: GPUTensor, act: GPUTensor, actBase: Int,
                                 out: GPUTensor, inDim: Int, outDim: Int, nTok: Int) throws {
        let rowBytes = inDim * 2
        let args = MetalRuntime.mulMMArgs(inDim: inDim, outDim: outDim, nTok: nTok,
                                          rowBytes: UInt64(rowBytes))
        let bcInp = (inDim % 32) != 0
        let bcOut = (outDim % 64) != 0 || (nTok % 32) != 0
        let pso = try rt.mulMMPipeline("kernel_mul_mm_f16_f32", bcInp: bcInp, bcOut: bcOut)
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(weight.buffer, offset: weight.byteOffset, index: 1)
        e.setBuffer(act.buffer, offset: act.byteOffset + actBase, index: 2)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 3)
        e.setThreadgroupMemoryLength(bcOut ? 8192 : 6144, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (nTok + 31) / 32, height: (outDim + 63) / 64, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Dense Q4_K matmul over a token-major activation matrix — the batched
    /// prefill on resident Q4-requantized weights (DS4_DENSE_Q4/DS4_QKV_Q4).
    /// Same portable kernel family as encodeMMDenseQ8; requires inDim % 256.
    public func encodeMMDenseQ4K(weight: GPUTensor, act: GPUTensor, actBase: Int,
                                 out: GPUTensor, inDim: Int, outDim: Int, nTok: Int) throws {
        precondition(inDim % 256 == 0)
        let rowBytes = (inDim / 256) * 144
        let args = MetalRuntime.mulMMArgs(inDim: inDim, outDim: outDim, nTok: nTok,
                                          rowBytes: UInt64(rowBytes))
        let bcOut = (outDim % 64) != 0 || (nTok % 32) != 0
        let pso = try rt.mulMMPipeline("kernel_mul_mm_q4_K_f32", bcInp: false, bcOut: bcOut)
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(weight.buffer, offset: weight.byteOffset, index: 1)
        e.setBuffer(act.buffer, offset: act.byteOffset + actBase, index: 2)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 3)
        e.setThreadgroupMemoryLength(bcOut ? 8192 : 6144, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (nTok + 31) / 32, height: (outDim + 63) / 64, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Q4_K twin of encodeMMDenseQ8Strided (grouped attention output when
    /// output_a is Q4-requantized). Same ne0-as-row-stride contract.
    public func encodeMMDenseQ4KStrided(weight: GPUTensor, weightOffset: Int,
                                        act: GPUTensor, actBase: Int, actRowStride: Int,
                                        out: GPUTensor, outBase: Int, outRowStrideElems: Int,
                                        inDim: Int, outDim: Int, nTok: Int) throws {
        precondition(inDim % 256 == 0)
        precondition(outRowStrideElems == outDim || outDim % 64 == 0,
                     "strided mm output requires full 64-wide tiles")
        let rowBytes = (inDim / 256) * 144
        let args = MetalRuntime.mulMMArgsStrided(inDim: inDim, nTok: nTok,
                                                 rowBytes: UInt64(rowBytes),
                                                 actRowStride: UInt64(actRowStride),
                                                 outRowStrideElems: outRowStrideElems)
        let bcOut = (nTok % 32) != 0
        let pso = try rt.mulMMPipeline("kernel_mul_mm_q4_K_f32", bcInp: false, bcOut: bcOut)
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(weight.buffer, offset: weight.byteOffset + weightOffset, index: 1)
        e.setBuffer(act.buffer, offset: act.byteOffset + actBase, index: 2)
        e.setBuffer(out.buffer, offset: out.byteOffset + outBase, index: 3)
        e.setThreadgroupMemoryLength(bcOut ? 8192 : 6144, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (nTok + 31) / 32, height: (outDim + 63) / 64, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Dense Q8_0 matmul over STRIDED activation/output row views — the
    /// batched prefill's grouped low-rank attention output: group g reads
    /// activation columns [g·groupDim, (g+1)·groupDim) of each token row
    /// (actBase + row stride actRowStride) and writes output columns
    /// [g·outDim, (g+1)·outDim) (outBase + row stride outRowStrideElems). The
    /// mm kernel derives BOTH the dst row stride and the r0 tile clamp from
    /// ne0, so a strided view requires outDim % 64 == 0 (every dispatched
    /// tile fully populated) — guarded here.
    public func encodeMMDenseQ8Strided(weight: GPUTensor, weightOffset: Int,
                                       act: GPUTensor, actBase: Int, actRowStride: Int,
                                       out: GPUTensor, outBase: Int, outRowStrideElems: Int,
                                       inDim: Int, outDim: Int, nTok: Int) throws {
        precondition(inDim % 32 == 0)
        precondition(outRowStrideElems == outDim || outDim % 64 == 0,
                     "strided mm output requires full 64-wide tiles")
        let rowBytes = (inDim / 32) * 34
        let args = MetalRuntime.mulMMArgsStrided(inDim: inDim, nTok: nTok,
                                                 rowBytes: UInt64(rowBytes),
                                                 actRowStride: UInt64(actRowStride),
                                                 outRowStrideElems: outRowStrideElems)
        let bcOut = (nTok % 32) != 0
        let pso = try rt.mulMMPipeline("kernel_mul_mm_q8_0_f32", bcInp: false, bcOut: bcOut)
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(weight.buffer, offset: weight.byteOffset + weightOffset, index: 1)
        e.setBuffer(act.buffer, offset: act.byteOffset + actBase, index: 2)
        e.setBuffer(out.buffer, offset: out.byteOffset + outBase, index: 3)
        e.setThreadgroupMemoryLength(bcOut ? 8192 : 6144, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (nTok + 31) / 32, height: (outDim + 63) / 64, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Down projection over the (already weighted, f16) mid rows via
    /// kernel_mul_mm_id_q2_K_f16 -> down6 [nTok x k x outDim] f32.
    public func encodeMMIdDownQ2K(down: GPUTensor, mid: GPUTensor,
                                  htpe: GPUTensor, hids: GPUTensor, out: GPUTensor,
                                  nTok: Int, kPerTok: Int, nExperts: Int,
                                  ffnDim: Int, outDim: Int) throws {
        precondition(ffnDim % 256 == 0 && ffnDim % 32 == 0)
        let rowBytes = (ffnDim / 256) * 84                      // block_q2_K
        let expertBytes = rowBytes * outDim
        let args = Self.mmIdArgs(inDim: ffnDim, rowBytes: rowBytes, expertBytes: expertBytes,
                                 nExperts: nExperts,
                                 actNe11: kPerTok, actNb10: 2,
                                 actNb11: ffnDim * 2, actNb12: kPerTok * ffnDim * 2,
                                 kPerTok: kPerTok, nTok: nTok, outDim: outDim, outNe1: kPerTok)
        // FC_mul_mm_bc_inp=false: ffnDim % 32 == 0 (guarded above); bc_out is
        // not referenced by the id kernel (extra constants are ignored).
        let pso = try rt.mulMMPipeline("kernel_mul_mm_id_q2_K_f16", bcInp: false, bcOut: false)
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(down.buffer, offset: down.byteOffset, index: 1)
        e.setBuffer(mid.buffer, offset: mid.byteOffset, index: 2)
        e.setBuffer(htpe.buffer, offset: htpe.byteOffset, index: 3)
        e.setBuffer(hids.buffer, offset: hids.byteOffset, index: 4)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 5)
        // Loop phase: sa 4096 + sb (32x32 half) 2048; final staging re-uses the
        // region as 32x64 floats = 8192 — allocate the max.
        e.setThreadgroupMemoryLength(8192, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (nTok + 31) / 32, height: (outDim + 63) / 64, depth: nExperts),
                               threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
}
