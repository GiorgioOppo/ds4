import Foundation
import Metal

/// Routed-expert quantization formats supported by the MoE matvec. All share the
/// `kernel_mul_mv_id<...>` wrapper; they differ only in GGUF block byte-size and
/// the rows-per-threadgroup (nr0). Maps from GGUF type codes (q4_K=12, q2_K=10,
/// iq2_xxs=16).
public enum MoEQuant: Sendable {
    case q4_K, q2_K, iq2_xxs
    public var kernel: String {
        switch self {
        case .q4_K:    return "kernel_mul_mv_id_q4_K_f32"
        case .q2_K:    return "kernel_mul_mv_id_q2_K_f32"
        case .iq2_xxs: return "kernel_mul_mv_id_iq2_xxs_f32"
        }
    }
    /// GGUF block bytes per 256 elements.
    public var blockBytes: Int {
        switch self { case .q4_K: return 144; case .q2_K: return 84; case .iq2_xxs: return 66 }
    }
    /// N_R0_* — rows per threadgroup (q4_K=2; q2_K/iq2_xxs=4).
    public var nr0: Int { self == .q4_K ? 2 : 4 }
    /// Threadgroup memory the kernel needs. iq2_xxs cooperatively loads the
    /// codebook into shared memory: svalues = uint64 grid[256] (2048 B) + ssigns
    /// = uint8[128] (128 B) = 2176 B. q2_K/q4_K only use the 256 B reduction
    /// scratch. Allocating only 256 B for iq2_xxs causes out-of-bounds threadgroup
    /// writes → garbage grid → wrong matvec on every gate/up expert.
    public var threadgroupBytes: Int { self == .iq2_xxs ? (256 * 8 + 128) : 256 }
    public static func from(ggufType: UInt32) -> MoEQuant? {
        switch ggufType { case 12: return .q4_K; case 10: return .q2_K; case 16: return .iq2_xxs; default: return nil }
    }
}

// Stage C: graph composition. Composes the validated Stage-B tensor-ops on
// GPUTensors into the DSV4 graph fragments. First fragments: token embedding
// (get_rows -> HC repeat) and the output head (final RMSNorm -> vocab matmul).
// Each fragment encodes into a GraphContext (one command buffer) and is validated
// against a CPU reference; the per-layer decode/prefill body builds on these.

extension GraphContext {
    /// Output head: logits = matmul_f32(outWeight, rmsNorm(hidden [* normWeight])).
    /// hidden: inDim, outWeight: vocab x inDim (F32 row-major), logits: vocab.
    public func outputHead(hidden: GPUTensor, normWeight: GPUTensor?, outWeight: GPUTensor,
                           normed: GPUTensor, logits: GPUTensor,
                           inDim: Int, vocab: Int, eps: Float) throws {
        try rmsNorm(hidden, weight: normWeight, out: normed, rows: 1, n: inDim, eps: eps)
        try matmulF32(weight: outWeight, x: normed, out: logits, inDim: inDim, outDim: vocab)
    }

    /// Token embedding to HC block: gather row `token` from an F16 table
    /// (vocab x nEmbd) into `embd`, then replicate across `nHC` HC streams into
    /// `hc` (nHC x nEmbd). Encodes get_rows_f16 + repeat_f32.
    public func embedTokenHC(table: GPUTensor, token: Int, embd: GPUTensor, hc: GPUTensor,
                             nEmbd: Int, nVocab: Int, nHC: Int) throws {
        try getRowsF16(table: table, id: token, out: embd, nEmbd: nEmbd, nVocab: nVocab)
        try repeatHC(src: embd, out: hc, nEmbd: nEmbd, nTokens: 1, nHC: nHC)
    }

    /// Shared-expert FFN block (pre-norm + SwiGLU MLP + residual), one token:
    ///   normed = rmsNorm(x [* normWeight])
    ///   mid    = swiglu(matmulQ8(gateW, normed), matmulQ8(upW, normed))
    ///   out    = x + matmulQ8(downW, mid)
    /// gateW/upW: ffnDim x inDim (Q8_0); downW: inDim x ffnDim (Q8_0).
    /// Scratch tensors (normed[inDim], gate[ffnDim], up[ffnDim], mid[ffnDim],
    /// down[inDim]) are caller-provided so the graph can reuse them across layers.
    public func ffnBlock(x: GPUTensor, normWeight: GPUTensor?,
                         gateW: GPUTensor, upW: GPUTensor, downW: GPUTensor,
                         normed: GPUTensor, gate: GPUTensor, up: GPUTensor, mid: GPUTensor,
                         down: GPUTensor, out: GPUTensor,
                         inDim: Int, ffnDim: Int, eps: Float) throws {
        try rmsNorm(x, weight: normWeight, out: normed, rows: 1, n: inDim, eps: eps)
        try matmulQ8_0(weight: gateW, x: normed, out: gate, inDim: inDim, outDim: ffnDim)
        try matmulQ8_0(weight: upW, x: normed, out: up, inDim: inDim, outDim: ffnDim)
        try swiglu(gate: gate, up: up, out: mid, n: ffnDim)
        try matmulQ8_0(weight: downW, x: mid, out: down, inDim: ffnDim, outDim: inDim)
        try add(x, down, out: out, width: inDim)
    }

    /// Decode FlashAttention core (MLA dk=dv=512), encode form. Converts the F32
    /// latent `kvF32` (nKeys x 512) to F16 (`kvF16` scratch) via cpy, then the
    /// vec + reduce kernels, all into the shared command buffer. K==V==latent,
    /// no mask. Requires nKeys % 32 == 0. `mask` must be a zeroed nKeys*2 byte
    /// tensor; sinks/pad/tmp are scratch sized per the C dispatch.
    public func flashAttnCore(q: GPUTensor, kvF32: GPUTensor, kvF16: GPUTensor,
                              mask: GPUTensor, sinks: GPUTensor, pad: GPUTensor, tmp: GPUTensor,
                              heads: GPUTensor, nHead: Int, nKeys: Int, rawStartRow: Int = 0,
                              hasSinks: Bool = false,
                              comp: GPUTensor? = nil, nComp: Int = 0) throws {
        let headDim = 512
        let ncpsg = 32, nwg = 32
        // Two-span attention: raw SWA rows (nKeys) followed by compressed rows (nComp),
        // contiguous in kvF16. The flash kernel then attends over the union.
        let total = nKeys + nComp
        let kvpad = (total % ncpsg) != 0
        var nsg = 1
        while 2 * nwg * nsg * ncpsg < total && nsg < 4 { nsg *= 2 }
        let e = encoder

        // 1) cpy F32 -> F16: raw rows (kvF32 -> kvF16[0..]) then comp rows (comp -> kvF16[nKeys..]).
        let cpyPso = try rt.pipeline("kernel_cpy_f32_f16")
        func cpyF32toF16(_ src: GPUTensor, srcOff: Int, dstOff: Int, count: Int) {
            let a = MetalRuntime.cpyArgs(n: count, srcElem: 4, dstElem: 2)
            var cnth = 32; let cmaxT = cpyPso.maxTotalThreadsPerThreadgroup
            while cnth < count && cnth < cmaxT { cnth *= 2 }
            if cnth > cmaxT { cnth = cmaxT }; if cnth > count { cnth = count }; if cnth == 0 { cnth = 1 }
            e.setComputePipelineState(cpyPso)
            a.withUnsafeBytes { e.setBytes($0.baseAddress!, length: a.count, index: 0) }
            e.setBuffer(src.buffer, offset: srcOff, index: 1)
            e.setBuffer(kvF16.buffer, offset: dstOff, index: 2)
            e.dispatchThreadgroups(MTLSize(width: (count + cnth - 1) / cnth, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: cnth, height: 1, depth: 1))
        }
        // Raw span: only the SWA window starting at `rawStartRow` (rows hold their
        // absolute-RoPE'd values, so a shifted span is exactly the C slid cache).
        // With a ring-buffer raw cache (count < contextSize) the window can wrap;
        // copy it in up to two segments so kvF16 holds it in chronological order.
        // With the full cache it never wraps -> a single copy identical to before.
        let rawRows = kvF32.count / headDim
        let physStart = ((rawStartRow % rawRows) + rawRows) % rawRows
        if physStart + nKeys <= rawRows {
            cpyF32toF16(kvF32, srcOff: physStart * headDim * 4, dstOff: 0, count: nKeys * headDim)
        } else {
            let seg1 = rawRows - physStart                       // older rows at the physical tail
            cpyF32toF16(kvF32, srcOff: physStart * headDim * 4, dstOff: 0, count: seg1 * headDim)
            cpyF32toF16(kvF32, srcOff: 0, dstOff: seg1 * headDim * 2, count: (nKeys - seg1) * headDim)
        }
        if let comp = comp, nComp > 0 {
            cpyF32toF16(comp, srcOff: comp.byteOffset, dstOff: nKeys * headDim * 2, count: nComp * headDim)
        }

        // 1b) pad the partial last block when total % 32 != 0 (K==V==kvF16, mask all-zero)
        if kvpad {
            let pArgs = MetalRuntime.flashPadArgs(nKeys: total, headDim: headDim)
            let padPso = try rt.flashPadPipeline(ncpsg: Int32(ncpsg))
            e.setComputePipelineState(padPso)
            pArgs.withUnsafeBytes { e.setBytes($0.baseAddress!, length: pArgs.count, index: 0) }
            e.setBuffer(kvF16.buffer, offset: 0, index: 1)
            e.setBuffer(kvF16.buffer, offset: 0, index: 2)
            e.setBuffer(mask.buffer, offset: 0, index: 3)
            e.setBuffer(pad.buffer, offset: 0, index: 4)
            e.dispatchThreadgroups(MTLSize(width: ncpsg, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }

        // 2) flash vec
        let scale = 1.0 / Float(headDim).squareRoot()
        let vargs = MetalRuntime.flashVecArgs(nHead: nHead, nKeys: total, headDim: headDim, scale: scale)
        let vec = try rt.flashVecPipeline(nsg: Int32(nsg), nwg: Int32(nwg), hasSinks: hasSinks, hasKvpad: kvpad)
        let alignUp = { (v: Int, a: Int) in (v + a - 1) & ~(a - 1) }
        let sharedElems = (alignUp(headDim, 128) + 4 * ncpsg + 2 * alignUp(headDim, 128)) * nsg
        let sharedBytes = alignUp(sharedElems * 2, 16)
        e.setComputePipelineState(vec)
        vargs.withUnsafeBytes { e.setBytes($0.baseAddress!, length: vargs.count, index: 0) }
        e.setBuffer(q.buffer, offset: 0, index: 1)
        e.setBuffer(kvF16.buffer, offset: 0, index: 2)
        e.setBuffer(kvF16.buffer, offset: 0, index: 3)
        e.setBuffer(mask.buffer, offset: 0, index: 4)
        e.setBuffer(sinks.buffer, offset: 0, index: 5)
        e.setBuffer(pad.buffer, offset: 0, index: 6)
        e.setBuffer(tmp.buffer, offset: 0, index: 7)
        e.setThreadgroupMemoryLength(sharedBytes, index: 0)
        e.dispatchThreadgroups(MTLSize(width: 1, height: nHead, depth: nwg),
                               threadsPerThreadgroup: MTLSize(width: 32, height: nsg, depth: 1))

        // 3) reduce
        let reduce = try rt.flashReducePipeline(dv: Int32(headDim), nwg: Int32(nwg))
        var reduceArgs = Int32(nHead)
        e.setComputePipelineState(reduce)
        e.setBytes(&reduceArgs, length: 4, index: 0)
        e.setBuffer(tmp.buffer, offset: 0, index: 1)
        e.setBuffer(heads.buffer, offset: 0, index: 2)
        e.dispatchThreadgroups(MTLSize(width: nHead, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32 * nwg, height: 1, depth: 1))
    }

    /// Encode-form HC expand4 (per-layer 4-stream HC update). See MetalHyperConnections.
    /// out[t][k][d] = (blockOut+blockAdd?)[t][d]*post[t][k] + sum_j comb[t][j][k]*residual[t][j][d].
    public func hcExpand4(blockOut: GPUTensor, residual: GPUTensor, post: GPUTensor, comb: GPUTensor,
                          blockAdd: GPUTensor?, out: GPUTensor, nEmbd: Int, nTokens: Int,
                          postByteOffset: Int = 0, combByteOffset: Int = 0) throws {
        let nHC = 4
        let hasAdd = blockAdd != nil
        let args = MetalRuntime.hcExpandArgs(nEmbd: nEmbd, nHC: nHC, nTokens: nTokens, hasAdd: hasAdd)
        let pso = try rt.pipeline("kernel_dsv4_hc_expand4")
        let nElem = nEmbd * nTokens
        let nth = min(256, max(1, nElem))
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(blockOut.buffer, offset: 0, index: 1)
        e.setBuffer(residual.buffer, offset: 0, index: 2)
        e.setBuffer(post.buffer, offset: postByteOffset, index: 3)
        e.setBuffer(comb.buffer, offset: combByteOffset, index: 4)
        e.setBuffer((blockAdd ?? blockOut).buffer, offset: 0, index: 5)
        e.setBuffer(out.buffer, offset: 0, index: 6)
        e.dispatchThreadgroups(MTLSize(width: (nElem + nth - 1) / nth, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }

    /// Encode-form HC mixer split (HC=4, Sinkhorn). mix: nRows x 24, scale [3],
    /// base [24] -> out nRows x 24 ([0:4] pre, [4:8] post, [8:24] comb).
    public func hcSplitSinkhorn(mix: GPUTensor, scale: GPUTensor, base: GPUTensor, out: GPUTensor,
                                nRows: Int, sinkhornIters: Int, eps: Float) throws {
        let nHc = 4, mixHc = 2 * nHc + nHc * nHc
        let args = MetalRuntime.hcSplitArgs(nHc: nHc, sinkhornIters: sinkhornIters, nRows: nRows, mixHc: mixHc, eps: eps)
        let pso = try rt.pipeline("kernel_dsv4_hc_split_sinkhorn")
        let nth = min(256, max(1, nRows))
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(mix.buffer, offset: 0, index: 1)
        e.setBuffer(scale.buffer, offset: 0, index: 2)
        e.setBuffer(base.buffer, offset: 0, index: 3)
        e.setBuffer(out.buffer, offset: 0, index: 4)
        e.dispatchThreadgroups(MTLSize(width: (nRows + nth - 1) / nth, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }

    /// Encode-form HC weighted sum (collapse n_hc streams): out[t][d] = sum_h x[t][h][d]*w[t][h].
    public func hcWeightedSum(x: GPUTensor, weights: GPUTensor, out: GPUTensor,
                              nEmbd: Int, nHC: Int, nTokens: Int) throws {
        let args = MetalRuntime.hcWeightedSumArgs(nEmbd: nEmbd, nHC: nHC, nTokens: nTokens)
        let pso = try rt.pipeline("kernel_dsv4_hc_weighted_sum")
        let nElem = nEmbd * nTokens
        let nth = min(256, max(1, nElem))
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(x.buffer, offset: 0, index: 1)
        e.setBuffer(weights.buffer, offset: 0, index: 2)
        e.setBuffer(out.buffer, offset: 0, index: 3)
        e.dispatchThreadgroups(MTLSize(width: (nElem + nth - 1) / nth, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }

    /// Encode-form partial RoPE (tail rotation, in-place) over [nTok][nHead][headDim].
    public func ropeTail(x: GPUTensor, nTok: Int, nHead: Int, headDim: Int, nRot: Int, nCtxOrig: Int,
                         freqBase: Float, freqScale: Float, extFactor: Float, attnFactor: Float,
                         betaFast: Float, betaSlow: Float, pos0: Int, posStep: Int, inverse: Bool = false) throws {
        var positions = [Int32](repeating: 0, count: nTok)
        for t in 0..<nTok { positions[t] = Int32(pos0 + t * posStep) }
        guard let posbuf = rt.device.makeBuffer(bytes: positions, length: nTok * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let args = MetalRuntime.ropeArgs(nTok: nTok, nHead: nHead, headDim: headDim, nRot: nRot,
                                         nCtxOrig: nCtxOrig, inverse: inverse, freqBase: freqBase,
                                         freqScale: freqScale, extFactor: extFactor, attnFactor: attnFactor,
                                         betaFast: betaFast, betaSlow: betaSlow)
        let pso = try rt.pipeline("kernel_dsv4_rope_tail_f32")
        let nth = min(headDim, 256)
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(x.buffer, offset: 0, index: 1)
        e.setBuffer(posbuf, offset: 0, index: 2)
        e.setBuffer(x.buffer, offset: 0, index: 3)
        e.setBuffer(x.buffer, offset: 0, index: 4)
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
        e.setBuffer(kv.buffer, offset: 0, index: 1)
        e.setBuffer(rawCache.buffer, offset: 0, index: 2)
        e.setThreadgroupMemoryLength(64 * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    /// Encode-form elementwise unary (single row). See MetalRuntime.UnaryOp.
    public func unary(_ x: GPUTensor, op: MetalRuntime.UnaryOp, out: GPUTensor, width: Int,
                      minV: Float = 0, maxV: Float = 0, scale: Float = 0, bias: Float = 0) throws {
        let args = MetalRuntime.unaryArgs(width: width, rows: 1, minV: minV, maxV: maxV, scale: scale, bias: bias, val: 0)
        let pso = try rt.unaryPipeline(op: op)
        var nthMax = pso.maxTotalThreadsPerThreadgroup; if nthMax > 256 { nthMax = 256 }
        var nth = width; if nth > nthMax { nth = nthMax }; if nth == 0 { nth = 1 }
        let nk0 = (width + nth - 1) / nth
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(x.buffer, offset: 0, index: 1)
        e.setBuffer(out.buffer, offset: 0, index: 2)
        e.dispatchThreadgroups(MTLSize(width: nk0, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }

    /// Encode-form router top-6 select over 256 experts (no bias/hash). `selected`
    /// is a 6-int32 GPUTensor.
    public func routerFinalizeTop6(probs: GPUTensor, selected: GPUTensor, bias: GPUTensor? = nil) throws {
        var args = [UInt8](repeating: 0, count: 20)
        func u32(_ off: Int, _ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { args[off+k] = $0[k] } } }
        u32(0, bias != nil ? 1 : 0); u32(4, 0); u32(8, 0); u32(12, 0); u32(16, 1)
        guard let hbuf = rt.device.makeBuffer(length: 4, options: .storageModeShared),
              let tbuf = rt.device.makeBuffer(length: 4, options: .storageModeShared) else { throw MetalError.bufferAlloc }
        let pso = try rt.pipeline("kernel_dsv4_router_finalize_one")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: 20, index: 0) }
        e.setBuffer(probs.buffer, offset: 0, index: 1)
        e.setBuffer((bias ?? probs).buffer, offset: 0, index: 2)
        e.setBuffer(hbuf, offset: 0, index: 3)
        e.setBuffer(tbuf, offset: 0, index: 4)
        e.setBuffer(selected.buffer, offset: 0, index: 5)
        e.setThreadgroupMemoryLength(256 * 4 + 256 * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Encode-form router weight normalization: w[i] = probs[sel[i]]/sum * 1.5.
    public func routerWeights(probs: GPUTensor, selected: GPUTensor, weights: GPUTensor) throws {
        let pso = try rt.pipeline("kernel_dsv4_router_weights_one")
        let e = encoder
        e.setComputePipelineState(pso)
        e.setBuffer(probs.buffer, offset: 0, index: 0)
        e.setBuffer(selected.buffer, offset: 0, index: 1)
        e.setBuffer(weights.buffer, offset: 0, index: 2)
        e.dispatchThreads(MTLSize(width: 6, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 6, height: 1, depth: 1))
    }

    /// Encode-form routed MoE matvec over typed experts (Q4_K/Q2_K/IQ2_XXS).
    /// K = ids.count slots. perExpertAct=false: all experts use the same
    /// `activation` row (gate/up); true: expert slot e uses activation row e (down).
    /// `out` is K x outDim. inDim multiple of 256. All three quants share the same
    /// `kernel_mul_mv_id<...>` wrapper (same 120-byte args, same dispatch) — only
    /// the block byte-size (nb00) and nr0 differ.
    public func moeMatvecID(_ quant: MoEQuant, experts: GPUTensor, ids: GPUTensor, activation: GPUTensor,
                            out: GPUTensor, k: Int, inDim: Int, outDim: Int, perExpertAct: Bool) throws {
        precondition(inDim % 256 == 0)
        let nsg = 4, nr0 = quant.nr0
        let blockBytes = quant.blockBytes
        let rowBytes = (inDim / 256) * blockBytes
        let expertBytes = rowBytes * outDim
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
        e.setBuffer(activation.buffer, offset: 0, index: 2)
        e.setBuffer(out.buffer, offset: 0, index: 3)
        e.setBuffer(ids.buffer, offset: 0, index: 4)
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
                              k: Int, inDim: Int, outDim: Int, clamp: Float) throws {
        let kernel: String
        switch quant {
        case .iq2_xxs: kernel = "kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32"
        case .q4_K:    kernel = "kernel_mul_mv_id_q4_K_pair_swiglu_f32"
        case .q2_K:    throw MetalError.missingKernel("no q2_K pair_swiglu kernel")
        }
        let nsg = 4, nr0 = quant.nr0
        let rowBytes = (inDim / 256) * quant.blockBytes
        let args = Self.mulMVIdArgsFull(nei0: k, nei1: 1, nbi1: UInt64(k * 4), ne00: inDim, ne01: outDim,
                                        nb00: UInt64(quant.blockBytes), nb01: UInt64(rowBytes),
                                        nb02: UInt64(rowBytes * outDim), ne10: inDim, ne11: 1,
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
        e.setBuffer(activation.buffer, offset: 0, index: 4)
        e.setBuffer(gateScratch.buffer, offset: 0, index: 5)
        e.setBuffer(upScratch.buffer, offset: 0, index: 6)
        e.setBuffer(mid.buffer, offset: 0, index: 7)
        e.setBuffer(ids.buffer, offset: 0, index: 8)
        e.setBuffer(weights.buffer, offset: 0, index: 9)
        e.setThreadgroupMemoryLength(quant.threadgroupBytes, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (outDim + nsg * nr0 - 1) / (nsg * nr0), height: 1, depth: k),
                               threadsPerThreadgroup: MTLSize(width: 32, height: nsg, depth: 1))
    }

    /// FUSED routed down-projection + sum over the 6 selected experts (1 dispatch
    /// instead of 2): out[outDim] = Σ_e down_e · mid[e]. The kernel hardcodes 6
    /// expert slots, so it requires k == 6. Only q2_K/q4_K exist.
    public func moeDownSum6(_ quant: MoEQuant, experts: GPUTensor, ids: GPUTensor,
                            mid: GPUTensor, out: GPUTensor, inDim: Int, outDim: Int) throws {
        let kernel: String
        switch quant {
        case .q2_K:    kernel = "kernel_mul_mv_id_q2_K_sum6_f32"
        case .q4_K:    kernel = "kernel_mul_mv_id_q4_K_sum6_f32"
        case .iq2_xxs: throw MetalError.missingKernel("no iq2_xxs down_sum6 kernel")
        }
        let nsg = 4, nr0 = quant.nr0
        let rowBytes = (inDim / 256) * quant.blockBytes
        let args = Self.mulMVIdArgsFull(nei0: 6, nei1: 1, nbi1: 6 * 4, ne00: inDim, ne01: outDim,
                                        nb00: UInt64(quant.blockBytes), nb01: UInt64(rowBytes),
                                        nb02: UInt64(rowBytes * outDim), ne10: inDim, ne11: 6,
                                        nb10: 4, nb11: UInt64(inDim * 4), nb12: UInt64(6 * inDim * 4),
                                        ne0: outDim, nb1: UInt64(outDim * 4), nr0: Int32(nr0))
        let pso = try rt.mulMVPipeline(kernel, nsg: Int16(nsg))
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(experts.buffer, offset: experts.byteOffset, index: 1)
        e.setBuffer(mid.buffer, offset: 0, index: 2)
        e.setBuffer(out.buffer, offset: 0, index: 3)
        e.setBuffer(ids.buffer, offset: 0, index: 4)
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
        e.setBuffer(gate.buffer, offset: 0, index: 1)
        e.setBuffer(up.buffer, offset: 0, index: 2)
        e.setBuffer(mid.buffer, offset: 0, index: 3)
        e.setBuffer(weights.buffer, offset: 0, index: 4)
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
        e.setBuffer(experts.buffer, offset: 0, index: 1)
        e.setBuffer(out.buffer, offset: 0, index: 2)
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

    /// Encode-form grouped low-rank attention output (kernel_dsv4_attn_out_low_q8_0_f32):
    /// low[g][rank] = outputA[g] (Q8) * heads[g] for each of nGroups groups.
    public func attnOutLowQ8(outputA: GPUTensor, heads: GPUTensor, low: GPUTensor,
                             nGroups: Int, groupDim: Int, rank: Int) throws {
        precondition(groupDim % 32 == 0)
        let rowABytes = (groupDim / 32) * 34
        let nsg: Int16 = 4, nr0 = 2
        let args = MetalRuntime.attnLowArgs(nGroups: nGroups, groupDim: groupDim, rank: rank, rowABytes: rowABytes)
        let pso = try rt.mulMVPipeline("kernel_dsv4_attn_out_low_q8_0_f32", nsg: nsg)
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(outputA.buffer, offset: outputA.byteOffset, index: 1)   // no-copy mmap weight
        e.setBuffer(heads.buffer, offset: 0, index: 2)
        e.setBuffer(low.buffer, offset: 0, index: 3)
        e.setThreadgroupMemoryLength(32 * 2 * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (rank + nr0 - 1) / nr0, height: 1, depth: nGroups),
                               threadsPerThreadgroup: MTLSize(width: 32, height: Int(nsg), depth: 1))
    }

    /// Output-head HC collapse weights: w[h] = sigmoid(pre[h]*scaleScalar + base[h]) + eps.
    /// Composition of unary(scale) + add(base) + unary(sigmoid) + unary(scale,+eps).
    public func outputHCWeights(pre: GPUTensor, scaleScalar: Float, base: GPUTensor,
                                weights: GPUTensor, tmp: GPUTensor, nHC: Int, eps: Float) throws {
        try unary(pre, op: .scale, out: tmp, width: nHC, scale: scaleScalar, bias: 0)
        try add(tmp, base, out: tmp, width: nHC)
        try unary(tmp, op: .sigmoid, out: weights, width: nHC)
        try unary(weights, op: .scale, out: weights, width: nHC, scale: 1, bias: eps)
    }

    /// Scratch sizes for flashAttnCore (bytes), given nHead/nKeys (headDim=512).
    public static func flashScratchBytes(nHead: Int, nKeys: Int) -> (kvF16: Int, mask: Int, sinks: Int, pad: Int, tmp: Int) {
        let headDim = 512, ncpsg = 32, nwg = 32
        return (nKeys * headDim * 2, nKeys * 2, max(4, nHead * 4),
                2 * ncpsg * headDim * 2 + ncpsg * 2,
                nHead * headDim * nwg * 4 + nHead * (2 * nwg) * 4)
    }

    // encode-form of get_rows_f16 (single row gather, F16 table -> F32 row)
    func getRowsF16(table: GPUTensor, id: Int, out: GPUTensor, nEmbd: Int, nVocab: Int) throws {
        let srcRowBytes = UInt64(nEmbd) * 2
        let args = MetalRuntime.getRowsArgs(ne00t: nEmbd, ne00: nEmbd, nb01: srcRowBytes,
                                            nb02: UInt64(nVocab) * srcRowBytes, ne10: 1, nb1: UInt64(nEmbd) * 4)
        let pso = try rt.pipeline("kernel_get_rows_f16")
        var nth = pso.maxTotalThreadsPerThreadgroup
        if nth > nEmbd { nth = nEmbd }; if nth == 0 { nth = 1 }
        var idv = Int32(id)
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(table.buffer, offset: table.byteOffset, index: 1)   // no-copy mmap embed table
        e.setBytes(&idv, length: 4, index: 2)
        e.setBuffer(out.buffer, offset: 0, index: 3)
        e.dispatchThreadgroups(MTLSize(width: (nEmbd + nth - 1) / nth, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }

    // encode-form of repeat (HC expand: src nTokens*nEmbd -> nTokens*nHC*nEmbd)
    func repeatHC(src: GPUTensor, out: GPUTensor, nEmbd: Int, nTokens: Int, nHC: Int) throws {
        let args = MetalRuntime.repeatArgs(nEmbd: nEmbd, nTokens: nTokens, nHC: nHC)
        let pso = try rt.pipeline("kernel_repeat_f32")
        var nth = nEmbd; let maxT = pso.maxTotalThreadsPerThreadgroup
        if nth > maxT { nth = maxT }; if nth == 0 { nth = 1 }
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(src.buffer, offset: 0, index: 1)
        e.setBuffer(out.buffer, offset: 0, index: 2)
        e.dispatchThreadgroups(MTLSize(width: nHC, height: nTokens, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }
}
