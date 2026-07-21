import Foundation
import Metal

// Batched prefill FlashAttention (MLA dk=dv=512): ONE dispatch consumes nQ
// query rows against the union KV span (raw SWA window + NSA compressed rows)
// with a per-query causal/window/comp mask — the C prefill's layer-major
// batched attention (ds4_gpu_encode_flash_attention_prefill_raw_heads_nonvec)
// instead of nQ single-query vec dispatches. Same arithmetic as the decode
// path over the same visible key set; only the accumulation order differs
// (simdgroup MMA blocks vs the vec kernel's per-chunk loop).

extension GraphContext {
    /// Fill the batched-prefill attention mask CPU-side (the buffer is read
    /// only by the command buffer this run commits). Row r belongs to absolute
    /// position `posFirst + r`; the key axis is `rawSpan` raw rows starting at
    /// absolute position `rawStart`, followed by `nComp` compressed rows.
    /// Visibility (identical to the per-token decode path):
    ///  - raw row at absolute pos pc: pc <= p AND p - pc < window (SWA);
    ///  - comp row c: c < nCompVis[r] (rows emitted up to and including token r).
    public static func fillPrefillAttnMask(_ mask: UnsafeMutablePointer<UInt16>,
                                           nQ: Int, posFirst: Int,
                                           rawStart: Int, rawSpan: Int, window: Int,
                                           nCompVis: [Int], nComp: Int) {
        let nKv = rawSpan + nComp
        let negInf: UInt16 = 0xFC00   // f16 -inf (the C fill's neg_inf_half)
        memset(mask, 0, nQ * nKv * 2)
        for r in 0..<nQ {
            let p = posFirst + r
            let row = mask + r * nKv
            // Raw span: -inf outside [lo, hi] (window start .. own position).
            let lo = max(0, max(0, p + 1 - window) - rawStart)
            let hi = p - rawStart
            for c in 0..<min(lo, rawSpan) { row[c] = negInf }
            if hi + 1 < rawSpan { for c in (hi + 1)..<rawSpan { row[c] = negInf } }
            // Comp rows emitted AFTER this token stay invisible to it.
            let vis = min(nCompVis[r], nComp)
            if vis < nComp { for c in vis..<nComp { row[rawSpan + c] = negInf } }
        }
    }

    /// Scratch sizes (bytes) for flashAttnPrefill given the run capacity.
    public static func flashPrefillScratchBytes(nHead: Int, nQ: Int, maxKv: Int)
        -> (q: Int, heads: Int, mask: Int, kvF16: Int, pad: Int, blk: Int) {
        let headDim = 512, ncpsg = 64, nqptg = 8
        let qBytes = nQ * nHead * headDim * 4
        let blk = ((maxKv + ncpsg - 1) / ncpsg) * ((nQ + nqptg - 1) / nqptg)
        return (qBytes, qBytes, nQ * maxKv * 2, maxKv * headDim * 2,
                ncpsg * (2 * headDim * 2 + nQ * 2), (blk + 31) & ~31)
    }

    /// Encode the batched prefill attention: stage the raw span (ring-aware)
    /// and comp rows to F16, pad the final partial 64-block, build the mask
    /// block map, then run the nqptg=8 MMA FlashAttention. `q` and `heads` are
    /// row-major [nQ][nHead][512] F32; `mask` must already be CPU-filled
    /// (fillPrefillAttnMask) for exactly (nQ, rawSpan + nComp).
    public func flashAttnPrefill(q: GPUTensor, kvF32: GPUTensor, kvF16: GPUTensor,
                                 mask: GPUTensor, sinks: GPUTensor, pad: GPUTensor,
                                 blk: GPUTensor, heads: GPUTensor,
                                 nHead: Int, nQ: Int, rawSpan: Int, rawStartRow: Int,
                                 comp: GPUTensor? = nil, nComp: Int = 0) throws {
        let headDim = 512
        let nqptg = 8, ncpsg = 64, nsg = 8
        let nKv = rawSpan + nComp
        precondition(nQ > 0 && rawSpan > 0 && nComp >= 0 && nHead > 0)
        precondition(nComp == 0 || comp != nil)
        precondition(q.byteLength >= nQ * nHead * headDim * 4)
        precondition(heads.byteLength >= nQ * nHead * headDim * 4)
        precondition(mask.byteLength >= nQ * nKv * 2)
        precondition(kvF16.byteLength >= nKv * headDim * 2)
        let e = encoder

        // 1) Raw span: circular gather from the F32 latent cache (rows carry
        //    their absolute RoPE, so a shifted span matches the C slid cache).
        let rawRows = kvF32.count / headDim
        precondition(rawRows > 0 && rawSpan <= rawRows)
        let physStart = ((rawStartRow % rawRows) + rawRows) % rawRows
        let cpyPso = try rt.pipeline("kernel_cpy_f32_f16")
        func cpyF32toF16(_ src: GPUTensor, srcOff: Int, dstOff: Int, count: Int) {
            let a = MetalRuntime.cpyArgs(n: count, srcElem: 4, dstElem: 2)
            var width = 32
            let maxWidth = cpyPso.maxTotalThreadsPerThreadgroup
            while width < count && width < maxWidth { width *= 2 }
            width = max(1, min(width, maxWidth, count))
            e.setComputePipelineState(cpyPso)
            a.withUnsafeBytes { e.setBytes($0.baseAddress!, length: a.count, index: 0) }
            e.setBuffer(src.buffer, offset: srcOff, index: 1)
            e.setBuffer(kvF16.buffer, offset: kvF16.byteOffset + dstOff, index: 2)
            e.dispatchThreadgroups(MTLSize(width: (count + width - 1) / width, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        }
        if physStart + rawSpan <= rawRows {
            cpyF32toF16(kvF32, srcOff: kvF32.byteOffset + physStart * headDim * 4,
                        dstOff: 0, count: rawSpan * headDim)
        } else {
            let ringPso = try rt.pipeline("kernel_dsv4_raw_ring_cpy_f32_f16")
            let args = [UInt32(rawSpan), UInt32(headDim), UInt32(rawRows), UInt32(physStart)]
            e.setComputePipelineState(ringPso)
            args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 0) }
            e.setBuffer(kvF32.buffer, offset: kvF32.byteOffset, index: 1)
            e.setBuffer(kvF16.buffer, offset: kvF16.byteOffset, index: 2)
            let width = min(headDim, max(1, ringPso.threadExecutionWidth))
            e.dispatchThreads(MTLSize(width: headDim, height: rawSpan, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        }
        if let comp, nComp > 0 {
            cpyF32toF16(comp, srcOff: comp.byteOffset,
                        dstOff: rawSpan * headDim * 2, count: nComp * headDim)
        }

        // 2) Pad the final partial 64-block of K/V/mask (per-query mask rows).
        let hasKvpad = (nKv % ncpsg) != 0
        if hasKvpad {
            precondition(pad.byteLength >= ncpsg * (2 * headDim * 2 + nQ * 2))
            let pArgs = MetalRuntime.flashPrefillPadArgs(nQ: nQ, nKv: nKv, headDim: headDim)
            let padPso = try rt.flashPadPipeline(ncpsg: Int32(ncpsg))
            e.setComputePipelineState(padPso)
            pArgs.withUnsafeBytes { e.setBytes($0.baseAddress!, length: pArgs.count, index: 0) }
            e.setBuffer(kvF16.buffer, offset: kvF16.byteOffset, index: 1)
            e.setBuffer(kvF16.buffer, offset: kvF16.byteOffset, index: 2)
            e.setBuffer(mask.buffer, offset: mask.byteOffset, index: 3)
            e.setBuffer(pad.buffer, offset: pad.byteOffset, index: 4)
            e.dispatchThreadgroups(MTLSize(width: ncpsg, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }

        // 3) Mask block map: skip fully-masked 8x64 tiles, drop the mask add on
        //    all-zero tiles. Required by the kernel whenever has_mask.
        let nblk0 = (nKv + ncpsg - 1) / ncpsg
        let nblk1 = (nQ + nqptg - 1) / nqptg
        precondition(blk.byteLength >= nblk0 * nblk1)
        let bArgs = MetalRuntime.flashBlkArgs(nQ: nQ, nKv: nKv)
        let blkPso = try rt.flashBlkPipeline(nqptg: Int32(nqptg), ncpsg: Int32(ncpsg))
        e.setComputePipelineState(blkPso)
        bArgs.withUnsafeBytes { e.setBytes($0.baseAddress!, length: bArgs.count, index: 0) }
        e.setBuffer(mask.buffer, offset: mask.byteOffset, index: 1)
        e.setBuffer(blk.buffer, offset: blk.byteOffset, index: 2)
        e.dispatchThreadgroups(MTLSize(width: nblk0, height: nblk1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        // 4) Batched FlashAttention: (ceil(nQ/8), nHead) threadgroups of 32x8.
        let scale = 1.0 / Float(headDim).squareRoot()
        let aArgs = MetalRuntime.flashPrefillArgs(nHead: nHead, nQ: nQ, nKv: nKv,
                                                  headDim: headDim, scale: scale)
        let pso = try rt.flashPrefillPipeline(nsg: Int32(nsg), hasSinks: true,
                                              hasKvpad: hasKvpad, bcMask: (nQ % nqptg) != 0)
        // C sizing: nqptg * (DK + 2*PAD2(DV,64) + 2*(2*ncpsg)) halves.
        let paddedV = (headDim + 63) & ~63
        let sharedElems = nqptg * (headDim + 2 * paddedV + 2 * (2 * ncpsg))
        let sharedBytes = (sharedElems * 2 + 15) & ~15
        e.setComputePipelineState(pso)
        aArgs.withUnsafeBytes { e.setBytes($0.baseAddress!, length: aArgs.count, index: 0) }
        e.setBuffer(q.buffer, offset: q.byteOffset, index: 1)
        e.setBuffer(kvF16.buffer, offset: kvF16.byteOffset, index: 2)
        e.setBuffer(kvF16.buffer, offset: kvF16.byteOffset, index: 3)
        e.setBuffer(mask.buffer, offset: mask.byteOffset, index: 4)
        e.setBuffer(sinks.buffer, offset: sinks.byteOffset, index: 5)
        e.setBuffer(pad.buffer, offset: pad.byteOffset, index: 6)
        e.setBuffer(blk.buffer, offset: blk.byteOffset, index: 7)
        e.setBuffer(heads.buffer, offset: heads.byteOffset, index: 8)
        e.setThreadgroupMemoryLength(sharedBytes, index: 0)
        e.dispatchThreadgroups(MTLSize(width: nblk1, height: nHead, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: nsg, depth: 1))
    }
}
