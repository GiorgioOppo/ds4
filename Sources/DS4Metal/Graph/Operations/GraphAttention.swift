import Foundation
import Metal

extension GraphContext {
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
        let ncpsg = 32
        // Two-span attention: raw SWA rows (nKeys) followed by compressed rows (nComp),
        // contiguous in kvF16. The flash kernel then attends over the union.
        let total = nKeys + nComp
        let kvpad = (total % ncpsg) != 0
        // Split-K ADATTIVO (DS4_ADAPTIVE_SPLITK, default on). Storico: nwg=32
        // SEMPRE — a contesto corto quasi tutti i workgroup uscivano subito dal
        // loop dei chunk ma scrivevano comunque il loro partial DV=512 (+S/M),
        // che il reduce rileggeva: ~2·nHead·512·32·4 B di traffico morto per
        // layer per token. Il loop del kernel assegna il chunk ic0=iwg·NSG+sgitg
        // con passo nwg·NSG: per QUALSIASI nwg >= nchunks ogni workgroup attivo
        // vede al più un chunk — esattamente gli stessi chunk del dispatch
        // storico, quindi partial bit-identici; spariscono solo i workgroup
        // vuoti. nwg è arrotondato alla potenza di 2 (≤32) per limitare le
        // varianti di PSO compilate (≤6). nsg resta calcolato con la formula
        // storica (nwg=32): supera 1 solo oltre 2048 chiavi, dove comunque
        // nchunks > 32 e quindi nwg = 32.
        let nchunks = (total + ncpsg - 1) / ncpsg
        var nwg = 32
        if GraphContext.adaptiveSplitK {
            nwg = 1
            while nwg < nchunks && nwg < 32 { nwg *= 2 }
        }
        var nsg = 1
        while 2 * 32 * nsg * ncpsg < total && nsg < 4 { nsg *= 2 }
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
            e.setBuffer(kvF16.buffer, offset: kvF16.byteOffset, index: 1)
            e.setBuffer(kvF16.buffer, offset: kvF16.byteOffset, index: 2)
            e.setBuffer(mask.buffer, offset: mask.byteOffset, index: 3)
            e.setBuffer(pad.buffer, offset: pad.byteOffset, index: 4)
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
        e.setBuffer(q.buffer, offset: q.byteOffset, index: 1)
        e.setBuffer(kvF16.buffer, offset: kvF16.byteOffset, index: 2)
        e.setBuffer(kvF16.buffer, offset: kvF16.byteOffset, index: 3)
        e.setBuffer(mask.buffer, offset: mask.byteOffset, index: 4)
        e.setBuffer(sinks.buffer, offset: sinks.byteOffset, index: 5)
        e.setBuffer(pad.buffer, offset: pad.byteOffset, index: 6)
        // Con nwg == 1 il kernel vec normalizza da solo (S = 1/ss[0], stesso
        // identico calcolo che il reduce farebbe con un solo lane attivo) e il
        // suo layout di scrittura dst4[rid·DV4 + i] coincide con quello del
        // reduce: si scrive direttamente in `heads` e il reduce si salta.
        let vecDst = nwg == 1 ? heads : tmp
        e.setBuffer(vecDst.buffer, offset: vecDst.byteOffset, index: 7)
        e.setThreadgroupMemoryLength(sharedBytes, index: 0)
        e.dispatchThreadgroups(MTLSize(width: 1, height: nHead, depth: nwg),
                               threadsPerThreadgroup: MTLSize(width: 32, height: nsg, depth: 1))

        // 3) reduce (solo con split-K reale: nwg == 1 ha già scritto in heads)
        if nwg > 1 {
            let reduce = try rt.flashReducePipeline(dv: Int32(headDim), nwg: Int32(nwg))
            var reduceArgs = Int32(nHead)
            e.setComputePipelineState(reduce)
            e.setBytes(&reduceArgs, length: 4, index: 0)
            e.setBuffer(tmp.buffer, offset: tmp.byteOffset, index: 1)
            e.setBuffer(heads.buffer, offset: heads.byteOffset, index: 2)
            e.dispatchThreadgroups(MTLSize(width: nHead, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 32 * nwg, height: 1, depth: 1))
        }
    }
}

