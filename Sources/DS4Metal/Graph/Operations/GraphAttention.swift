import Foundation
import Metal

extension GraphContext {
    /// Number of decode FlashAttention workgroups for the live row count.
    /// Arbitrary NWG values in 1...32 are supported by both vec and reduce
    /// kernels; using the exact chunk count avoids the 4 -> 8 cliff at 129 rows.
    static func splitKWorkgroups(totalRows: Int, adaptive: Bool) -> Int {
        guard adaptive else { return 32 }
        let chunks = max(1, (max(0, totalRows) + 31) / 32)
        return min(32, chunks)
    }

    /// Decode FlashAttention core (MLA dk=dv=512), encode form. Converts the F32
    /// latent `kvF32` (nKeys x 512) and optional compressed F32 rows to the F16
    /// `kvF16` scratch, then runs vec + reduce in the shared command buffer.
    /// K==V==latent. `mask` may carry the NSA indexer's compressed-key selection.
    /// A partial final 32-row block is padded internally.
    ///
    /// `fusedStage` / `vectorCopy` are explicit A/B overrides used by numerical
    /// tests and the benchmark harness. nil selects the decoder-level env knobs.
    public func flashAttnCore(q: GPUTensor, kvF32: GPUTensor, kvF16: GPUTensor,
                              mask: GPUTensor, sinks: GPUTensor, pad: GPUTensor, tmp: GPUTensor,
                              heads: GPUTensor, nHead: Int, nKeys: Int, rawStartRow: Int = 0,
                              hasSinks: Bool = false,
                              comp: GPUTensor? = nil, nComp: Int = 0,
                              fusedStage: Bool? = nil, vectorCopy: Bool? = nil) throws {
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
        // con passo nwg·NSG. NWG non deve essere una potenza di due: il reduce
        // usa un simdgroup da 32 lane, neutralizza quelle >= NWG e distribuisce
        // DV4 fra gli NWG simdgroup del threadgroup. Usiamo quindi esattamente
        // ceil(total/32), fino a 32, evitando il salto 4 -> 8 a 129 righe.
        // NSG resta 1/2/4 come nel kernel storico.
        let nwg = GraphContext.splitKWorkgroups(totalRows: total,
                                                adaptive: GraphContext.adaptiveSplitK)
        var nsg = 1
        while 2 * 32 * nsg * ncpsg < total && nsg < 4 { nsg *= 2 }
        let e = encoder

        precondition(nKeys > 0 && nComp >= 0 && total > 0)
        precondition(nComp == 0 || comp != nil)

        // Raw span: only the SWA window starting at `rawStartRow` (rows hold their
        // absolute-RoPE'd values, so a shifted span is exactly the C slid cache).
        let rawRows = kvF32.count / headDim
        precondition(rawRows > 0 && nKeys <= rawRows)
        let physStart = ((rawStartRow % rawRows) + rawRows) % rawRows

        let useFusedStage = fusedStage ?? GraphContext.fusedFlashKVStage
        let useVectorCopy = vectorCopy ?? GraphContext.vectorCopies

        if useFusedStage {
            // One shape-specialized dispatch: circular gather + both F32->F16
            // spans + optional final-block K/V/mask padding.
            let stagePso = try rt.pipeline("kernel_dsv4_flash_kv_stage_f32")
            let args = [UInt32(rawRows), UInt32(physStart), UInt32(nKeys), UInt32(nComp),
                        UInt32(ncpsg), UInt32(kvpad ? 1 : 0)]
            e.setComputePipelineState(stagePso)
            args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 0) }
            e.setBuffer(kvF32.buffer, offset: kvF32.byteOffset, index: 1)
            let compTensor = comp ?? kvF32 // never read when nComp == 0
            e.setBuffer(compTensor.buffer, offset: compTensor.byteOffset, index: 2)
            e.setBuffer(kvF16.buffer, offset: kvF16.byteOffset, index: 3)
            e.setBuffer(mask.buffer, offset: mask.byteOffset, index: 4)
            e.setBuffer(pad.buffer, offset: pad.byteOffset, index: 5)
            let rowVectors = headDim / 4
            let stageThreads = total * rowVectors + (kvpad ? ncpsg * rowVectors + ncpsg : 0)
            let stageWidth = min(256, stagePso.maxTotalThreadsPerThreadgroup)
            e.dispatchThreads(MTLSize(width: stageThreads, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: stageWidth, height: 1, depth: 1))
        } else {
            // Exact fallback graph. Contiguous spans may still use the simpler
            // packed conversion kernel; DS4_VECTOR_COPY=0 restores the original
            // generic tensor-indexing kernel for a clean three-way A/B.
            let cpyPso = try rt.pipeline(useVectorCopy
                                         ? "kernel_cpy_contig_f32_f16_4"
                                         : "kernel_cpy_f32_f16")
            func cpyF32toF16(_ src: GPUTensor, srcOff: Int,
                             dstOff: Int, count: Int) {
                e.setComputePipelineState(cpyPso)
                if useVectorCopy {
                    var count32 = UInt32(count)
                    e.setBytes(&count32, length: 4, index: 0)
                    e.setBuffer(src.buffer, offset: srcOff, index: 1)
                    e.setBuffer(kvF16.buffer, offset: kvF16.byteOffset + dstOff, index: 2)
                    let threads = (count + 3) / 4
                    let width = min(256, cpyPso.maxTotalThreadsPerThreadgroup)
                    e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
                } else {
                    let a = MetalRuntime.cpyArgs(n: count, srcElem: 4, dstElem: 2)
                    var width = 32
                    let maxWidth = cpyPso.maxTotalThreadsPerThreadgroup
                    while width < count && width < maxWidth { width *= 2 }
                    width = max(1, min(width, maxWidth, count))
                    a.withUnsafeBytes { e.setBytes($0.baseAddress!, length: a.count, index: 0) }
                    e.setBuffer(src.buffer, offset: srcOff, index: 1)
                    e.setBuffer(kvF16.buffer, offset: kvF16.byteOffset + dstOff, index: 2)
                    e.dispatchThreadgroups(MTLSize(width: (count + width - 1) / width,
                                                   height: 1, depth: 1),
                                           threadsPerThreadgroup: MTLSize(width: width,
                                                                          height: 1, depth: 1))
                }
            }

            if physStart + nKeys <= rawRows {
                cpyF32toF16(kvF32,
                            srcOff: kvF32.byteOffset + physStart * headDim * 4,
                            dstOff: 0, count: nKeys * headDim)
            } else {
                let ringPso = try rt.pipeline("kernel_dsv4_raw_ring_cpy_f32_f16")
                let args = [UInt32(nKeys), UInt32(headDim), UInt32(rawRows), UInt32(physStart)]
                e.setComputePipelineState(ringPso)
                args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 0) }
                e.setBuffer(kvF32.buffer, offset: kvF32.byteOffset, index: 1)
                e.setBuffer(kvF16.buffer, offset: kvF16.byteOffset, index: 2)
                let width = min(headDim, max(1, ringPso.threadExecutionWidth))
                e.dispatchThreads(MTLSize(width: headDim, height: nKeys, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
            }
            if let comp, nComp > 0 {
                cpyF32toF16(comp, srcOff: comp.byteOffset,
                            dstOff: nKeys * headDim * 2, count: nComp * headDim)
            }

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
