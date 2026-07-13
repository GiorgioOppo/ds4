import Foundation
import Metal

extension GraphContext {
    /// Encode-form grouped low-rank attention output (kernel_dsv4_attn_out_low_q8_0_f32):
    /// low[g][rank] = outputA[g] (Q8) * heads[g] for each of nGroups groups.
    public func attnOutLowQ8(outputA: GPUTensor, heads: GPUTensor, low: GPUTensor,
                             nGroups: Int, groupDim: Int, rank: Int) throws {
        precondition(groupDim % 32 == 0)
        let rowABytes = (groupDim / 32) * 34
        let nsg = GraphContext.q8NSG, nr0 = 2   // nsg tunable via DS4_Q8_NSG (default 4 = reference)
        let args = MetalRuntime.attnLowArgs(nGroups: nGroups, groupDim: groupDim, rank: rank, rowABytes: rowABytes)
        let pso = try rt.mulMVPipeline("kernel_dsv4_attn_out_low_q8_0_f32", nsg: nsg)
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(outputA.buffer, offset: outputA.byteOffset, index: 1)   // no-copy mmap weight
        e.setBuffer(heads.buffer, offset: heads.byteOffset, index: 2)
        e.setBuffer(low.buffer, offset: low.byteOffset, index: 3)
        e.setThreadgroupMemoryLength(32 * 2 * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (rank + nr0 - 1) / nr0, height: 1, depth: nGroups),
                               threadsPerThreadgroup: MTLSize(width: 32, height: Int(nsg), depth: 1))
    }

    /// Q4_K grouped low-rank attention output with the fixed identity group map.
    public func attnOutLowQ4(outputA: GPUTensor, heads: GPUTensor, low: GPUTensor,
                             nGroups: Int, groupDim: Int, rank: Int) throws {
        precondition(groupDim % 256 == 0)
        let rowABytes = (groupDim / 256) * 144
        let nsg = Int(GraphContext.denseQ4NSG), nr0 = 2
        let args = MetalRuntime.attnLowArgs(nGroups: nGroups, groupDim: groupDim,
                                            rank: rank, rowABytes: rowABytes,
                                            blockBytes: 144)
        let pso = try rt.mulMVPipeline("kernel_dsv4_attn_out_low_q4_K_f32", nsg: Int16(nsg))
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(outputA.buffer, offset: outputA.byteOffset, index: 1)
        e.setBuffer(heads.buffer, offset: heads.byteOffset, index: 2)
        e.setBuffer(low.buffer, offset: low.byteOffset, index: 3)
        e.setThreadgroupMemoryLength(256, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (rank + nsg * nr0 - 1) / (nsg * nr0),
                                       height: 1, depth: nGroups),
                               threadsPerThreadgroup: MTLSize(width: 32, height: nsg, depth: 1))
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
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 3)
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
        e.setBuffer(src.buffer, offset: src.byteOffset, index: 1)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 2)
        e.dispatchThreadgroups(MTLSize(width: nHC, height: nTokens, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }
}

