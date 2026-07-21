import Foundation
import Metal

extension GraphContext {
    /// Encode-form HC expand4 (per-layer 4-stream HC update). See MetalHyperConnections.
    /// out[t][k][d] = (blockOut+blockAdd?)[t][d]*post[t][k] + sum_j comb[t][j][k]*residual[t][j][d].
    public func hcExpand4(blockOut: GPUTensor, residual: GPUTensor, post: GPUTensor, comb: GPUTensor,
                          blockAdd: GPUTensor?, out: GPUTensor, nEmbd: Int, nTokens: Int,
                          postByteOffset: Int = 0, combByteOffset: Int = 0,
                          splitTokenStride: Int? = nil) throws {
        let nHC = 4
        let hasAdd = blockAdd != nil
        let args = MetalRuntime.hcExpandArgs(nEmbd: nEmbd, nHC: nHC, nTokens: nTokens, hasAdd: hasAdd,
                                             postTokenStride: splitTokenStride,
                                             combTokenStride: splitTokenStride)
        let pso = try rt.pipeline("kernel_dsv4_hc_expand4")
        let nElem = nEmbd * nTokens
        let nth = min(256, max(1, nElem))
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(blockOut.buffer, offset: blockOut.byteOffset, index: 1)
        e.setBuffer(residual.buffer, offset: residual.byteOffset, index: 2)
        e.setBuffer(post.buffer, offset: post.byteOffset + postByteOffset, index: 3)
        e.setBuffer(comb.buffer, offset: comb.byteOffset + combByteOffset, index: 4)
        let add = blockAdd ?? blockOut
        e.setBuffer(add.buffer, offset: add.byteOffset, index: 5)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 6)
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
        e.setBuffer(mix.buffer, offset: mix.byteOffset, index: 1)
        e.setBuffer(scale.buffer, offset: scale.byteOffset, index: 2)
        e.setBuffer(base.buffer, offset: base.byteOffset, index: 3)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 4)
        e.dispatchThreadgroups(MTLSize(width: (nRows + nth - 1) / nth, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }

    /// Encode-form FUSED HC-reduce tail (the C decode release path): mixer
    /// split (HC=4 Sinkhorn) + pre-weighted collapse of the 4 HC streams +
    /// weighted RMSNorm in ONE dispatch (kernel_dsv4_hc_split_weighted_sum_
    /// norm4) instead of three. Lane 0 computes the 24-value split exactly
    /// like kernel_dsv4_hc_split_sinkhorn (same sigmoid/softmax/Sinkhorn
    /// sequence, same eps placement); every lane then collapses its slice of
    /// the row from threadgroup memory and the norm reuses the just-collapsed
    /// values. Outputs match the unfused triple: `split` (nRows x 24), `embd`
    /// (collapsed row, kept for diagnostics/parity), `normOut` (normed row).
    /// x is [nRows][nHC=4][nEmbd] F32 contiguous; nEmbd multiple of 4.
    public func hcSplitWeightedSumNorm4(mix: GPUTensor, scale: GPUTensor, base: GPUTensor,
                                        x: GPUTensor, split: GPUTensor, embd: GPUTensor,
                                        normWeight: GPUTensor, normOut: GPUTensor,
                                        nEmbd: Int, nRows: Int, sinkhornIters: Int,
                                        eps: Float, normEps: Float) throws {
        precondition(nEmbd % 4 == 0)
        let args = MetalRuntime.hcSplitWeightedSumNormArgs(nEmbd: nEmbd, nHc: 4,
                                                           sinkhornIters: sinkhornIters,
                                                           nRows: nRows, eps: eps, normEps: normEps)
        let pso = try rt.pipeline("kernel_dsv4_hc_split_weighted_sum_norm4")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        // Every GPUTensor binding honors its logical view offset.
        e.setBuffer(mix.buffer, offset: mix.byteOffset, index: 1)
        e.setBuffer(scale.buffer, offset: scale.byteOffset, index: 2)
        e.setBuffer(base.buffer, offset: base.byteOffset, index: 3)
        e.setBuffer(x.buffer, offset: x.byteOffset, index: 4)
        e.setBuffer(split.buffer, offset: split.byteOffset, index: 5)
        e.setBuffer(embd.buffer, offset: embd.byteOffset, index: 6)
        e.setBuffer(normWeight.buffer, offset: normWeight.byteOffset, index: 7)
        e.setBuffer(normOut.buffer, offset: normOut.byteOffset, index: 8)
        // shared = row_shmem[nEmbd floats] + pre_shmem[4] + sum_shmem[32].
        e.setThreadgroupMemoryLength((nEmbd + 4 + 32) * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: nRows, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Encode-form HC weighted sum (collapse n_hc streams): out[t][d] = sum_h x[t][h][d]*w[t][h].
    public func hcWeightedSum(x: GPUTensor, weights: GPUTensor, out: GPUTensor,
                              nEmbd: Int, nHC: Int, nTokens: Int,
                              weightsTokenStride: Int? = nil) throws {
        let args = MetalRuntime.hcWeightedSumArgs(nEmbd: nEmbd, nHC: nHC, nTokens: nTokens,
                                                  weightsTokenStride: weightsTokenStride)
        let pso = try rt.pipeline("kernel_dsv4_hc_weighted_sum")
        let nElem = nEmbd * nTokens
        let nth = min(256, max(1, nElem))
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(x.buffer, offset: x.byteOffset, index: 1)
        e.setBuffer(weights.buffer, offset: weights.byteOffset, index: 2)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 3)
        e.dispatchThreadgroups(MTLSize(width: (nElem + nth - 1) / nth, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }
}

