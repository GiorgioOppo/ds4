import Foundation
import Metal

extension GraphContext {
    /// Encode-form router top-6 select over the Flash (256) or Pro (384) experts.
    /// `selected` is a 6-int32
    /// GPUTensor. `bias` (exp_probs_b) shifts probs for SELECTION only. With
    /// `hashTable` (ffn_gate_tid2eid, I32 [6 x hashRows]) the kernel ignores the
    /// scores and copies row min(token, hashRows-1) — the C hash-layer routing
    /// (ds4.c layer_hash_selected_experts / ds4_gpu_router_select_tensor).
    public func routerFinalizeTop6(probs: GPUTensor, selected: GPUTensor, bias: GPUTensor? = nil,
                                   hashTable: GPUTensor? = nil, hashRows: Int = 0, token: Int = 0,
                                   weights: GPUTensor? = nil, nExperts: Int,
                                   expertWeightScale: Float) throws {
        guard nExperts == 256 || nExperts == 384 else {
            throw MetalError.unsupported("router expert count \(nExperts); expected 256 or 384")
        }
        precondition(expertWeightScale.isFinite && expertWeightScale >= 0)
        precondition(probs.count >= nExperts)
        if let bias { precondition(bias.count >= nExperts) }
        precondition(selected.byteLength >= 6 * 4)
        if let weights { precondition(weights.count >= 6) }

        let sortWidth = nExperts == 256 ? 256 : 512
        var args = [UInt8](repeating: 0, count: 32)
        func u32(_ off: Int, _ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { args[off+k] = $0[k] } } }
        // has_bias, hash_mode, use_token_buffer, token, hash_rows, write_weights,
        // n_experts, expert_weight_scale
        u32(0, bias != nil ? 1 : 0); u32(4, hashTable != nil ? 1 : 0); u32(8, 0)
        u32(12, UInt32(max(0, token))); u32(16, UInt32(max(1, hashRows)))
        u32(20, weights != nil ? 1 : 0)
        u32(24, UInt32(nExperts)); u32(28, expertWeightScale.bitPattern)
        let pso = try rt.pipeline("kernel_dsv4_router_finalize_one")
        guard sortWidth <= pso.maxTotalThreadsPerThreadgroup else {
            throw MetalError.unsupported("router requires \(sortWidth) threads, GPU supports \(pso.maxTotalThreadsPerThreadgroup)")
        }
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: 32, index: 0) }
        e.setBuffer(probs.buffer, offset: probs.byteOffset, index: 1)
        // Unused slots get probs as a placeholder: the kernel dereferences bias
        // only with has_bias, hash only with hash_mode, tokens never (use_token_buffer=0).
        e.setBuffer((bias ?? probs).buffer, offset: (bias ?? probs).byteOffset, index: 2)
        e.setBuffer((hashTable ?? probs).buffer, offset: (hashTable ?? probs).byteOffset, index: 3)
        e.setBuffer(probs.buffer, offset: probs.byteOffset, index: 4)
        e.setBuffer(selected.buffer, offset: selected.byteOffset, index: 5)
        e.setBuffer((weights ?? probs).buffer, offset: (weights ?? probs).byteOffset, index: 6)
        e.setThreadgroupMemoryLength(sortWidth * 4 + sortWidth * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: sortWidth, height: 1, depth: 1))
    }

    /// Encode-form router weight normalization using the architecture's scale.
    public func routerWeights(probs: GPUTensor, selected: GPUTensor, weights: GPUTensor,
                              nExperts: Int, expertWeightScale: Float) throws {
        guard nExperts == 256 || nExperts == 384 else {
            throw MetalError.unsupported("router expert count \(nExperts); expected 256 or 384")
        }
        precondition(expertWeightScale.isFinite && expertWeightScale >= 0)
        precondition(probs.count >= nExperts)
        precondition(selected.byteLength >= 6 * 4)
        precondition(weights.count >= 6)

        var args = [UInt8](repeating: 0, count: 8)
        func u32(_ off: Int, _ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { args[off+k] = $0[k] } } }
        u32(0, UInt32(nExperts)); u32(4, expertWeightScale.bitPattern)
        let pso = try rt.pipeline("kernel_dsv4_router_weights_one")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: 8, index: 0) }
        e.setBuffer(probs.buffer, offset: probs.byteOffset, index: 1)
        e.setBuffer(selected.buffer, offset: selected.byteOffset, index: 2)
        e.setBuffer(weights.buffer, offset: weights.byteOffset, index: 3)
        e.dispatchThreads(MTLSize(width: 6, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 6, height: 1, depth: 1))
    }

    /// Backwards-compatible Flash overload. New model-aware callers should pass
    /// `nExperts` and `expertWeightScale` explicitly.
    public func routerFinalizeTop6(probs: GPUTensor, selected: GPUTensor, bias: GPUTensor? = nil,
                                   hashTable: GPUTensor? = nil, hashRows: Int = 0, token: Int = 0,
                                   weights: GPUTensor? = nil) throws {
        try routerFinalizeTop6(probs: probs, selected: selected, bias: bias,
                               hashTable: hashTable, hashRows: hashRows, token: token,
                               weights: weights, nExperts: 256, expertWeightScale: 1.5)
    }

    /// Backwards-compatible Flash overload.
    public func routerWeights(probs: GPUTensor, selected: GPUTensor,
                              weights: GPUTensor) throws {
        try routerWeights(probs: probs, selected: selected, weights: weights,
                          nExperts: 256, expertWeightScale: 1.5)
    }
}
