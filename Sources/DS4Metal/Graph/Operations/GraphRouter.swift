import Foundation
import Metal

extension GraphContext {
    /// Encode-form router top-6 select over 256 experts. `selected` is a 6-int32
    /// GPUTensor. `bias` (exp_probs_b) shifts probs for SELECTION only. With
    /// `hashTable` (ffn_gate_tid2eid, I32 [6 x hashRows]) the kernel ignores the
    /// scores and copies row min(token, hashRows-1) — the C hash-layer routing
    /// (ds4.c layer_hash_selected_experts / ds4_gpu_router_select_tensor).
    public func routerFinalizeTop6(probs: GPUTensor, selected: GPUTensor, bias: GPUTensor? = nil,
                                   hashTable: GPUTensor? = nil, hashRows: Int = 0, token: Int = 0,
                                   weights: GPUTensor? = nil) throws {
        var args = [UInt8](repeating: 0, count: 24)
        func u32(_ off: Int, _ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { args[off+k] = $0[k] } } }
        // has_bias, hash_mode, use_token_buffer, token, hash_rows
        u32(0, bias != nil ? 1 : 0); u32(4, hashTable != nil ? 1 : 0); u32(8, 0)
        u32(12, UInt32(max(0, token))); u32(16, UInt32(max(1, hashRows)))
        u32(20, weights != nil ? 1 : 0)
        let pso = try rt.pipeline("kernel_dsv4_router_finalize_one")
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: 24, index: 0) }
        e.setBuffer(probs.buffer, offset: probs.byteOffset, index: 1)
        // Unused slots get probs as a placeholder: the kernel dereferences bias
        // only with has_bias, hash only with hash_mode, tokens never (use_token_buffer=0).
        e.setBuffer((bias ?? probs).buffer, offset: (bias ?? probs).byteOffset, index: 2)
        e.setBuffer((hashTable ?? probs).buffer, offset: (hashTable ?? probs).byteOffset, index: 3)
        e.setBuffer(probs.buffer, offset: probs.byteOffset, index: 4)
        e.setBuffer(selected.buffer, offset: selected.byteOffset, index: 5)
        e.setBuffer((weights ?? probs).buffer, offset: (weights ?? probs).byteOffset, index: 6)
        e.setThreadgroupMemoryLength(256 * 4 + 256 * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Encode-form router weight normalization: w[i] = probs[sel[i]]/sum * 1.5.
    public func routerWeights(probs: GPUTensor, selected: GPUTensor, weights: GPUTensor) throws {
        let pso = try rt.pipeline("kernel_dsv4_router_weights_one")
        let e = encoder
        e.setComputePipelineState(pso)
        e.setBuffer(probs.buffer, offset: probs.byteOffset, index: 0)
        e.setBuffer(selected.buffer, offset: selected.byteOffset, index: 1)
        e.setBuffer(weights.buffer, offset: weights.byteOffset, index: 2)
        e.dispatchThreads(MTLSize(width: 6, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 6, height: 1, depth: 1))
    }
}

