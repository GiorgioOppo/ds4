import Foundation
import Metal

/// DeepSeek-style MoE FFN: shared experts (always active) + routed experts
/// (top-k via softmax gate). SwiGLU activation, like V3.
public final class MoEFFN {
    public let gate: Linear                         // [n_routed_experts, hidden]
    public let routedUp: [Linear]                   // each [moe_inter, hidden]
    public let routedGate: [Linear]                 // each [moe_inter, hidden]
    public let routedDown: [Linear]                 // each [hidden, moe_inter]
    public let sharedUp: Linear?
    public let sharedGate: Linear?
    public let sharedDown: Linear?
    public let topK: Int
    public let nExperts: Int

    private let pipelineGate: MTLComputePipelineState

    public init(gate: Linear,
                routedUp: [Linear], routedGate: [Linear], routedDown: [Linear],
                sharedUp: Linear?, sharedGate: Linear?, sharedDown: Linear?,
                topK: Int) {
        self.gate = gate
        self.routedUp = routedUp
        self.routedGate = routedGate
        self.routedDown = routedDown
        self.sharedUp = sharedUp
        self.sharedGate = sharedGate
        self.sharedDown = sharedDown
        self.topK = topK
        self.nExperts = routedUp.count
        self.pipelineGate = Device.shared.makePipeline("moe_topk_gate")
    }

    /// Decode-only forward (M=1). Prefill MoE needs token-permutation
    /// scatter/gather and grouped GEMM — see roadmap in README.
    public func callAsFunction(_ x: Tensor, in cmd: MTLCommandBuffer) -> Tensor {
        precondition(x.shape.count == 2 && x.shape[0] == 1,
                     "MoE forward currently supports only single-token decode")
        let hidden = x.shape[1]

        // gate logits
        let gateLogits = gate(x, in: cmd)               // [1, nExperts]

        // top-k gating — raw buffers since indices are u32, not part of the Tensor abstraction
        let idxBuf = Device.shared.mtl.makeBuffer(length: topK * 4, options: .storageModeShared)!
        let wBuf = Device.shared.mtl.makeBuffer(length: topK * 4, options: .storageModeShared)!
        do {
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipelineGate)
            enc.setBuffer(gateLogits.buffer, offset: 0, index: 0)
            enc.setBuffer(idxBuf, offset: 0, index: 1)
            enc.setBuffer(wBuf, offset: 0, index: 2)
            var dims = SIMD3<UInt32>(1, UInt32(nExperts), UInt32(topK))
            enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)
            enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            enc.endEncoding()
        }

        // Read indices/weights back to host so we can dispatch the right experts.
        // For batch=1 decode this is a tiny CPU-GPU sync, acceptable.
        cmd.commit(); cmd.waitUntilCompleted()
        let idxPtr = idxBuf.contents().bindMemory(to: UInt32.self, capacity: topK)
        let wPtr = wBuf.contents().bindMemory(to: Float.self, capacity: topK)
        let activeIdx = (0..<topK).map { Int(idxPtr[$0]) }
        let activeW = (0..<topK).map { wPtr[$0] }

        // New command buffer for the expert MLPs.
        let cmd2 = Device.shared.queue.makeCommandBuffer()!
        var sum: Tensor? = nil
        for (k, e) in activeIdx.enumerated() {
            let h = swiglu(x, up: routedUp[e], gateProj: routedGate[e], in: cmd2)
            let out = routedDown[e](h, in: cmd2)
            if let acc = sum {
                Elementwise.axpy(acc, out, alpha: activeW[k], in: cmd2)
            } else {
                sum = Elementwise.scale(out, by: activeW[k], in: cmd2)
            }
        }
        if let su = sharedUp, let sg = sharedGate, let sd = sharedDown {
            let h = swiglu(x, up: su, gateProj: sg, in: cmd2)
            let out = sd(h, in: cmd2)
            if let acc = sum {
                Elementwise.addInPlace(acc, out, in: cmd2)
            } else {
                sum = out
            }
        }
        cmd2.commit(); cmd2.waitUntilCompleted()
        return sum ?? Tensor.empty(shape: [1, hidden], dtype: .f32)
    }

    private func swiglu(_ x: Tensor, up: Linear, gateProj: Linear,
                        in cmd: MTLCommandBuffer) -> Tensor {
        let g = gateProj(x, in: cmd)
        let u = up(x, in: cmd)
        return Elementwise.siluMul(g, u, in: cmd)
    }
}
