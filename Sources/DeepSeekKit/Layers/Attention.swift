import Foundation
import Metal

/// Hybrid attention dispatcher: routes to CSA or HCA based on layer index.
///
/// CSA and HCA are not implemented yet — both backends trap. Once the
/// reference Python is available, fill in `forwardCSA` / `forwardHCA`
/// and replace the trap kernels in `attention_csa.metal` / `attention_hca.metal`.
public final class HybridAttention {
    public enum Mode { case csa, hca }

    public let mode: Mode
    public let qProj: Linear
    public let kProj: Linear
    public let vProj: Linear
    public let oProj: Linear
    public let rope: RoPE
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let compressionRatio: Int

    public init(mode: Mode,
                qProj: Linear, kProj: Linear, vProj: Linear, oProj: Linear,
                rope: RoPE,
                numHeads: Int, numKVHeads: Int, headDim: Int,
                compressionRatio: Int) {
        self.mode = mode
        self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.oProj = oProj
        self.rope = rope
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.compressionRatio = compressionRatio
    }

    public func callAsFunction(_ x: Tensor, cache: KVCache, in cmd: MTLCommandBuffer) -> Tensor {
        switch mode {
        case .csa: return forwardCSA(x, cache: cache, in: cmd)
        case .hca: return forwardHCA(x, cache: cache, in: cmd)
        }
    }

    private func forwardCSA(_ x: Tensor, cache: KVCache, in cmd: MTLCommandBuffer) -> Tensor {
        fatalError("CSA forward not implemented — see Sources/DeepSeekKit/Kernels/attention_csa.metal")
    }

    private func forwardHCA(_ x: Tensor, cache: KVCache, in cmd: MTLCommandBuffer) -> Tensor {
        fatalError("HCA forward not implemented — see Sources/DeepSeekKit/Kernels/attention_hca.metal")
    }
}
