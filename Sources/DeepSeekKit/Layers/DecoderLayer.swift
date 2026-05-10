import Foundation
import Metal

/// One transformer block: pre-norm → hybrid attention → mHC residual
/// → pre-norm → MoE FFN → mHC residual.
///
/// The mHC residual mix is currently a plain add. Replace with the proper
/// manifold-constrained mixing once `mhc.metal` is implemented.
public final class DecoderLayer {
    public let attnNorm: RMSNorm
    public let ffnNorm: RMSNorm
    public let attention: HybridAttention
    public let ffn: MoEFFN     // for dense layers (first_k_dense_replace), substitute a single dense FFN of the same shape

    public init(attnNorm: RMSNorm, ffnNorm: RMSNorm,
                attention: HybridAttention, ffn: MoEFFN) {
        self.attnNorm = attnNorm
        self.ffnNorm = ffnNorm
        self.attention = attention
        self.ffn = ffn
    }

    public func callAsFunction(_ x: Tensor, cache: KVCache, in cmd: MTLCommandBuffer) -> Tensor {
        let h1 = attnNorm(x, in: cmd)
        let attnOut = attention(h1, cache: cache, in: cmd)
        Elementwise.addInPlace(x, attnOut, in: cmd)

        let h2 = ffnNorm(x, in: cmd)
        let ffnOut = ffn(h2, in: cmd)
        Elementwise.addInPlace(x, ffnOut, in: cmd)
        return x
    }
}
