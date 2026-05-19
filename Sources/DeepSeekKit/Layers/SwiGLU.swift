import Foundation
import Metal

/// Dense SwiGLU feed-forward block (TODO §10.2 / T2). Llama-family
/// architectures use this where DeepSeek uses `MoEFFN`:
///
///   y = wDown( silu(wGate(x)) * wUp(x) )
///
/// `wGate` and `wUp` project up to `intermediate_size`, the
/// elementwise SwiGLU body lives in `Elementwise.siluMul`, and
/// `wDown` projects back to `hidden_size`. No bias on Llama; the
/// `Linear` instances here are bias-less.
///
/// The three Linears can be any quantization the engine supports
/// (F32 / BF16 / FP8 / INT8 / INT4) — the dispatch already routes
/// through `Linear.callAsFunction`, which picks the right GEMM kernel
/// based on `weight.dtype`. Most GGUF-quantized Llama checkpoints
/// land here as F32 after `GGUFLoader.load(...)` dequant-on-load.
public final class SwiGLU {
    public let wGate: Linear
    public let wUp: Linear
    public let wDown: Linear

    public init(wGate: Linear, wUp: Linear, wDown: Linear) {
        self.wGate = wGate
        self.wUp = wUp
        self.wDown = wDown
    }

    /// Forward pass. Accepts a 2-D `[M, hidden]` tensor or a 3-D
    /// `[B, S, hidden]` tensor and preserves the input rank in the
    /// output. Input must be F32 (matches the rest of the
    /// transformer's intermediate dtype).
    public func callAsFunction(_ x: Tensor,
                                in cmd: MTLCommandBuffer) -> Tensor
    {
        precondition(x.dtype == .f32, "SwiGLU: input must be f32")
        // wGate / wUp / wDown each preserve x's leading dims and
        // swap the last one (`Linear` does
        // `outShape = x.shape.dropLast() + [outFeatures]`).
        let gate = wGate(x, in: cmd)
        let up   = wUp(x, in: cmd)
        let hidden = Elementwise.siluMul(gate, up, in: cmd)
        return wDown(hidden, in: cmd)
    }
}
