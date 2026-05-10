import Foundation
import Metal

/// KV compressor: gated softmax-pooling over `compressRatio` consecutive
/// tokens. Mirrors the `Compressor` module in
/// `Reference/inference/model.py` lines 279–377.
///
/// When ratio == 4, uses overlapping windows for smoother boundaries; for
/// other ratios it operates on disjoint windows. Maintains internal state
/// across decode steps so the compression can run incrementally.
///
/// The forward pass is mathematically clear but logistically heavy:
///   - prefill (start_pos == 0): compresses each consecutive ratio-block
///     in parallel, stashes any tail remainder into kv_state/score_state for
///     the next decode step
///   - decode (start_pos > 0): accumulates one token into the state buffer;
///     when the buffer fills (start_pos + 1 % ratio == 0) it emits one
///     compressed token into the KV cache
///
/// The KV cache and YaRN cos/sin freqs are wired in by the parent Attention
/// module after construction.
public final class Compressor {
    public let dim: Int
    public let headDim: Int
    public let ropeHeadDim: Int
    public let nopeHeadDim: Int
    public let compressRatio: Int
    public let overlap: Bool
    public let rotate: Bool

    public let ape: Tensor                  // [ratio, coff*headDim] f32
    public let wkv: Linear                  // [coff*headDim, dim] f32
    public let wgate: Linear                // [coff*headDim, dim] f32
    public let norm: RMSNorm                // RMSNorm(headDim)

    /// Internal state — sized by parent. coff = 2 if overlap else 1.
    public var kvState: Tensor              // [maxBatch, coff*ratio, coff*headDim] f32
    public var scoreState: Tensor           // [maxBatch, coff*ratio, coff*headDim] f32

    public weak var owningCache: KVCache?   // not yet wired
    public var rope: RoPE?                  // assigned by parent Attention

    public init(config: ModelConfig, compressRatio: Int, headDim: Int, rotate: Bool,
                ape: Tensor, wkv: Linear, wgate: Linear, norm: RMSNorm,
                kvState: Tensor, scoreState: Tensor) {
        self.dim = config.dim
        self.headDim = headDim
        self.ropeHeadDim = config.ropeHeadDim
        self.nopeHeadDim = headDim - config.ropeHeadDim
        self.compressRatio = compressRatio
        self.overlap = compressRatio == 4
        self.rotate = rotate
        self.ape = ape
        self.wkv = wkv
        self.wgate = wgate
        self.norm = norm
        self.kvState = kvState
        self.scoreState = scoreState
    }

    /// `forward(x, startPos)` from the reference. Returns the compressed KV
    /// row(s) when one is produced, or nil otherwise.
    ///
    /// NOT IMPLEMENTED. The reference uses several non-trivial PyTorch ops:
    ///   - `score.unflatten(1, (-1, ratio)).softmax(dim=2)` over consecutive
    ///     ratio-blocks
    ///   - `kv = (kv * score).sum(dim=2)` weighted pooling
    ///   - `overlap_transform` shuffle that interleaves overlapping windows
    ///   - lazy population of kv_state across decode steps
    ///   - in-place `act_quant` / `fp4_act_quant` on the compressed output
    ///
    /// All require companion Metal kernels (gather/scatter, softmax along a
    /// non-last dim, fp4_act_quant). See README.
    public func callAsFunction(_ x: Tensor, startPos: Int, in cmd: MTLCommandBuffer) -> Tensor? {
        fatalError("Compressor.forward not implemented — porting target: model.py:316")
    }
}
