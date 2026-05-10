import Foundation
import Metal

/// Embedding table + lookup. Stripped down from `ParallelEmbedding`
/// in model.py:83 — single-rank (no tensor parallel).
public final class ParallelEmbedding {
    public let vocabSize: Int
    public let dim: Int
    public let weight: Tensor      // [vocab, dim]

    public init(vocabSize: Int, dim: Int, weight: Tensor) {
        self.vocabSize = vocabSize; self.dim = dim; self.weight = weight
    }
}

/// LM head with HC mixing applied to the input. Mirrors `ParallelHead`
/// in model.py:703–735. Logits are computed only on the LAST sequence
/// position via `get_logits(x[:, -1])`.
public final class ParallelHead {
    public let vocabSize: Int
    public let dim: Int
    public let normEps: Float
    public let hcEps: Float
    public let weight: Tensor      // [vocab, dim] f32

    public init(vocabSize: Int, dim: Int, normEps: Float, hcEps: Float, weight: Tensor) {
        self.vocabSize = vocabSize; self.dim = dim
        self.normEps = normEps; self.hcEps = hcEps
        self.weight = weight
    }
}

/// DeepSeek-V4 transformer. Mirrors `Transformer` in
/// `Reference/inference/model.py` lines 769–809.
///
/// forward(input_ids, start_pos):
///   h = embed(input_ids).unsqueeze(2).repeat(1, 1, hc_mult, 1)   // expand to hc copies
///   for layer in layers: h = layer(h, start_pos, input_ids)
///   logits = head(h, hc_head_fn, hc_head_scale, hc_head_base, norm)
public final class Transformer {
    public let config: ModelConfig
    public let embed: ParallelEmbedding
    public let layers: [Block]
    public let mtp: [MTPBlock]
    public let norm: RMSNorm
    public let head: ParallelHead

    // Top-level HC head parameters
    public let hcHeadFn: Tensor          // [hc_mult, hc_mult*dim] f32
    public let hcHeadBase: Tensor        // [hc_mult] f32
    public let hcHeadScale: Tensor       // [1] f32

    public init(config: ModelConfig,
                embed: ParallelEmbedding,
                layers: [Block],
                mtp: [MTPBlock],
                norm: RMSNorm,
                head: ParallelHead,
                hcHeadFn: Tensor, hcHeadBase: Tensor, hcHeadScale: Tensor) {
        self.config = config
        self.embed = embed
        self.layers = layers
        self.mtp = mtp
        self.norm = norm
        self.head = head
        self.hcHeadFn = hcHeadFn
        self.hcHeadBase = hcHeadBase
        self.hcHeadScale = hcHeadScale
    }

    /// Forward a batch of new tokens. Returns logits of shape [b, vocab].
    /// NOT IMPLEMENTED — needs the full chain (embed lookup → HC expand →
    /// per-layer Block → ParallelHead). All sublayer forwards are still stubs.
    public func forward(inputIds: [[Int]], startPos: Int) -> Tensor {
        fatalError("Transformer.forward not implemented — porting target: model.py:802")
    }
}
