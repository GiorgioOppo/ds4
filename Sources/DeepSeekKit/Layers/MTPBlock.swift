import Foundation
import Metal

/// Multi-Token Prediction block. Mirrors `MTPBlock` in
/// `Original/DeepSeek-V4-Pro/inference/model.py` lines 738–766.
///
/// Inherits the structure of `Block` but adds:
///   - e_proj, h_proj projections that fuse the new-token embedding with the
///     previous block's hidden state
///   - enorm, hnorm pre-projection RMSNorms
///   - norm + per-block hc_head_fn / hc_head_base / hc_head_scale that are
///     used by ParallelHead at the end
///   - holds (non-owning) references to the shared embedding table and head
///
/// forward:
///   e = enorm(embed(input_ids))
///   x = hnorm(x)
///   x = e_proj(e).unsqueeze(2) + h_proj(x)
///   x = Block.forward(x, start_pos, input_ids)
///   logits = head(x, hc_head_fn, hc_head_scale, hc_head_base, norm)
public final class MTPBlock {
    public let block: Block
    public let eProj: Linear
    public let hProj: Linear
    public let eNorm: RMSNorm
    public let hNorm: RMSNorm
    public let norm: RMSNorm
    public let hcHeadFn: Tensor          // [hc_mult, hc_mult*dim] f32
    public let hcHeadBase: Tensor        // [hc_mult] f32
    public let hcHeadScale: Tensor       // [1] f32

    /// Embed and head are owned by Transformer; MTPBlock holds non-owning refs.
    public weak var embed: ParallelEmbedding?
    public weak var head: ParallelHead?

    public init(block: Block,
                eProj: Linear, hProj: Linear,
                eNorm: RMSNorm, hNorm: RMSNorm, norm: RMSNorm,
                hcHeadFn: Tensor, hcHeadBase: Tensor, hcHeadScale: Tensor) {
        self.block = block
        self.eProj = eProj; self.hProj = hProj
        self.eNorm = eNorm; self.hNorm = hNorm; self.norm = norm
        self.hcHeadFn = hcHeadFn; self.hcHeadBase = hcHeadBase; self.hcHeadScale = hcHeadScale
    }
}
