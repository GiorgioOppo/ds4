import Foundation
import Metal

/// Multi-Token Prediction block. Mirrors `MTPBlock` in
/// `Reference/inference/model.py` lines 738–766.
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

    /// Forward port of `MTPBlock.forward` (model.py:756-766).
    ///
    /// `x`: [B, S, hc, dim] — hidden state from the previous block.
    /// `inputIds`: flattened [B*S] token ids; the *next* token after the
    /// current position is what we predict.
    /// Returns logits `[B, vocab]` for the speculative prediction.
    public func callAsFunction(_ x: Tensor, startPos: Int, inputIds: [Int32],
                                in cmd: inout MTLCommandBuffer) -> Tensor {
        precondition(x.dtype == .f32 && x.shape.count == 4)
        guard let embed = embed, let head = head else {
            fatalError("MTPBlock.embed and .head must be wired by the parent Transformer")
        }
        let B = x.shape[0], S = x.shape[1], HC = x.shape[2], D = x.shape[3]
        let N = B * S

        // 1. e = embed(input_ids).reshape([N, dim])
        let e = embed.lookup(inputIds, in: cmd)         // [N, dim]
        let eN = eNorm(e, in: cmd)

        // 2. x = hnorm(x.flatten(2))   then projected back to [N, hc, dim]
        // The reference applies hnorm on the [b, s, hc*dim] flattened view.
        let xN = hNorm(x.reshape([N, HC * D]), in: cmd).reshape([N, HC, D])

        // 3. x = e_proj(e).unsqueeze(2) + h_proj(x)
        // e_proj output: [N, dim] → broadcast across hc dim
        // h_proj output: [N, hc*dim] reshaped to [N, hc, dim]
        let eProjOut = eProj(eN, in: cmd).reshape([N, 1, D])
        let hProjOut = hProj(xN.reshape([N, HC * D]), in: cmd).reshape([N, HC, D])

        // y[N, hc, d] = e_proj_out[N, 0, d] + h_proj_out[N, hc, d] (broadcast)
        let combined = Tensor.empty(shape: [N, HC, D], dtype: .f32)
        broadcastAddSecondAxis(out: combined, base: hProjOut, addend: eProjOut,
                                N: N, HC: HC, D: D, in: cmd)

        // 4. Run the inner Block.forward with the fused input.
        let combined4 = combined.reshape([B, S, HC, D])
        let after = block(combined4, startPos: startPos, inputIds: inputIds, in: &cmd)

        // 5. Head with the MTP-block's own (hc_head_fn, base, scale, norm).
        return head(after, hcFn: hcHeadFn, hcScale: hcHeadScale,
                    hcBase: hcHeadBase, norm: norm, in: cmd)
    }

    /// out[n, h, d] = base[n, h, d] + addend[n, 0, d]   (addend broadcasts
    /// over the hc axis). Implemented as a one-shot kernel dispatch using
    /// the existing addInPlace + a manual broadcast copy. To avoid adding
    /// a new kernel we do it in two passes on small tensors.
    private func broadcastAddSecondAxis(out: Tensor, base: Tensor, addend: Tensor,
                                         N: Int, HC: Int, D: Int,
                                         in cmd: MTLCommandBuffer) {
        // 1. out := base (blit copy)
        let blit = cmd.makeBlitCommandEncoder()!
        blit.copy(from: base.buffer, sourceOffset: base.offset,
                  to: out.buffer, destinationOffset: 0,
                  size: N * HC * D * MemoryLayout<Float>.size)
        blit.endEncoding()
        // 2. out += addend, broadcast across HC.
        // Use Elementwise.addInPlace HC times, each on a [N, D] slice.
        for h in 0..<HC {
            let outSlice = Tensor(shape: [N, D], dtype: .f32,
                                   buffer: out.buffer,
                                   offset: h * D * MemoryLayout<Float>.size)
            // The slice as written above is wrong because it's strided in
            // memory ([N, HC, D] row-major). Fall back to a direct kernel
            // dispatch using broadcast_row_mul-style pattern... actually
            // simpler: call a small helper that adds addend[N, 1, D] into
            // out[N, h, D] via a custom dispatch below.
            _ = outSlice
            addendIntoSlice(out: out, addend: addend, N: N, HC: HC, D: D, h: h, in: cmd)
        }
    }

    /// out[n, h, d] += addend[n, 0, d] for the fixed `h`. Uses Metal's
    /// existing `add_inplace_f32` kernel by binding the right offset views.
    private func addendIntoSlice(out: Tensor, addend: Tensor,
                                   N: Int, HC: Int, D: Int, h: Int,
                                   in cmd: MTLCommandBuffer) {
        // out[n, h, d] lives at offset (n * HC + h) * D + d. We need a
        // strided add, which add_inplace_f32 doesn't do. Use a tiny
        // per-h compute encoder that walks (n, d).
        let pAdd = Device.shared.makePipeline("add_inplace_f32")
        // We synthesise [N, D] views by doing N independent blit-style
        // calls — but add_inplace_f32 expects contiguous y and x of the
        // same length. For correctness across the HC stride we serialise
        // N rows of D floats at a time.
        let bytesPerRow = D * MemoryLayout<Float>.size
        for n in 0..<N {
            let outOff = ((n * HC + h) * D) * MemoryLayout<Float>.size
            let addOff = (n * D) * MemoryLayout<Float>.size + addend.offset
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pAdd)
            enc.setBuffer(out.buffer, offset: outOff, index: 0)
            enc.setBuffer(addend.buffer, offset: addOff, index: 1)
            var nn = UInt32(D)
            enc.setBytes(&nn, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: D, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(D, 256),
                                                                height: 1, depth: 1))
            enc.endEncoding()
            _ = bytesPerRow
        }
    }
}
