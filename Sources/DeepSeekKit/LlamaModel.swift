import Foundation
import Metal

/// Minimal config for Llama-family architectures (TODO §10.2 / T2).
/// Mirrors the fields `LlamaConfig` carries in HuggingFace
/// Transformers and `llama.cpp` reads out of GGUF metadata. No HC,
/// no MLA, no compress-ratios — that DeepSeek-specific state lives
/// in `ModelConfig`.
public struct LlamaConfig: Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let nHeads: Int
    public let nKVHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let nLayers: Int
    public let maxSeqLen: Int
    public let normEps: Float
    public let ropeTheta: Float
    /// Element count of rotary application. Llama 1/2 rotates the
    /// full head; Llama 3 / Mistral are the same. Some variants
    /// (CodeLlama) partial-rotate — `< headDim`.
    public let ropeHeadDim: Int

    public init(vocabSize: Int, hiddenSize: Int,
                nHeads: Int, nKVHeads: Int, headDim: Int,
                intermediateSize: Int, nLayers: Int,
                maxSeqLen: Int, normEps: Float,
                ropeTheta: Float, ropeHeadDim: Int? = nil)
    {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.intermediateSize = intermediateSize
        self.nLayers = nLayers
        self.maxSeqLen = maxSeqLen
        self.normEps = normEps
        self.ropeTheta = ropeTheta
        self.ropeHeadDim = ropeHeadDim ?? headDim
    }
}

/// Llama-family transformer (TODO §10.2 / T2). Embedding + N decoder
/// layers + final RMSNorm + LM head. Matches the structure shipped
/// by every Llama / Mistral / Qwen / CodeLlama checkpoint after the
/// `LlamaForCausalLM` HF schema landed.
///
/// `forward(inputIds:startPos:)` returns logits for the LAST token
/// of each batch row only — the only position the sampler needs, and
/// it skips a (full × vocab) GEMM per row × position. Matches what
/// `ParallelHead` does for the DeepSeek path.
public final class LlamaModel {
    public let config: LlamaConfig
    public let embed: ParallelEmbedding
    public let layers: [LlamaDecoderLayer]
    public let norm: RMSNorm
    public let lmHead: Linear

    public init(config: LlamaConfig,
                embed: ParallelEmbedding,
                layers: [LlamaDecoderLayer],
                norm: RMSNorm,
                lmHead: Linear)
    {
        self.config = config
        self.embed = embed
        self.layers = layers
        self.norm = norm
        self.lmHead = lmHead
    }

    /// `inputIds`: outer = batch, inner = seqlen. All rows must
    /// share the same seqlen (no ragged batches). `startPos` is the
    /// absolute position of `inputIds[:, 0]` in the model's context
    /// — 0 for prefill, growing by 1 on each decoded token.
    public func forward(inputIds: [[Int]], startPos: Int) -> Tensor {
        let B = inputIds.count
        precondition(B > 0)
        let S = inputIds[0].count
        for row in inputIds {
            precondition(row.count == S, "ragged batch not supported")
        }

        let flatIds: [Int32] = inputIds.flatMap { $0.map(Int32.init) }
        var cmd = Device.shared.queue.makeCommandBuffer()!

        // 1. Token embedding → [B*S, dim] then reshape to [B, S, dim].
        let embedded = embed.lookup(flatIds, in: cmd)
        var x = embedded.reshape([B, S, config.hiddenSize])

        // 2. Residual stream through every decoder layer. Each layer
        //    commits its own work via the shared command buffer
        //    (mutated by `inout cmd` inside `StandardMHA.callAsFunction`).
        for layer in layers {
            x = layer(x, startPos: startPos, in: &cmd)
        }

        // 3. Final RMSNorm.
        let normed = norm(x, in: cmd)

        // 4. Take the last token per batch row via a blit.
        let dim = config.hiddenSize
        let lastTok = Tensor.empty(shape: [B, dim], dtype: .f32)
        let blit = cmd.makeBlitCommandEncoder()!
        let bytesPerRow = dim * MemoryLayout<Float>.size
        for b in 0..<B {
            let src = ((b * S) + (S - 1)) * bytesPerRow
            let dst = b * bytesPerRow
            blit.copy(from: normed.buffer,
                       sourceOffset: normed.offset + src,
                       to: lastTok.buffer,
                       destinationOffset: dst,
                       size: bytesPerRow)
        }
        blit.endEncoding()

        // 5. LM head — F32 logits straight into the sampler. Casting
        //    to BF16 here would warp top-K / temperature ties.
        let logits = lmHead(lastTok, in: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        return logits
    }

    /// Drop every layer's KV cache. Used between unrelated prompts
    /// so the pages return to the system.
    public func releaseCache() {
        for l in layers { l.releaseCache() }
    }
}
