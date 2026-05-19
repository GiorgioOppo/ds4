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

// MARK: - GGUF factory

extension LlamaModel {
    /// Construct a `LlamaModel` from a GGUF file (TODO §10.2 / T2
    /// final piece). Pulls every dimension off the file's metadata,
    /// dequants the weight tensors through `GGUFFile.load` (which
    /// dispatches the Q8_0 / Q4_0 / Q4_K kernels added earlier in
    /// T2), and wires up Linears / RMSNorms / RoPE / KV caches so
    /// `forward(...)` is immediately callable.
    ///
    /// `maxSeqLen` defaults to the GGUF's `context_length` metadata
    /// but can be capped by `maxSeqLenOverride` so a 128k-context
    /// checkpoint doesn't try to allocate KV caches sized for that
    /// full window on a small Mac. Cap is per-layer:
    /// `[B, maxSeqLen, Hkv, D] × n_layers × 2 (K + V)` so it adds
    /// up fast.
    ///
    /// Architectures recognized: every value
    /// `ModelArchitecture.fromGGUFArchString(_:)` resolves to
    /// `.llama` (llama, mistral, qwen / qwen2 / qwen3, codellama).
    /// Tensor naming convention is the post-`llama.cpp`-`convert.py`
    /// canonical form: `token_embd.weight` for the embedding,
    /// `blk.N.{attn_norm, attn_q, attn_k, attn_v, attn_output,
    /// ffn_norm, ffn_gate, ffn_up, ffn_down}.weight`,
    /// `output_norm.weight`, optional `output.weight` (tied to
    /// `token_embd` when absent — the common Llama 3 setup).
    /// `weightDtype` controls the dequant target for quantized
    /// weights (Q8_0 / Q4_0 / Q4_K). `.f32` (default) is safer
    /// numerically; `.bf16` halves the resident memory of those
    /// weights at no measurable accuracy loss in practice. Q5_K /
    /// Q6_K silently fall back to F32 — their BF16 kernels haven't
    /// landed yet. Pass-through dtypes in the GGUF (F32, F16, BF16)
    /// ignore this knob and load as-is.
    public static func fromGGUF(_ gguf: GGUFFile,
                                  maxSeqLenOverride: Int? = nil,
                                  weightDtype: DType = .f32) throws
        -> LlamaModel
    {
        let meta = gguf.header.metadata

        guard case .string(let arch)? = meta["general.architecture"] else {
            throw GGUFError.malformed(
                "missing general.architecture metadata")
        }
        guard ModelArchitecture.fromGGUFArchString(arch) == .llama else {
            throw GGUFError.malformed(
                "general.architecture '\(arch)' is not Llama-family; "
                + "use the DeepSeek-V4 path instead")
        }
        let p = arch.lowercased()  // metadata key prefix

        // Required scalar dimensions.
        let hiddenSize       = try metaInt(meta, "\(p).embedding_length")
        let nLayers          = try metaInt(meta, "\(p).block_count")
        let nHeads           = try metaInt(meta, "\(p).attention.head_count")
        let intermediateSize = try metaInt(meta, "\(p).feed_forward_length")
        let ctxLen           = try metaInt(meta, "\(p).context_length")

        // Optional with sensible Llama defaults.
        let nKVHeads = (try? metaInt(meta, "\(p).attention.head_count_kv"))
            ?? nHeads
        let normEps = (try? metaFloat(meta, "\(p).attention.layer_norm_rms_epsilon"))
            ?? 1e-5
        let ropeTheta = (try? metaFloat(meta, "\(p).rope.freq_base"))
            ?? 10_000.0
        let headDim = hiddenSize / nHeads
        let ropeHeadDim = (try? metaInt(meta, "\(p).rope.dimension_count"))
            ?? headDim
        let maxSeqLen = maxSeqLenOverride ?? ctxLen

        // Vocab size: prefer explicit metadata, fall back to the
        // tokens array length (every Llama GGUF has one or the other).
        let vocabSize: Int
        if let v = try? metaInt(meta, "\(p).vocab_size") {
            vocabSize = v
        } else if case .array(let toks)? = meta["tokenizer.ggml.tokens"] {
            vocabSize = toks.count
        } else {
            throw GGUFError.malformed(
                "GGUF: cannot determine vocab size (no \(p).vocab_size "
                + "and no tokenizer.ggml.tokens)")
        }

        let config = LlamaConfig(
            vocabSize: vocabSize,
            hiddenSize: hiddenSize,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            intermediateSize: intermediateSize,
            nLayers: nLayers,
            maxSeqLen: maxSeqLen,
            normEps: normEps,
            ropeTheta: ropeTheta,
            ropeHeadDim: ropeHeadDim)

        // Shared RoPE freqs. Llama doesn't use YaRN (factor=1,
        // originalSeqLen=0 bypasses the YaRN correction logic in
        // `precomputeFreqsCis`).
        let freqsArr = YaRN.precomputeFreqsCis(
            dim: ropeHeadDim,
            seqlen: maxSeqLen,
            originalSeqLen: 0,
            base: ropeTheta,
            factor: 1.0,
            betaFast: 32, betaSlow: 1)
        let freqs = freqsArr.withUnsafeBytes { raw in
            Tensor.from(bytes: raw,
                         shape: [maxSeqLen, ropeHeadDim / 2, 2],
                         dtype: .f32)
        }
        let rope = RoPE(ropeHeadDim: ropeHeadDim, freqs: freqs)

        // Embedding (also reused as the LM-head weight when the
        // checkpoint ties them — common in Llama 3).
        let embedW = try gguf.load("token_embd.weight",
                                     outputDtype: weightDtype)
        let embed = ParallelEmbedding(
            vocabSize: vocabSize, dim: hiddenSize, weight: embedW)

        // Decoder layers.
        var layers: [LlamaDecoderLayer] = []
        layers.reserveCapacity(nLayers)
        for layerId in 0..<nLayers {
            let prefix = "blk.\(layerId)."

            // RMSNorms ship as F32 in GGUF (pass-through, no
            // dequant); the `outputDtype` arg only kicks in for
            // quantized tensors, so threading it through unchanged
            // is safe.
            let attnNormW = try gguf.load("\(prefix)attn_norm.weight",
                                            outputDtype: weightDtype)
            let attnNorm  = RMSNorm(weight: attnNormW, eps: normEps)

            let wqW = try gguf.load("\(prefix)attn_q.weight",
                                      outputDtype: weightDtype)
            let wkW = try gguf.load("\(prefix)attn_k.weight",
                                      outputDtype: weightDtype)
            let wvW = try gguf.load("\(prefix)attn_v.weight",
                                      outputDtype: weightDtype)
            let woW = try gguf.load("\(prefix)attn_output.weight",
                                      outputDtype: weightDtype)
            let wQ = Linear(inFeatures: hiddenSize,
                             outFeatures: nHeads * headDim,
                             weight: wqW, scale: nil)
            let wK = Linear(inFeatures: hiddenSize,
                             outFeatures: nKVHeads * headDim,
                             weight: wkW, scale: nil)
            let wV = Linear(inFeatures: hiddenSize,
                             outFeatures: nKVHeads * headDim,
                             weight: wvW, scale: nil)
            let wO = Linear(inFeatures: nHeads * headDim,
                             outFeatures: hiddenSize,
                             weight: woW, scale: nil)
            let attn = StandardMHA(
                nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                maxSeq: maxSeqLen,
                wQ: wQ, wK: wK, wV: wV, wO: wO, rope: rope)

            let ffnNormW = try gguf.load("\(prefix)ffn_norm.weight",
                                           outputDtype: weightDtype)
            let ffnNorm  = RMSNorm(weight: ffnNormW, eps: normEps)

            let gateW = try gguf.load("\(prefix)ffn_gate.weight",
                                        outputDtype: weightDtype)
            let upW   = try gguf.load("\(prefix)ffn_up.weight",
                                        outputDtype: weightDtype)
            let downW = try gguf.load("\(prefix)ffn_down.weight",
                                        outputDtype: weightDtype)
            let wGate = Linear(inFeatures: hiddenSize,
                                outFeatures: intermediateSize,
                                weight: gateW, scale: nil)
            let wUp   = Linear(inFeatures: hiddenSize,
                                outFeatures: intermediateSize,
                                weight: upW, scale: nil)
            let wDown = Linear(inFeatures: intermediateSize,
                                outFeatures: hiddenSize,
                                weight: downW, scale: nil)
            let ffn = SwiGLU(wGate: wGate, wUp: wUp, wDown: wDown)

            layers.append(LlamaDecoderLayer(
                layerId: layerId,
                attnNorm: attnNorm, attn: attn,
                ffnNorm: ffnNorm, ffn: ffn))
        }

        // Final norm + LM head.
        let outputNormW = try gguf.load("output_norm.weight",
                                          outputDtype: weightDtype)
        let norm = RMSNorm(weight: outputNormW, eps: normEps)

        // Tied embeddings: when `output.weight` is absent the LM
        // head reuses `token_embd.weight`. Detection is a name
        // lookup against the tensor table — `info(name:)` doesn't
        // touch the data, just the in-memory index.
        let headW: Tensor
        if gguf.info(name: "output.weight") != nil {
            headW = try gguf.load("output.weight",
                                    outputDtype: weightDtype)
        } else {
            headW = embedW
        }
        let lmHead = Linear(inFeatures: hiddenSize,
                             outFeatures: vocabSize,
                             weight: headW, scale: nil)

        return LlamaModel(config: config,
                           embed: embed,
                           layers: layers,
                           norm: norm,
                           lmHead: lmHead)
    }
}

// MARK: - GGUF metadata helpers

/// Read a numeric metadata value as Int. Accepts int64 / uint64 /
/// float64 (GGUF stores `block_count` etc. as uint64 typically) and
/// throws if the key is missing or non-numeric.
private func metaInt(_ meta: [String: GGUFValue],
                      _ key: String) throws -> Int
{
    guard let v = meta[key] else {
        throw GGUFError.malformed("missing GGUF metadata key '\(key)'")
    }
    switch v {
    case .int64(let n):   return Int(n)
    case .uint64(let n):  return Int(n)
    case .float64(let n): return Int(n)
    default:
        throw GGUFError.malformed("'\(key)' is not numeric")
    }
}

/// Like `metaInt` but returns Float. Accepts the same three numeric
/// cases; common for epsilons / RoPE base.
private func metaFloat(_ meta: [String: GGUFValue],
                        _ key: String) throws -> Float
{
    guard let v = meta[key] else {
        throw GGUFError.malformed("missing GGUF metadata key '\(key)'")
    }
    switch v {
    case .float64(let n): return Float(n)
    case .int64(let n):   return Float(n)
    case .uint64(let n):  return Float(n)
    default:
        throw GGUFError.malformed("'\(key)' is not numeric")
    }
}
