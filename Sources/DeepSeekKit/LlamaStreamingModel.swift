import Foundation
import Metal

/// Streaming variant of `LlamaModel`: the layers' weights are
/// dequantized lazily per forward pass instead of materialized
/// upfront, and the source mmap pages are advised `MADV_DONTNEED`
/// between layers so the OS is free to evict them under memory
/// pressure. Matches the `.streaming` semantics on the safetensors
/// path (`WeightLoader.streamingEnabled`).
///
/// When to reach for this:
///   - You're on a 16 GB Mac and the model would otherwise
///     dequantize to more than `availableRAM × 0.7`.
///   - You're OK trading ~5× lower per-token throughput (dequant
///     runs every forward, vs. once at load) for a working set
///     that doesn't push the system into jetsam.
///
/// Architecture differences vs `LlamaModel`:
///   - `embed`, `norm`, `lmHead`, `rope` stay materialized — they
///     account for a small slice of total weights (mostly the
///     embedding table) and are used at *every* forward, so
///     evicting them would just thrash the page cache.
///   - Per-layer weights live as GGUF tensor info refs. Each
///     forward, the layer's 9 weight tensors are dequantized into
///     fresh MTLBuffers, the layer is rebuilt around them, run,
///     then dropped. The dequant outputs are held until the
///     command buffer commits via `forwardScratch`.
///   - KV caches are stashed externally (in this class) and
///     restored onto each freshly built `StandardMHA` instance —
///     the layer rebuild would otherwise lose the cache.
public final class LlamaStreamingModel: LlamaForwardModel {
    public let gguf: GGUFFile
    public let config: LlamaConfig
    public let embed: ParallelEmbedding
    public let norm: RMSNorm
    public let lmHead: Linear
    public let rope: RoPE
    public let weightDtype: DType

    /// Per-layer KV cache, persisted across forward calls. Each
    /// element is `nil` until the layer runs its first forward; the
    /// `ensureCache(B:)` inside `StandardMHA` then sizes it on
    /// demand.
    private var kCaches: [Tensor?]
    private var vCaches: [Tensor?]
    private var cachedB: Int = 0

    /// Strong refs to every per-forward weight tensor, so the
    /// MTLBuffers survive until the outer command buffer commits.
    /// Cleared after `cmd.waitUntilCompleted()`. The encoder
    /// retains its buffers internally, but we don't want to rely
    /// on the undocumented exact semantics — explicit retention
    /// here is the safe pattern.
    private var forwardScratch: [Tensor] = []

    public init(gguf: GGUFFile,
                maxSeqLenOverride: Int? = nil,
                weightDtype: DType = .f32) throws
    {
        self.gguf = gguf
        self.weightDtype = weightDtype

        // Parse config out of the GGUF metadata. Identical logic to
        // `LlamaModel.fromGGUF` — refactoring it into a shared helper
        // would force two consumers to share an init style that
        // doesn't cleanly fit either, so we duplicate.
        let meta = gguf.header.metadata
        guard case .string(let arch)? = meta["general.architecture"] else {
            throw GGUFError.malformed("missing general.architecture metadata")
        }
        guard ModelArchitecture.fromGGUFArchString(arch) == .llama else {
            throw GGUFError.malformed(
                "general.architecture '\(arch)' is not Llama-family")
        }
        let p = arch.lowercased()
        let hiddenSize = try Self.metaInt(meta, "\(p).embedding_length")
        let nLayers    = try Self.metaInt(meta, "\(p).block_count")
        let nHeads     = try Self.metaInt(meta, "\(p).attention.head_count")
        let intermediateSize = try Self.metaInt(meta, "\(p).feed_forward_length")
        let ctxLen     = try Self.metaInt(meta, "\(p).context_length")
        let nKVHeads = (try? Self.metaInt(meta, "\(p).attention.head_count_kv"))
            ?? nHeads
        let normEps = (try? Self.metaFloat(meta,
                "\(p).attention.layer_norm_rms_epsilon")) ?? 1e-5
        let ropeTheta = (try? Self.metaFloat(meta, "\(p).rope.freq_base"))
            ?? 10_000.0
        let headDim = hiddenSize / nHeads
        let ropeHeadDim = (try? Self.metaInt(meta,
            "\(p).rope.dimension_count")) ?? headDim
        let maxSeqLen = maxSeqLenOverride ?? ctxLen

        let vocabSize: Int
        if let v = try? Self.metaInt(meta, "\(p).vocab_size") {
            vocabSize = v
        } else if case .array(let toks)? = meta["tokenizer.ggml.tokens"] {
            vocabSize = toks.count
        } else {
            throw GGUFError.malformed("cannot determine vocab size")
        }

        self.config = LlamaConfig(
            vocabSize: vocabSize, hiddenSize: hiddenSize,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            intermediateSize: intermediateSize, nLayers: nLayers,
            maxSeqLen: maxSeqLen, normEps: normEps,
            ropeTheta: ropeTheta, ropeHeadDim: ropeHeadDim)

        // Shared RoPE freqs.
        let freqsArr = YaRN.precomputeFreqsCis(
            dim: ropeHeadDim, seqlen: maxSeqLen, originalSeqLen: 0,
            base: ropeTheta, factor: 1.0, betaFast: 32, betaSlow: 1)
        let freqs = freqsArr.withUnsafeBytes { raw in
            Tensor.from(bytes: raw,
                         shape: [maxSeqLen, ropeHeadDim / 2, 2],
                         dtype: .f32)
        }
        self.rope = RoPE(ropeHeadDim: ropeHeadDim, freqs: freqs)

        // Embedding / final norm / LM head are materialized once —
        // they're used at every forward and small enough relative to
        // the layer weights that streaming them buys nothing.
        let embedW = try gguf.load("token_embd.weight",
                                     outputDtype: weightDtype)
        self.embed = ParallelEmbedding(
            vocabSize: vocabSize, dim: hiddenSize, weight: embedW)
        let outputNormW = try gguf.load("output_norm.weight",
                                          outputDtype: weightDtype)
        self.norm = RMSNorm(weight: outputNormW, eps: normEps)
        let headW: Tensor
        if gguf.info(name: "output.weight") != nil {
            headW = try gguf.load("output.weight",
                                    outputDtype: weightDtype)
        } else {
            headW = embedW
        }
        self.lmHead = Linear(inFeatures: hiddenSize,
                              outFeatures: vocabSize,
                              weight: headW, scale: nil)

        self.kCaches = Array(repeating: nil, count: nLayers)
        self.vCaches = Array(repeating: nil, count: nLayers)
    }

    /// Forward pass. Same contract as `LlamaModel.forward`: returns
    /// `[B, vocabSize]` F32 logits for the last position of each
    /// batch row. Internally, every layer's weights are dequantized
    /// once at use time and dropped after; the mmap pages backing
    /// those weights get a `POSIX_MADV_DONTNEED` hint immediately
    /// after, telling the kernel it's safe to evict them.
    public func forward(inputIds: [[Int]], startPos: Int) -> Tensor {
        let B = inputIds.count
        precondition(B > 0)
        let S = inputIds[0].count
        for row in inputIds {
            precondition(row.count == S, "ragged batch not supported")
        }

        let flatIds: [Int32] = inputIds.flatMap { $0.map(Int32.init) }
        var cmd = Device.shared.queue.makeCommandBuffer()!

        // 1. Embedding.
        let embedded = embed.lookup(flatIds, in: cmd)
        var x = embedded.reshape([B, S, config.hiddenSize])

        // 2. Per-layer streaming. The scratch holds references to
        //    the dequantized weights so they survive until the GPU
        //    consumes them at commit time.
        forwardScratch.removeAll(keepingCapacity: true)
        for layerId in 0..<config.nLayers {
            x = runLayer(x, layerId: layerId, startPos: startPos,
                          B: B, in: &cmd)
            // Hint the kernel: this layer's bytes can go. The
            // kernel may keep them anyway if there's no memory
            // pressure (POSIX_MADV_DONTNEED is advisory), which is
            // exactly what we want on machines that have headroom.
            let prefix = "blk.\(layerId)."
            if let range = gguf.byteRange(forNamePrefix: prefix) {
                gguf.madviseRange(offset: range.offset,
                                   length: range.length,
                                   advice: POSIX_MADV_DONTNEED)
            }
        }

        // 3. Final norm + last-token slice + LM head.
        let normed = norm(x, in: cmd)
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
        let logits = lmHead(lastTok, in: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        // GPU is done — every weight in forwardScratch can drop.
        forwardScratch.removeAll(keepingCapacity: true)
        return logits
    }

    /// Drop every layer's KV cache. ARC frees the underlying
    /// MTLBuffers; the next forward re-allocates.
    public func releaseCache() {
        for i in 0..<kCaches.count {
            kCaches[i] = nil
            vCaches[i] = nil
        }
        cachedB = 0
    }

    // MARK: - Per-layer dequant + run

    private func runLayer(_ x: Tensor, layerId: Int, startPos: Int,
                           B: Int, in cmd: inout MTLCommandBuffer) -> Tensor
    {
        let prefix = "blk.\(layerId)."

        // Dequant 9 weights for this layer. Each `try!` is OK because
        // `LlamaStreamingModel.init` already validated the layer names
        // via `LlamaConfig.nLayers` against the GGUF block_count.
        // The append-to-forwardScratch keeps the MTLBuffers alive
        // until the outer cmd commits.
        let attnNormW = try! gguf.load("\(prefix)attn_norm.weight",
                                         outputDtype: weightDtype)
        let wqW = try! gguf.load("\(prefix)attn_q.weight",
                                   outputDtype: weightDtype)
        let wkW = try! gguf.load("\(prefix)attn_k.weight",
                                   outputDtype: weightDtype)
        let wvW = try! gguf.load("\(prefix)attn_v.weight",
                                   outputDtype: weightDtype)
        let woW = try! gguf.load("\(prefix)attn_output.weight",
                                   outputDtype: weightDtype)
        let ffnNormW = try! gguf.load("\(prefix)ffn_norm.weight",
                                        outputDtype: weightDtype)
        let gateW = try! gguf.load("\(prefix)ffn_gate.weight",
                                     outputDtype: weightDtype)
        let upW   = try! gguf.load("\(prefix)ffn_up.weight",
                                     outputDtype: weightDtype)
        let downW = try! gguf.load("\(prefix)ffn_down.weight",
                                     outputDtype: weightDtype)
        forwardScratch.append(contentsOf:
            [attnNormW, wqW, wkW, wvW, woW,
             ffnNormW, gateW, upW, downW])

        // Build sublayers around the just-loaded weights.
        let attnNorm = RMSNorm(weight: attnNormW, eps: config.normEps)
        let wQ = Linear(inFeatures: config.hiddenSize,
                         outFeatures: config.nHeads * config.headDim,
                         weight: wqW, scale: nil)
        let wK = Linear(inFeatures: config.hiddenSize,
                         outFeatures: config.nKVHeads * config.headDim,
                         weight: wkW, scale: nil)
        let wV = Linear(inFeatures: config.hiddenSize,
                         outFeatures: config.nKVHeads * config.headDim,
                         weight: wvW, scale: nil)
        let wO = Linear(inFeatures: config.nHeads * config.headDim,
                         outFeatures: config.hiddenSize,
                         weight: woW, scale: nil)
        let attn = StandardMHA(
            nHeads: config.nHeads, nKVHeads: config.nKVHeads,
            headDim: config.headDim, maxSeq: config.maxSeqLen,
            wQ: wQ, wK: wK, wV: wV, wO: wO, rope: rope)

        // Restore KV cache from the previous forward. The first call
        // for this layer has nil here, which `ensureCache(B:)` inside
        // StandardMHA will materialize.
        attn.kCache = kCaches[layerId]
        attn.vCache = vCaches[layerId]
        attn.cachedB = cachedB

        let ffnNorm = RMSNorm(weight: ffnNormW, eps: config.normEps)
        let wGate = Linear(inFeatures: config.hiddenSize,
                            outFeatures: config.intermediateSize,
                            weight: gateW, scale: nil)
        let wUp   = Linear(inFeatures: config.hiddenSize,
                            outFeatures: config.intermediateSize,
                            weight: upW, scale: nil)
        let wDown = Linear(inFeatures: config.intermediateSize,
                            outFeatures: config.hiddenSize,
                            weight: downW, scale: nil)
        let ffn = SwiGLU(wGate: wGate, wUp: wUp, wDown: wDown)

        let layer = LlamaDecoderLayer(
            layerId: layerId,
            attnNorm: attnNorm, attn: attn,
            ffnNorm: ffnNorm, ffn: ffn)
        let y = layer(x, startPos: startPos, in: &cmd)

        // Stash the (possibly newly-allocated) KV cache back so the
        // next forward picks up where we left off.
        kCaches[layerId] = attn.kCache
        vCaches[layerId] = attn.vCache
        cachedB = attn.cachedB
        // Keep the KV cache buffers alive across the eventual ARC
        // drop of `attn` — kCaches/vCaches already hold strong refs,
        // so we don't double-add them to forwardScratch.
        return y
    }

    // MARK: - Metadata helpers (duplicated from LlamaModel.fromGGUF)

    private static func metaInt(_ meta: [String: GGUFValue],
                                  _ key: String) throws -> Int {
        guard let v = meta[key] else {
            throw GGUFError.malformed("missing GGUF metadata key '\(key)'")
        }
        switch v {
        case .int64(let n):   return Int(n)
        case .uint64(let n):  return Int(n)
        case .float64(let n): return Int(n)
        default: throw GGUFError.malformed("'\(key)' is not numeric")
        }
    }

    private static func metaFloat(_ meta: [String: GGUFValue],
                                    _ key: String) throws -> Float {
        guard let v = meta[key] else {
            throw GGUFError.malformed("missing GGUF metadata key '\(key)'")
        }
        switch v {
        case .float64(let n): return Float(n)
        case .int64(let n):   return Float(n)
        case .uint64(let n):  return Float(n)
        default: throw GGUFError.malformed("'\(key)' is not numeric")
        }
    }
}
