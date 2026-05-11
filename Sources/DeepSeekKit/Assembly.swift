import Foundation
import Metal

/// Factory functions that build a `Transformer` from a `ModelConfig`.
///
/// Two paths:
///   - `Transformer.randomInit(config:)`  — fills every weight with a small
///     pseudo-random f32 value. Good enough to smoke-test the entire forward
///     chain end-to-end without weights on disk.
///   - `Transformer.load(config:from:)`   — reads a directory of safetensors
///     shards using the canonical V4 weight name conventions (see
///     `Reference/inference/convert.py`). For names that aren't found we
///     fall back to random init and log a warning. This lets a partially-
///     converted checkpoint still produce a forward pass instead of crashing.
public extension Transformer {

    static func randomInit(config: ModelConfig) -> Transformer {
        var rng = MiniRNG(seed: 0xDEADC0DE)
        let dim = config.dim
        let hc = config.hcMult
        let mixHc = (2 + hc) * hc
        let nLayers = config.nLayers
        let nExperts = config.nRoutedExperts

        // Top-level
        let embedW = AssemblyHelpers.randomTensor([config.vocabSize, dim], rng: &rng, scale: 0.02)
        let embed = ParallelEmbedding(vocabSize: config.vocabSize, dim: dim, weight: embedW)

        let normW = AssemblyHelpers.onesTensor([dim])
        let norm = RMSNorm(weight: normW, eps: config.normEps)

        let lmHeadW = AssemblyHelpers.randomTensor([config.vocabSize, dim], rng: &rng, scale: 0.02)
        let head = ParallelHead(vocabSize: config.vocabSize, dim: dim,
                                normEps: config.normEps, hcEps: config.hcEps,
                                weight: lmHeadW)

        let hcHeadFn = AssemblyHelpers.randomTensor([hc, hc * dim], rng: &rng, scale: 0.02)
        let hcHeadBase = AssemblyHelpers.randomTensor([hc], rng: &rng, scale: 0.0)
        let hcHeadScale = AssemblyHelpers.randomTensor([1], rng: &rng, scale: 0.5)

        // Per-layer state
        var blocks: [Block] = []
        for i in 0..<nLayers {
            let attnNorm = RMSNorm(weight: AssemblyHelpers.onesTensor([dim]), eps: config.normEps)
            let ffnNorm = RMSNorm(weight: AssemblyHelpers.onesTensor([dim]), eps: config.normEps)

            // ---- MLA ----
            let wqA = AssemblyHelpers.linear(in: dim, out: config.qLoraRank, rng: &rng)
            let qNorm = RMSNorm(weight: AssemblyHelpers.onesTensor([config.qLoraRank]),
                                 eps: config.normEps)
            let wqB = AssemblyHelpers.linear(in: config.qLoraRank,
                                              out: config.nHeads * config.headDim, rng: &rng)
            let wkv = AssemblyHelpers.linear(in: dim, out: config.headDim, rng: &rng)
            let kvNorm = RMSNorm(weight: AssemblyHelpers.onesTensor([config.headDim]),
                                  eps: config.normEps)
            let perGroupD = config.nHeads * config.headDim / config.oGroups
            let woA = AssemblyHelpers.linear(in: perGroupD,
                                              out: config.oGroups * config.oLoraRank, rng: &rng)
            let woB = AssemblyHelpers.linear(in: config.oGroups * config.oLoraRank,
                                              out: dim, rng: &rng)
            let attnSink = AssemblyHelpers.randomTensor([config.nHeads], rng: &rng, scale: 0.1)

            let ratio = config.compressRatios[i]
            var compressor: Compressor? = nil
            var indexer: Indexer? = nil
            if ratio > 0 {
                compressor = AssemblyHelpers.makeCompressor(config: config, ratio: ratio,
                                                             headDim: config.headDim,
                                                             rotate: false, rng: &rng)
                if ratio == 4 {
                    indexer = AssemblyHelpers.makeIndexer(config: config, ratio: ratio, rng: &rng)
                }
            }

            let kvCacheRows = config.windowSize +
                (ratio > 0 ? config.maxSeqLen / ratio : 0)
            let kvCache = Tensor.empty(shape: [config.maxBatchSize, max(kvCacheRows, 1), config.headDim],
                                        dtype: .f32)

            let rope = RoPE(ropeHeadDim: config.ropeHeadDim,
                            freqs: RoPE.makeFreqs(config: config, useYarn: ratio > 0))

            let mla = MLA(config: config, layerId: i,
                          wqA: wqA, qNorm: qNorm, wqB: wqB,
                          wkv: wkv, kvNorm: kvNorm,
                          woA: woA, woB: woB,
                          attnSink: attnSink,
                          rope: rope,
                          compressor: compressor, indexer: indexer,
                          kvCache: kvCache)

            // ---- MoE FFN ----
            let gateW = AssemblyHelpers.linear(in: dim, out: nExperts, rng: &rng)
            let gateBias: Tensor? = i < config.nHashLayers ? nil :
                AssemblyHelpers.randomTensor([nExperts], rng: &rng, scale: 0.0)
            let gate = Gate(config: config, layerId: i,
                            weight: gateW, bias: gateBias, tid2eid: nil)

            var experts: [Expert?] = []
            for _ in 0..<nExperts {
                let w1 = AssemblyHelpers.linear(in: dim, out: config.moeInterDim, rng: &rng)
                let w2 = AssemblyHelpers.linear(in: config.moeInterDim, out: dim, rng: &rng)
                let w3 = AssemblyHelpers.linear(in: dim, out: config.moeInterDim, rng: &rng)
                experts.append(Expert(w1: w1, w2: w2, w3: w3, swigluLimit: config.swigluLimit))
            }
            let sharedExpert = Expert(
                w1: AssemblyHelpers.linear(in: dim, out: config.moeInterDim, rng: &rng),
                w2: AssemblyHelpers.linear(in: config.moeInterDim, out: dim, rng: &rng),
                w3: AssemblyHelpers.linear(in: dim, out: config.moeInterDim, rng: &rng),
                swigluLimit: config.swigluLimit
            )
            let moe = MoEFFN(config: config, gate: gate, experts: experts, shared: sharedExpert)

            // ---- HC params ----
            let hcAttnFn = AssemblyHelpers.randomTensor([mixHc, hc * dim], rng: &rng, scale: 0.02)
            let hcAttnBase = AssemblyHelpers.randomTensor([mixHc], rng: &rng, scale: 0.0)
            let hcAttnScale = AssemblyHelpers.randomTensor([3], rng: &rng, scale: 0.5)
            let hcFfnFn = AssemblyHelpers.randomTensor([mixHc, hc * dim], rng: &rng, scale: 0.02)
            let hcFfnBase = AssemblyHelpers.randomTensor([mixHc], rng: &rng, scale: 0.0)
            let hcFfnScale = AssemblyHelpers.randomTensor([3], rng: &rng, scale: 0.5)

            blocks.append(Block(layerId: i, config: config,
                                attn: mla, ffn: moe,
                                attnNorm: attnNorm, ffnNorm: ffnNorm,
                                hcAttnFn: hcAttnFn, hcAttnBase: hcAttnBase, hcAttnScale: hcAttnScale,
                                hcFfnFn: hcFfnFn, hcFfnBase: hcFfnBase, hcFfnScale: hcFfnScale))
        }

        return Transformer(config: config, embed: embed, layers: blocks, mtp: [],
                           norm: norm, head: head,
                           hcHeadFn: hcHeadFn, hcHeadBase: hcHeadBase, hcHeadScale: hcHeadScale)
    }

    /// Load a Transformer from a directory of safetensors shards.
    ///
    /// Expected layout: the post-`convert.py` form. Run
    ///   python Reference/inference/convert.py \
    ///     --hf-ckpt-path <huggingface_dir> \
    ///     --save-path <converted_dir> \
    ///     --n-experts <N> --model-parallel 1
    /// to produce a directory containing `model0-mp1.safetensors` (single-
    /// rank, names already in the canonical form this loader expects).
    ///
    /// Names that can't be found are filled in with random init plus a
    /// stderr summary at the end. This lets a partially-converted or
    /// pruned checkpoint still produce a forward pass.
    static func load(config: ModelConfig, from weightsDir: URL) throws -> Transformer {
        let loader = try WeightLoader(directory: weightsDir)
        FileHandle.standardError.write(Data(
            "Indexed \(loader.totalKnownNames) tensors across \(loader.shardCount) shard(s).\n".utf8))

        var rng = MiniRNG(seed: 0xDEADC0DE)
        let dim = config.dim
        let hc = config.hcMult
        let mixHc = (2 + hc) * hc
        let nLayers = config.nLayers
        let nExperts = config.nRoutedExperts

        // ---------- Top-level ----------
        let embedW = (try loader.tryLoad(["embed.weight", "model.embed.weight"]))
            ?? AssemblyHelpers.randomTensor([config.vocabSize, dim], rng: &rng, scale: 0.02)
        let embed = ParallelEmbedding(vocabSize: config.vocabSize, dim: dim, weight: embedW)

        let normW = (try loader.tryLoad(["norm.weight"]))
            ?? AssemblyHelpers.onesTensor([dim])
        let norm = RMSNorm(weight: normW, eps: config.normEps)

        let lmHeadW = (try loader.tryLoad(["head.weight", "lm_head.weight"]))
            ?? AssemblyHelpers.randomTensor([config.vocabSize, dim], rng: &rng, scale: 0.02)
        let head = ParallelHead(vocabSize: config.vocabSize, dim: dim,
                                normEps: config.normEps, hcEps: config.hcEps,
                                weight: lmHeadW)

        let hcHeadFn = (try loader.tryLoad(["hc_head_fn"]))
            ?? AssemblyHelpers.randomTensor([hc, hc * dim], rng: &rng, scale: 0.02)
        let hcHeadBase = (try loader.tryLoad(["hc_head_base"]))
            ?? AssemblyHelpers.randomTensor([hc], rng: &rng, scale: 0.0)
        let hcHeadScale = (try loader.tryLoad(["hc_head_scale"]))
            ?? AssemblyHelpers.randomTensor([1], rng: &rng, scale: 0.5)

        // ---------- Per-layer ----------
        var blocks: [Block] = []
        for i in 0..<nLayers {
            let lp = "layers.\(i)"
            let attnNorm = RMSNorm(
                weight: (try loader.tryLoad(["\(lp).attn_norm.weight"]))
                    ?? AssemblyHelpers.onesTensor([dim]),
                eps: config.normEps)
            let ffnNorm = RMSNorm(
                weight: (try loader.tryLoad(["\(lp).ffn_norm.weight"]))
                    ?? AssemblyHelpers.onesTensor([dim]),
                eps: config.normEps)

            // ---- MLA ----
            let wqA = try loadLinear(loader, base: "\(lp).attn.wq_a",
                                      inF: dim, outF: config.qLoraRank, rng: &rng)
            let qNorm = RMSNorm(
                weight: (try loader.tryLoad(["\(lp).attn.q_norm.weight"]))
                    ?? AssemblyHelpers.onesTensor([config.qLoraRank]),
                eps: config.normEps)
            let wqB = try loadLinear(loader, base: "\(lp).attn.wq_b",
                                      inF: config.qLoraRank,
                                      outF: config.nHeads * config.headDim, rng: &rng)
            let wkv = try loadLinear(loader, base: "\(lp).attn.wkv",
                                      inF: dim, outF: config.headDim, rng: &rng)
            let kvNorm = RMSNorm(
                weight: (try loader.tryLoad(["\(lp).attn.kv_norm.weight"]))
                    ?? AssemblyHelpers.onesTensor([config.headDim]),
                eps: config.normEps)
            let perGroupD = config.nHeads * config.headDim / config.oGroups
            let woA = try loadLinear(loader, base: "\(lp).attn.wo_a",
                                      inF: perGroupD,
                                      outF: config.oGroups * config.oLoraRank, rng: &rng)
            let woB = try loadLinear(loader, base: "\(lp).attn.wo_b",
                                      inF: config.oGroups * config.oLoraRank,
                                      outF: dim, rng: &rng)
            let attnSink = (try loader.tryLoad(["\(lp).attn.attn_sink"]))
                ?? AssemblyHelpers.randomTensor([config.nHeads], rng: &rng, scale: 0.1)

            let ratio = config.compressRatios[i]
            var compressor: Compressor? = nil
            var indexer: Indexer? = nil
            if ratio > 0 {
                compressor = try AssemblyHelpers.loadCompressor(
                    loader, base: "\(lp).attn.compressor",
                    config: config, ratio: ratio, headDim: config.headDim,
                    rotate: false, rng: &rng)
                if ratio == 4 {
                    indexer = try AssemblyHelpers.loadIndexer(
                        loader, base: "\(lp).attn.indexer",
                        config: config, ratio: ratio, rng: &rng)
                }
            }

            let kvCacheRows = config.windowSize +
                (ratio > 0 ? config.maxSeqLen / ratio : 0)
            let kvCache = Tensor.empty(shape: [config.maxBatchSize,
                                                max(kvCacheRows, 1),
                                                config.headDim],
                                        dtype: .f32)

            let rope = RoPE(ropeHeadDim: config.ropeHeadDim,
                            freqs: RoPE.makeFreqs(config: config, useYarn: ratio > 0))

            let mla = MLA(config: config, layerId: i,
                          wqA: wqA, qNorm: qNorm, wqB: wqB,
                          wkv: wkv, kvNorm: kvNorm,
                          woA: woA, woB: woB,
                          attnSink: attnSink, rope: rope,
                          compressor: compressor, indexer: indexer,
                          kvCache: kvCache)

            // ---- MoE FFN ----
            let gateW = try loadLinear(loader, base: "\(lp).ffn.gate",
                                        inF: dim, outF: nExperts, rng: &rng)
            let gateBias: Tensor? = i < config.nHashLayers ? nil :
                ((try loader.tryLoad(["\(lp).ffn.gate.bias"]))
                 ?? AssemblyHelpers.randomTensor([nExperts], rng: &rng, scale: 0.0))
            let tid2eid: Tensor? = i < config.nHashLayers
                ? (try loader.tryLoad(["\(lp).ffn.gate.tid2eid"]))
                : nil
            let gate = Gate(config: config, layerId: i,
                            weight: gateW, bias: gateBias, tid2eid: tid2eid)

            var experts: [Expert?] = []
            for j in 0..<nExperts {
                let ep = "\(lp).ffn.experts.\(j)"
                experts.append(Expert(
                    w1: try loadLinear(loader, base: "\(ep).w1",
                                        inF: dim, outF: config.moeInterDim, rng: &rng),
                    w2: try loadLinear(loader, base: "\(ep).w2",
                                        inF: config.moeInterDim, outF: dim, rng: &rng),
                    w3: try loadLinear(loader, base: "\(ep).w3",
                                        inF: dim, outF: config.moeInterDim, rng: &rng),
                    swigluLimit: config.swigluLimit))
            }
            let sep = "\(lp).ffn.shared_experts"
            let sharedExpert = Expert(
                w1: try loadLinear(loader, base: "\(sep).w1",
                                    inF: dim, outF: config.moeInterDim, rng: &rng),
                w2: try loadLinear(loader, base: "\(sep).w2",
                                    inF: config.moeInterDim, outF: dim, rng: &rng),
                w3: try loadLinear(loader, base: "\(sep).w3",
                                    inF: dim, outF: config.moeInterDim, rng: &rng),
                swigluLimit: config.swigluLimit)
            let moe = MoEFFN(config: config, gate: gate,
                             experts: experts, shared: sharedExpert)

            // ---- HC params ----
            let hcAttnFn = (try loader.tryLoad(["\(lp).hc_attn_fn"]))
                ?? AssemblyHelpers.randomTensor([mixHc, hc * dim], rng: &rng, scale: 0.02)
            let hcAttnBase = (try loader.tryLoad(["\(lp).hc_attn_base"]))
                ?? AssemblyHelpers.randomTensor([mixHc], rng: &rng, scale: 0.0)
            let hcAttnScale = (try loader.tryLoad(["\(lp).hc_attn_scale"]))
                ?? AssemblyHelpers.randomTensor([3], rng: &rng, scale: 0.5)
            let hcFfnFn = (try loader.tryLoad(["\(lp).hc_ffn_fn"]))
                ?? AssemblyHelpers.randomTensor([mixHc, hc * dim], rng: &rng, scale: 0.02)
            let hcFfnBase = (try loader.tryLoad(["\(lp).hc_ffn_base"]))
                ?? AssemblyHelpers.randomTensor([mixHc], rng: &rng, scale: 0.0)
            let hcFfnScale = (try loader.tryLoad(["\(lp).hc_ffn_scale"]))
                ?? AssemblyHelpers.randomTensor([3], rng: &rng, scale: 0.5)

            blocks.append(Block(layerId: i, config: config,
                                attn: mla, ffn: moe,
                                attnNorm: attnNorm, ffnNorm: ffnNorm,
                                hcAttnFn: hcAttnFn, hcAttnBase: hcAttnBase,
                                hcAttnScale: hcAttnScale,
                                hcFfnFn: hcFfnFn, hcFfnBase: hcFfnBase,
                                hcFfnScale: hcFfnScale))
        }

        if !loader.missing.isEmpty {
            FileHandle.standardError.write(Data("""
            \(loader.missing.count) tensor name(s) were not found in the
            checkpoint and were filled with random init. First few:
              \(loader.missing.prefix(8).joined(separator: "\n  "))

            """.utf8))
        }

        return Transformer(config: config, embed: embed, layers: blocks, mtp: [],
                           norm: norm, head: head,
                           hcHeadFn: hcHeadFn, hcHeadBase: hcHeadBase,
                           hcHeadScale: hcHeadScale)
    }
}

// MARK: - Loading helpers

internal func loadLinear(_ loader: WeightLoader, base: String,
                         inF: Int, outF: Int,
                         rng: inout MiniRNG) throws -> Linear {
    if let w = try loader.tryLoad(["\(base).weight"]) {
        let scale = try loader.tryLoad(["\(base).scale"])
        return Linear(inFeatures: inF, outFeatures: outF, weight: w, scale: scale)
    }
    return AssemblyHelpers.linear(in: inF, out: outF, rng: &rng)
}

// MARK: - Helpers

internal struct MiniRNG {
    var state: UInt64
    init(seed: UInt64) { self.state = seed | 1 }
    mutating func nextUnit() -> Float {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Float(Double(state >> 11) / Double(1 << 53))
    }
}

internal enum AssemblyHelpers {

    static func randomTensor(_ shape: [Int], rng: inout MiniRNG, scale: Float) -> Tensor {
        let n = shape.reduce(1, *)
        var arr = [Float](repeating: 0, count: n)
        if scale != 0 {
            for i in 0..<n { arr[i] = (rng.nextUnit() - 0.5) * 2 * scale }
        }
        return arr.withUnsafeBytes { Tensor.from(bytes: $0, shape: shape, dtype: .f32) }
    }

    static func onesTensor(_ shape: [Int]) -> Tensor {
        let n = shape.reduce(1, *)
        let arr = [Float](repeating: 1.0, count: n)
        return arr.withUnsafeBytes { Tensor.from(bytes: $0, shape: shape, dtype: .f32) }
    }

    static func linear(`in` inFeatures: Int, out outFeatures: Int,
                       rng: inout MiniRNG, scale: Float = 0.02) -> Linear {
        let w = randomTensor([outFeatures, inFeatures], rng: &rng, scale: scale)
        return Linear(inFeatures: inFeatures, outFeatures: outFeatures, weight: w, scale: nil)
    }

    static func makeCompressor(config: ModelConfig, ratio: Int, headDim: Int,
                               rotate: Bool, rng: inout MiniRNG) -> Compressor {
        let coff = ratio == 4 ? 2 : 1
        let coffHeadDim = coff * headDim
        let ape = randomTensor([ratio, coffHeadDim], rng: &rng, scale: 0.05)
        let wkv = linear(in: config.dim, out: coffHeadDim, rng: &rng)
        let wgate = linear(in: config.dim, out: coffHeadDim, rng: &rng)
        let norm = RMSNorm(weight: onesTensor([headDim]), eps: config.normEps)
        let kvState = Tensor.empty(shape: [config.maxBatchSize, coff * ratio, coffHeadDim],
                                    dtype: .f32)
        let scoreState = Compressor.makeScoreState(
            shape: [config.maxBatchSize, coff * ratio, coffHeadDim])
        return Compressor(config: config, compressRatio: ratio, headDim: headDim, rotate: rotate,
                          ape: ape, wkv: wkv, wgate: wgate, norm: norm,
                          kvState: kvState, scoreState: scoreState)
    }

    static func makeIndexer(config: ModelConfig, ratio: Int,
                            rng: inout MiniRNG) -> Indexer {
        let wqB = linear(in: config.qLoraRank,
                         out: config.indexNHeads * config.indexHeadDim, rng: &rng)
        let weightsProj = linear(in: config.dim, out: config.indexNHeads, rng: &rng)
        let compressor = makeCompressor(config: config, ratio: ratio,
                                         headDim: config.indexHeadDim,
                                         rotate: true, rng: &rng)
        let kvCache = Tensor.empty(shape: [config.maxBatchSize,
                                            config.maxSeqLen / ratio,
                                            config.indexHeadDim], dtype: .f32)
        return Indexer(config: config, compressRatio: ratio,
                       wqB: wqB, weightsProj: weightsProj,
                       compressor: compressor, kvCache: kvCache)
    }

    // ---- Loader-aware variants used by Transformer.load ----

    static func loadCompressor(_ loader: WeightLoader, base: String,
                               config: ModelConfig, ratio: Int, headDim: Int,
                               rotate: Bool, rng: inout MiniRNG) throws -> Compressor {
        let coff = ratio == 4 ? 2 : 1
        let coffHeadDim = coff * headDim
        let ape = (try loader.tryLoad(["\(base).ape"]))
            ?? randomTensor([ratio, coffHeadDim], rng: &rng, scale: 0.05)
        let wkv = try loadLinear(loader, base: "\(base).wkv",
                                  inF: config.dim, outF: coffHeadDim, rng: &rng)
        let wgate = try loadLinear(loader, base: "\(base).wgate",
                                    inF: config.dim, outF: coffHeadDim, rng: &rng)
        let norm = RMSNorm(
            weight: (try loader.tryLoad(["\(base).norm.weight"])) ?? onesTensor([headDim]),
            eps: config.normEps)
        let kvState = Tensor.empty(shape: [config.maxBatchSize, coff * ratio, coffHeadDim],
                                    dtype: .f32)
        let scoreState = Compressor.makeScoreState(
            shape: [config.maxBatchSize, coff * ratio, coffHeadDim])
        return Compressor(config: config, compressRatio: ratio, headDim: headDim, rotate: rotate,
                          ape: ape, wkv: wkv, wgate: wgate, norm: norm,
                          kvState: kvState, scoreState: scoreState)
    }

    static func loadIndexer(_ loader: WeightLoader, base: String,
                            config: ModelConfig, ratio: Int,
                            rng: inout MiniRNG) throws -> Indexer {
        let wqB = try loadLinear(loader, base: "\(base).wq_b",
                                  inF: config.qLoraRank,
                                  outF: config.indexNHeads * config.indexHeadDim, rng: &rng)
        let weightsProj = try loadLinear(loader, base: "\(base).weights_proj",
                                          inF: config.dim, outF: config.indexNHeads, rng: &rng)
        let compressor = try loadCompressor(loader, base: "\(base).compressor",
                                             config: config, ratio: ratio,
                                             headDim: config.indexHeadDim,
                                             rotate: true, rng: &rng)
        let kvCache = Tensor.empty(shape: [config.maxBatchSize,
                                            config.maxSeqLen / ratio,
                                            config.indexHeadDim], dtype: .f32)
        return Indexer(config: config, compressRatio: ratio,
                       wqB: wqB, weightsProj: weightsProj,
                       compressor: compressor, kvCache: kvCache)
    }
}
