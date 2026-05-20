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
            // Gate logits stay in FP32 (see real-load path comment).
            let gateWWeight = AssemblyHelpers.randomTensor([nExperts, dim], rng: &rng, scale: 0.02)
            let gateW = Linear(inFeatures: dim, outFeatures: nExperts,
                                weight: gateWWeight, scale: nil,
                                castOutputToBF16: false)
            let gateBias: Tensor? = i < config.nHashLayers ? nil :
                AssemblyHelpers.randomTensor([nExperts], rng: &rng, scale: 0.0)
            let gate = Gate(config: config, layerId: i,
                            weight: gateW, bias: gateBias, tid2eid: nil)

            // Honor `config.prunedExperts[i]` even in the random-init
            // path so unit tests can exercise the loader's skip
            // behavior without writing real safetensors.
            let droppedForLayer: Set<Int> =
                (i < config.prunedExperts.count)
                ? Set(config.prunedExperts[i])
                : []
            var experts: [Expert?] = []
            for j in 0..<nExperts {
                if droppedForLayer.contains(j) {
                    experts.append(nil)
                    continue
                }
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
            moe.layerId = i

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
    static func load(config: ModelConfig, from weightsDir: URL,
                      strategyOverride: String? = nil,
                      forceLoad: Bool = false,
                      warmupOnLoad: Bool = false,
                      useMapSharedWeights: Bool = false,
                      kvCacheFile: KVCacheFile? = nil) throws -> Transformer {
        MemoryLogger.snapshot("load:start", force: true)
        let plan = try LoadPlan.decide(modelDir: weightsDir,
                                        override: strategyOverride,
                                        forceLoad: forceLoad)
        FileHandle.standardError.write(Data(plan.summary().utf8))
        let loader = try WeightLoader(plan: plan, useMapShared: useMapSharedWeights)

        // Layout del KV cache cross-restart: se fornito un KVCacheFile,
        // calcola gli offset per ogni layer e li usa per allocare i
        // Tensor di stato sui slice del file mmappato. Senza file →
        // path classico Tensor.empty (KV solo in memoria, perso al
        // restart).
        let kvLayout: KVCacheLayout? = kvCacheFile != nil
            ? KVCacheLayout.compute(config: config)
            : nil
        if let l = kvLayout {
            FileHandle.standardError.write(Data(l.summary().utf8))
        }
        MemoryLogger.snapshot("load:after-mmap", force: true)

        // Optional ds4-style pages warmup: pre-fault tutte le pagine
        // dei shards prima del primo forward, riduce il
        // time-to-first-token. Skip automaticamente se la model size
        // > physical RAM × 1.5 (vedi `WeightLoader.warmupAllShards`).
        // Default off — opt-in via flag.
        if warmupOnLoad {
            loader.warmupAllShards()
            MemoryLogger.snapshot("load:after-warmup", force: true)
        }
        FileHandle.standardError.write(Data(
            "Indexed \(loader.totalKnownNames) tensors across \(loader.shardCount) shard(s).\n".utf8))

        // Patch missing/stale config fields from actual tensor shapes so a
        // partial config.json (only head_dim + compress_ratios + n_routed_experts,
        // for instance) doesn't leave the rest at toy defaults that mismatch
        // the real checkpoint and produce garbage logits.
        let config = config.inferred(from: loader)

        // Refuse early if the projected KV cache size at the chosen
        // (max_seq_len, max_batch_size) would blow the budget. KV
        // caches are dense storageModeShared MTLBuffers (not mmap),
        // streaming hints don't help. Common trap: the HF config
        // ships max_position_embeddings = 1M, which on a 16 GB Mac
        // tries to allocate ~50 GB of KV state at load time and
        // jetsams the process.
        let kvProjected = config.projectedKVCacheBytes
        let kvBudget = SystemProbe.effectiveProcessBudget()
        if kvBudget > 0, kvProjected > kvBudget {
            throw LoadStrategyError.kvCacheTooLarge(
                projected: kvProjected,
                available: kvBudget,
                maxSeqLen: config.maxSeqLen,
                maxBatchSize: config.maxBatchSize)
        }
        FileHandle.standardError.write(Data(String(
            format: "Projected KV cache: %.2f GB at max_seq_len=%d, max_batch_size=%d.\n",
            Double(kvProjected) / 1_073_741_824.0,
            config.maxSeqLen, config.maxBatchSize).utf8))

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
        MemoryLogger.snapshot("load:embed+head-built", force: true)

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

            // Backing fisico opzionale: se KVCacheFile fornito,
            // estrai gli offset dal layout per questo layer e wrappali
            // come backing per Compressor / Indexer / kvCache.
            let layerOffs = kvLayout?.layers[i]
            var compBacking: AssemblyHelpers.CompressorBacking? = nil
            var idxBacking: AssemblyHelpers.IndexerBacking? = nil
            if let kvFile = kvCacheFile,
               let off = layerOffs
            {
                if let ks = off.compressorKVState,
                   let ss = off.compressorScoreState
                {
                    compBacking = AssemblyHelpers.CompressorBacking(
                        file: kvFile, kvState: ks, scoreState: ss)
                }
                if let ikv = off.indexerKVCache,
                   let iks = off.indexerCompressorKVState,
                   let iss = off.indexerCompressorScoreState
                {
                    idxBacking = AssemblyHelpers.IndexerBacking(
                        file: kvFile,
                        kvCache: ikv,
                        compressor: AssemblyHelpers.CompressorBacking(
                            file: kvFile, kvState: iks, scoreState: iss))
                }
            }

            if ratio > 0 {
                compressor = try AssemblyHelpers.loadCompressor(
                    loader, base: "\(lp).attn.compressor",
                    config: config, ratio: ratio, headDim: config.headDim,
                    rotate: false, rng: &rng,
                    backing: compBacking)
                if ratio == 4 {
                    indexer = try AssemblyHelpers.loadIndexer(
                        loader, base: "\(lp).attn.indexer",
                        config: config, ratio: ratio, rng: &rng,
                        backing: idxBacking)
                }
            }

            let kvCacheRows = config.windowSize +
                (ratio > 0 ? config.maxSeqLen / ratio : 0)
            let kvCacheShape = [config.maxBatchSize,
                                 max(kvCacheRows, 1),
                                 config.headDim]
            let kvCache: Tensor
            if let kvFile = kvCacheFile, let off = layerOffs {
                kvCache = kvFile.tensor(at: off.attnKVCache,
                                         shape: kvCacheShape, dtype: .f32)
            } else {
                kvCache = Tensor.empty(shape: kvCacheShape, dtype: .f32)
            }

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
            // Gate logits MUST stay in FP32: model.py:566 spells out
            //   scores = linear(x.float(), self.weight.float())
            // i.e. the projection is run in FP32 explicitly. Quantising the
            // logits to BF16 (~7 mantissa bits) before sqrt(softplus) + topk
            // perturbs which experts get selected, and on V4-Flash that
            // perturbation shows up as an 8.4× residual-stream cliff at the
            // first SCORE-routed layer (= the first layer past
            // `n_hash_layers`). Hash-routed layers are spared because their
            // expert indices come from a precomputed token→expert table —
            // only the weights are affected, not the routing.
            let gateW = try loadLinear(loader, base: "\(lp).ffn.gate",
                                        inF: dim, outF: nExperts,
                                        castOutputToBF16: false, rng: &rng)
            let gateBias: Tensor? = i < config.nHashLayers ? nil :
                ((try loader.tryLoad(["\(lp).ffn.gate.bias"]))
                 ?? AssemblyHelpers.randomTensor([nExperts], rng: &rng, scale: 0.0))
            let tid2eid: Tensor? = i < config.nHashLayers
                ? (try loader.tryLoad(["\(lp).ffn.gate.tid2eid"])).map(AssemblyHelpers.castIntToI32)
                : nil
            let gate = Gate(config: config, layerId: i,
                            weight: gateW, bias: gateBias, tid2eid: tid2eid)

            // Expert-prune support: skip allocation for experts marked
            // as dropped in config.json's `pruned_experts[i]`. The MoE
            // dispatch path already handles `nil` slots in
            // `MoEFFN.experts` (Layers/MoE.swift), and the gate weight
            // rows for these ids were set to large-negative by the
            // rewriter so the top-K kernel never picks them.
            let droppedForLayer: Set<Int> =
                (i < config.prunedExperts.count)
                ? Set(config.prunedExperts[i])
                : []
            var experts: [Expert?] = []
            for j in 0..<nExperts {
                if droppedForLayer.contains(j) {
                    experts.append(nil)
                    continue
                }
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
            moe.layerId = i

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
        MemoryLogger.snapshot("load:layers-built", force: true)

        if !loader.missing.isEmpty {
            FileHandle.standardError.write(Data("""
            \(loader.missing.count) tensor name(s) were not found in the
            checkpoint and were filled with random init. First few:
              \(loader.missing.prefix(8).joined(separator: "\n  "))

            """.utf8))
        }

        let model = Transformer(config: config, embed: embed, layers: blocks, mtp: [],
                                 norm: norm, head: head,
                                 hcHeadFn: hcHeadFn, hcHeadBase: hcHeadBase,
                                 hcHeadScale: hcHeadScale)
        // Park the WeightLoader on the model so its `shardLayers`
        // index lives as long as the model and `forward(...)` can
        // call `prefetchLayer` / `releaseLayer` between blocks.
        // This matters for `.streaming` strategy; for `.mmap` /
        // `.preload` the loader is held but never queried.
        model.weightLoader = loader

        // Wire the lazy-expert hook on every MoEFFN (main + MTP)
        // unconditionally — the hook checks `StreamingPool
        // .lazyExpertEnabled` at call time, so flipping the UI toggle
        // (or `DEEPSEEK_LAZY_EXPERT` env var) takes effect on the next
        // token without a model reload. `.mmap` / `.preload` leave
        // `pool` nil so `ensureExperts` returns immediately either way.
        let lazyHook: MoEFFN.EnsureExpertsHook = { [weak loader] layerK, indices in
            loader?.ensureExperts(layer: layerK, indices: indices)
        }
        for block in model.layers {
            block.ffn.ensureExpertsHook = lazyHook
        }
        for mtp in model.mtp {
            mtp.block.ffn.ensureExpertsHook = lazyHook
        }

        MemoryLogger.snapshot("load:complete", force: true)
        return model
    }
}

// MARK: - Loading helpers

internal func loadLinear(_ loader: WeightLoader, base: String,
                         inF: Int, outF: Int,
                         castOutputToBF16: Bool = false,
                         rng: inout MiniRNG) throws -> Linear {
    if var w = try loader.tryLoad(["\(base).weight"]) {
        // V4-Flash-HF stores routed-expert FP4 weights as raw I8/U8
        // bytes (each byte packs two E2M1 nibbles) because safetensors
        // has no native FP4 dtype. Our `Linear.callAsFunction` switch
        // on `.i8` dispatches the W8A16 INT8 kernel, which reads the
        // bytes as signed [-128, 127] integers and multiplies by an
        // F16 group scale — completely the wrong math, so expert
        // outputs blow up to 1e25 and NaN the whole block.
        //
        // Reinterpret the tensor as FP4 with the LOGICAL shape (last
        // dim doubled) when the name matches a routed expert. The
        // converter's CLI already does the same reinterpretation
        // (Sources/converter/main.swift:568) — this is just the
        // inference-side equivalent so we can run the HF checkpoint
        // directly without converting.
        if base.contains(".experts.") && (w.dtype == .i8) {
            let packedLast = w.shape.last ?? 0
            let logicalShape = Array(w.shape.dropLast()) + [packedLast * 2]
            w = Tensor(shape: logicalShape, dtype: .fp4E2M1,
                       buffer: w.buffer, offset: w.offset)
        }

        // Only quantized dtypes carry a `.scale` companion. Asking for
        // one on bf16/f32 paths adds noise to the missing-tensor report
        // and does nothing useful (Linear's switch ignores `scale` for
        // those dtypes).
        let needsScale = w.dtype == .i8
                      || w.dtype == .i4
                      || w.dtype == .i2
                      || w.dtype == .fp8E4M3
                      || w.dtype == .fp4E2M1
        // Names: post-converter we use `<base>.scale`, but HF-native
        // FP8/FP4 release stores it as `<base>.weight_scale_inv`. Try
        // both so the same code path serves both directories.
        let scale: Tensor? = needsScale
            ? try loader.tryLoad(["\(base).scale", "\(base).weight_scale_inv"])
            : nil
        return Linear(inFeatures: inF, outFeatures: outF, weight: w, scale: scale,
                      castOutputToBF16: castOutputToBF16)
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

    /// Returns a copy of `t` as an i32 tensor. Used at load time for
    /// integer tables (e.g. tid2eid) that the on-disk safetensors stores
    /// as i64 but downstream Swift code (Gate.tidPtr binding) expects as
    /// Int32. Idempotent for already-i32 input.
    static func castIntToI32(_ t: Tensor) -> Tensor {
        if t.dtype == .i32 { return t }
        let n = t.count
        let dst = Tensor.empty(shape: t.shape, dtype: .i32)
        let dstP = dst.buffer.contents().bindMemory(to: Int32.self, capacity: n)
        let srcRaw = t.buffer.contents().advanced(by: t.offset)
        switch t.dtype {
        case .i64:
            let srcP = srcRaw.bindMemory(to: Int64.self, capacity: n)
            for i in 0..<n { dstP[i] = Int32(truncatingIfNeeded: srcP[i]) }
        case .i8:
            let srcP = srcRaw.bindMemory(to: Int8.self, capacity: n)
            for i in 0..<n { dstP[i] = Int32(srcP[i]) }
        default:
            fatalError("castIntToI32: unsupported source dtype \(t.dtype)")
        }
        return dst
    }

    static func linear(`in` inFeatures: Int, out outFeatures: Int,
                       rng: inout MiniRNG, scale: Float = 0.02) -> Linear {
        let w = randomTensor([outFeatures, inFeatures], rng: &rng, scale: scale)
        return Linear(inFeatures: inFeatures, outFeatures: outFeatures, weight: w, scale: nil)
    }

    /// Backing storage opzionale per il KV state del Compressor.
    /// Quando fornito, `kvState` e `scoreState` vengono allocati come
    /// slice del KVCacheFile invece di MTLBuffer indipendenti — abilita
    /// cross-restart resume. Vedi `KVCacheLayout` per il layout
    /// binario.
    struct CompressorBacking {
        let file: KVCacheFile
        let kvState: KVCacheLayout.Region
        let scoreState: KVCacheLayout.Region
    }

    static func makeCompressor(config: ModelConfig, ratio: Int, headDim: Int,
                               rotate: Bool, rng: inout MiniRNG,
                               backing: CompressorBacking? = nil) -> Compressor {
        let coff = ratio == 4 ? 2 : 1
        let coffHeadDim = coff * headDim
        let ape = randomTensor([ratio, coffHeadDim], rng: &rng, scale: 0.05)
        let wkv = linear(in: config.dim, out: coffHeadDim, rng: &rng)
        let wgate = linear(in: config.dim, out: coffHeadDim, rng: &rng)
        let norm = RMSNorm(weight: onesTensor([headDim]), eps: config.normEps)
        let stateShape = [config.maxBatchSize, coff * ratio, coffHeadDim]
        let kvState: Tensor
        let scoreState: Tensor
        if let b = backing {
            // KVCacheFile-backed: zero-copy slice del payload mmappato.
            kvState = b.file.tensor(at: b.kvState,
                                     shape: stateShape, dtype: .f32)
            // Per scoreState, il backing buffer è zeroed dal sizing del
            // file ma noi vogliamo -inf (semantica makeScoreState).
            // Inizializziamo CPU-side via storageModeShared.
            scoreState = b.file.tensor(at: b.scoreState,
                                         shape: stateShape, dtype: .f32)
            let n = scoreState.count
            let p = scoreState.buffer.contents()
                .advanced(by: scoreState.offset)
                .bindMemory(to: Float.self, capacity: n)
            for i in 0..<n { p[i] = -Float.infinity }
        } else {
            kvState = Tensor.empty(shape: stateShape, dtype: .f32)
            scoreState = Compressor.makeScoreState(shape: stateShape)
        }
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
                               rotate: Bool, rng: inout MiniRNG,
                               backing: CompressorBacking? = nil) throws -> Compressor {
        let coff = ratio == 4 ? 2 : 1
        let coffHeadDim = coff * headDim
        let ape = (try loader.tryLoad(["\(base).ape"]))
            ?? randomTensor([ratio, coffHeadDim], rng: &rng, scale: 0.05)
        // Compressor's wkv / wgate are FP32 in the reference (model.py:297-298
        // — the comment notes BF16 storage but FP32 parameters at runtime).
        // The forward at model.py:322-324 explicitly does `x = x.float()`
        // before these Linears, so their outputs propagate in FP32.
        let wkv = try loadLinear(loader, base: "\(base).wkv",
                                  inF: config.dim, outF: coffHeadDim,
                                  castOutputToBF16: false, rng: &rng)
        let wgate = try loadLinear(loader, base: "\(base).wgate",
                                    inF: config.dim, outF: coffHeadDim,
                                    castOutputToBF16: false, rng: &rng)
        let norm = RMSNorm(
            weight: (try loader.tryLoad(["\(base).norm.weight"])) ?? onesTensor([headDim]),
            eps: config.normEps)
        let stateShape = [config.maxBatchSize, coff * ratio, coffHeadDim]
        let kvState: Tensor
        let scoreState: Tensor
        if let b = backing {
            kvState = b.file.tensor(at: b.kvState,
                                     shape: stateShape, dtype: .f32)
            scoreState = b.file.tensor(at: b.scoreState,
                                         shape: stateShape, dtype: .f32)
            let n = scoreState.count
            let p = scoreState.buffer.contents()
                .advanced(by: scoreState.offset)
                .bindMemory(to: Float.self, capacity: n)
            for i in 0..<n { p[i] = -Float.infinity }
        } else {
            kvState = Tensor.empty(shape: stateShape, dtype: .f32)
            scoreState = Compressor.makeScoreState(shape: stateShape)
        }
        return Compressor(config: config, compressRatio: ratio, headDim: headDim, rotate: rotate,
                          ape: ape, wkv: wkv, wgate: wgate, norm: norm,
                          kvState: kvState, scoreState: scoreState)
    }

    /// Backing storage opzionale per l'Indexer: kvCache + sub-compressor.
    struct IndexerBacking {
        let file: KVCacheFile
        let kvCache: KVCacheLayout.Region
        let compressor: CompressorBacking
    }

    static func loadIndexer(_ loader: WeightLoader, base: String,
                            config: ModelConfig, ratio: Int,
                            rng: inout MiniRNG,
                            backing: IndexerBacking? = nil) throws -> Indexer {
        let wqB = try loadLinear(loader, base: "\(base).wq_b",
                                  inF: config.qLoraRank,
                                  outF: config.indexNHeads * config.indexHeadDim, rng: &rng)
        let weightsProj = try loadLinear(loader, base: "\(base).weights_proj",
                                          inF: config.dim, outF: config.indexNHeads, rng: &rng)
        let compressor = try loadCompressor(loader, base: "\(base).compressor",
                                             config: config, ratio: ratio,
                                             headDim: config.indexHeadDim,
                                             rotate: true, rng: &rng,
                                             backing: backing?.compressor)
        let kvCacheShape = [config.maxBatchSize,
                             config.maxSeqLen / ratio,
                             config.indexHeadDim]
        let kvCache: Tensor
        if let b = backing {
            kvCache = b.file.tensor(at: b.kvCache,
                                     shape: kvCacheShape, dtype: .f32)
        } else {
            kvCache = Tensor.empty(shape: kvCacheShape, dtype: .f32)
        }
        return Indexer(config: config, compressRatio: ratio,
                       wqB: wqB, weightsProj: weightsProj,
                       compressor: compressor, kvCache: kvCache)
    }
}
