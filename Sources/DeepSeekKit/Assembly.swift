import Foundation
import Metal
import MLX

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
            let rope = RoPE(ropeHeadDim: config.ropeHeadDim,
                            freqs: RoPE.makeFreqs(config: config, useYarn: ratio > 0))

            let mla = MLA(config: config, layerId: i,
                          wqA: wqA, qNorm: qNorm, wqB: wqB,
                          wkv: wkv, kvNorm: kvNorm,
                          woA: woA, woB: woB,
                          attnSink: attnSink,
                          rope: rope,
                          kvCache: nil)

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
                      useMapSharedWeights: Bool = false) throws -> Transformer {
        MemoryLogger.snapshot("load:start", force: true)

        // Bound MLX's internal buffer pool. Without this MLX keeps freed
        // GPU/unified-memory buffers around to satisfy future allocs
        // and the pool grows to the lifetime peak — on V4-Pro that's
        // the 256-expert MoE layer dequant burst, ~10+ GB. With a
        // 1 GB cap MLX returns excess buffers to the system instead
        // of hoarding them, at the cost of a few extra allocs per
        // forward. Tunable via DEEPSEEK_MLX_CACHE_MB.
        let cacheLimitMB: Int = {
            if let raw = ProcessInfo.processInfo.environment["DEEPSEEK_MLX_CACHE_MB"],
               let n = Int(raw), n >= 0 { return n }
            return 1024
        }()
        MLX.GPU.set(cacheLimit: cacheLimitMB * 1024 * 1024)

        // Opt-in: route routed-expert Linears through MLX's fused
        // 4-bit quantized GEMM (groupSize=64). One-time re-quant at
        // expert-load time (in Linear.getMLXQuant) eliminates the
        // [outFeatures, inFeatures] bf16 dequant temporary per call —
        // ≈32 MB per expert × 8 active experts × 7 layers in decode.
        // Numerical drift: bf16 → int4 round-trip; validate before
        // shipping on by default. See plan file.
        let mlxQuantExperts: Bool = ProcessInfo.processInfo
            .environment["DEEPSEEK_MLX_QUANT"] == "1"
        if mlxQuantExperts {
            FileHandle.standardError.write(Data(
                "[config] DEEPSEEK_MLX_QUANT=1 — routed-expert Linears "
                + "will use MLXFast.quantizedMatmul (groupSize=64, "
                + "bits=4) after one-time re-quant.\n".utf8))
        }

        let loader = try WeightLoader(directory: weightsDir)

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

        // Caches will be managed dynamically by MLX.
        let kvProjected = config.projectedKVCacheBytes
        FileHandle.standardError.write(Data(String(
            format: "Projected KV cache: %.2f GB at max_seq_len=%d, max_batch_size=%d.\n",
            Double(kvProjected) / 1_073_741_824.0,
            config.maxSeqLen, config.maxBatchSize).utf8))

        // MoE expert counts — printed at load so the effective routing
        // width is visible without `--print-config`. `nActivatedExperts`
        // already reflects any `DEEPSEEK_TOPK_EXPERTS` override.
        FileHandle.standardError.write(Data(String(
            format: "MoE experts: routed=%d activated/token=%d shared=%d.\n",
            config.nRoutedExperts, config.nActivatedExperts,
            config.nSharedExperts).utf8))

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
        var blocksOpt: [Block?] = Array(repeating: nil, count: nLayers)
        let blocksLock = NSLock()
        
        DispatchQueue.concurrentPerform(iterations: nLayers) { i in
            var rng = MiniRNG(seed: 0xDEADC0DE &+ UInt64(i))
            let lp = "layers.\(i)"
            let attnNorm = RMSNorm(
                weight: (try? loader.tryLoad(["\(lp).attn_norm.weight"]))
                    ?? AssemblyHelpers.onesTensor([dim]),
                eps: config.normEps)
            let ffnNorm = RMSNorm(
                weight: (try? loader.tryLoad(["\(lp).ffn_norm.weight"]))
                    ?? AssemblyHelpers.onesTensor([dim]),
                eps: config.normEps)

            // ---- MLA ----
            let wqA = try! loadLinear(loader, base: "\(lp).attn.wq_a",
                                      inF: dim, outF: config.qLoraRank, rng: &rng)
            let qNorm = RMSNorm(
                weight: (try? loader.tryLoad(["\(lp).attn.q_norm.weight"]))
                    ?? AssemblyHelpers.onesTensor([config.qLoraRank]),
                eps: config.normEps)
            let wqB = try! loadLinear(loader, base: "\(lp).attn.wq_b",
                                      inF: config.qLoraRank,
                                      outF: config.nHeads * config.headDim, rng: &rng)
            let wkv = try! loadLinear(loader, base: "\(lp).attn.wkv",
                                      inF: dim, outF: config.headDim, rng: &rng)
            let kvNorm = RMSNorm(
                weight: (try? loader.tryLoad(["\(lp).attn.kv_norm.weight"]))
                    ?? AssemblyHelpers.onesTensor([config.headDim]),
                eps: config.normEps)
            let perGroupD = config.nHeads * config.headDim / config.oGroups
            let woA = try! loadLinear(loader, base: "\(lp).attn.wo_a",
                                      inF: perGroupD,
                                      outF: config.oGroups * config.oLoraRank, rng: &rng)
            let woB = try! loadLinear(loader, base: "\(lp).attn.wo_b",
                                      inF: config.oGroups * config.oLoraRank,
                                      outF: dim, rng: &rng)
            let attnSink = (try? loader.tryLoad(["\(lp).attn.attn_sink"]))
                ?? AssemblyHelpers.randomTensor([config.nHeads], rng: &rng, scale: 0.1)

            let ratio = config.compressRatios[i]

            let rope = RoPE(ropeHeadDim: config.ropeHeadDim,
                            freqs: RoPE.makeFreqs(config: config, useYarn: ratio > 0))

            let mla = MLA(config: config, layerId: i,
                          wqA: wqA, qNorm: qNorm, wqB: wqB,
                          wkv: wkv, kvNorm: kvNorm,
                          woA: woA, woB: woB,
                          attnSink: attnSink, rope: rope,
                          kvCache: nil)

            // ---- MoE FFN ----
            let gateW = try! loadLinear(loader, base: "\(lp).ffn.gate",
                                        inF: dim, outF: nExperts,
                                        castOutputToBF16: false, rng: &rng)
            let gateBias: Tensor? = i < config.nHashLayers ? nil :
                ((try? loader.tryLoad(["\(lp).ffn.gate.bias"]))
                 ?? AssemblyHelpers.randomTensor([nExperts], rng: &rng, scale: 0.0))
            let tid2eid: Tensor? = i < config.nHashLayers
                ? (try? loader.tryLoad(["\(lp).ffn.gate.tid2eid"])).map(AssemblyHelpers.castIntToI32)
                : nil
            let gate = Gate(config: config, layerId: i,
                            weight: gateW, bias: gateBias, tid2eid: tid2eid)

            // Shared expert (same layout in both checkpoint families).
            let sep = "\(lp).ffn.shared_experts"
            let sharedExpert = Expert(
                w1: try! loadLinear(loader, base: "\(sep).w1",
                                    inF: dim, outF: config.moeInterDim, rng: &rng),
                w2: try! loadLinear(loader, base: "\(sep).w2",
                                    inF: config.moeInterDim, outF: dim, rng: &rng),
                w3: try! loadLinear(loader, base: "\(sep).w3",
                                    inF: dim, outF: config.moeInterDim, rng: &rng),
                swigluLimit: config.swigluLimit)

            // Routed experts: branch by checkpoint format.
            //  • MLX-native (mlx-community): single packed tensor per
            //    projection, dispatched via SwitchMoEFFN +
            //    MLXFast.gatherQuantizedMatmul.
            //  • Custom DeepSeek (original): 256 separate Linear×3 per
            //    layer, dispatched via MoEFFN's per-expert loop.
            let ffn: any FFNModule
            if config.isMLXNative {
                // Per-tensor quant spec from config.json. Falls back to
                // (groupSize=32, bits=4, mode="mxfp4") which is the
                // observed default for switch_mlp.{gate,up,down}_proj
                // in the mlx-community checkpoint.
                let baseGate = "\(lp).ffn.switch_mlp.gate_proj"
                let baseUp   = "\(lp).ffn.switch_mlp.up_proj"
                let baseDown = "\(lp).ffn.switch_mlp.down_proj"
                let specGate = config.mlxQuantSpec(for: "model." + baseGate)
                    ?? ModelConfig.MLXQuantSpec(groupSize: 32, bits: 4, mode: "mxfp4")
                let specUp = config.mlxQuantSpec(for: "model." + baseUp)
                    ?? specGate
                let specDown = config.mlxQuantSpec(for: "model." + baseDown)
                    ?? specGate

                let gateProj = SwitchProj(
                    nExperts: nExperts,
                    inFeatures: dim, outFeatures: config.moeInterDim,
                    base: baseGate,
                    groupSize: specGate.groupSize, bits: specGate.bits,
                    mode: specGate.mode, loader: loader)
                let upProj = SwitchProj(
                    nExperts: nExperts,
                    inFeatures: dim, outFeatures: config.moeInterDim,
                    base: baseUp,
                    groupSize: specUp.groupSize, bits: specUp.bits,
                    mode: specUp.mode, loader: loader)
                let downProj = SwitchProj(
                    nExperts: nExperts,
                    inFeatures: config.moeInterDim, outFeatures: dim,
                    base: baseDown,
                    groupSize: specDown.groupSize, bits: specDown.bits,
                    mode: specDown.mode, loader: loader)

                let switchFFN = SwitchMoEFFN(
                    config: config, gate: gate,
                    gateProj: gateProj, upProj: upProj, downProj: downProj,
                    sharedExpert: sharedExpert)
                switchFFN.layerId = i
                ffn = switchFFN
            } else {
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
                    let w1 = try! loadLinear(loader, base: "\(ep).w1",
                                              inF: dim, outF: config.moeInterDim, rng: &rng)
                    let w2 = try! loadLinear(loader, base: "\(ep).w2",
                                              inF: config.moeInterDim, outF: dim, rng: &rng)
                    let w3 = try! loadLinear(loader, base: "\(ep).w3",
                                              inF: dim, outF: config.moeInterDim, rng: &rng)
                    if mlxQuantExperts {
                        w1.useMLXQuant = true
                        w2.useMLXQuant = true
                        w3.useMLXQuant = true
                    }
                    experts.append(Expert(
                        w1: w1, w2: w2, w3: w3,
                        swigluLimit: config.swigluLimit))
                }
                let moe = MoEFFN(config: config, gate: gate,
                                 experts: experts, shared: sharedExpert)
                moe.layerId = i
                ffn = moe
            }

            // ---- HC params ----
            let hcAttnFn = (try? loader.tryLoad(["\(lp).hc_attn_fn"]))
                ?? AssemblyHelpers.randomTensor([mixHc, hc * dim], rng: &rng, scale: 0.02)
            let hcAttnBase = (try? loader.tryLoad(["\(lp).hc_attn_base"]))
                ?? AssemblyHelpers.randomTensor([mixHc], rng: &rng, scale: 0.0)
            let hcAttnScale = (try? loader.tryLoad(["\(lp).hc_attn_scale"]))
                ?? AssemblyHelpers.randomTensor([3], rng: &rng, scale: 0.5)
            let hcFfnFn = (try? loader.tryLoad(["\(lp).hc_ffn_fn"]))
                ?? AssemblyHelpers.randomTensor([mixHc, hc * dim], rng: &rng, scale: 0.02)
            let hcFfnBase = (try? loader.tryLoad(["\(lp).hc_ffn_base"]))
                ?? AssemblyHelpers.randomTensor([mixHc], rng: &rng, scale: 0.0)
            let hcFfnScale = (try? loader.tryLoad(["\(lp).hc_ffn_scale"]))
                ?? AssemblyHelpers.randomTensor([3], rng: &rng, scale: 0.5)

            let block = Block(layerId: i, config: config,
                                attn: mla, ffn: ffn,
                                attnNorm: attnNorm, ffnNorm: ffnNorm,
                                hcAttnFn: hcAttnFn, hcAttnBase: hcAttnBase,
                                hcAttnScale: hcAttnScale,
                                hcFfnFn: hcFfnFn, hcFfnBase: hcFfnBase,
                                hcFfnScale: hcFfnScale)
            
            blocksLock.lock()
            blocksOpt[i] = block
            blocksLock.unlock()
        }
        
        let blocks = blocksOpt.compactMap { $0 }
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

        // Wire per-MoE expert streaming. Each MoEFFN pulls in only its
        // `topK` active routed experts from disk after the gate runs
        // and releases them before the next layer, keeping the
        // resident MoE working-set proportional to `topK` instead of
        // `nRoutedExperts`. `Transformer.forward` keeps the
        // non-expert prefetch / release on the layer level.
        for block in blocks {
            block.ffn.weightLoader = loader
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
    
    let wNames = ["\(base).weight"]
    let sNames = ["\(base).scale", "\(base).weight_scale_inv"]
    
    if let dt = loader.dtype(ofAny: wNames) {
        let weightName = wNames.first { loader.dtype(of: $0) != nil }!
        let scaleName = sNames.first { loader.dtype(of: $0) != nil }
        
        var wDtype = dt
        if base.contains(".experts.") && wDtype == .i8 {
            wDtype = .fp4E2M1
        }
        
        return Linear(inFeatures: inF, outFeatures: outF,
                      weightName: weightName,
                      scaleName: scaleName,
                      weightDType: wDtype,
                      loader: loader,
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
        return Tensor(array: t.array.asType(.int32), dtype: .i32)
    }

    static func linear(`in` inFeatures: Int, out outFeatures: Int,
                       rng: inout MiniRNG, scale: Float = 0.02) -> Linear {
        let w = randomTensor([outFeatures, inFeatures], rng: &rng, scale: scale)
        return Linear(inFeatures: inFeatures, outFeatures: outFeatures, weight: w, scale: nil)
    }


}
