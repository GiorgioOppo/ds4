import Foundation
import Metal
import Accelerate
#if canImport(Darwin)
import Darwin
#endif

/// Mix wall-clock nanoseconds + pid into a 64-bit LCG seed. Avoids
/// the trap where two runs of the CLI with `--temperature > 0` would
/// produce identical "random" output because every `SamplingOptions`
/// instance was seeded with the same compile-time constant.
///
/// `@usableFromInline` (not `public`) so it can be referenced from a
/// public initializer's default-argument expression without becoming
/// part of the module's public ABI surface.
@usableFromInline
@inline(__always)
internal func defaultSamplerSeed() -> UInt64 {
    var seed: UInt64 = UInt64(DispatchTime.now().uptimeNanoseconds)
    seed ^= UInt64(bitPattern: Int64(getpid())) &* 0x9E37_79B9_7F4A_7C15
    seed = seed &* 6364136223846793005 &+ 1442695040888963407
    return seed | 1   // keep odd to avoid degenerate LCG cycles
}

/// Sampling options collected for one decode step. Pipeline order:
/// temperature → repetition/frequency/presence penalties → top-K → min-P
/// → tail-free → typical → top-P → (Gumbel-max OR mirostat). When
/// `temperature == 0` and every filter is disabled, we shortcut to
/// greedy argmax.
///
/// All filter params use "neutral" defaults that disable the step:
/// `topK=0`, `topP=1`, `minP=0`, `tailFree=1`, `typical=1`,
/// `repetitionPenalty=1`, `frequencyPenalty=0`, `presencePenalty=0`,
/// `mirostatTau=0`.
public struct SamplingOptions {
    public var temperature: Float = 1.0
    public var topK: Int = 0                    // 0 = disabled
    public var topP: Float = 1.0                // 1.0 = disabled
    /// Filter tokens with probability < `minP × max_prob`. Standard
    /// llama.cpp range: 0.05 – 0.1. `0` disables.
    public var minP: Float = 0.0
    /// Tail-free sampling z-parameter. Cuts the tail when the
    /// second-derivative of the sorted probs grows past `z`. `1` disables.
    public var tailFree: Float = 1.0
    /// Locally-typical sampling p-parameter. Keeps tokens whose
    /// information content sits within mass `typical` around the
    /// distribution entropy. `1` disables.
    public var typical: Float = 1.0
    public var repetitionPenalty: Float = 1.0   // 1.0 = disabled
    /// OpenAI-style frequency penalty. Subtracts `frequencyPenalty *
    /// count(token)` from logits. `0` disables.
    public var frequencyPenalty: Float = 0.0
    /// OpenAI-style presence penalty. Subtracts `presencePenalty` once
    /// for any token that appears in history. `0` disables.
    public var presencePenalty: Float = 0.0
    /// Mirostat v2 target surprise. `0` disables (falls through to
    /// Gumbel-max). Typical value: 5.0.
    public var mirostatTau: Float = 0.0
    /// Mirostat v2 learning rate. Default 0.1.
    public var mirostatEta: Float = 0.1
    /// Mirostat running estimate of surprise. Initialised to
    /// `2 * mirostatTau` by convention; updated in place across calls.
    public var mirostatMu: Float = 10.0
    /// Per-instance LCG state. Pass an explicit value for reproducibility;
    /// the default is wall-clock + pid mixed, so distinct runs really
    /// produce distinct streams.
    public var rngState: UInt64 = defaultSamplerSeed()

    public init(temperature: Float = 1.0, topK: Int = 0, topP: Float = 1.0,
                minP: Float = 0.0, tailFree: Float = 1.0, typical: Float = 1.0,
                repetitionPenalty: Float = 1.0,
                frequencyPenalty: Float = 0.0, presencePenalty: Float = 0.0,
                mirostatTau: Float = 0.0, mirostatEta: Float = 0.1,
                mirostatMu: Float = 10.0,
                rngState: UInt64 = defaultSamplerSeed()) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.minP = minP
        self.tailFree = tailFree
        self.typical = typical
        self.repetitionPenalty = repetitionPenalty
        self.frequencyPenalty = frequencyPenalty
        self.presencePenalty = presencePenalty
        self.mirostatTau = mirostatTau
        self.mirostatEta = mirostatEta
        self.mirostatMu = mirostatMu
        self.rngState = rngState
    }

    /// True iff every filter / penalty is at its disabled default, so the
    /// caller can shortcut to greedy argmax.
    @inline(__always)
    fileprivate var allFiltersDisabled: Bool {
        return temperature == 0 && topK == 0 && topP == 1.0
            && minP == 0.0 && tailFree == 1.0 && typical == 1.0
            && repetitionPenalty == 1.0
            && frequencyPenalty == 0.0 && presencePenalty == 0.0
            && mirostatTau == 0.0
    }
}

/// All sampling beyond the GPU-side temperature scaling is host-side: at
/// each decode step we already pay a CPU/GPU sync to read the chosen
/// token id, so doing top-K / top-P / Gumbel-max in Swift on the small
/// vocabSize-length logits buffer is essentially free and avoids new
/// Metal kernels.
public enum Sampler {
    private static let argmaxP = Device.shared.makePipeline("argmax_f32")
    private static let argmax1P = Device.shared.makePipeline("argmax_f32_stage1")
    private static let argmax2P = Device.shared.makePipeline("argmax_f32_stage2")
    private static let tempP = Device.shared.makePipeline("apply_temperature")

    /// Soglia oltre cui l'argmax usa il path multi-stage. Sotto, il
    /// single-threadgroup kernel è già abbastanza veloce. Empirico:
    /// per V < 8192 il dispatch overhead supera il guadagno della
    /// parallelizzazione fra threadgroup.
    private static let argmaxMultiStageThreshold = 8192

    /// Dimensione del tile per il path multi-stage. Ogni threadgroup
    /// gestisce `argmaxTileSize` elementi. Per V=130k → 64
    /// threadgroup paralleli (= sufficienti per saturare una Apple
    /// GPU da 10-40 core).
    private static let argmaxTileSize = 2048

    /// Greedy argmax. GPU-side reduction.
    ///
    /// Per V piccolo usa un singolo threadgroup (256 thread).
    /// Per V grande (≥ 8192 logit) usa multi-stage:
    ///   - Stage 1: M = ceil(V/2048) threadgroup paralleli, ognuno
    ///     produce il proprio (val, idx) parziale.
    ///   - Stage 2: 1 threadgroup riduce gli M parziali al risultato
    ///     finale.
    /// Su vocab 130k passa da ~1 shader core attivo a ~M core,
    /// riducendo la latenza dell'argmax di ~5-8×.
    public static func argmax(_ logits: Tensor) -> Int {
        precondition(logits.dtype == .f32 && logits.shape.count == 2 && logits.shape[0] == 1)
        let V = logits.shape[1]
        let outBuf = Device.shared.mtl.makeBuffer(length: 4, options: .storageModeShared)!

        if V < argmaxMultiStageThreshold {
            // Path singolo threadgroup (legacy).
            let cmd = Device.shared.queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(argmaxP)
            enc.setBuffer(logits.buffer, offset: logits.offset, index: 0)
            enc.setBuffer(outBuf, offset: 0, index: 1)
            var v = UInt32(V)
            enc.setBytes(&v, length: 4, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit(); cmd.waitUntilCompleted()
            return Int(outBuf.contents().load(as: UInt32.self))
        }

        // Path multi-stage.
        let tileSize = argmaxTileSize
        let M = (V + tileSize - 1) / tileSize
        // Buffer per i (val, idx) parziali. private storage va bene
        // — il bufer è prodotto e consumato dalla GPU, mai letto dal
        // host.
        let partVBuf = Device.shared.mtl.makeBuffer(
            length: M * MemoryLayout<Float>.size, options: .storageModePrivate)!
        let partIBuf = Device.shared.mtl.makeBuffer(
            length: M * MemoryLayout<UInt32>.size, options: .storageModePrivate)!

        let cmd = Device.shared.queue.makeCommandBuffer()!

        // Stage 1.
        let enc1 = cmd.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(argmax1P)
        enc1.setBuffer(logits.buffer, offset: logits.offset, index: 0)
        enc1.setBuffer(partVBuf, offset: 0, index: 1)
        enc1.setBuffer(partIBuf, offset: 0, index: 2)
        var dims = SIMD2<UInt32>(UInt32(V), UInt32(tileSize))
        enc1.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)
        let simdW1 = argmax1P.threadExecutionWidth
        let tg1 = max(simdW1, min(256, argmax1P.maxTotalThreadsPerThreadgroup))
        enc1.dispatchThreadgroups(MTLSize(width: M, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tg1, height: 1, depth: 1))
        enc1.endEncoding()

        // Stage 2.
        let enc2 = cmd.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(argmax2P)
        enc2.setBuffer(partVBuf, offset: 0, index: 0)
        enc2.setBuffer(partIBuf, offset: 0, index: 1)
        enc2.setBuffer(outBuf, offset: 0, index: 2)
        var m32 = UInt32(M)
        enc2.setBytes(&m32, length: 4, index: 3)
        let simdW2 = argmax2P.threadExecutionWidth
        let tg2 = max(simdW2, min(M, 256))
        enc2.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tg2, height: 1, depth: 1))
        enc2.endEncoding()

        cmd.commit(); cmd.waitUntilCompleted()
        return Int(outBuf.contents().load(as: UInt32.self))
    }

    /// In-place GPU temperature scaling. T == 0 leaves logits untouched
    /// (caller should switch to argmax).
    public static func applyTemperature(_ logits: Tensor, _ T: Float) {
        precondition(logits.dtype == .f32)
        if T == 0.0 { return }
        let V = logits.count
        let cmd = Device.shared.queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(tempP)
        enc.setBuffer(logits.buffer, offset: logits.offset, index: 0)
        var v = UInt32(V); var t = T
        enc.setBytes(&v, length: 4, index: 1)
        enc.setBytes(&t, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: V, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
    }

    /// Full sampling pipeline. Reads the logits buffer to host once and
    /// performs the rest in pure Swift. Updates `options.rngState` and,
    /// when active, `options.mirostatMu` across calls.
    public static func sample(_ logits: Tensor, history: [Int],
                              options: inout SamplingOptions) -> Int {
        precondition(logits.dtype == .f32 && logits.shape.count == 2 && logits.shape[0] == 1)
        let V = logits.shape[1]

        if options.allFiltersDisabled {
            return argmax(logits)
        }

        var arr = logits.toFloatArray()

        // 1. Temperature.
        if options.temperature > 0 && options.temperature != 1.0 {
            let inv = 1.0 / max(options.temperature, 1e-5)
            for i in 0..<V { arr[i] *= inv }
        }

        // 2a. Repetition penalty: divide positive logits by penalty (and
        //     multiply negative logits by it) for tokens already in history.
        if options.repetitionPenalty != 1.0 {
            let p = options.repetitionPenalty
            for id in history where id >= 0 && id < V {
                arr[id] = arr[id] >= 0 ? arr[id] / p : arr[id] * p
            }
        }

        // 2b. Frequency + presence penalty (OpenAI-style).
        //     freq: subtract freq_pen × count(token)
        //     pres: subtract pres_pen if token appears at all
        //     Both are additive in logit space, distinct from the
        //     repetition penalty above.
        if options.frequencyPenalty != 0.0 || options.presencePenalty != 0.0 {
            var counts = [Int](repeating: 0, count: V)
            for id in history where id >= 0 && id < V { counts[id] += 1 }
            let fp = options.frequencyPenalty
            let pp = options.presencePenalty
            for i in 0..<V where counts[i] > 0 {
                arr[i] -= fp * Float(counts[i])
                arr[i] -= pp
            }
        }

        // Mirostat path: it sets its own filter + sample strategy and
        // does not use top-K/top-P/min-P/tail-free/typical/Gumbel-max.
        if options.mirostatTau > 0.0 {
            return mirostatV2Sample(&arr, vocabSize: V, options: &options)
        }

        // 3. Top-K filter.
        if options.topK > 0 && options.topK < V {
            let kth = nthLargest(arr, k: options.topK)
            for i in 0..<V where arr[i] < kth { arr[i] = -.infinity }
        }

        // 4. Min-P filter. Compute max prob (stable softmax max-shift),
        //    then keep only tokens with prob >= minP × p_max.
        if options.minP > 0.0 {
            applyMinP(&arr, vocabSize: V, minP: options.minP)
        }

        // 5. Tail-free sampling. Cuts the tail where the absolute
        //    second derivative of sorted probs grows past `tailFree`.
        if options.tailFree < 1.0 && options.tailFree > 0.0 {
            applyTailFree(&arr, vocabSize: V, z: options.tailFree)
        }

        // 6. Locally typical sampling. Keeps tokens whose surprise
        //    `-log(p_i)` is closest to the distribution entropy until
        //    cumulative mass `typical` is reached.
        if options.typical < 1.0 && options.typical > 0.0 {
            applyTypical(&arr, vocabSize: V, p: options.typical)
        }

        // 7. Top-P filter (nucleus).
        if options.topP < 1.0 {
            applyTopP(&arr, vocabSize: V, topP: options.topP)
        }

        // 8. Multinomial via Gumbel-max trick. argmax(log(p) + g) where
        //    g ~ Gumbel(0,1). Reference: generate.py:19-24.
        var rng = options.rngState
        var bestI = 0
        var bestV = -Float.infinity
        let mLog = arr.max() ?? 0
        var anyFinite = false
        for i in 0..<V {
            if arr[i] == -.infinity { continue }
            anyFinite = true
            let u = nextUnit(&rng)
            // Gumbel(0,1) = -log(-log(u)) but the trick equivalently uses
            //   key = log(p_i) + g_i, which here is logit_i (already log
            //   numerator) - log(-log(u)).
            // We work in log-space directly: skip softmax normalization
            // (constant subtraction does not change argmax) so we avoid
            // computing the partition function twice.
            let g = -log(max(-log(max(u, 1e-12)), 1e-30))
            let key = (arr[i] - mLog) + g
            if key > bestV { bestV = key; bestI = i }
        }
        options.rngState = rng
        // Defensive fallback: if every token got filtered out (can happen
        // with extreme tfs/typical/topP combos), pick the unfiltered argmax
        // of the original logits to keep generation alive.
        if !anyFinite { return argmax(logits) }
        return bestI
    }

    // MARK: - Filter helpers

    /// In-place min-P filter on a logits array. Vectorizzato con
    /// Accelerate analogamente a `softmaxDouble` — `vvexp` per
    /// l'esponenziazione SIMD, `vDSP_*` per max/sum/normalize.
    @inline(__always)
    private static func applyMinP(_ arr: inout [Float], vocabSize V: Int, minP: Float) {
        // max(arr) via vDSP_maxv.
        var m: Float = 0
        arr.withUnsafeBufferPointer { p in
            vDSP_maxv(p.baseAddress!, 1, &m, vDSP_Length(V))
        }
        // shifted[i] = Double(arr[i] - m); -infinity passa through.
        var shifted = [Double](repeating: 0, count: V)
        for i in 0..<V { shifted[i] = Double(arr[i] - m) }

        var probs = [Double](repeating: 0, count: V)
        var n = Int32(V)
        vvexp(&probs, shifted, &n)

        var sum: Double = 0
        vDSP_sveD(probs, 1, &sum, vDSP_Length(V))
        if sum <= 0 { return }

        var inv = 1.0 / sum
        vDSP_vsmulD(probs, 1, &inv, &probs, 1, vDSP_Length(V))

        // pMax = max(probs).
        var pMax: Double = 0
        vDSP_maxvD(probs, 1, &pMax, vDSP_Length(V))

        let threshold = pMax * Double(minP)
        for i in 0..<V where probs[i] < threshold { arr[i] = -.infinity }
    }

    /// In-place tail-free filter. The "tail" is defined as the region
    /// where the second derivative |p_{i+2} − 2p_{i+1} + p_i| of the
    /// descending-sorted probability curve sums past `z` of its total.
    @inline(__always)
    private static func applyTailFree(_ arr: inout [Float], vocabSize V: Int, z: Float) {
        let probs = softmaxDouble(arr, vocabSize: V)
        let order = (0..<V).sorted { probs[$0] > probs[$1] }
        if order.count < 3 { return }
        var d2 = [Double](repeating: 0, count: order.count - 2)
        var total: Double = 0
        for i in 0..<(order.count - 2) {
            let v = abs(probs[order[i + 2]] - 2.0 * probs[order[i + 1]] + probs[order[i]])
            d2[i] = v
            total += v
        }
        if total <= 0 { return }
        var cum: Double = 0
        var lastKept = order.count - 1
        for i in 0..<d2.count {
            cum += d2[i] / total
            if cum >= Double(z) { lastKept = i + 1; break }
        }
        for i in (lastKept + 1)..<order.count { arr[order[i]] = -.infinity }
    }

    /// In-place locally-typical filter. Keeps the tokens whose
    /// |surprise − entropy| is smallest until cumulative mass ≥ p.
    @inline(__always)
    private static func applyTypical(_ arr: inout [Float], vocabSize V: Int, p: Float) {
        let probs = softmaxDouble(arr, vocabSize: V)
        var H: Double = 0
        for q in probs where q > 0 { H -= q * log(q) }
        // Sort by absolute deviation of surprise from entropy.
        let order = (0..<V).sorted { i, j in
            let si = probs[i] > 0 ? -log(probs[i]) : .infinity
            let sj = probs[j] > 0 ? -log(probs[j]) : .infinity
            return abs(si - H) < abs(sj - H)
        }
        var cum: Double = 0
        var cutoff = order.count
        for (idx, ti) in order.enumerated() {
            cum += probs[ti]
            if cum >= Double(p) { cutoff = idx + 1; break }
        }
        if cutoff >= order.count { return }
        for i in cutoff..<order.count { arr[order[i]] = -.infinity }
    }

    /// In-place top-P (nucleus) filter — factored out from the original
    /// inline implementation so the pipeline reads linearly.
    @inline(__always)
    private static func applyTopP(_ arr: inout [Float], vocabSize V: Int, topP: Float) {
        let probs = softmaxDouble(arr, vocabSize: V)
        let order = (0..<V).sorted { probs[$0] > probs[$1] }
        var cum: Double = 0
        var threshold = -Double.infinity
        for idx in order {
            cum += probs[idx]
            if cum >= Double(topP) {
                threshold = probs[idx]
                break
            }
        }
        for i in 0..<V where probs[i] < threshold { arr[i] = -.infinity }
    }

    /// Mirostat v2: surprise-controlled top-k where k is implied by the
    /// running estimate `μ`. After sampling, `μ ← μ − η × (S_t − τ)`.
    ///
    /// Reference: Basu et al. 2020 ("Mirostat: A Neural Text Decoding
    /// Algorithm that Directly Controls Perplexity").
    @inline(__always)
    private static func mirostatV2Sample(_ arr: inout [Float], vocabSize V: Int,
                                         options: inout SamplingOptions) -> Int {
        let probs = softmaxDouble(arr, vocabSize: V)
        // Sort indices by descending prob.
        let order = (0..<V).sorted { probs[$0] > probs[$1] }
        // Truncate: keep prefix where surprise -log(p_i) < μ.
        let mu = Double(options.mirostatMu)
        var keep = 1
        for (k, idx) in order.enumerated() {
            let surprise = probs[idx] > 0 ? -log(probs[idx]) : .infinity
            if surprise > mu { keep = max(1, k); break }
            keep = k + 1
        }
        // Mask the rest.
        for i in keep..<order.count { arr[order[i]] = -.infinity }
        // Gumbel-max sample over the kept set.
        var rng = options.rngState
        var bestI = order[0]
        var bestV = -Float.infinity
        let mLog = arr.max() ?? 0
        for k in 0..<keep {
            let i = order[k]
            let u = nextUnit(&rng)
            let g = -log(max(-log(max(u, 1e-12)), 1e-30))
            let key = (arr[i] - mLog) + g
            if key > bestV { bestV = key; bestI = i }
        }
        options.rngState = rng
        // Update μ ← μ − η × (S_t − τ).
        let pSel = probs[bestI]
        let St = pSel > 0 ? -log(pSel) : Double(options.mirostatTau)
        let err = Float(St) - options.mirostatTau
        options.mirostatMu = max(0.01, options.mirostatMu - options.mirostatEta * err)
        return bestI
    }

    /// Stable softmax → Double[] without mutating the input. Skips
    /// `-infinity` entries (treats them as zero mass).
    ///
    /// Implementazione vettorizzata con Accelerate:
    ///   1. max-reduce via `vDSP_maxv` (Float, single SIMD scan)
    ///   2. shift + Float→Double convert con loop scalare (gestisce
    ///      `-infinity` esplicitamente: lo mappa a -INF in Double
    ///      così `vvexp` ritorna 0 lì, semantica identica al loop
    ///      originale che skippava la entry)
    ///   3. exp per-element via `vvexp` (SIMD vForce)
    ///   4. sum via `vDSP_sveD`
    ///   5. divisione in-place via `vDSP_vsmulD` con `1/sum`
    ///
    /// Speedup atteso vs il loop Swift puro: ~3-5× per V=130k
    /// (vForce vvexp è il guadagno principale).
    @inline(__always)
    private static func softmaxDouble(_ arr: [Float], vocabSize V: Int) -> [Double] {
        // 1) max(arr) — vDSP_maxv ignora -infinity correttamente
        //    (lo confronta con `>` invece di `>=`).
        var m: Float = 0
        arr.withUnsafeBufferPointer { p in
            vDSP_maxv(p.baseAddress!, 1, &m, vDSP_Length(V))
        }

        // 2) shifted[i] = Double(arr[i] - m). Per i = -infinity in
        //    arr, shifted[i] resta -infinity (così vvexp ritorna 0,
        //    semantica del vecchio loop "treats as zero mass").
        var shifted = [Double](repeating: 0, count: V)
        for i in 0..<V {
            shifted[i] = Double(arr[i] - m)
        }

        // 3) exp per-element via vForce.
        var exps = [Double](repeating: 0, count: V)
        var n = Int32(V)
        vvexp(&exps, shifted, &n)

        // 4) sum (Double precision per stabilità numerica).
        var sum: Double = 0
        vDSP_sveD(exps, 1, &sum, vDSP_Length(V))

        // 5) normalize in-place. `vDSP_vsmulD` non gestisce sum<=0
        //    (divisione per zero); ritorniamo gli exp non normalizzati
        //    (tutto 0 se -INF dominante) come fallback safe.
        if sum > 0 {
            var inv = 1.0 / sum
            vDSP_vsmulD(exps, 1, &inv, &exps, 1, vDSP_Length(V))
        }
        return exps
    }

    // MARK: - Pure-Swift helpers

    /// Returns the K-th largest value in `arr`. Uses a partial sort via
    /// quickselect-style nth_element. O(N) average.
    private static func nthLargest(_ arr: [Float], k: Int) -> Float {
        var copy = arr
        let target = k - 1
        var lo = 0, hi = copy.count - 1
        while lo < hi {
            let pivot = copy[(lo + hi) / 2]
            var i = lo, j = hi
            while i <= j {
                while copy[i] > pivot { i += 1 }
                while copy[j] < pivot { j -= 1 }
                if i <= j { copy.swapAt(i, j); i += 1; j -= 1 }
            }
            if target <= j { hi = j }
            else if target >= i { lo = i }
            else { return copy[target] }
        }
        return copy[target]
    }

    /// LCG next unit Float in (0, 1). Inline so the inner sample loop
    /// stays dependency-free.
    private static func nextUnit(_ state: inout UInt64) -> Float {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Float(Double(state >> 11) / Double(1 << 53))
    }
}
