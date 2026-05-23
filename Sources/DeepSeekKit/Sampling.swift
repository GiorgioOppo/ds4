import Foundation
import Metal
import Accelerate
#if canImport(Darwin)
import Darwin
#endif
import MLX

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

    // MARK: - T5 sampler residui (TODO §10.5)

    /// Per-token additive bias applied at stage 0 (before temperature
    /// scaling). Mirrors the OpenAI API `logit_bias` field — the caller
    /// can hard-block a token by passing `[id: -100]` or boost one with
    /// `[id: 5]`. Empty (the default) is the no-op identity.
    public var logitBias: [Int32: Float] = [:]

    /// DRY sampler ("Don't Repeat Yourself"). Multiplicative penalty on
    /// candidates that would extend an n-gram already present in the
    /// history. `dryMultiplier == 0` disables. Standard tunables:
    ///   - `dryMultiplier`: 0.5–1.0 typical; 0 disables.
    ///   - `dryBase`: 1.5–2.0 typical (penalty grows exponentially in
    ///     the matched n-gram length beyond `dryAllowedLength`).
    ///   - `dryAllowedLength`: short n-grams below this are not
    ///     penalized — keeps natural repetition (function words,
    ///     punctuation) intact.
    public var dryMultiplier: Float = 0.0
    public var dryBase: Float = 1.75
    public var dryAllowedLength: Int = 2

    /// Mirostat v1 toggle. When true *and* `mirostatTau > 0`, the
    /// sampler uses the smoothed `μ`-update path (averaging the recent
    /// `mirostatM` surprises) instead of the single-step v2. Slight
    /// stability win in long generations at the cost of a per-step
    /// `[Float]` allocation. Default false (v2).
    public var mirostatV1: Bool = false

    /// Window size for the v1 smoothed update. Ignored when
    /// `mirostatV1 == false`. The reference paper uses ~100; smaller
    /// values track surprise faster but oscillate more.
    public var mirostatM: Int = 100

    /// Internal: rolling buffer of recent surprise values for the
    /// v1 update. The sampler appends to this on every call and
    /// trims to `mirostatM` entries. Persisted across calls so the
    /// running mean reflects history within the same generation.
    public var mirostatV1History: [Float] = []

    /// Optional schema-constrained-decoding mask (TODO §10.3 / T3).
    /// When non-nil, the sampler applies the mask at stage 0
    /// (`-INF`s every token whose decoded bytes would steer the
    /// output off the schema), and after sampling calls
    /// `mask.advance(token:)` so the next step's allowed-set
    /// reflects the new consumed prefix.
    ///
    /// Reference type by design — the consumed-prefix state lives
    /// on the mask object and persists across calls without the
    /// caller having to thread it back through `options`.
    public var schemaMask: SchemaMask? = nil

    public init(temperature: Float = 1.0, topK: Int = 0, topP: Float = 1.0,
                minP: Float = 0.0, tailFree: Float = 1.0, typical: Float = 1.0,
                repetitionPenalty: Float = 1.0,
                frequencyPenalty: Float = 0.0, presencePenalty: Float = 0.0,
                mirostatTau: Float = 0.0, mirostatEta: Float = 0.1,
                mirostatMu: Float = 10.0,
                logitBias: [Int32: Float] = [:],
                dryMultiplier: Float = 0.0, dryBase: Float = 1.75,
                dryAllowedLength: Int = 2,
                mirostatV1: Bool = false, mirostatM: Int = 100,
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
        self.logitBias = logitBias
        self.dryMultiplier = dryMultiplier
        self.dryBase = dryBase
        self.dryAllowedLength = dryAllowedLength
        self.mirostatV1 = mirostatV1
        self.mirostatM = mirostatM
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
            && logitBias.isEmpty && dryMultiplier == 0.0
            && schemaMask == nil
    }
}

/// All sampling beyond the GPU-side temperature scaling is host-side: at
/// each decode step we already pay a CPU/GPU sync to read the chosen
/// token id, so doing top-K / top-P / Gumbel-max in Swift on the small
/// vocabSize-length logits buffer is essentially free and avoids new
/// Metal kernels.
public enum Sampler {

    public static func argmax(_ logits: Tensor) -> Int {
        precondition(logits.dtype == .f32 && logits.shape.count == 2 && logits.shape[0] == 1)
        MLX.eval(logits.array)
        return MLX.argMax(logits.array, axis: 1).item(Int.self)
    }

    public static func applyTemperature(_ logits: Tensor, _ T: Float) {
        precondition(logits.dtype == .f32)
        if T == 0.0 { return }
        logits.array = logits.array / MLXArray(T)
        MLX.eval(logits.array)
    }

    /// Full sampling pipeline. Reads the logits buffer to host once and
    /// performs the rest in pure Swift. Updates `options.rngState` and,
    /// when active, `options.mirostatMu` across calls.
    public static func sample(_ logits: Tensor, history: [Int],
                              options: inout SamplingOptions) -> Int {
        let id = sampleCore(logits, history: history, options: &options)
        // T3: advance the schema mask's consumed-prefix state so the
        // next call's allowed-set reflects what we just emitted.
        // Stop tokens are a no-op inside the mask itself.
        options.schemaMask?.advance(token: Int32(id))
        return id
    }

    /// Internal: the actual sampling body. Split out so the public
    /// `sample(...)` can wrap it with the `SchemaMask.advance(...)`
    /// post-step without smearing that concern across the four
    /// existing return paths (early greedy / mirostat v1 / mirostat
    /// v2 / Gumbel-max). All other behavior is identical to the
    /// pre-T3 sampler.
    private static func sampleCore(_ logits: Tensor, history: [Int],
                                    options: inout SamplingOptions) -> Int {
        precondition(logits.dtype == .f32 && logits.shape.count == 2 && logits.shape[0] == 1)
        let V = logits.shape[1]

        if options.allFiltersDisabled {
            return argmax(logits)
        }

        var arr = logits.toFloatArray()

        // 0a. Schema mask — `-INF`s every token whose decoded
        //     string would steer the running output off the
        //     compiled schema. Applied first so downstream
        //     temperature / penalties operate on the already
        //     constrained support. Trivial cost (precomputed
        //     allowed set in the mask).
        if let mask = options.schemaMask {
            let allowed = mask.allowedTokens()
            if allowed.isEmpty {
                // Pathological: caller compiled a mask with no
                // accepting tokens. Bail to the unconstrained
                // argmax so generation doesn't hang.
                return argmax(logits)
            }
            for i in 0..<V where !allowed.contains(Int32(i)) {
                arr[i] = -.infinity
            }
        }

        // 0b. Logit bias — additive in raw-logit space (OpenAI's
        //     `logit_bias` semantics). Applied before temperature so
        //     `-100` is a hard block regardless of T. Trivial cost.
        if !options.logitBias.isEmpty {
            for (id, bias) in options.logitBias where id >= 0 && Int(id) < V {
                arr[Int(id)] += bias
            }
        }

        // 1. Temperature — vectorized via vDSP_vsmul (~3-5× faster
        //    than the per-element Swift loop for V=130k).
        if options.temperature > 0 && options.temperature != 1.0 {
            var inv = 1.0 / max(options.temperature, 1e-5)
            arr.withUnsafeMutableBufferPointer { p in
                vDSP_vsmul(p.baseAddress!, 1, &inv,
                            p.baseAddress!, 1, vDSP_Length(V))
            }
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

        // 2c. DRY penalty — discourage candidates that would extend an
        //     n-gram already present in history. Cost is O(H × L_max)
        //     where H is the (bounded) history window and L_max is the
        //     longest match we bother extending; both are small in
        //     practice. Disabled at `dryMultiplier == 0`.
        if options.dryMultiplier > 0.0 && history.count >= 2 {
            applyDRY(&arr, vocabSize: V, history: history, options: options)
        }

        // Mirostat path: it sets its own filter + sample strategy and
        // does not use top-K/top-P/min-P/tail-free/typical/Gumbel-max.
        if options.mirostatTau > 0.0 {
            if options.mirostatV1 {
                return mirostatV1Sample(&arr, vocabSize: V, options: &options)
            }
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
        // shifted[i] = Double(arr[i] - m), vectorized in due pass
        // SIMD (vDSP_vsadd shift Float + vDSP_vspdp Float→Double).
        // -infinity passa through entrambi i pass.
        let n64 = vDSP_Length(V)
        var negM = -m
        var shiftedF = [Float](repeating: 0, count: V)
        arr.withUnsafeBufferPointer { src in
            shiftedF.withUnsafeMutableBufferPointer { dst in
                vDSP_vsadd(src.baseAddress!, 1, &negM,
                            dst.baseAddress!, 1, n64)
            }
        }
        var shifted = [Double](repeating: 0, count: V)
        shiftedF.withUnsafeBufferPointer { srcF in
            shifted.withUnsafeMutableBufferPointer { dstD in
                vDSP_vspdp(srcF.baseAddress!, 1, dstD.baseAddress!, 1, n64)
            }
        }

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

    /// Mirostat v1 (TODO §10.5): same surprise-controlled truncation
    /// as v2 — the math literature distinguishes them by the μ
    /// estimation rule, not the sampling rule — but the update step
    /// averages the recent `mirostatM` surprises instead of using only
    /// the current step. Smoother in long generations at the cost of a
    /// per-call append/trim of a small `[Float]` buffer.
    @inline(__always)
    private static func mirostatV1Sample(_ arr: inout [Float], vocabSize V: Int,
                                          options: inout SamplingOptions) -> Int {
        let probs = softmaxDouble(arr, vocabSize: V)
        let order = (0..<V).sorted { probs[$0] > probs[$1] }
        let mu = Double(options.mirostatMu)
        var keep = 1
        for (k, idx) in order.enumerated() {
            let surprise = probs[idx] > 0 ? -log(probs[idx]) : .infinity
            if surprise > mu { keep = max(1, k); break }
            keep = k + 1
        }
        for i in keep..<order.count { arr[order[i]] = -.infinity }
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

        // v1 update: append this step's surprise to the rolling
        // window, trim to `mirostatM`, drive μ toward the window mean.
        let pSel = probs[bestI]
        let St = pSel > 0 ? -log(pSel) : Double(options.mirostatTau)
        options.mirostatV1History.append(Float(St))
        let m = max(1, options.mirostatM)
        if options.mirostatV1History.count > m {
            options.mirostatV1History.removeFirst(
                options.mirostatV1History.count - m)
        }
        let sum = options.mirostatV1History.reduce(0, +)
        let avg = sum / Float(options.mirostatV1History.count)
        let err = avg - options.mirostatTau
        options.mirostatMu = max(0.01, options.mirostatMu - options.mirostatEta * err)
        return bestI
    }

    /// DRY penalty (TODO §10.5). Walks the (bounded) history window
    /// looking for previous occurrences of the most recent token; for
    /// each occurrence it computes the matching n-gram length L and
    /// — if L ≥ `dryAllowedLength` — subtracts
    /// `multiplier * base^(L - allowedLength)` from the logit of the
    /// token that previously followed that n-gram. That candidate is
    /// the one that would extend the repetition.
    ///
    /// Bounded scan: only the last 1024 history tokens participate, and
    /// individual matches stop extending at 32 tokens. Both caps are
    /// generous — typical hits are L ≤ 8 on conversational text.
    @inline(__always)
    private static func applyDRY(_ arr: inout [Float], vocabSize V: Int,
                                  history: [Int], options: SamplingOptions)
    {
        let historyCap = 1024
        let maxMatchLen = 32
        let end = history.count - 1
        guard end >= 1 else { return }
        let lastTok = history[end]
        let scanStart = max(0, end - historyCap)
        // Iterate previous positions p in the recent window where
        // history[p] == lastTok. p == end - 1 is fine (a directly
        // preceding occurrence still counts); the next position
        // history[p + 1] is the candidate that would extend the
        // n-gram and gets penalized.
        for p in scanStart..<end where history[p] == lastTok {
            // Extend backward as long as tokens match and we're
            // within both the scan window and the per-match cap.
            var L = 1
            while L < maxMatchLen
                && p - L >= scanStart
                && end - L >= 0
                && history[p - L] == history[end - L]
            {
                L += 1
            }
            // Identify the candidate that would extend this n-gram.
            let candIdx = p + 1
            guard candIdx <= end else { continue }
            let candidate = history[candIdx]
            guard candidate >= 0 && candidate < V else { continue }
            if L >= options.dryAllowedLength {
                let exponent = Float(L - options.dryAllowedLength)
                let penalty = options.dryMultiplier
                    * pow(options.dryBase, exponent)
                arr[candidate] -= penalty
            }
        }
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

        // 2) shifted[i] = Double(arr[i] - m), vectorized in due
        //    pass SIMD: (a) `vDSP_vsadd` per la shift Float, (b)
        //    `vDSP_vspdp` per promozione Float→Double. Per
        //    -infinity in input: lo shift preserva -infinity,
        //    la promozione preserva -infinity → vvexp ritorna 0
        //    (semantica del vecchio loop "treats as zero mass").
        let n64 = vDSP_Length(V)
        var negM = -m
        var shiftedF = [Float](repeating: 0, count: V)
        arr.withUnsafeBufferPointer { src in
            shiftedF.withUnsafeMutableBufferPointer { dst in
                vDSP_vsadd(src.baseAddress!, 1, &negM,
                            dst.baseAddress!, 1, n64)
            }
        }
        var shifted = [Double](repeating: 0, count: V)
        shiftedF.withUnsafeBufferPointer { srcF in
            shifted.withUnsafeMutableBufferPointer { dstD in
                vDSP_vspdp(srcF.baseAddress!, 1, dstD.baseAddress!, 1, n64)
            }
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
