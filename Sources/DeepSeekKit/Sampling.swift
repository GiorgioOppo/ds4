import Foundation
import Metal

/// Sampling options collected for one decode step. Mirrors the standard
/// HF sampling pipeline: temperature → repetition penalty → top-K → top-P
/// → multinomial. When `temperature == 0` we shortcut to greedy argmax.
public struct SamplingOptions {
    public var temperature: Float = 1.0
    public var topK: Int = 0                    // 0 = disabled
    public var topP: Float = 1.0                // 1.0 = disabled
    public var repetitionPenalty: Float = 1.0   // 1.0 = disabled
    public var rngState: UInt64 = 0xA17EBABE

    public init(temperature: Float = 1.0, topK: Int = 0, topP: Float = 1.0,
                repetitionPenalty: Float = 1.0, rngState: UInt64 = 0xA17EBABE) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.rngState = rngState
    }
}

/// All sampling beyond the GPU-side temperature scaling is host-side: at
/// each decode step we already pay a CPU/GPU sync to read the chosen
/// token id, so doing top-K / top-P / Gumbel-max in Swift on the small
/// vocabSize-length logits buffer is essentially free and avoids new
/// Metal kernels.
public enum Sampler {
    private static let argmaxP = Device.shared.makePipeline("argmax_f32")
    private static let tempP = Device.shared.makePipeline("apply_temperature")

    /// Greedy argmax. GPU-side reduction.
    public static func argmax(_ logits: Tensor) -> Int {
        precondition(logits.dtype == .f32 && logits.shape.count == 2 && logits.shape[0] == 1)
        let V = logits.shape[1]
        let outBuf = Device.shared.mtl.makeBuffer(length: 4, options: .storageModeShared)!

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

    /// Full sampling pipeline: temperature → repetition penalty → top-K
    /// → top-P → multinomial. Reads the logits buffer to host once and
    /// performs the rest in pure Swift. Updates `options.rngState` for
    /// repeatable streams across calls.
    public static func sample(_ logits: Tensor, history: [Int],
                              options: inout SamplingOptions) -> Int {
        precondition(logits.dtype == .f32 && logits.shape.count == 2 && logits.shape[0] == 1)
        let V = logits.shape[1]

        if options.temperature == 0 && options.repetitionPenalty == 1.0
            && options.topK == 0 && options.topP == 1.0 {
            return argmax(logits)
        }

        var arr = logits.toFloatArray()

        // 1. Temperature.
        if options.temperature > 0 && options.temperature != 1.0 {
            let inv = 1.0 / max(options.temperature, 1e-5)
            for i in 0..<V { arr[i] *= inv }
        }

        // 2. Repetition penalty: divide positive logits by penalty (and
        //    multiply negative logits by it) for tokens already in history.
        if options.repetitionPenalty != 1.0 {
            let p = options.repetitionPenalty
            for id in history where id >= 0 && id < V {
                arr[id] = arr[id] >= 0 ? arr[id] / p : arr[id] * p
            }
        }

        // 3. Top-K filter.
        if options.topK > 0 && options.topK < V {
            let kth = nthLargest(arr, k: options.topK)
            for i in 0..<V where arr[i] < kth { arr[i] = -.infinity }
        }

        // 4. Top-P filter (nucleus).
        if options.topP < 1.0 {
            // Stable softmax over the (possibly already filtered) logits.
            let m = arr.max() ?? 0
            var sum: Double = 0
            var probs = [Double](repeating: 0, count: V)
            for i in 0..<V {
                let e = exp(Double(arr[i] - m))
                probs[i] = e
                sum += e
            }
            for i in 0..<V { probs[i] /= sum }

            // Sort indices by descending probability.
            let order = (0..<V).sorted { probs[$0] > probs[$1] }
            var cum: Double = 0
            var threshold = -Double.infinity
            for idx in order {
                cum += probs[idx]
                if cum >= Double(options.topP) {
                    threshold = probs[idx]
                    break
                }
            }
            for i in 0..<V where probs[i] < threshold { arr[i] = -.infinity }
        }

        // 5. Multinomial via Gumbel-max trick. argmax(log(p) + g) where
        //    g ~ Gumbel(0,1). Reference: generate.py:19-24.
        var rng = options.rngState
        var bestI = 0
        var bestV = -Float.infinity
        let mLog = arr.max() ?? 0
        for i in 0..<V {
            if arr[i] == -.infinity { continue }
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
        return bestI
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
