import Foundation
import DS4Core

// Scalar CPU oracles for the Laguna S 2.1 decode path, ported from the
// reference `laguna-s2.1` branch (`laguna_graph_forward_token` in ds4.c and
// `metal/laguna.metal`). Laguna specifics baked in: Qwen-style per-head
// RMSNorm on Q/K followed by NeoX rotary pairs on the head prefix, YaRN only
// on the full-attention blocks (the sliding-window blocks use plain RoPE on
// an independent frequency base), a per-head softplus gate on the attention
// output, an F16 ring KV cache, SwiGLU without clamp, and the GLM-shared
// sigmoid router with 10 active experts. These references are deliberately
// Metal-free so the future graph/kernel work has a stable correctness
// boundary that runs in plain unit tests.

/// Rotary/window parameters of one Laguna block kind.
public struct LagunaAttentionSpec: Sendable, Equatable {
    public let headCount: Int
    public let kvHeadCount: Int
    public let headDim: Int
    public let rotationDims: Int
    public let cacheCapacity: Int
    public let ropeFrequencyBase: Float
    public let ropeFrequencyScale: Float
    public let extrapolationFactor: Float
    public let attentionFactor: Float
    public let betaFast: Float
    public let betaSlow: Float
    public let ropeOriginalContext: Int
    public let rmsEpsilon: Float

    public init(headCount: Int, kvHeadCount: Int, headDim: Int, rotationDims: Int,
                cacheCapacity: Int, ropeFrequencyBase: Float, ropeFrequencyScale: Float,
                extrapolationFactor: Float, attentionFactor: Float,
                betaFast: Float, betaSlow: Float, ropeOriginalContext: Int,
                rmsEpsilon: Float) {
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.headDim = headDim
        self.rotationDims = rotationDims
        self.cacheCapacity = cacheCapacity
        self.ropeFrequencyBase = ropeFrequencyBase
        self.ropeFrequencyScale = ropeFrequencyScale
        self.extrapolationFactor = extrapolationFactor
        self.attentionFactor = attentionFactor
        self.betaFast = betaFast
        self.betaSlow = betaSlow
        self.ropeOriginalContext = ropeOriginalContext
        self.rmsEpsilon = rmsEpsilon
    }

    public var queryWidth: Int { headCount * headDim }
    public var keyValueWidth: Int { kvHeadCount * headDim }

    /// Full-attention blocks (every fourth layer, 48 heads): 64 YaRN-scaled
    /// rotary dims over the 8192-token original context, cache bounded only
    /// by the configured context.
    public static func fullAttention(shape: LagunaShape = .s2_1,
                                     contextSize: Int) -> LagunaAttentionSpec {
        LagunaAttentionSpec(
            headCount: Int(shape.nHeadFull),
            kvHeadCount: Int(shape.nHeadKV),
            headDim: Int(shape.nHeadDim),
            rotationDims: Int(shape.nRot),
            cacheCapacity: contextSize,
            ropeFrequencyBase: shape.ropeFrequencyBase,
            ropeFrequencyScale: 1 / shape.ropeScaleFactor,
            extrapolationFactor: 1,
            attentionFactor: shape.ropeYarnAttnFactor,
            betaFast: shape.ropeYarnBetaFast,
            betaSlow: shape.ropeYarnBetaSlow,
            ropeOriginalContext: Int(shape.ropeOriginalContext),
            rmsEpsilon: shape.rmsEpsilon
        )
    }

    /// Sliding-window blocks (72 heads): plain NeoX RoPE over the full
    /// 128-dim head on the independent base, 512-row ring cache.
    public static func slidingWindow(shape: LagunaShape = .s2_1) -> LagunaAttentionSpec {
        LagunaAttentionSpec(
            headCount: Int(shape.nHead),
            kvHeadCount: Int(shape.nHeadKV),
            headDim: Int(shape.nHeadDim),
            rotationDims: Int(shape.nRotSWA),
            cacheCapacity: Int(shape.nSWA),
            ropeFrequencyBase: shape.ropeFrequencyBaseSWA,
            ropeFrequencyScale: 1,
            extrapolationFactor: 0,
            attentionFactor: 1,
            betaFast: 0,
            betaSlow: 0,
            ropeOriginalContext: Int(shape.contextLength),
            rmsEpsilon: shape.rmsEpsilon
        )
    }

    public static func spec(forLayer layer: Int,
                            shape: LagunaShape = .s2_1,
                            contextSize: Int) -> LagunaAttentionSpec {
        shape.isSlidingWindowLayer(layer)
            ? slidingWindow(shape: shape)
            : fullAttention(shape: shape, contextSize: contextSize)
    }
}

/// Scalar oracle for the Laguna router.
///
/// Upstream routes Laguna through the SAME primitive as GLM
/// (`ds4_gpu_glm_router_select`): selection by `sigmoid(logit) + bias`, the
/// normalized route weight deliberately uses the unbiased sigmoid value, and
/// exact score ties prefer the lower expert id. Laguna differs from GLM only
/// in the active count (10) — the scale stays 2.5 with normalization on.
public enum LagunaRouterReference {
    public static let expertCount = 256
    public static let expertsUsed = 10
    public static let expertWeightScale: Float = 2.5

    public struct Output: Sendable, Equatable {
        public let selected: [Int32]
        public let weights: [Float]
        public let probabilities: [Float]
    }

    public static func route(logits: [Float], bias: [Float]) throws -> Output {
        guard logits.count == expertCount else {
            throw MetalError.unsupported(
                "Laguna router logits count \(logits.count); expected \(expertCount)"
            )
        }
        guard bias.count == expertCount else {
            throw MetalError.unsupported(
                "Laguna router bias count \(bias.count); expected \(expertCount)"
            )
        }

        let probabilities = logits.map(GLM52RouterReference.sigmoid)
        let selected = (0..<expertCount).sorted { lhs, rhs in
            let a = probabilities[lhs] + bias[lhs]
            let b = probabilities[rhs] + bias[rhs]
            return a == b ? lhs < rhs : a > b
        }.prefix(expertsUsed).map(Int32.init)

        var sum: Float = 0
        for expert in selected { sum += probabilities[Int(expert)] }
        sum = max(sum, 6.103515625e-5)
        let weights = selected.map {
            probabilities[Int($0)] / sum * expertWeightScale
        }
        return Output(selected: selected, weights: weights,
                      probabilities: probabilities)
    }
}

/// F16 ring KV cache mirroring the Metal `key_cache`/`value_cache` layout:
/// `capacity` rows of `kvHeadCount * headDim` half floats, written at
/// `position % capacity`. Values round through F16 exactly like the GPU
/// store, so reference scores see the same quantized keys the kernels see.
public final class LagunaReferenceKVCache {
    public let capacity: Int
    public let rowWidth: Int
    public private(set) var keys: [Float]
    public private(set) var values: [Float]

    public init(capacity: Int, rowWidth: Int) {
        precondition(capacity > 0 && rowWidth > 0)
        self.capacity = capacity
        self.rowWidth = rowWidth
        self.keys = [Float](repeating: 0, count: capacity * rowWidth)
        self.values = [Float](repeating: 0, count: capacity * rowWidth)
    }

    public func store(position: Int, keyRow: [Float], valueRow: [Float]) {
        precondition(keyRow.count == rowWidth && valueRow.count == rowWidth)
        let row = position % capacity
        let base = row * rowWidth
        for i in 0..<rowWidth {
            keys[base + i] = Half.float(Half.bits(keyRow[i]))
            values[base + i] = Half.float(Half.bits(valueRow[i]))
        }
    }
}

public enum LagunaLayerReference {
    // MARK: Scalar primitives

    public static func rmsNorm(_ values: [Float], weight: [Float],
                               epsilon: Float) throws -> [Float] {
        guard values.count == weight.count, !values.isEmpty else {
            throw MetalError.unsupported(
                "Laguna rmsNorm width \(values.count) vs weight \(weight.count)"
            )
        }
        var sum: Float = 0
        for v in values { sum += v * v }
        let inv = 1 / (sum / Float(values.count) + epsilon).squareRoot()
        return (0..<values.count).map { values[$0] * inv * weight[$0] }
    }

    /// Upstream silu via the stable sigmoid the router also uses.
    public static func silu(_ value: Float) -> Float {
        value * GLM52RouterReference.sigmoid(value)
    }

    /// Laguna SwiGLU: `silu(gate) * up`, without a clamp (the graph passes
    /// clamp 0 to the shared swiglu primitive).
    public static func swiGLU(gate: [Float], up: [Float]) throws -> [Float] {
        guard gate.count == up.count else {
            throw MetalError.unsupported(
                "Laguna swiGLU width \(gate.count) vs \(up.count)"
            )
        }
        return (0..<gate.count).map { silu(gate[$0]) * up[$0] }
    }

    /// The per-head attention gate: softplus with the kernel's overflow
    /// guard (`g > 20 ? g : log(1 + exp(g))`).
    public static func softplusGate(_ value: Float) -> Float {
        value > 20 ? value : Foundation.log(1 + Foundation.exp(value))
    }

    /// `weight` holds `outputWidth` rows of `inputWidth` columns (the GGUF
    /// 2D `[in, out]` layout after dequantization).
    public static func matvec(weight: [Float], inputWidth: Int, outputWidth: Int,
                              input: [Float]) throws -> [Float] {
        guard input.count == inputWidth,
              weight.count == inputWidth * outputWidth else {
            throw MetalError.unsupported(
                "Laguna matvec \(weight.count) weights for \(inputWidth)x\(outputWidth)"
            )
        }
        var out = [Float](repeating: 0, count: outputWidth)
        for row in 0..<outputWidth {
            var acc: Float = 0
            let base = row * inputWidth
            for column in 0..<inputWidth {
                acc += weight[base + column] * input[column]
            }
            out[row] = acc
        }
        return out
    }

    // MARK: RoPE (NeoX pairs + YaRN)

    /// `rope_yarn_corr_dims` from the shared Metal helpers.
    public static func ropeYarnCorrectionDims(
        rotationDims: Int, originalContext: Int, frequencyBase: Float,
        betaFast: Float, betaSlow: Float
    ) -> (low: Float, high: Float) {
        func factor(_ rotations: Float) -> Float {
            Float(rotationDims)
                * Foundation.log(Float(originalContext) / (rotations * 2 * Float.pi))
                / (2 * Foundation.log(frequencyBase))
        }
        let low = max(0, factor(betaFast).rounded(.down))
        let high = min(Float(rotationDims - 1), factor(betaSlow).rounded(.up))
        return (low, high)
    }

    /// `rope_yarn`: extrapolation/interpolation mix plus the YaRN magnitude
    /// scale, exactly as in `metal/laguna.metal` (which reuses the shared
    /// DeepSeek helper).
    public static func ropeYarn(
        thetaExtrapolated: Float, frequencyScale: Float,
        correctionDims: (low: Float, high: Float), pairIndex i0: Int,
        extrapolationFactor: Float, magnitudeScale: Float
    ) -> (cos: Float, sin: Float) {
        let thetaInterpolated = frequencyScale * thetaExtrapolated
        var theta = thetaInterpolated
        var mscale = magnitudeScale
        if extrapolationFactor != 0 {
            let y = (Float(i0 / 2) - correctionDims.low)
                / max(0.001, correctionDims.high - correctionDims.low)
            let ramp = (1 - min(1, max(0, y))) * extrapolationFactor
            theta = thetaInterpolated * (1 - ramp) + thetaExtrapolated * ramp
            mscale *= 1 + 0.1 * Foundation.log(1 / frequencyScale)
        }
        return (Foundation.cos(theta) * mscale, Foundation.sin(theta) * mscale)
    }

    /// Per-head RMS norm followed by NeoX rotary pairs on the head prefix
    /// (`laguna_head_rms_norm_rope_neox`): the whole head is normalized with
    /// the shared per-dimension weight, then dimension pairs `(i, i+nRot/2)`
    /// of the first `rotationDims` dimensions rotate; any remainder is left
    /// unrotated.
    public static func headRMSNormRope(
        _ head: [Float], weight: [Float], spec: LagunaAttentionSpec,
        position: Int
    ) throws -> [Float] {
        guard head.count == spec.headDim, weight.count == spec.headDim,
              spec.rotationDims <= spec.headDim,
              spec.rotationDims % 2 == 0 else {
            throw MetalError.unsupported(
                "Laguna head norm/rope: head \(head.count), weight \(weight.count), rot \(spec.rotationDims)"
            )
        }
        var row = try rmsNorm(head, weight: weight, epsilon: spec.rmsEpsilon)

        let halfRotation = spec.rotationDims / 2
        let correction = spec.extrapolationFactor != 0
            ? ropeYarnCorrectionDims(
                rotationDims: spec.rotationDims,
                originalContext: spec.ropeOriginalContext,
                frequencyBase: spec.ropeFrequencyBase,
                betaFast: spec.betaFast,
                betaSlow: spec.betaSlow)
            : (low: Float(0), high: Float(0))
        for pair in 0..<halfRotation {
            let i0 = pair * 2
            let thetaExtrapolated = Float(position)
                * Foundation.pow(spec.ropeFrequencyBase,
                                 -Float(i0) / Float(spec.rotationDims))
            let rotation = ropeYarn(
                thetaExtrapolated: thetaExtrapolated,
                frequencyScale: spec.ropeFrequencyScale,
                correctionDims: correction,
                pairIndex: i0,
                extrapolationFactor: spec.extrapolationFactor,
                magnitudeScale: spec.attentionFactor
            )
            let x0 = row[pair]
            let x1 = row[pair + halfRotation]
            row[pair] = x0 * rotation.cos - x1 * rotation.sin
            row[pair + halfRotation] = x0 * rotation.sin + x1 * rotation.cos
        }
        return row
    }

    /// Apply the per-head norm/rope to a full projection
    /// (`headCount * headDim` values sharing one `headDim` weight vector).
    public static func projectionHeadRMSNormRope(
        _ projection: [Float], weight: [Float], headCount: Int,
        spec: LagunaAttentionSpec, position: Int
    ) throws -> [Float] {
        guard projection.count == headCount * spec.headDim else {
            throw MetalError.unsupported(
                "Laguna projection width \(projection.count); expected \(headCount * spec.headDim)"
            )
        }
        var out = [Float]()
        out.reserveCapacity(projection.count)
        for head in 0..<headCount {
            let base = head * spec.headDim
            out += try headRMSNormRope(
                Array(projection[base..<base + spec.headDim]),
                weight: weight, spec: spec, position: position
            )
        }
        return out
    }

    // MARK: Attention

    /// Gated GQA attention over the ring cache
    /// (`kernel_laguna_attention_decode_gqa_f16`): softmax of
    /// `q·k / sqrt(headDim)` over the last `min(position+1, capacity)`
    /// cached rows, value mix, then the per-head softplus gate.
    /// The query row for `position` must already be stored in the cache.
    public static func attend(
        query: [Float], gate: [Float], cache: LagunaReferenceKVCache,
        position: Int, spec: LagunaAttentionSpec
    ) throws -> [Float] {
        guard query.count == spec.queryWidth,
              gate.count == spec.headCount,
              cache.rowWidth == spec.keyValueWidth,
              cache.capacity == spec.cacheCapacity,
              spec.headCount % spec.kvHeadCount == 0 else {
            throw MetalError.unsupported("Laguna attend: inconsistent shapes")
        }
        let keyCount = min(position + 1, spec.cacheCapacity)
        let keyStart = position + 1 - keyCount
        let headsPerKV = spec.headCount / spec.kvHeadCount
        let scale = 1 / Float(spec.headDim).squareRoot()

        var out = [Float](repeating: 0, count: spec.queryWidth)
        for head in 0..<spec.headCount {
            let kvHead = head / headsPerKV
            let qBase = head * spec.headDim

            var scores = [Float](repeating: 0, count: keyCount)
            for i in 0..<keyCount {
                let cacheRow = (keyStart + i) % spec.cacheCapacity
                let kvBase = cacheRow * cache.rowWidth + kvHead * spec.headDim
                var dot: Float = 0
                for d in 0..<spec.headDim {
                    dot += query[qBase + d] * cache.keys[kvBase + d]
                }
                scores[i] = dot * scale
            }

            let maxScore = scores.max() ?? 0
            var sum: Float = 0
            for i in 0..<keyCount {
                scores[i] = Foundation.exp(scores[i] - maxScore)
                sum += scores[i]
            }
            let invSum = sum > 0 ? 1 / sum : 0

            let gateScale = softplusGate(gate[head])
            for i in 0..<keyCount {
                let probability = scores[i] * invSum
                let cacheRow = (keyStart + i) % spec.cacheCapacity
                let kvBase = cacheRow * cache.rowWidth + kvHead * spec.headDim
                for d in 0..<spec.headDim {
                    out[qBase + d] += probability * cache.values[kvBase + d]
                }
            }
            for d in 0..<spec.headDim {
                out[qBase + d] *= gateScale
            }
        }
        return out
    }

    // MARK: Attention block

    /// Dequantized F32 weights of one Laguna attention block.
    public struct AttentionWeights: Sendable {
        public let attnNorm: [Float]
        public let query: [Float]
        public let key: [Float]
        public let value: [Float]
        public let gate: [Float]
        public let queryNorm: [Float]
        public let keyNorm: [Float]
        public let output: [Float]

        public init(attnNorm: [Float], query: [Float], key: [Float],
                    value: [Float], gate: [Float], queryNorm: [Float],
                    keyNorm: [Float], output: [Float]) {
            self.attnNorm = attnNorm
            self.query = query
            self.key = key
            self.value = value
            self.gate = gate
            self.queryNorm = queryNorm
            self.keyNorm = keyNorm
            self.output = output
        }
    }

    /// One decode step of the attention half of a Laguna block, storing this
    /// position's K/V into the ring cache and returning
    /// `hidden + attn_output(heads)` (the residual add of the graph).
    public static func attentionBlock(
        hidden: [Float], weights: AttentionWeights,
        cache: LagunaReferenceKVCache, position: Int,
        spec: LagunaAttentionSpec, embeddingWidth: Int
    ) throws -> [Float] {
        guard hidden.count == embeddingWidth else {
            throw MetalError.unsupported(
                "Laguna attention block hidden width \(hidden.count)"
            )
        }
        let normed = try rmsNorm(hidden, weight: weights.attnNorm,
                                 epsilon: spec.rmsEpsilon)
        let q = try matvec(weight: weights.query, inputWidth: embeddingWidth,
                           outputWidth: spec.queryWidth, input: normed)
        let k = try matvec(weight: weights.key, inputWidth: embeddingWidth,
                           outputWidth: spec.keyValueWidth, input: normed)
        let v = try matvec(weight: weights.value, inputWidth: embeddingWidth,
                           outputWidth: spec.keyValueWidth, input: normed)
        let gate = try matvec(weight: weights.gate, inputWidth: embeddingWidth,
                              outputWidth: spec.headCount, input: normed)

        let ropedQ = try projectionHeadRMSNormRope(
            q, weight: weights.queryNorm, headCount: spec.headCount,
            spec: spec, position: position
        )
        let ropedK = try projectionHeadRMSNormRope(
            k, weight: weights.keyNorm, headCount: spec.kvHeadCount,
            spec: spec, position: position
        )
        cache.store(position: position, keyRow: ropedK, valueRow: v)

        let heads = try attend(query: ropedQ, gate: gate, cache: cache,
                               position: position, spec: spec)
        let projected = try matvec(weight: weights.output,
                                   inputWidth: spec.queryWidth,
                                   outputWidth: embeddingWidth, input: heads)
        return (0..<embeddingWidth).map { hidden[$0] + projected[$0] }
    }

    // MARK: FFN blocks

    /// Dense SwiGLU FFN of the leading block:
    /// `hidden + down(swiglu(gate·norm, up·norm))`.
    public static func denseFFNBlock(
        hidden: [Float], ffnNorm: [Float], gate: [Float], up: [Float],
        down: [Float], hiddenWidth: Int, intermediateWidth: Int,
        epsilon: Float
    ) throws -> [Float] {
        let normed = try rmsNorm(hidden, weight: ffnNorm, epsilon: epsilon)
        let g = try matvec(weight: gate, inputWidth: hiddenWidth,
                           outputWidth: intermediateWidth, input: normed)
        let u = try matvec(weight: up, inputWidth: hiddenWidth,
                           outputWidth: intermediateWidth, input: normed)
        let mid = try swiGLU(gate: g, up: u)
        let out = try matvec(weight: down, inputWidth: intermediateWidth,
                             outputWidth: hiddenWidth, input: mid)
        return (0..<hiddenWidth).map { hidden[$0] + out[$0] }
    }

    /// Dequantized F32 weights of one routed MoE block (per-expert slabs are
    /// concatenated in expert order, mirroring the GGUF 3D layout).
    public struct MoEWeights: Sendable {
        public let ffnNorm: [Float]
        public let routerWeight: [Float]
        public let routerBias: [Float]
        public let expertGate: [Float]
        public let expertUp: [Float]
        public let expertDown: [Float]
        public let sharedGate: [Float]
        public let sharedUp: [Float]
        public let sharedDown: [Float]

        public init(ffnNorm: [Float], routerWeight: [Float], routerBias: [Float],
                    expertGate: [Float], expertUp: [Float], expertDown: [Float],
                    sharedGate: [Float], sharedUp: [Float], sharedDown: [Float]) {
            self.ffnNorm = ffnNorm
            self.routerWeight = routerWeight
            self.routerBias = routerBias
            self.expertGate = expertGate
            self.expertUp = expertUp
            self.expertDown = expertDown
            self.sharedGate = sharedGate
            self.sharedUp = sharedUp
            self.sharedDown = sharedDown
        }
    }

    /// One decode step of a routed Laguna block:
    /// `hidden + Σ_e w_e · down_e(swiglu_e) + down_s(swiglu_s)`.
    /// Router logits come from the F32 `ffn_gate_inp` matvec on the FFN norm;
    /// selection and weighting follow `LagunaRouterReference`.
    public static func moeFFNBlock(
        hidden: [Float], weights: MoEWeights, hiddenWidth: Int,
        expertWidth: Int, sharedWidth: Int, epsilon: Float
    ) throws -> [Float] {
        let normed = try rmsNorm(hidden, weight: weights.ffnNorm, epsilon: epsilon)
        let logits = try matvec(weight: weights.routerWeight,
                                inputWidth: hiddenWidth,
                                outputWidth: LagunaRouterReference.expertCount,
                                input: normed)
        let route = try LagunaRouterReference.route(logits: logits,
                                                    bias: weights.routerBias)

        var accumulated = [Float](repeating: 0, count: hiddenWidth)
        let gateSlab = hiddenWidth * expertWidth
        let downSlab = expertWidth * hiddenWidth
        for (rank, expert) in route.selected.enumerated() {
            let e = Int(expert)
            let g = try matvec(
                weight: Array(weights.expertGate[e * gateSlab..<(e + 1) * gateSlab]),
                inputWidth: hiddenWidth, outputWidth: expertWidth, input: normed
            )
            let u = try matvec(
                weight: Array(weights.expertUp[e * gateSlab..<(e + 1) * gateSlab]),
                inputWidth: hiddenWidth, outputWidth: expertWidth, input: normed
            )
            let mid = try swiGLU(gate: g, up: u)
            let out = try matvec(
                weight: Array(weights.expertDown[e * downSlab..<(e + 1) * downSlab]),
                inputWidth: expertWidth, outputWidth: hiddenWidth, input: mid
            )
            let routeWeight = route.weights[rank]
            for i in 0..<hiddenWidth { accumulated[i] += routeWeight * out[i] }
        }

        let sharedG = try matvec(weight: weights.sharedGate,
                                 inputWidth: hiddenWidth,
                                 outputWidth: sharedWidth, input: normed)
        let sharedU = try matvec(weight: weights.sharedUp,
                                 inputWidth: hiddenWidth,
                                 outputWidth: sharedWidth, input: normed)
        let sharedMid = try swiGLU(gate: sharedG, up: sharedU)
        let sharedOut = try matvec(weight: weights.sharedDown,
                                   inputWidth: sharedWidth,
                                   outputWidth: hiddenWidth, input: sharedMid)

        return (0..<hiddenWidth).map {
            hidden[$0] + accumulated[$0] + sharedOut[$0]
        }
    }

    /// Final norm + LM head: `output · rmsNorm(hidden, output_norm)`.
    public static func outputHead(
        hidden: [Float], outputNorm: [Float], output: [Float],
        hiddenWidth: Int, vocabularySize: Int, epsilon: Float
    ) throws -> [Float] {
        let normed = try rmsNorm(hidden, weight: outputNorm, epsilon: epsilon)
        return try matvec(weight: output, inputWidth: hiddenWidth,
                          outputWidth: vocabularySize, input: normed)
    }
}
