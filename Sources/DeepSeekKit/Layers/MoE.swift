import Foundation
import MLX
import MLXNN

public enum ScoreFunc: Int, Sendable {
    case softmax = 0
    case sigmoid = 1
    case sqrtsoftplus = 2

    public init(_ s: String) {
        switch s {
        case "softmax": self = .softmax
        case "sigmoid": self = .sigmoid
        case "sqrtsoftplus": self = .sqrtsoftplus
        default: fatalError("unknown score_func: \(s)")
        }
    }
}

public final class Gate {
    public let topK: Int
    public let nExperts: Int
    public let scoreFunc: ScoreFunc
    public let routeScale: Float
    public let hashRouting: Bool
    public let weight: Linear
    public let bias: Tensor?
    public let tid2eid: Tensor?

    public init(config: ModelConfig, layerId: Int,
                weight: Linear, bias: Tensor?, tid2eid: Tensor?) {
        self.nExperts = config.nRoutedExperts
        self.scoreFunc = ScoreFunc(config.scoreFunc)
        self.routeScale = config.routeScale
        self.hashRouting = layerId < config.nHashLayers
        self.weight = weight
        self.bias = bias
        self.tid2eid = tid2eid
        if self.hashRouting, let tid = tid2eid {
            self.topK = tid.shape[1]
        } else {
            self.topK = config.nActivatedExperts
        }
    }

    public func callAsFunction(_ x: Tensor, inputIds: [Int32]) -> (weights: MLXArray, indices: MLXArray) {
        let xArr = x.array
        let N = xArr.shape[0]

        // Reference: scores = activation(linear(x, weight)); bias shifts the
        // scores used for topk only, then gather() picks original_scores.
        // Renormalization runs only when score_func != "softmax".
        let logits = weight(x).array
        let scores: MLXArray = {
            switch scoreFunc {
            case .softmax: return softmax(logits, axis: 1)
            case .sigmoid: return sigmoid(logits)
            case .sqrtsoftplus:
                let absL = abs(logits)
                let sp = maximum(logits, MLXArray(Float(0))) + log(MLXArray(Float(1)) + exp(-absL))
                return sqrt(sp)
            }
        }()

        if hashRouting {
            guard let tid = tid2eid else { fatalError("hash routing requires tid2eid") }
            let inputIdsArr = MLXArray(inputIds).reshaped([N])
            let indices = tid.array[inputIdsArr]

            var w = takeAlong(scores, indices, axis: 1)
            if scoreFunc != .softmax {
                let sumW = w.sum(axes: [1], keepDims: true)
                w = w / maximum(sumW, MLXArray(Float(1e-12)))
            }
            w = w * routeScale
            return (w, indices)
        }

        // Score-based routing: bias is applied for top-K *selection* only.
        let scoresForTopK = bias.map { scores + $0.array } ?? scores
        let sortedIndices = argSort(scoresForTopK, axis: 1)[0..., (nExperts - topK)..<nExperts]
        let indicesR = sortedIndices[0..., .stride(by: -1)]

        var w = takeAlong(scores, indicesR, axis: 1)
        if scoreFunc != .softmax {
            let sumW = w.sum(axes: [1], keepDims: true)
            w = w / maximum(sumW, MLXArray(Float(1e-12)))
        }
        w = w * routeScale
        return (w, indicesR)
    }
}

public final class Expert {
    public let w1: Linear
    public let w2: Linear
    public let w3: Linear
    public let swigluLimit: Float

    public init(w1: Linear, w2: Linear, w3: Linear, swigluLimit: Float) {
        self.w1 = w1; self.w2 = w2; self.w3 = w3
        self.swigluLimit = swigluLimit
    }

    public func callAsFunction(_ x: Tensor) -> Tensor {
        let g = w1(x).array
        let u = w3(x).array
        let h = (g * sigmoid(g)) * u
        return w2(Tensor(array: h, dtype: x.dtype))
    }
}

public final class MoEFFN {
    public let gate: Gate
    public let experts: [Expert?]
    public let sharedExpert: Expert
    public let dim: Int
    public let nExperts: Int
    public let topK: Int
    public var layerId: Int = -1
    /// Per-token expert streaming: after the gate runs, only the
    /// `topK` active experts are pulled in from disk; the rest stay
    /// off-RAM. Wired by `Assembly.load` for non-eager strategies.
    public weak var weightLoader: WeightLoader? = nil

    public init(config: ModelConfig, gate: Gate, experts: [Expert?], shared: Expert) {
        self.gate = gate
        self.experts = experts
        self.sharedExpert = shared
        self.dim = config.dim
        self.nExperts = config.nRoutedExperts
        self.topK = gate.topK
    }

    public func callAsFunction(_ x: Tensor, inputIds: [Int32]) -> Tensor {
        let shape = x.shape
        let N = shape.dropLast().reduce(1, *)
        let xFlat = Tensor(array: x.array.reshaped([N, dim]), dtype: x.dtype)

        let (weights, indices) = gate(xFlat, inputIds: inputIds)
        // indices: [N, topK] int, weights: [N, topK] float

        // Compute per-expert routed-token counts in one pass, then sync once
        // so we can skip experts with zero tokens. The reference does this
        // via `torch.bincount`. Without the skip we'd run every expert on
        // every token — defeats sparse MoE.
        let allExpertIds = MLXArray((0..<nExperts).map { Int32($0) })
        let idxFlat = indices.reshaped([N * topK])
        let oneHot = (idxFlat.expandedDimensions(axes: [1]) .== allExpertIds.expandedDimensions(axes: [0]))
        let counts = oneHot.asType(.int32).sum(axes: [0])
        MLX.eval(counts)
        let countsArr = counts.asArray(Int32.self)

        // Active expert list for this batch. Per-expert streaming only
        // pays off when the active set is much smaller than
        // `nRoutedExperts`. With N=128 prefill tokens × topK=8 = 1024
        // routings across 256 experts, by pigeonhole nearly every
        // expert gets some traffic — `activeExperts` ≈ nExperts. In
        // that regime "stream only what's active" reduces to "stream
        // all 256 individually", which is the worst of both worlds:
        // full memory footprint AND maximum disk I/O contention. Use
        // the decode path (N==1, topK=8 active) for the win it was
        // designed for, and bulk-load the full routed set on prefill.
        var activeExperts: [Int] = []
        for e in 0..<nExperts {
            if countsArr[e] > 0, experts[e] != nil {
                activeExperts.append(e)
            }
        }
        let bulkLoad = (N > 1)
        let toLoad: [Int] = bulkLoad
            ? (0..<nExperts).filter { experts[$0] != nil }
            : activeExperts
        weightLoader?.ensureExperts(layer: layerId, indices: toLoad)

        var yFlat = MLXArray.zeros([N, dim]).asType(x.array.dtype)

        for e in activeExperts {
            // We already filtered out nil entries above.
            let expert = experts[e]!

            // Per-token weight for expert e: sum(weights * (indices==e)) over topK.
            // The reference's `weights[idx, top, None]` indexing collapses to
            // exactly this when each token routes each expert at most once,
            // which holds for V4 (topk picks distinct expert ids).
            let mask = (indices .== e).asType(weights.dtype)   // [N, topK]
            let perTokenW = (weights * mask).sum(axes: [1])    // [N]

            let expertOut = expert(xFlat).array                // [N, dim]
            yFlat = yFlat + expertOut * perTokenW.expandedDimensions(axes: [1])

            // In the prefill/bulk-load path we materialize after every
            // expert. The lazy graph would otherwise hold the
            // dequantized weight temporary (~30–60 MB at bf16 for a
            // 4096×4096 expert) AND the matmul output for every iter
            // simultaneously — for the full 256-expert layer that's
            // ~15 GB of transient live in addition to the 6 GB of
            // resident expert weights, which is what pushed peak RAM
            // to >25 GB. Decode (per-expert streaming) doesn't need
            // this since `activeExperts.count` is at most `topK`.
            if bulkLoad {
                MLX.eval(yFlat)
                MLX.GPU.clearCache()
            }
        }
        // Materialize before releasing weights — MLX is lazy and the
        // unevaluated expert(xFlat) graph nodes still reference the
        // MLXArrays we're about to drop.
        MLX.eval(yFlat)
        weightLoader?.releaseExperts(layer: layerId, indices: toLoad)
        // Drain MLX's buffer pool: the dequantized weight temporaries
        // for each of `topK` experts (~30–60 MB at bf16 for a 4096×4096
        // expert linear) are otherwise retained for reuse, and on a
        // tight-RAM run the high-water-mark stays elevated even
        // though our cache says the experts are gone.
        MLX.GPU.clearCache()

        let sharedOut = sharedExpert(xFlat).array
        yFlat = yFlat + sharedOut

        return Tensor(array: yFlat.reshaped(shape), dtype: x.dtype)
    }
}
