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

/// Polymorphic interface implemented by both the per-expert
/// `MoEFFN` (custom DeepSeek FP4/FP8 checkpoint with 256 separate
/// Linear instances) and the packed-expert `SwitchMoEFFN` (MLX-native
/// checkpoint with 3 quantized tensors per layer). `Block` holds an
/// `any FFNModule` so the same decoder layer can run either layout.
public protocol FFNModule: AnyObject {
    var layerId: Int { get set }
    var weightLoader: WeightLoader? { get set }
    func callAsFunction(_ x: Tensor, inputIds: [Int32]) -> Tensor
}

public final class MoEFFN: FFNModule {
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

        // Active expert list for this batch.
        var activeExperts: [Int] = []
        for e in 0..<nExperts {
            if countsArr[e] > 0, experts[e] != nil {
                activeExperts.append(e)
            }
        }
        // Chunk size for the expert dispatch loop. In prefill the active
        // set may cover most/all of `nRoutedExperts` (256 on V4-Pro,
        // each ~24 MB of FP4 weights); bulk-loading the lot would pin
        // ~6 GB of routed-expert weights resident for the whole
        // dispatch. Chunking caps the resident routed footprint at
        // `chunkSize × per-expert-size` (32 → ~768 MB) at the cost of
        // re-issuing the disk fetch per chunk. Decode (N==1) bypasses
        // chunking and falls back to per-expert load.
        let chunkSize: Int = {
            if let raw = ProcessInfo.processInfo.environment["DEEPSEEK_MOE_CHUNK_EXPERTS"],
               let n = Int(raw), n > 0 { return n }
            return 32
        }()

        var yFlat = MLXArray.zeros([N, dim]).asType(x.array.dtype)

        var dispatchOffset = 0
        while dispatchOffset < activeExperts.count {
            let dispatchEnd = min(dispatchOffset + chunkSize, activeExperts.count)
            let chunk = Array(activeExperts[dispatchOffset..<dispatchEnd])

            // Pull only this chunk's experts in from disk (parallel
            // load bounded by WeightLoader's load-concurrency cap).
            weightLoader?.ensureExperts(layer: layerId, indices: chunk)

            for e in chunk {
                // We already filtered out nil entries above.
                let expert = experts[e]!

                // Per-token weight for expert e: sum(weights * (indices==e))
                // over topK. The reference's `weights[idx, top, None]`
                // indexing collapses to exactly this when each token
                // routes each expert at most once, which holds for V4.
                let mask = (indices .== e).asType(weights.dtype)   // [N, topK]
                let perTokenW = (weights * mask).sum(axes: [1])    // [N]

                let expertOut = expert(xFlat).array                // [N, dim]
                yFlat = yFlat + expertOut * perTokenW.expandedDimensions(axes: [1])

                // Materialize after every expert to keep the dequant
                // temporary (~30–60 MB at bf16) from accumulating in
                // the lazy graph. Without this MLX could hold all
                // `chunkSize` experts' temps live for the whole
                // dispatch chunk.
                MLX.eval(yFlat)
            }

            // Release this chunk's experts before loading the next.
            // This is what bounds the resident routed-expert
            // footprint per layer to `chunkSize × per-expert-size`.
            weightLoader?.releaseExperts(layer: layerId, indices: chunk)
            MLX.GPU.clearCache()

            dispatchOffset = dispatchEnd
        }

        let sharedOut = sharedExpert(xFlat).array
        yFlat = yFlat + sharedOut

        return Tensor(array: yFlat.reshaped(shape), dtype: x.dtype)
    }
}
