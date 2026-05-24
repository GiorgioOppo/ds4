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

        if hashRouting {
            guard let tid = tid2eid else { fatalError("hash routing requires tid2eid") }
            let inputIdsArr = MLXArray(inputIds).reshaped([N])
            let indices = tid.array[inputIdsArr]
            
            let logits = weight(x).array
            let gatheredLogits = takeAlong(logits, indices, axis: 1)
            
            var w = gatheredLogits
            if scoreFunc == .sqrtsoftplus {
                let absW = abs(w)
                let sp = maximum(w, MLXArray(0.0)) + log(MLXArray(1.0) + exp(-absW))
                w = sqrt(sp)
            } else if scoreFunc == .sigmoid {
                w = sigmoid(w)
            }
            
            let sumW = w.sum(axes: [1], keepDims: true)
            w = (w / maximum(sumW, 1e-12)) * routeScale
            
            return (w, indices)
        }

        let logits = weight(x).array
        let sortedIndices = argSort(logits, axis: 1)[0..., (nExperts - topK)..<nExperts]
        let indicesR = sortedIndices[0..., .stride(by: -1)]
        
        let valuesR = takeAlong(logits, indicesR, axis: 1)
        
        var w = valuesR
        if scoreFunc == .softmax {
            w = softmax(w, axis: 1)
        } else if scoreFunc == .sigmoid {
            w = sigmoid(w)
        } else if scoreFunc == .sqrtsoftplus {
            let absW = abs(w)
            let sp = maximum(w, MLXArray(0.0)) + log(MLXArray(1.0) + exp(-absW))
            w = sqrt(sp)
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

        var yFlat = MLXArray.zeros([N, dim]).asType(x.array.dtype)
        
        for e in 0..<nExperts {
            guard let expert = experts[e] else { continue }
            let mask = (indices .== e)
            let tokenHasExpert = mask.any(axes: [1])
            
            let wIndices = argMax(mask.asType(.int32), axis: 1)
            let w = takeAlong(weights, wIndices.expandedDimensions(axes: [1]), axis: 1).reshaped([N])
            
            let expertOut = expert(xFlat).array
            let scaledOut = expertOut * w.expandedDimensions(axes: [1])
            let contribution = MLX.where(tokenHasExpert.expandedDimensions(axes: [1]), scaledOut, MLXArray.zeros(like: scaledOut))
            
            yFlat = yFlat + contribution
            
            if e % 8 == 7 {
                MLX.eval(yFlat)
            }
        }
        MLX.eval(yFlat)

        let sharedOut = sharedExpert(xFlat).array
        yFlat = yFlat + sharedOut

        return Tensor(array: yFlat.reshaped(shape), dtype: x.dtype)
    }
}
