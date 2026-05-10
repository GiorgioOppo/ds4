import Foundation
import Metal

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

/// MoE Gate. Mirrors `Gate` in
/// `Reference/inference/model.py` lines 546–584.
public final class Gate {
    public let topK: Int
    public let nExperts: Int
    public let scoreFunc: ScoreFunc
    public let routeScale: Float
    public let hashRouting: Bool       // first n_hash_layers route by token id
    public let weight: Linear          // gate weight (FP32 in checkpoint)
    public let bias: Tensor?           // [n_experts] f32, nil when hashRouting
    public let tid2eid: Tensor?        // [vocab, top_k] i32, only when hashRouting

    public init(config: ModelConfig, layerId: Int,
                weight: Linear, bias: Tensor?, tid2eid: Tensor?) {
        self.topK = config.nActivatedExperts
        self.nExperts = config.nRoutedExperts
        self.scoreFunc = ScoreFunc(config.scoreFunc)
        self.routeScale = config.routeScale
        self.hashRouting = layerId < config.nHashLayers
        self.weight = weight
        self.bias = bias
        self.tid2eid = tid2eid
    }
}

/// Single MoE expert: SwiGLU FFN (w1 = gate_proj, w3 = up_proj, w2 = down_proj).
public final class Expert {
    public let w1: Linear
    public let w2: Linear
    public let w3: Linear
    public let swigluLimit: Float

    public init(w1: Linear, w2: Linear, w3: Linear, swigluLimit: Float) {
        self.w1 = w1; self.w2 = w2; self.w3 = w3
        self.swigluLimit = swigluLimit
    }
}

/// MoE container. Mirrors `MoE` in model.py:609–644.
///
/// Forward routing logic is left unimplemented because the reference uses
/// `torch.bincount` + `torch.where(indices == i)` to scatter tokens to
/// experts, which on Metal becomes a token-permutation kernel + grouped
/// GEMM dispatch. See README "Roadmap → MoE prefill path".
public final class MoEFFN {
    public let gate: Gate
    public let experts: [Expert?]      // length n_routed_experts; nil when sharded out
    public let sharedExpert: Expert
    public let dim: Int

    public init(config: ModelConfig, gate: Gate, experts: [Expert?], shared: Expert) {
        self.gate = gate
        self.experts = experts
        self.sharedExpert = shared
        self.dim = config.dim
    }

    public func callAsFunction(_ x: Tensor, inputIds: [Int], in cmd: MTLCommandBuffer) -> Tensor {
        fatalError("MoEFFN.forward not implemented — porting target: model.py:629")
    }
}
