import Foundation
import MLX
import MLXNN
import MLXFast

/// Packed quantized projection used by `SwitchMoEFFN`. In the
/// mlx-community checkpoint each layer's routed-expert FFN ships its
/// three projections (`gate_proj`, `up_proj`, `down_proj`) as a
/// single quantized tensor per projection of shape
/// `[nExperts, out, in]` (with bits-4 packing along the last axis and
/// matching `scales` / `biases` tensors).
///
/// At dispatch time, the SwitchMoEFFN forward gathers per-token slices
/// of the packed tensor via `MLXFast.gatherQuantizedMatmul`, so the
/// kernel sees `topK × N` matmuls fused into a single launch.
public final class SwitchProj {
    public let nExperts: Int
    public let inFeatures: Int
    public let outFeatures: Int
    public let groupSize: Int
    public let bits: Int
    public let mode: String

    public let weightName: String   // e.g. "layers.0.ffn.switch_mlp.gate_proj.weight"
    public let scalesName: String
    public let biasesName: String
    public weak var loader: WeightLoader?

    public init(nExperts: Int,
                inFeatures: Int,
                outFeatures: Int,
                base: String,
                groupSize: Int,
                bits: Int,
                mode: String,
                loader: WeightLoader) {
        self.nExperts = nExperts
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.weightName = "\(base).weight"
        self.scalesName = "\(base).scales"
        self.biasesName = "\(base).biases"
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.loader = loader
    }

    /// Returns the (packed weight, scales, biases) triplet from the
    /// loader cache (or via on-demand disk read if not cached). All
    /// three must resolve or this returns nil — the caller falls
    /// back to a no-op contribution for this layer.
    public func getPackedTriple() -> (w: MLXArray,
                                       scales: MLXArray,
                                       biases: MLXArray)? {
        guard let loader = loader else { return nil }
        guard let qW = (try? loader.load(weightName))?.array,
              let qS = (try? loader.load(scalesName))?.array,
              let qB = (try? loader.load(biasesName))?.array
        else { return nil }
        return (qW, qS, qB)
    }
}

/// Packed-expert MoE FFN for the MLX-native (mlx-community)
/// DeepSeek-V4 checkpoint. Counterpart to `MoEFFN`, which handles the
/// per-expert layout of the original DeepSeek FP4/FP8 checkpoint.
///
/// Forward (matching the reference MoE math):
///   weights, indices = gate(x)             // [N, topK]
///   gate_out  = gather_qmm(x, gateW, ...)  // [N*topK, inter]
///   up_out    = gather_qmm(x, upW, ...)    // [N*topK, inter]
///   h         = silu(gate_out) * up_out
///   down_out  = gather_qmm(h, downW, ...)  // [N*topK, dim]
///   y         = sum(weights[:,:,None] * down_out.reshape([N, topK, dim]), axis=1)
///   y         = y + shared_expert(x)
///
/// The dispatch uses one `MLXFast.gatherQuantizedMatmul` call per
/// projection (3 per layer instead of 256×3 separate matmuls), with
/// `lhsIndices` selecting the source token row and `rhsIndices`
/// selecting the destination expert slice in the packed weight.
public final class SwitchMoEFFN: FFNModule {
    public let gate: Gate
    public let gateProj: SwitchProj   // gate_proj (SwiGLU "gate", == w1)
    public let upProj: SwitchProj     // up_proj   (SwiGLU "up",   == w3)
    public let downProj: SwitchProj   // down_proj (output,        == w2)
    public let sharedExpert: Expert
    public let dim: Int
    public let nExperts: Int
    public let topK: Int
    public let swigluLimit: Float
    public var layerId: Int = -1
    public weak var weightLoader: WeightLoader? = nil

    public init(config: ModelConfig,
                gate: Gate,
                gateProj: SwitchProj,
                upProj: SwitchProj,
                downProj: SwitchProj,
                sharedExpert: Expert) {
        self.gate = gate
        self.gateProj = gateProj
        self.upProj = upProj
        self.downProj = downProj
        self.sharedExpert = sharedExpert
        self.dim = config.dim
        self.nExperts = config.nRoutedExperts
        self.topK = gate.topK
        self.swigluLimit = config.swigluLimit
    }

    public func callAsFunction(_ x: Tensor, inputIds: [Int32]) -> Tensor {
        let shape = x.shape
        let N = shape.dropLast().reduce(1, *)
        let xFlat = Tensor(array: x.array.reshaped([N, dim]), dtype: x.dtype)

        // Gate: weights [N, topK], indices [N, topK] (int32).
        let (weights, indices) = gate(xFlat, inputIds: inputIds)
        let K = self.topK

        // Pull packed triplets for the three projections. Missing any
        // of them → fall back to shared-expert only (better than NaN).
        guard let gateT = gateProj.getPackedTriple(),
              let upT   = upProj.getPackedTriple(),
              let downT = downProj.getPackedTriple()
        else {
            FileHandle.standardError.write(Data(
                "[SwitchMoE layer=\(layerId)] missing packed weight; "
                + "routed-expert contribution skipped this step.\n".utf8))
            let sharedOnly = sharedExpert(xFlat).array
            return Tensor(array: sharedOnly.reshaped(shape), dtype: x.dtype)
        }

        // lhs_indices: for output position i ∈ [0, N*K), which row of x?
        //   i / K → token row (each token contributes K outputs).
        // rhs_indices: which expert slice of the packed weight?
        //   indices.flatten()[i] → expert id for this output position.
        let lhsIdxs = MLXArray((0..<N).flatMap {
            Array(repeating: Int32($0), count: K)
        })
        let rhsIdxs = indices.reshaped([N * K]).asType(.int32)

        // Activations in bf16 to match the quantized GEMM kernel.
        let xBf16 = xFlat.array.dtype == .bfloat16
            ? xFlat.array
            : xFlat.array.asType(.bfloat16)

        // gate_proj  · x  →  [N*K, inter_dim]
        let yGate = MLXFast.gatherQuantizedMatmul(
            xBf16, gateT.w,
            scales: gateT.scales, biases: gateT.biases,
            lhsIndices: lhsIdxs, rhsIndices: rhsIdxs,
            transpose: true,
            groupSize: gateProj.groupSize, bits: gateProj.bits)

        // up_proj    · x  →  [N*K, inter_dim]
        let yUp = MLXFast.gatherQuantizedMatmul(
            xBf16, upT.w,
            scales: upT.scales, biases: upT.biases,
            lhsIndices: lhsIdxs, rhsIndices: rhsIdxs,
            transpose: true,
            groupSize: upProj.groupSize, bits: upProj.bits)

        // SwiGLU: silu(gate) * up. Optional clamp to swigluLimit per
        // the reference (Expert.callAsFunction reference logic) —
        // present in the trained model for stability.
        var hMid = (yGate * sigmoid(yGate)) * yUp
        if swigluLimit > 0 {
            let lim = MLXArray(Float(swigluLimit)).asType(.bfloat16)
            let nlim = MLXArray(Float(-swigluLimit)).asType(.bfloat16)
            hMid = MLX.minimum(MLX.maximum(hMid, nlim), lim)
        }

        // down_proj  · h  →  [N*K, dim]
        // h is already per-output row (no further lhs_indices gather);
        // we still need rhs_indices to pick the right expert slice.
        let lhsIdxsDown = MLXArray((0..<(N * K)).map { Int32($0) })
        let yDown = MLXFast.gatherQuantizedMatmul(
            hMid, downT.w,
            scales: downT.scales, biases: downT.biases,
            lhsIndices: lhsIdxsDown, rhsIndices: rhsIdxs,
            transpose: true,
            groupSize: downProj.groupSize, bits: downProj.bits)

        // Per-token weighted sum across topK.
        let yDownReshaped = yDown.reshaped([N, K, dim])
        let wExpanded = weights.asType(.bfloat16)
            .expandedDimensions(axes: [2])             // [N, K, 1]
        let weightedDown = (yDownReshaped * wExpanded).sum(axes: [1])
        // weightedDown: [N, dim] bf16

        // Materialize the routed contribution before doing the shared
        // expert, so MLX can free the gather intermediates and the
        // memory footprint stays bounded.
        MLX.eval(weightedDown)
        MLX.GPU.clearCache()

        // Shared expert runs in the same path as MoEFFN (its Linears
        // auto-detect MLX-native triplet via Linear.hasMLXNativeTriplet
        // and route through MLXFast.quantizedMatmul).
        let sharedOut = sharedExpert(xFlat).array

        // Combine. Cast routed to the shared dtype (typically f32) so
        // the add doesn't promote/demote unexpectedly.
        let routedF = weightedDown.dtype == sharedOut.dtype
            ? weightedDown
            : weightedDown.asType(sharedOut.dtype)
        let yFinal = routedF + sharedOut

        return Tensor(array: yFinal.reshaped(shape), dtype: x.dtype)
    }
}
