import Foundation
import DS4Core

// CPU correctness oracles for the GLM 5.2 feed-forward stages, ports of
// upstream's F32 reference path (layer_glm_dense_ffn_one_f32_ref,
// layer_glm_routed_moe_one_f32_ref, layer_glm_shared_ffn_one_f32_ref,
// output_logits_glm_one_f32_ref): F32 activations against dequantized
// weights, never the requantized-activation fast path. Weights arrive as
// row-major [Float] the caller dequantized with the `Quantize` references —
// quantization error belongs to the fixture, evaluation order to the oracle.
//
// GLM specifics baked in: SwiGLU has NO clamp (swiglu_clamp_exp = 0), the
// router weight (already normalized ×2.5 by GLM52RouterReference) multiplies
// each expert's SwiGLU mid BEFORE the down projection, routed and shared
// outputs add, and RMSNorm accumulates squares in Double like upstream.

public struct GLM52FFNGeometry: Sendable, Equatable {
    /// Model embedding width (matvec input and residual width).
    public let embeddingWidth: Int
    /// Hidden width of the three leading dense FFN blocks.
    public let denseHiddenWidth: Int
    /// Hidden width of every routed expert and of the shared expert.
    public let expertHiddenWidth: Int
    /// Routed experts consumed per token (router top-k).
    public let expertsUsed: Int

    /// GLM 5.2: 6144 wide, dense 12288, experts 2048, top-8.
    public static let v5_2 = GLM52FFNGeometry(
        embeddingWidth: 6_144,
        denseHiddenWidth: 12_288,
        expertHiddenWidth: 2_048,
        expertsUsed: 8
    )

    public init(embeddingWidth: Int,
                denseHiddenWidth: Int,
                expertHiddenWidth: Int,
                expertsUsed: Int) {
        self.embeddingWidth = embeddingWidth
        self.denseHiddenWidth = denseHiddenWidth
        self.expertHiddenWidth = expertHiddenWidth
        self.expertsUsed = expertsUsed
    }

    var isValid: Bool {
        embeddingWidth > 0 && denseHiddenWidth > 0
            && expertHiddenWidth > 0 && expertsUsed > 0
    }
}

/// Dequantized weights of ONE routed expert, row-major matvec layout:
/// gate/up are `[hidden][embedding]`, down is `[embedding][hidden]`.
public struct GLM52ExpertWeights: Sendable, Equatable {
    public let gate: [Float]
    public let up: [Float]
    public let down: [Float]

    public init(gate: [Float], up: [Float], down: [Float]) {
        self.gate = gate
        self.up = up
        self.down = down
    }
}

public enum GLM52FFNReferenceError: Error, Sendable, Equatable,
    CustomStringConvertible {
    case invalidGeometry(GLM52FFNGeometry)
    case invalidDimensions(component: String, expected: Int, got: Int)
    case nonFiniteValue(component: String, index: Int)

    public var description: String {
        switch self {
        case .invalidGeometry(let geometry):
            return "invalid GLM FFN geometry \(geometry)"
        case .invalidDimensions(let component, let expected, let got):
            return "GLM FFN \(component) has \(got) elements, expected \(expected)"
        case .nonFiniteValue(let component, let index):
            return "GLM FFN \(component)[\(index)] is not finite"
        }
    }
}

public enum GLM52FFNCPUReference {
    /// Upstream rms_norm_weight: Double sum of squares, then
    /// `1/sqrt(mean + epsilon)` in Float, elementwise weight.
    public static func rmsNorm(_ values: [Float],
                               weight: [Float],
                               epsilon: Float = 1e-5) throws -> [Float] {
        guard weight.count == values.count, !values.isEmpty else {
            throw GLM52FFNReferenceError.invalidDimensions(
                component: "rmsNorm.weight",
                expected: values.count, got: weight.count)
        }
        try requireFinite(values, component: "rmsNorm.values")
        try requireFinite(weight, component: "rmsNorm.weight")
        var sumSquares = 0.0
        for value in values { sumSquares += Double(value) * Double(value) }
        let scale = 1 / (Float(sumSquares / Double(values.count)) + epsilon)
            .squareRoot()
        return (0..<values.count).map { values[$0] * scale * weight[$0] }
    }

    /// Upstream silu via the stable sigmoid GLM's router also uses.
    public static func silu(_ value: Float) -> Float {
        value * GLM52RouterReference.sigmoid(value)
    }

    /// GLM SwiGLU: `silu(gate) * up`, deliberately without the DeepSeek clamp
    /// (GLM ships swiglu_clamp_exp = 0, so upstream never clips).
    public static func swiGLU(gate: [Float], up: [Float]) throws -> [Float] {
        guard gate.count == up.count, !gate.isEmpty else {
            throw GLM52FFNReferenceError.invalidDimensions(
                component: "swiGLU.up", expected: gate.count, got: up.count)
        }
        try requireFinite(gate, component: "swiGLU.gate")
        try requireFinite(up, component: "swiGLU.up")
        return (0..<gate.count).map { silu(gate[$0]) * up[$0] }
    }

    /// Plain row-major matvec: `rows` is `[rowCount][inputWidth]`.
    public static func matvec(rows: [Float],
                              input: [Float],
                              rowCount: Int) throws -> [Float] {
        guard !input.isEmpty, rowCount > 0,
              rows.count == rowCount * input.count else {
            throw GLM52FFNReferenceError.invalidDimensions(
                component: "matvec.rows",
                expected: rowCount * input.count, got: rows.count)
        }
        try requireFinite(rows, component: "matvec.rows")
        try requireFinite(input, component: "matvec.input")
        let width = input.count
        var output = [Float](repeating: 0, count: rowCount)
        for row in 0..<rowCount {
            var accumulator: Float = 0
            let base = row * width
            for i in 0..<width {
                accumulator += rows[base + i] * input[i]
            }
            output[row] = accumulator
        }
        return output
    }

    /// One dense or shared-expert FFN block:
    /// `down( swiglu(gate·x, up·x) )` with `hiddenWidth` mid rows.
    public static func ffnBlock(input: [Float],
                                gate: [Float],
                                up: [Float],
                                down: [Float],
                                hiddenWidth: Int) throws -> [Float] {
        let mid = try swiGLU(
            gate: matvec(rows: gate, input: input, rowCount: hiddenWidth),
            up: matvec(rows: up, input: input, rowCount: hiddenWidth))
        return try matvec(rows: down, input: mid, rowCount: input.count)
    }

    /// Dense FFN of the three leading blocks (hidden = denseHiddenWidth).
    public static func denseFFN(geometry: GLM52FFNGeometry,
                                input: [Float],
                                gate: [Float],
                                up: [Float],
                                down: [Float]) throws -> [Float] {
        try validate(geometry: geometry, input: input)
        return try ffnBlock(input: input, gate: gate, up: up, down: down,
                            hiddenWidth: geometry.denseHiddenWidth)
    }

    /// Shared expert (hidden = expertHiddenWidth), summed with the routed
    /// output by `sparseFFN`.
    public static func sharedFFN(geometry: GLM52FFNGeometry,
                                 input: [Float],
                                 gate: [Float],
                                 up: [Float],
                                 down: [Float]) throws -> [Float] {
        try validate(geometry: geometry, input: input)
        return try ffnBlock(input: input, gate: gate, up: up, down: down,
                            hiddenWidth: geometry.expertHiddenWidth)
    }

    /// Routed experts: for every selected expert e (router rank order),
    /// `out += down_e( w_e · swiglu(gate_e·x, up_e·x) )` — the router weight
    /// multiplies the SwiGLU mid BEFORE the down projection, upstream's exact
    /// association. `weights` come straight from `GLM52RouterReference.route`
    /// and already carry the ×2.5 normalization.
    public static func routedFFN(geometry: GLM52FFNGeometry,
                                 input: [Float],
                                 experts: [GLM52ExpertWeights],
                                 weights: [Float]) throws -> [Float] {
        try validate(geometry: geometry, input: input)
        guard experts.count == geometry.expertsUsed,
              weights.count == geometry.expertsUsed else {
            throw GLM52FFNReferenceError.invalidDimensions(
                component: "routedFFN.experts",
                expected: geometry.expertsUsed,
                got: experts.count == geometry.expertsUsed
                    ? weights.count : experts.count)
        }
        try requireFinite(weights, component: "routedFFN.weights")
        var output = [Float](repeating: 0, count: input.count)
        for (rank, expert) in experts.enumerated() {
            var mid = try swiGLU(
                gate: matvec(rows: expert.gate, input: input,
                             rowCount: geometry.expertHiddenWidth),
                up: matvec(rows: expert.up, input: input,
                           rowCount: geometry.expertHiddenWidth))
            let weight = weights[rank]
            for i in 0..<mid.count { mid[i] *= weight }
            let contribution = try matvec(rows: expert.down, input: mid,
                                          rowCount: input.count)
            for i in 0..<output.count { output[i] += contribution[i] }
        }
        try requireFinite(output, component: "routedFFN.output")
        return output
    }

    /// Sparse layer FFN: routed experts plus the shared expert, plain sum.
    public static func sparseFFN(geometry: GLM52FFNGeometry,
                                 input: [Float],
                                 experts: [GLM52ExpertWeights],
                                 weights: [Float],
                                 sharedGate: [Float],
                                 sharedUp: [Float],
                                 sharedDown: [Float]) throws -> [Float] {
        let routed = try routedFFN(geometry: geometry, input: input,
                                   experts: experts, weights: weights)
        let shared = try sharedFFN(geometry: geometry, input: input,
                                   gate: sharedGate, up: sharedUp,
                                   down: sharedDown)
        return (0..<routed.count).map { routed[$0] + shared[$0] }
    }

    /// Output head: RMSNorm with output_norm, then the vocab matvec.
    /// No extra scale and no softcap, exactly upstream.
    public static func outputHead(hidden: [Float],
                                  normWeight: [Float],
                                  head: [Float],
                                  vocabularySize: Int,
                                  epsilon: Float = 1e-5) throws -> [Float] {
        let normalized = try rmsNorm(hidden, weight: normWeight,
                                     epsilon: epsilon)
        return try matvec(rows: head, input: normalized,
                          rowCount: vocabularySize)
    }

    // MARK: - Validation

    private static func validate(geometry: GLM52FFNGeometry,
                                 input: [Float]) throws {
        guard geometry.isValid else {
            throw GLM52FFNReferenceError.invalidGeometry(geometry)
        }
        guard input.count == geometry.embeddingWidth else {
            throw GLM52FFNReferenceError.invalidDimensions(
                component: "input",
                expected: geometry.embeddingWidth, got: input.count)
        }
        try requireFinite(input, component: "input")
    }

    private static func requireFinite(_ values: [Float],
                                      component: String) throws {
        for (index, value) in values.enumerated() where !value.isFinite {
            throw GLM52FFNReferenceError.nonFiniteValue(
                component: component, index: index)
        }
    }
}
