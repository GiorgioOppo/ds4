import Foundation
import DS4Core

// CPU oracle of one full GLM 5.2 layer and of the first-token forward chain,
// the port of upstream's F32 reference path (layer_glm_first_token_one_f32_ref
// and forward_glm_first_token_cpu_f32_ref). Single token at position 0, no
// cache: attention over one row is softmax of one score, so the Q path
// vanishes and the layer reduces to the value/output projections — exactly
// upstream's first-token shortcut. This is the independent baseline the
// roadmap's tensor-by-tensor comparison (step 4) judges the GPU graph
// against; it is deliberately not a decode path.

public struct GLM52LayerGeometry: Sendable, Equatable {
    public let embeddingWidth: Int
    public let headCount: Int
    public let kvLoraRank: Int
    public let ropeDimension: Int
    public let valueDimension: Int
    public let denseHiddenWidth: Int
    public let expertHiddenWidth: Int
    public let expertsUsed: Int

    /// attn_kv_a_mqa output width: KV-LoRA prefix plus the K-RoPE tail.
    public var kvRawWidth: Int { kvLoraRank + ropeDimension }
    public var ffnGeometry: GLM52FFNGeometry {
        GLM52FFNGeometry(embeddingWidth: embeddingWidth,
                         denseHiddenWidth: denseHiddenWidth,
                         expertHiddenWidth: expertHiddenWidth,
                         expertsUsed: expertsUsed)
    }

    /// GLM 5.2: 6144 wide, 64 heads, KV-LoRA 512 + RoPE 64, value 256,
    /// dense 12288, experts 2048, top-8.
    public static let v5_2 = GLM52LayerGeometry(
        embeddingWidth: 6_144, headCount: 64, kvLoraRank: 512,
        ropeDimension: 64, valueDimension: 256, denseHiddenWidth: 12_288,
        expertHiddenWidth: 2_048, expertsUsed: 8)

    public init(embeddingWidth: Int, headCount: Int, kvLoraRank: Int,
                ropeDimension: Int, valueDimension: Int,
                denseHiddenWidth: Int, expertHiddenWidth: Int,
                expertsUsed: Int) {
        self.embeddingWidth = embeddingWidth
        self.headCount = headCount
        self.kvLoraRank = kvLoraRank
        self.ropeDimension = ropeDimension
        self.valueDimension = valueDimension
        self.denseHiddenWidth = denseHiddenWidth
        self.expertHiddenWidth = expertHiddenWidth
        self.expertsUsed = expertsUsed
    }
}

/// Dequantized attention weights of one layer, row-major matvec layouts:
/// `kvA` is `[kvRawWidth][embedding]`, `valueB` is `[head*value][kvLora]`
/// (attn_v_b flattened), `attnOutput` is `[embedding][head*value]`.
public struct GLM52LayerAttentionWeightsF32: Sendable {
    public let attnNorm: [Float]
    public let kvA: [Float]
    public let kvANorm: [Float]
    public let valueB: [Float]
    public let attnOutput: [Float]

    public init(attnNorm: [Float], kvA: [Float], kvANorm: [Float],
                valueB: [Float], attnOutput: [Float]) {
        self.attnNorm = attnNorm
        self.kvA = kvA
        self.kvANorm = kvANorm
        self.valueB = valueB
        self.attnOutput = attnOutput
    }
}

/// One layer's FFN: the three leading blocks are dense; every later block is
/// sparse. The sparse case fetches ONLY the router-selected experts through
/// the provider closure — mirroring the streaming reality where unselected
/// experts are never read.
public enum GLM52LayerFFNWeightsF32 {
    case dense(gate: [Float], up: [Float], down: [Float])
    case sparse(routerRows: [Float],
                routerBias: [Float],
                shared: GLM52ExpertWeights,
                expertProvider: (UInt32) throws -> GLM52ExpertWeights)
}

public enum GLM52LayerCPUReference {
    /// Upstream first-token attention shortcut (pos 0, one visible row):
    /// softmax over one score is 1, so attention output is the token's own
    /// value projection — the Q path is never evaluated.
    public static func firstTokenAttention(
        geometry: GLM52LayerGeometry,
        input: [Float],
        weights: GLM52LayerAttentionWeightsF32) throws -> [Float] {
        try requireCount(input, geometry.embeddingWidth, "input")
        try requireCount(weights.attnNorm, geometry.embeddingWidth,
                         "attention.attnNorm")
        try requireCount(weights.kvA,
                         geometry.kvRawWidth * geometry.embeddingWidth,
                         "attention.kvA")
        try requireCount(weights.kvANorm, geometry.kvLoraRank,
                         "attention.kvANorm")
        let headsWidth = geometry.headCount * geometry.valueDimension
        try requireCount(weights.valueB, headsWidth * geometry.kvLoraRank,
                         "attention.valueB")
        try requireCount(weights.attnOutput,
                         geometry.embeddingWidth * headsWidth,
                         "attention.attnOutput")

        let normed = try GLM52FFNCPUReference.rmsNorm(
            input, weight: weights.attnNorm)
        let kvRaw = try GLM52FFNCPUReference.matvec(
            rows: weights.kvA, input: normed, rowCount: geometry.kvRawWidth)
        let kvNorm = try GLM52FFNCPUReference.rmsNorm(
            Array(kvRaw[0..<geometry.kvLoraRank]), weight: weights.kvANorm)
        let heads = try GLM52FFNCPUReference.matvec(
            rows: weights.valueB, input: kvNorm, rowCount: headsWidth)
        return try GLM52FFNCPUReference.matvec(
            rows: weights.attnOutput, input: heads,
            rowCount: geometry.embeddingWidth)
    }

    /// One full pre-norm layer:
    /// `afterAttn = x + attn(x)`, `out = afterAttn + ffn(rmsNorm(afterAttn))`.
    /// Returns the router output for sparse layers (nil on dense) so callers
    /// can assert selections without recomputing them.
    public static func firstTokenLayer(
        geometry: GLM52LayerGeometry,
        input: [Float],
        attention: GLM52LayerAttentionWeightsF32,
        ffnNorm: [Float],
        ffn: GLM52LayerFFNWeightsF32) throws
        -> (output: [Float], routing: GLM52RouterOutput?) {
        let attnOut = try firstTokenAttention(
            geometry: geometry, input: input, weights: attention)
        let afterAttn = (0..<input.count).map { input[$0] + attnOut[$0] }
        return try ffnStage(geometry: geometry, afterAttention: afterAttn,
                            ffnNorm: ffnNorm, ffn: ffn)
    }

    /// The residual FFN half of a layer, shared by the first-token and decode
    /// paths: `out = afterAttention + ffn(rmsNorm(afterAttention))`.
    static func ffnStage(
        geometry: GLM52LayerGeometry,
        afterAttention: [Float],
        ffnNorm: [Float],
        ffn: GLM52LayerFFNWeightsF32) throws
        -> (output: [Float], routing: GLM52RouterOutput?) {
        let ffnIn = try GLM52FFNCPUReference.rmsNorm(
            afterAttention, weight: ffnNorm)

        let ffnOut: [Float]
        var routing: GLM52RouterOutput?
        switch ffn {
        case .dense(let gate, let up, let down):
            ffnOut = try GLM52FFNCPUReference.denseFFN(
                geometry: geometry.ffnGeometry, input: ffnIn,
                gate: gate, up: up, down: down)
        case .sparse(let routerRows, let routerBias, let shared,
                     let expertProvider):
            let logits = try GLM52FFNCPUReference.matvec(
                rows: routerRows, input: ffnIn,
                rowCount: GLM52RouterReference.expertCount)
            let routed = try GLM52RouterReference.route(
                logits: logits, bias: routerBias)
            routing = routed
            let experts = try routed.selected.map {
                try expertProvider(UInt32(bitPattern: $0))
            }
            ffnOut = try GLM52FFNCPUReference.sparseFFN(
                geometry: geometry.ffnGeometry, input: ffnIn,
                experts: experts, weights: routed.weights,
                sharedGate: shared.gate, sharedUp: shared.up,
                sharedDown: shared.down)
        }
        return ((0..<afterAttention.count).map { afterAttention[$0] + ffnOut[$0] },
                routing)
    }

    /// The first-token forward chain: the embedded token through every
    /// autoregressive layer in order. The output head (RMSNorm + vocab
    /// matvec) stays a separate call — `GLM52FFNCPUReference.outputHead`.
    public static func firstTokenForward(
        geometry: GLM52LayerGeometry,
        embeddedToken: [Float],
        layers: [(attention: GLM52LayerAttentionWeightsF32,
                  ffnNorm: [Float],
                  ffn: GLM52LayerFFNWeightsF32)]) throws -> [Float] {
        var hidden = embeddedToken
        for layer in layers {
            hidden = try firstTokenLayer(
                geometry: geometry, input: hidden,
                attention: layer.attention, ffnNorm: layer.ffnNorm,
                ffn: layer.ffn).output
        }
        return hidden
    }

    private static func requireCount(_ values: [Float], _ expected: Int,
                                     _ component: String) throws {
        guard values.count == expected else {
            throw GLM52FFNReferenceError.invalidDimensions(
                component: component, expected: expected, got: values.count)
        }
    }
}
