import DS4Core
import Foundation
import Metal

// First GPU composition of a GLM 5.2 layer: the first-token forward with
// EVERY quantized matvec dispatched through the validated kernels, judged
// against GLM52LayerCPUReference. Deliberate validation-level split: the
// heavy Q8_0/K-quant matvecs (attention projections, dense/shared/routed FFN)
// run on GPU; the cheap glue — RMSNorm over one embedding row, residual adds,
// the tiny F32 router matvec (256 rows, the GGUF stores ffn_gate_inp in F32)
// — stays on the CPU oracle implementations, so every difference from the
// oracle is attributable to a GPU kernel. The persistent-GPUTensor graph that
// keeps activations resident comes later; this is its correctness baseline,
// not a decode loop.

/// One layer's attention weights as the GGUF stores them: Q8_0 projections,
/// F32 norms. Row layouts match `GLM52LayerAttentionWeightsF32`.
public struct GLM52QuantizedLayerAttention: Sendable {
    public let attnNorm: [Float]
    public let kvA: [UInt8]
    public let kvANorm: [Float]
    public let valueB: [UInt8]
    public let attnOutput: [UInt8]

    public init(attnNorm: [Float], kvA: [UInt8], kvANorm: [Float],
                valueB: [UInt8], attnOutput: [UInt8]) {
        self.attnNorm = attnNorm
        self.kvA = kvA
        self.kvANorm = kvANorm
        self.valueB = valueB
        self.attnOutput = attnOutput
    }
}

/// One layer's FFN weights: Q8_0 for dense and shared, K-quant expert records
/// through the provider (slot cache, payload reader, or fixture bytes).
public enum GLM52QuantizedLayerFFN {
    case dense(gate: [UInt8], up: [UInt8], down: [UInt8])
    case sparse(routerRows: [Float],
                routerBias: [Float],
                sharedGate: [UInt8], sharedUp: [UInt8], sharedDown: [UInt8],
                expertProvider: (UInt32) throws -> GLM52QuantizedExpert)
}

extension MetalRuntime {
    /// First-token layer forward (position 0, upstream's shortcut: attention
    /// over one visible row is the token's own value projection). Returns the
    /// router output on sparse layers so callers can assert selections.
    public func glm52FirstTokenLayer(
        geometry: GLM52LayerGeometry,
        input: [Float],
        attention: GLM52QuantizedLayerAttention,
        ffnNorm: [Float],
        ffn: GLM52QuantizedLayerFFN) throws
        -> (output: [Float], routing: GLM52RouterOutput?) {
        let g = geometry
        guard input.count == g.embeddingWidth else {
            throw MetalError.unsupported(
                "GLM 5.2 layer input must be \(g.embeddingWidth) wide")
        }
        let q8 = GLM52TensorSchema.q8_0
        let headsWidth = g.headCount * g.valueDimension

        // Attention: CPU norms, GPU Q8_0 projections.
        let normed = try GLM52FFNCPUReference.rmsNorm(
            input, weight: attention.attnNorm)
        let kvRaw = try glm52MoEDown(
            mid: normed, downRows: attention.kvA,
            weightType: q8, outputWidth: g.kvRawWidth)
        let kvNorm = try GLM52FFNCPUReference.rmsNorm(
            Array(kvRaw[0..<g.kvLoraRank]), weight: attention.kvANorm)
        let heads = try glm52MoEDown(
            mid: kvNorm, downRows: attention.valueB,
            weightType: q8, outputWidth: headsWidth)
        let attnOut = try glm52MoEDown(
            mid: heads, downRows: attention.attnOutput,
            weightType: q8, outputWidth: g.embeddingWidth)
        let afterAttn = (0..<input.count).map { input[$0] + attnOut[$0] }
        return try glm52LayerFFNStage(
            geometry: g, afterAttention: afterAttn, ffnNorm: ffnNorm, ffn: ffn)
    }

    /// The residual FFN half of a layer on GPU quantized blocks (CPU norm and
    /// router glue), shared by the first-token and decode compositions.
    func glm52LayerFFNStage(
        geometry: GLM52LayerGeometry,
        afterAttention: [Float],
        ffnNorm: [Float],
        ffn: GLM52QuantizedLayerFFN) throws
        -> (output: [Float], routing: GLM52RouterOutput?) {
        let g = geometry
        let q8 = GLM52TensorSchema.q8_0
        let ffnIn = try GLM52FFNCPUReference.rmsNorm(
            afterAttention, weight: ffnNorm)
        let ffnOut: [Float]
        var routing: GLM52RouterOutput?
        switch ffn {
        case .dense(let gate, let up, let down):
            ffnOut = try glm52FFNBlock(
                input: ffnIn, gateRows: gate, upRows: up, downRows: down,
                weightType: q8, hiddenWidth: g.denseHiddenWidth)
        case .sparse(let routerRows, let routerBias,
                     let sharedGate, let sharedUp, let sharedDown,
                     let expertProvider):
            let logits = try GLM52FFNCPUReference.matvec(
                rows: routerRows, input: ffnIn,
                rowCount: GLM52RouterReference.expertCount)
            let routed = try glm52Route(logits: logits, bias: routerBias)
            routing = routed
            let experts = try routed.selected.map {
                try expertProvider(UInt32(bitPattern: $0))
            }
            let routedOut = try glm52RoutedFFN(
                input: ffnIn, experts: experts, weights: routed.weights,
                hiddenWidth: g.expertHiddenWidth)
            let sharedOut = try glm52FFNBlock(
                input: ffnIn, gateRows: sharedGate, upRows: sharedUp,
                downRows: sharedDown, weightType: q8,
                hiddenWidth: g.expertHiddenWidth)
            ffnOut = (0..<routedOut.count).map { routedOut[$0] + sharedOut[$0] }
        }
        return ((0..<afterAttention.count).map { afterAttention[$0] + ffnOut[$0] },
                routing)
    }

    /// The first-token forward chain on GPU matvecs: embedded token through
    /// every layer in order, then the output head (CPU RMSNorm + GPU vocab
    /// matvec). The judge is `GLM52LayerCPUReference.firstTokenForward` plus
    /// `GLM52FFNCPUReference.outputHead` on the dequantized weights.
    public func glm52FirstTokenForward(
        geometry: GLM52LayerGeometry,
        embeddedToken: [Float],
        layers: [(attention: GLM52QuantizedLayerAttention,
                  ffnNorm: [Float],
                  ffn: GLM52QuantizedLayerFFN)],
        outputNorm: [Float],
        outputHead: [UInt8],
        vocabularySize: Int) throws -> [Float] {
        var hidden = embeddedToken
        for layer in layers {
            hidden = try glm52FirstTokenLayer(
                geometry: geometry, input: hidden,
                attention: layer.attention, ffnNorm: layer.ffnNorm,
                ffn: layer.ffn).output
        }
        let normalized = try GLM52FFNCPUReference.rmsNorm(
            hidden, weight: outputNorm)
        return try glm52OutputHeadLogits(
            normalized: normalized, headRows: outputHead,
            weightType: GLM52TensorSchema.q8_0,
            vocabularySize: vocabularySize)
    }
}
