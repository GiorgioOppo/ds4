import Foundation
import Metal

extension GraphContext {
    /// Output head: logits = matmul_f32(outWeight, rmsNorm(hidden [* normWeight])).
    /// hidden: inDim, outWeight: vocab x inDim (F32 row-major), logits: vocab.
    public func outputHead(hidden: GPUTensor, normWeight: GPUTensor?, outWeight: GPUTensor,
                           normed: GPUTensor, logits: GPUTensor,
                           inDim: Int, vocab: Int, eps: Float) throws {
        try rmsNorm(hidden, weight: normWeight, out: normed, rows: 1, n: inDim, eps: eps)
        try matmulF32(weight: outWeight, x: normed, out: logits, inDim: inDim, outDim: vocab)
    }

    /// Token embedding to HC block: gather row `token` from an F16 table
    /// (vocab x nEmbd) into `embd`, then replicate across `nHC` HC streams into
    /// `hc` (nHC x nEmbd). Encodes get_rows_f16 + repeat_f32.
    public func embedTokenHC(table: GPUTensor, token: Int, embd: GPUTensor, hc: GPUTensor,
                             nEmbd: Int, nVocab: Int, nHC: Int) throws {
        try getRowsF16(table: table, id: token, out: embd, nEmbd: nEmbd, nVocab: nVocab)
        try repeatHC(src: embd, out: hc, nEmbd: nEmbd, nTokens: 1, nHC: nHC)
    }

    /// Shared-expert FFN block (pre-norm + SwiGLU MLP + residual), one token:
    ///   normed = rmsNorm(x [* normWeight])
    ///   mid    = swiglu(matmulQ8(gateW, normed), matmulQ8(upW, normed))
    ///   out    = x + matmulQ8(downW, mid)
    /// gateW/upW: ffnDim x inDim (Q8_0); downW: inDim x ffnDim (Q8_0).
    /// Scratch tensors (normed[inDim], gate[ffnDim], up[ffnDim], mid[ffnDim],
    /// down[inDim]) are caller-provided so the graph can reuse them across layers.
    public func ffnBlock(x: GPUTensor, normWeight: GPUTensor?,
                         gateW: GPUTensor, upW: GPUTensor, downW: GPUTensor,
                         normed: GPUTensor, gate: GPUTensor, up: GPUTensor, mid: GPUTensor,
                         down: GPUTensor, out: GPUTensor,
                         inDim: Int, ffnDim: Int, eps: Float) throws {
        try rmsNorm(x, weight: normWeight, out: normed, rows: 1, n: inDim, eps: eps)
        try matmulQ8_0(weight: gateW, x: normed, out: gate, inDim: inDim, outDim: ffnDim)
        try matmulQ8_0(weight: upW, x: normed, out: up, inDim: inDim, outDim: ffnDim)
        try swiglu(gate: gate, up: up, out: mid, n: ffnDim)
        try matmulQ8_0(weight: downW, x: mid, out: down, inDim: ffnDim, outDim: inDim)
        try add(x, down, out: out, width: inDim)
    }
}
