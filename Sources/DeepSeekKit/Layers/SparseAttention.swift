import Foundation
import Metal

/// Sparse multi-head attention with FlashAttention-style online softmax.
/// Mirrors `sparse_attn` in
/// `Reference/inference/kernel.py` lines 355–368.
public enum SparseAttention {
    private static let pipeline = Device.shared.makePipeline("sparse_attn_f32")

    /// `q`: [B, M, H, D] f32. `kv`: [B, N, D] f32. `sink`: [H] f32.
    /// `topkIdxs`: [B, M, K] i32 (use -1 for padding entries).
    /// Returns `o`: [B, M, H, D] f32.
    public static func apply(q: Tensor, kv: Tensor, sink: Tensor,
                             topkIdxs: Tensor, scale: Float,
                             in cmd: MTLCommandBuffer) -> Tensor {
        precondition(q.dtype == .f32 && q.shape.count == 4)
        precondition(kv.dtype == .f32 && kv.shape.count == 3)
        precondition(sink.dtype == .f32 && sink.shape.count == 1)
        precondition(topkIdxs.dtype == .i32 && topkIdxs.shape.count == 3)

        let B = q.shape[0], M = q.shape[1], H = q.shape[2], D = q.shape[3]
        let N = kv.shape[1]
        let K = topkIdxs.shape[2]
        precondition(kv.shape[0] == B && kv.shape[2] == D)
        precondition(sink.shape[0] == H)
        precondition(topkIdxs.shape[0] == B && topkIdxs.shape[1] == M)

        let o = Tensor.empty(shape: [B, M, H, D], dtype: .f32)

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q.buffer, offset: q.offset, index: 0)
        enc.setBuffer(kv.buffer, offset: kv.offset, index: 1)
        enc.setBuffer(sink.buffer, offset: sink.offset, index: 2)
        enc.setBuffer(topkIdxs.buffer, offset: topkIdxs.offset, index: 3)
        enc.setBuffer(o.buffer, offset: 0, index: 4)
        var dims = SIMD4<UInt32>(UInt32(B), UInt32(M), UInt32(N), UInt32(D))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 5)
        var misc = SIMD2<UInt32>(UInt32(H), UInt32(K))
        enc.setBytes(&misc, length: MemoryLayout.size(ofValue: misc), index: 6)
        var s = scale
        enc.setBytes(&s, length: 4, index: 7)

        let tg = MTLSize(width: min(H, 32), height: 1, depth: 1)
        enc.dispatchThreads(MTLSize(width: H, height: M, depth: B),
                            threadsPerThreadgroup: tg)
        enc.endEncoding()
        return o
    }

    // MARK: - Pure-Swift reference

    /// Same math, host-side. Used by tests.
    public static func referenceCPU(q: [Float], kv: [Float], sink: [Float], tk: [Int32],
                                     B: Int, M: Int, N: Int, H: Int, D: Int, K: Int,
                                     scale: Float) -> [Float] {
        var o = [Float](repeating: 0, count: B * M * H * D)
        for b in 0..<B {
            for m in 0..<M {
                for h in 0..<H {
                    var acc = [Float](repeating: 0, count: D)
                    var sumExp: Float = 0
                    var sMax: Float = -Float.infinity
                    let qOff = ((b * M + m) * H + h) * D
                    let tkOff = (b * M + m) * K
                    for k in 0..<K {
                        let idx = Int(tk[tkOff + k])
                        if idx < 0 { continue }
                        let kvOff = (b * N + idx) * D
                        var score: Float = 0
                        for d in 0..<D { score += q[qOff + d] * kv[kvOff + d] }
                        score *= scale
                        let newMax = max(sMax, score)
                        let fac = (sMax == -Float.infinity) ? 0.0 : exp(sMax - newMax)
                        let e = exp(score - newMax)
                        for d in 0..<D { acc[d] = acc[d] * fac + e * kv[kvOff + d] }
                        sumExp = sumExp * fac + e
                        sMax = newMax
                    }
                    if sMax == -Float.infinity {
                        sumExp = 1.0
                    } else {
                        sumExp += exp(sink[h] - sMax)
                    }
                    for d in 0..<D { o[qOff + d] = acc[d] / sumExp }
                }
            }
        }
        return o
    }
}
