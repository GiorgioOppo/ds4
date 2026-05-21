import Foundation
import Metal

/// Sparse multi-head attention with FlashAttention-style online softmax.
/// Mirrors `sparse_attn` in
/// `Reference/inference/kernel.py` lines 355–368.
public enum SparseAttention {
    private static let pipeline = Device.shared.makePipeline("sparse_attn_f32")

    /// Query-tile size. `sparse_attn_f32` runs one thread per
    /// (head, query, batch) with a scalar K-loop and a device-memory
    /// accumulator — heavy per thread. A long prefill (M = prompt
    /// length) packed into a single dispatch produces a multi-second
    /// command buffer that the macOS GPU watchdog aborts with
    /// `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`.
    ///
    /// `apply` therefore tiles the query dimension: each tile is its
    /// own dispatch in its own command buffer (commit + wait), so no
    /// single buffer monopolises the GPU. Tunable via
    /// `DEEPSEEK_SPARSE_ATTN_TILE`; 128 is conservative enough to stay
    /// well under the watchdog for realistic H/K/D. Decode (M == 1)
    /// runs as a single one-row tile.
    private static let tileM: Int = {
        if let v = ProcessInfo.processInfo.environment["DEEPSEEK_SPARSE_ATTN_TILE"],
           let n = Int(v), n > 0 { return n }
        return 128
    }()

    /// `q`: [B, M, H, D] f32. `kv`: [B, N, D] f32. `sink`: [H] f32.
    /// `topkIdxs`: [B, M, K] i32 (use -1 for padding entries).
    /// Returns `o`: [B, M, H, D] f32.
    ///
    /// `cmd` is `inout`: each query tile is committed on its own buffer
    /// and `cmd` is replaced with a fresh one, so the caller continues
    /// encoding onto the swapped value.
    public static func apply(q: Tensor, kv: Tensor, sink: Tensor,
                             topkIdxs: Tensor, scale: Float,
                             in cmd: inout MTLCommandBuffer) -> Tensor {
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

        // `dims` always carries the global M — the kernel indexes
        // q/tk/o with the absolute query position (`gid.y + mOffset`).
        var dims = SIMD4<UInt32>(UInt32(B), UInt32(M), UInt32(N), UInt32(D))
        var misc = SIMD2<UInt32>(UInt32(H), UInt32(K))
        var s = scale
        let tg = MTLSize(width: min(H, 32), height: 1, depth: 1)

        var m0 = 0
        while m0 < M {
            let rows = min(tileM, M - m0)
            let enc = cmd.makeComputeCommandEncoder()!
            enc.label = "sparse_attn m=[\(m0)..\(m0 + rows)) of \(M) H=\(H) K=\(K)"
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(q.buffer, offset: q.offset, index: 0)
            enc.setBuffer(kv.buffer, offset: kv.offset, index: 1)
            enc.setBuffer(sink.buffer, offset: sink.offset, index: 2)
            enc.setBuffer(topkIdxs.buffer, offset: topkIdxs.offset, index: 3)
            enc.setBuffer(o.buffer, offset: 0, index: 4)
            enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 5)
            enc.setBytes(&misc, length: MemoryLayout.size(ofValue: misc), index: 6)
            enc.setBytes(&s, length: 4, index: 7)
            var mOff = UInt32(m0)
            enc.setBytes(&mOff, length: 4, index: 8)

            enc.dispatchThreads(MTLSize(width: H, height: rows, depth: B),
                                threadsPerThreadgroup: tg)
            enc.endEncoding()

            // One command buffer per tile — bounds each buffer's GPU
            // time under the interactivity watchdog threshold.
            cmd.commit(); cmd.waitUntilCompleted()
            cmd = Device.shared.makeCommandBuffer()
            m0 += rows
        }
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
