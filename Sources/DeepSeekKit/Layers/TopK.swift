import Foundation
import Metal

/// Top-K along the last axis of an [N, V] tensor. Returns descending-sorted
/// values and corresponding Int32 indices. Mirrors `tensor.topk(k, dim=-1)`.
///
/// Two kernels back this API:
///
///   - `topk_f32`           — register-resident running heap, K ≤ 32.
///     One thread per row, no shared memory. Cheapest path; used for
///     decode-time sparse attention (window kWin + small kComp).
///
///   - `topk_f32_bitonic`   — in-place descending bitonic sort in
///     threadgroup-shared memory. Handles any K up to V, with V ≤ 4096
///     (the threadgroup memory budget). Used by the Indexer at prefill
///     when `index_topk` is large (e.g. 512 for V4-class checkpoints).
public enum TopK {
    /// Upper bound supported by the register-heap kernel.
    public static let maxK = 32
    /// Upper bound on V (row width) supported by the bitonic kernel.
    /// 4096 floats + 4096 int32 = 32 KiB threadgroup memory — fits all
    /// Apple-silicon GPUs we target.
    public static let maxLargeV = 4096
    /// Max threads per threadgroup the bitonic kernel will request.
    /// Metal's hard limit is 1024; the kernel handles V_padded > T by
    /// looping internally.
    public static let maxBitonicThreads = 1024

    private static let pipelineSmall = Device.shared.makePipeline("topk_f32")
    private static let pipelineLarge = Device.shared.makePipeline("topk_f32_bitonic")

    public struct Output {
        public let values: Tensor    // [N, K] f32
        public let indices: Tensor   // [N, K] i32
    }

    public static func apply(_ scores: Tensor, k: Int, in cmd: MTLCommandBuffer) -> Output {
        precondition(scores.dtype == .f32 && scores.shape.count == 2)
        precondition(k > 0, "k must be positive")
        let N = scores.shape[0]
        let V = scores.shape[1]
        precondition(k <= V, "k > vocab size")

        if k <= maxK {
            return applySmall(scores, N: N, V: V, k: k, in: cmd)
        }
        precondition(V <= maxLargeV,
                     "TopK bitonic kernel supports V <= \(maxLargeV); got V=\(V), k=\(k). A tiled top-K is needed for wider rows.")
        return applyLarge(scores, N: N, V: V, k: k, in: cmd)
    }

    private static func applySmall(_ scores: Tensor,
                                    N: Int, V: Int, k: Int,
                                    in cmd: MTLCommandBuffer) -> Output {
        let values = Tensor.empty(shape: [N, k], dtype: .f32)
        let indices = Tensor.empty(shape: [N, k], dtype: .i32)
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipelineSmall)
        enc.setBuffer(scores.buffer, offset: scores.offset, index: 0)
        enc.setBuffer(values.buffer, offset: 0, index: 1)
        enc.setBuffer(indices.buffer, offset: 0, index: 2)
        var dims = SIMD3<UInt32>(UInt32(N), UInt32(V), UInt32(k))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)

        // Threadgroup size = almeno un simdgroup intero (32 su Apple
        // GPU). Per N<simdWidth, alcune lane fanno `if (row >= N)
        // return` ma il simdgroup viene comunque schedulato in modo
        // coerente (no partial-warp penalty).
        let simdWidth = pipelineSmall.threadExecutionWidth
        let tgSize = max(simdWidth, min(N, 256))
        enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()
        return Output(values: values, indices: indices)
    }

    private static func applyLarge(_ scores: Tensor,
                                    N: Int, V: Int, k: Int,
                                    in cmd: MTLCommandBuffer) -> Output {
        let values = Tensor.empty(shape: [N, k], dtype: .f32)
        let indices = Tensor.empty(shape: [N, k], dtype: .i32)

        // The kernel pads V to the next power of two; pick a thread
        // count proportional to that so each thread covers a constant
        // chunk of compares per stage (Vp / T). Capped al limite hw
        // del pipeline (= 1024 su Apple GPU, ma leggiamo runtime per
        // portabilità), e arrotondato al simdWidth più vicino per
        // evitare lane SIMD parzialmente sprecate.
        let vPadded = nextPowerOfTwo(max(V, 2))
        let simdWidth = pipelineLarge.threadExecutionWidth
        let maxTG = min(maxBitonicThreads,
                         pipelineLarge.maxTotalThreadsPerThreadgroup)
        let raw = min(vPadded, maxTG)
        // Round-down al multiplo di simdWidth più vicino, ma almeno 1
        // simdgroup. Per V molto piccolo (raw < simdWidth) lasciamo
        // raw così com'è — non perdiamo nulla.
        let threads: Int
        if raw >= simdWidth {
            threads = (raw / simdWidth) * simdWidth
        } else {
            threads = raw
        }

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipelineLarge)
        enc.setBuffer(scores.buffer, offset: scores.offset, index: 0)
        enc.setBuffer(values.buffer, offset: 0, index: 1)
        enc.setBuffer(indices.buffer, offset: 0, index: 2)
        var dims = SIMD3<UInt32>(UInt32(N), UInt32(V), UInt32(k))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)
        // One threadgroup per row → row id = threadgroup z index.
        // Limitazione architetturale: per N piccolo (es. decode-time
        // single-token) abbiamo pochi threadgroup attivi e la GPU
        // resta sotto-utilizzata. Un vero fix richiede un tile/merge
        // multi-pass; al momento la bitonic single-row resta sotto i
        // 100µs su V≤4096 e non è il bottleneck inference dominante.
        let tgCount = MTLSize(width: 1, height: 1, depth: N)
        let tgSize  = MTLSize(width: threads, height: 1, depth: 1)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
        return Output(values: values, indices: indices)
    }

    public static func referenceCPU(_ scores: [Float], N: Int, V: Int, k: Int)
            -> (values: [Float], indices: [Int32]) {
        var values = [Float](repeating: 0, count: N * k)
        var indices = [Int32](repeating: 0, count: N * k)
        for n in 0..<N {
            let pairs = (0..<V).map { (scores[n * V + $0], Int32($0)) }
            let sorted = pairs.sorted { $0.0 > $1.0 }
            for i in 0..<k {
                values[n * k + i] = sorted[i].0
                indices[n * k + i] = sorted[i].1
            }
        }
        return (values, indices)
    }
}

@inline(__always)
private func nextPowerOfTwo(_ x: Int) -> Int {
    var p = 1
    while p < x { p <<= 1 }
    return p
}
