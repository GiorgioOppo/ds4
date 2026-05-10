import Foundation
import Metal

/// Top-K along the last axis of an [N, V] tensor. Returns descending-sorted
/// values and corresponding Int32 indices. Mirrors `tensor.topk(k, dim=-1)`.
public enum TopK {
    public static let maxK = 32
    private static let pipeline = Device.shared.makePipeline("topk_f32")

    public struct Output {
        public let values: Tensor    // [N, K] f32
        public let indices: Tensor   // [N, K] i32
    }

    public static func apply(_ scores: Tensor, k: Int, in cmd: MTLCommandBuffer) -> Output {
        precondition(scores.dtype == .f32 && scores.shape.count == 2)
        precondition(k > 0 && k <= maxK,
                     "TopK kernel supports k <= \(maxK); for larger k a tiled implementation is needed")
        let N = scores.shape[0]
        let V = scores.shape[1]
        precondition(k <= V, "k > vocab size")

        let values = Tensor.empty(shape: [N, k], dtype: .f32)
        let indices = Tensor.empty(shape: [N, k], dtype: .i32)

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(scores.buffer, offset: scores.offset, index: 0)
        enc.setBuffer(values.buffer, offset: 0, index: 1)
        enc.setBuffer(indices.buffer, offset: 0, index: 2)
        var dims = SIMD3<UInt32>(UInt32(N), UInt32(V), UInt32(k))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)
        enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(N, 64), height: 1, depth: 1))
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
