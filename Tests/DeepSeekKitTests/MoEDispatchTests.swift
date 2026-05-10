import XCTest
import Metal
@testable import DeepSeekKit

final class MoEDispatchTests: XCTestCase {

    /// Gather then scatter with weight=1 should reconstruct the original
    /// (each token routed to one expert).
    func testGatherScatterRoundTripIdentity() throws {
        let N = 4, D = 8, topK = 1, nExperts = 3
        let x = (0..<N*D).map { Float($0) }
        // Each token to a different expert (round-robin).
        let indices = (0..<N).map { Int32($0 % nExperts) }
        let weights = [Float](repeating: 1.0, count: N)

        let xT = x.withUnsafeBytes { Tensor.from(bytes: $0, shape: [N, D], dtype: .f32) }
        let plan = MoEDispatch.prepare(indices: indices, weights: weights,
                                        N: N, topK: topK, nExperts: nExperts)
        XCTAssertEqual(plan.totalAssignments, N)

        let cmd = Device.shared.queue.makeCommandBuffer()!
        let gathered = MoEDispatch.gather(xT, plan: plan, in: cmd)
        // "expert" is identity — outs == gathered.
        let yT = Tensor.empty(shape: [N, D], dtype: .f32)
        // zero y first via blit
        let blit = cmd.makeBlitCommandEncoder()!
        blit.fill(buffer: yT.buffer, range: 0..<yT.byteCount, value: 0)
        blit.endEncoding()
        MoEDispatch.scatter(y: yT, outs: gathered, plan: plan, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let result = yT.toFloatArray()

        for i in 0..<x.count {
            XCTAssertEqual(result[i], x[i], accuracy: 1e-6)
        }
        _ = x
    }

    /// topK=2: each token is routed to two experts with weights [w0, w1].
    /// scatter should sum w0·out0 + w1·out1 == (w0+w1) · x[n] for an identity expert.
    func testTopK2WeightedSum() throws {
        let N = 3, D = 4, topK = 2, nExperts = 4
        let x = (0..<N*D).map { Float($0) * 0.1 }
        // indices: every token goes to experts 0 and 1.
        var indices = [Int32](); var weights = [Float]()
        for _ in 0..<N {
            indices.append(0); indices.append(1)
            weights.append(0.4); weights.append(0.6)
        }

        let xT = x.withUnsafeBytes { Tensor.from(bytes: $0, shape: [N, D], dtype: .f32) }
        let plan = MoEDispatch.prepare(indices: indices, weights: weights,
                                        N: N, topK: topK, nExperts: nExperts)
        XCTAssertEqual(plan.totalAssignments, N * topK)

        let cmd = Device.shared.queue.makeCommandBuffer()!
        let gathered = MoEDispatch.gather(xT, plan: plan, in: cmd)
        let y = Tensor.empty(shape: [N, D], dtype: .f32)
        let blit = cmd.makeBlitCommandEncoder()!
        blit.fill(buffer: y.buffer, range: 0..<y.byteCount, value: 0)
        blit.endEncoding()
        MoEDispatch.scatter(y: y, outs: gathered, plan: plan, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let result = y.toFloatArray()

        for i in 0..<result.count {
            XCTAssertEqual(result[i], x[i] * (0.4 + 0.6), accuracy: 1e-5, "i=\(i)")
        }
        _ = x
    }
}
