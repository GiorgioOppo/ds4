import Foundation
import Metal

/// HC mixing-tensor splitter with Sinkhorn-normalized comb matrix.
/// Mirrors `hc_split_sinkhorn` in
/// `Original/DeepSeek-V4-Pro/inference/kernel.py` lines 371–438.
public final class HCSinkhorn {
    public let hcMult: Int
    public let sinkhornIters: Int
    public let hcEps: Float

    private let pipeline: MTLComputePipelineState

    public init(hcMult: Int, sinkhornIters: Int, hcEps: Float) {
        self.hcMult = hcMult
        self.sinkhornIters = sinkhornIters
        self.hcEps = hcEps

        let constants = MTLFunctionConstantValues()
        var hc = UInt32(hcMult)
        var iters = UInt32(sinkhornIters)
        var eps = hcEps
        constants.setConstantValue(&hc, type: .uint, index: 0)
        constants.setConstantValue(&iters, type: .uint, index: 1)
        constants.setConstantValue(&eps, type: .float, index: 2)

        let lib = Device.shared.library
        do {
            let fn = try lib.makeFunction(name: "hc_split_sinkhorn_f32", constantValues: constants)
            self.pipeline = try Device.shared.mtl.makeComputePipelineState(function: fn)
        } catch {
            fatalError("HCSinkhorn pipeline failed: \(error)")
        }
    }

    public struct Output {
        public let pre: Tensor      // [n, hc]
        public let post: Tensor     // [n, hc]
        public let comb: Tensor     // [n, hc, hc]
    }

    /// `mixes`: [n, (2+hc)*hc] f32. `hcScale`: [3] f32. `hcBase`: [(2+hc)*hc] f32.
    public func split(mixes: Tensor, hcScale: Tensor, hcBase: Tensor,
                      in cmd: MTLCommandBuffer) -> Output {
        precondition(mixes.dtype == .f32 && hcScale.dtype == .f32 && hcBase.dtype == .f32)
        precondition(mixes.shape.count == 2)
        let n = mixes.shape[0]
        let mixHc = mixes.shape[1]
        precondition(mixHc == (2 + hcMult) * hcMult)
        precondition(hcScale.count == 3)
        precondition(hcBase.count == mixHc)

        let pre = Tensor.empty(shape: [n, hcMult], dtype: .f32)
        let post = Tensor.empty(shape: [n, hcMult], dtype: .f32)
        let comb = Tensor.empty(shape: [n, hcMult, hcMult], dtype: .f32)

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(mixes.buffer, offset: mixes.offset, index: 0)
        enc.setBuffer(hcScale.buffer, offset: hcScale.offset, index: 1)
        enc.setBuffer(hcBase.buffer, offset: hcBase.offset, index: 2)
        enc.setBuffer(pre.buffer, offset: 0, index: 3)
        enc.setBuffer(post.buffer, offset: 0, index: 4)
        enc.setBuffer(comb.buffer, offset: 0, index: 5)

        // shared: hc*hc (comb) + hc (rowSum) + hc (colSum)
        let sharedBytes = (hcMult * hcMult + 2 * hcMult) * MemoryLayout<Float>.size
        enc.setThreadgroupMemoryLength(sharedBytes, index: 0)

        let tg = MTLSize(width: hcMult * hcMult, height: 1, depth: 1)
        enc.dispatchThreadgroups(MTLSize(width: n, height: 1, depth: 1),
                                 threadsPerThreadgroup: tg)
        enc.endEncoding()

        return Output(pre: pre, post: post, comb: comb)
    }

    // MARK: - Pure-Swift reference (test only)

    public struct CPURow {
        public var pre: [Float]
        public var post: [Float]
        public var comb: [[Float]]
    }

    /// Apply the same math on host. Used as ground truth in tests.
    public func referenceCPU(mixes: [Float], hcScale: [Float], hcBase: [Float]) -> CPURow {
        let hc = hcMult
        precondition(mixes.count == (2 + hc) * hc)
        precondition(hcScale.count == 3)
        precondition(hcBase.count == (2 + hc) * hc)

        var pre = [Float](repeating: 0, count: hc)
        var post = [Float](repeating: 0, count: hc)
        for j in 0..<hc {
            let p = sigmoid(mixes[j] * hcScale[0] + hcBase[j]) + hcEps
            let q = 2 * sigmoid(mixes[j + hc] * hcScale[1] + hcBase[j + hc])
            pre[j] = p
            post[j] = q
        }

        var comb = Array(repeating: Array(repeating: Float(0), count: hc), count: hc)
        for j in 0..<hc {
            for k in 0..<hc {
                let idx = j * hc + k + 2 * hc
                comb[j][k] = mixes[idx] * hcScale[2] + hcBase[idx]
            }
        }
        // row softmax
        for j in 0..<hc {
            let m = comb[j].max()!
            var s: Float = 0
            for k in 0..<hc { comb[j][k] = exp(comb[j][k] - m); s += comb[j][k] }
            for k in 0..<hc { comb[j][k] = comb[j][k] / s + hcEps }
        }
        // initial col-norm
        for k in 0..<hc {
            var s: Float = 0
            for j in 0..<hc { s += comb[j][k] }
            for j in 0..<hc { comb[j][k] /= (s + hcEps) }
        }
        // sinkhorn iters - 1 alternations
        for _ in 1..<sinkhornIters {
            for j in 0..<hc {
                var s: Float = 0
                for k in 0..<hc { s += comb[j][k] }
                for k in 0..<hc { comb[j][k] /= (s + hcEps) }
            }
            for k in 0..<hc {
                var s: Float = 0
                for j in 0..<hc { s += comb[j][k] }
                for j in 0..<hc { comb[j][k] /= (s + hcEps) }
            }
        }

        return CPURow(pre: pre, post: post, comb: comb)
    }

    private func sigmoid(_ x: Float) -> Float { 1.0 / (1.0 + exp(-x)) }
}
