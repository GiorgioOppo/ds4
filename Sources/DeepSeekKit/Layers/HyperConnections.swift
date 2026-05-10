import Foundation
import Metal

/// Hyper-Connections pre/post mixing.
/// Mirrors `hc_pre` and `hc_post` in
/// `Reference/inference/model.py` lines 673–686.
public final class HyperConnections {
    public let hcMult: Int
    public let normEps: Float
    public let hcEps: Float
    public let dim: Int
    public let sinkhornIters: Int

    private let sinkhorn: HCSinkhorn
    private let pRsqrt: MTLComputePipelineState
    private let pBroadcastMul: MTLComputePipelineState
    private let pCollapse: MTLComputePipelineState
    private let pCompose: MTLComputePipelineState

    public init(config: ModelConfig, dim: Int) {
        self.hcMult = config.hcMult
        self.normEps = config.normEps
        self.hcEps = config.hcEps
        self.dim = dim
        self.sinkhornIters = config.hcSinkhornIters
        self.sinkhorn = HCSinkhorn(hcMult: config.hcMult,
                                   sinkhornIters: config.hcSinkhornIters,
                                   hcEps: config.hcEps)
        self.pRsqrt = Device.shared.makePipeline("rsqrt_mean_square_f32")
        self.pBroadcastMul = Device.shared.makePipeline("broadcast_row_mul_f32")
        self.pCollapse = Device.shared.makePipeline("hc_collapse_f32")
        self.pCompose = Device.shared.makePipeline("hc_post_compose_f32")
    }

    public struct PreOutput {
        public let y: Tensor       // [N, d]   — the collapsed input for the sublayer
        public let post: Tensor    // [N, hc]
        public let comb: Tensor    // [N, hc, hc]
    }

    /// `x`: [N, hc, d] f32, where N = b*s.
    /// `hcFn`: [(2+hc)*hc, hc*d] f32 — the projection weight (no bias).
    /// `hcScale`: [3] f32. `hcBase`: [(2+hc)*hc] f32.
    public func pre(x: Tensor, hcFn: Tensor, hcScale: Tensor, hcBase: Tensor,
                    in cmd: MTLCommandBuffer) -> PreOutput {
        precondition(x.dtype == .f32 && x.shape.count == 3)
        let N = x.shape[0], hc = x.shape[1], d = x.shape[2]
        precondition(hc == hcMult, "hc dim mismatch")
        let hcD = hc * d
        let mixHc = (2 + hc) * hc

        let xFlat = x.reshape([N, hcD])

        // 1. rsqrt(mean(x²) + eps)
        let rsqrt = Tensor.empty(shape: [N], dtype: .f32)
        do {
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pRsqrt)
            enc.setBuffer(xFlat.buffer, offset: xFlat.offset, index: 0)
            enc.setBuffer(rsqrt.buffer, offset: 0, index: 1)
            var dim = UInt32(hcD); var eps = normEps
            enc.setBytes(&dim, length: 4, index: 2)
            enc.setBytes(&eps, length: 4, index: 3)
            let tgWidth = 256
            enc.setThreadgroupMemoryLength(tgWidth * MemoryLayout<Float>.size, index: 0)
            enc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
            enc.endEncoding()
        }

        // 2. mixes = xFlat @ hcFnᵀ via Linear (f32 dense)
        precondition(hcFn.dtype == .f32 && hcFn.shape == [mixHc, hcD])
        let lin = Linear(inFeatures: hcD, outFeatures: mixHc, weight: hcFn, scale: nil)
        let mixes = lin(xFlat, in: cmd)

        // 3. mixes *= rsqrt (broadcast over rows)
        do {
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pBroadcastMul)
            enc.setBuffer(mixes.buffer, offset: 0, index: 0)
            enc.setBuffer(rsqrt.buffer, offset: 0, index: 1)
            var dims = SIMD2<UInt32>(UInt32(N), UInt32(mixHc))
            enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 2)
            enc.dispatchThreads(MTLSize(width: mixHc, height: N, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.endEncoding()
        }

        // 4. (pre, post, comb) = hc_split_sinkhorn(mixes)
        let split = sinkhorn.split(mixes: mixes, hcScale: hcScale, hcBase: hcBase, in: cmd)

        // 5. y[N, d] = Σ_h pre[N, h] · x[N, h, d]
        let y = Tensor.empty(shape: [N, d], dtype: .f32)
        do {
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pCollapse)
            enc.setBuffer(x.buffer, offset: x.offset, index: 0)
            enc.setBuffer(split.pre.buffer, offset: 0, index: 1)
            enc.setBuffer(y.buffer, offset: 0, index: 2)
            var dims = SIMD3<UInt32>(UInt32(N), UInt32(hc), UInt32(d))
            enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)
            enc.dispatchThreads(MTLSize(width: d, height: N, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.endEncoding()
        }

        return PreOutput(y: y, post: split.post, comb: split.comb)
    }

    /// `x`: [N, d] (sublayer output). `residual`: [N, hc, d] (pre-sublayer state).
    /// `post`: [N, hc]. `comb`: [N, hc, hc]. Returns [N, hc, d].
    public func post(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor,
                     in cmd: MTLCommandBuffer) -> Tensor {
        precondition(x.dtype == .f32 && x.shape.count == 2)
        precondition(residual.dtype == .f32 && residual.shape.count == 3)
        let N = x.shape[0], d = x.shape[1]
        let hc = residual.shape[1]
        precondition(residual.shape[0] == N && residual.shape[2] == d)
        precondition(hc == hcMult)
        precondition(post.shape == [N, hc])
        precondition(comb.shape == [N, hc, hc])

        let y = Tensor.empty(shape: [N, hc, d], dtype: .f32)
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pCompose)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        enc.setBuffer(residual.buffer, offset: residual.offset, index: 1)
        enc.setBuffer(post.buffer, offset: post.offset, index: 2)
        enc.setBuffer(comb.buffer, offset: comb.offset, index: 3)
        enc.setBuffer(y.buffer, offset: 0, index: 4)
        var dims = SIMD3<UInt32>(UInt32(N), UInt32(hc), UInt32(d))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 5)
        enc.dispatchThreads(MTLSize(width: d, height: hc, depth: N),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 4, depth: 4))
        enc.endEncoding()
        return y
    }

    // MARK: - Pure-Swift reference

    public static func referencePreCPU(
        x: [Float], hcFn: [Float], hcScale: [Float], hcBase: [Float],
        N: Int, hc: Int, d: Int,
        sinkhornIters: Int, normEps: Float, hcEps: Float
    ) -> (y: [Float], post: [Float], comb: [Float]) {
        let hcD = hc * d
        let mixHc = (2 + hc) * hc

        // rsqrt
        var rsqrt = [Float](repeating: 0, count: N)
        for n in 0..<N {
            var s: Float = 0
            for k in 0..<hcD { let v = x[n * hcD + k]; s += v * v }
            rsqrt[n] = 1.0 / (s / Float(hcD) + normEps).squareRoot()
        }

        // mixes = xFlat @ hcFnᵀ
        var mixes = [Float](repeating: 0, count: N * mixHc)
        for n in 0..<N {
            for m in 0..<mixHc {
                var acc: Float = 0
                for k in 0..<hcD { acc += hcFn[m * hcD + k] * x[n * hcD + k] }
                mixes[n * mixHc + m] = acc * rsqrt[n]
            }
        }

        // sinkhorn split per row
        let sk = HCSinkhorn(hcMult: hc, sinkhornIters: sinkhornIters, hcEps: hcEps)
        var preArr = [Float](repeating: 0, count: N * hc)
        var postArr = [Float](repeating: 0, count: N * hc)
        var combArr = [Float](repeating: 0, count: N * hc * hc)
        for n in 0..<N {
            let row = Array(mixes[n * mixHc ..< (n + 1) * mixHc])
            let r = sk.referenceCPU(mixes: row, hcScale: hcScale, hcBase: hcBase)
            for j in 0..<hc {
                preArr[n * hc + j] = r.pre[j]
                postArr[n * hc + j] = r.post[j]
                for k in 0..<hc { combArr[(n * hc + j) * hc + k] = r.comb[j][k] }
            }
        }

        // collapse: y[n, d_idx] = Σ_h pre[n, h] · x[n, h, d_idx]
        var y = [Float](repeating: 0, count: N * d)
        for n in 0..<N {
            for di in 0..<d {
                var acc: Float = 0
                for h in 0..<hc { acc += preArr[n * hc + h] * x[(n * hc + h) * d + di] }
                y[n * d + di] = acc
            }
        }
        return (y, postArr, combArr)
    }

    public static func referencePostCPU(
        x: [Float], residual: [Float], post: [Float], comb: [Float],
        N: Int, hc: Int, d: Int
    ) -> [Float] {
        var y = [Float](repeating: 0, count: N * hc * d)
        for n in 0..<N {
            for j in 0..<hc {
                for di in 0..<d {
                    let term1 = post[n * hc + j] * x[n * d + di]
                    var term2: Float = 0
                    for k in 0..<hc {
                        term2 += comb[(n * hc + k) * hc + j] * residual[(n * hc + k) * d + di]
                    }
                    y[(n * hc + j) * d + di] = term1 + term2
                }
            }
        }
        return y
    }
}
