import Foundation
import Metal

/// MoE token-permutation helpers. Builds the host-side assignment tables
/// from the gate's `(weights, indices)` output, then exposes Metal kernels
/// that gather tokens to per-expert rows and scatter expert outputs back
/// into a dense [N, D] tensor weighted by the gating coefficients.
public enum MoEDispatch {
    private static let pGather = Device.shared.makePipeline("moe_gather")
    private static let pScatter = Device.shared.makePipeline("moe_scatter")

    /// Output of `prepare`: host-side tables + the per-expert offsets.
    public struct Plan {
        public let totalAssignments: Int                // N * topK
        public let perExpertOffsets: [Int]              // [nExperts + 1]
        public let assignTok: Tensor                    // [T] i32 — source token per t
        public let weights: Tensor                      // [T] f32 — gating weight per t
        public let tokSlotStart: Tensor                 // [N+1] i32 — for scatter
        public let tokSlotIdx: Tensor                   // [N * topK] i32 — for scatter
    }

    /// Build a Plan from gate output. `indices`: [N, topK] i32, `weights`:
    /// [N, topK] f32. The plan groups assignments by expert, in expert-id
    /// order, so caller can iterate `perExpertOffsets[i]..<perExpertOffsets[i+1]`
    /// to find the contiguous slice in `gathered` for expert i.
    public static func prepare(indices: [Int32], weights: [Float],
                               N: Int, topK: Int, nExperts: Int) -> Plan {
        precondition(indices.count == N * topK)
        precondition(weights.count == N * topK)

        // Bucket assignments by expert.
        var byExpert: [[Int]] = Array(repeating: [], count: nExperts)   // values are flat (n*topK + k) indices
        for n in 0..<N {
            for k in 0..<topK {
                let e = Int(indices[n * topK + k])
                if e >= 0 && e < nExperts {
                    byExpert[e].append(n * topK + k)
                }
            }
        }

        var offsets = [Int](repeating: 0, count: nExperts + 1)
        for e in 0..<nExperts { offsets[e + 1] = offsets[e] + byExpert[e].count }
        let total = offsets[nExperts]

        // Flat assignment tables.
        var assignTok = [Int32](repeating: -1, count: total)
        var assignWeights = [Float](repeating: 0, count: total)
        // Reverse lookup for scatter: per-token list of (t) values.
        var tokToTs: [[Int32]] = Array(repeating: [], count: N)

        var t = 0
        for e in 0..<nExperts {
            for orig in byExpert[e] {
                let n = orig / topK
                assignTok[t] = Int32(n)
                assignWeights[t] = weights[orig]
                tokToTs[n].append(Int32(t))
                t += 1
            }
        }

        // tokSlotStart prefix sum + tokSlotIdx flat list.
        var slotStart = [Int32](repeating: 0, count: N + 1)
        for n in 0..<N { slotStart[n + 1] = slotStart[n] + Int32(tokToTs[n].count) }
        var slotIdx = [Int32](repeating: -1, count: Int(slotStart[N]))
        var s = 0
        for n in 0..<N {
            for tt in tokToTs[n] { slotIdx[s] = tt; s += 1 }
        }

        let assignTokT = assignTok.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [max(total, 1)], dtype: .i32)
        }
        let weightsT = assignWeights.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [max(total, 1)], dtype: .f32)
        }
        let slotStartT = slotStart.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [N + 1], dtype: .i32)
        }
        let slotIdxT = slotIdx.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [max(slotIdx.count, 1)], dtype: .i32)
        }
        return Plan(totalAssignments: total,
                    perExpertOffsets: offsets,
                    assignTok: assignTokT,
                    weights: weightsT,
                    tokSlotStart: slotStartT,
                    tokSlotIdx: slotIdxT)
    }

    /// Permute `x[N, D]` into `gathered[T, D]` such that
    /// `gathered[plan.perExpertOffsets[i] ..< [i+1]]` is the rows that
    /// expert i should process. Sets unused (-1) rows to 0.
    public static func gather(_ x: Tensor, plan: Plan,
                              in cmd: MTLCommandBuffer) -> Tensor {
        precondition(x.dtype == .f32 && x.shape.count == 2)
        let N = x.shape[0], D = x.shape[1]
        _ = N
        let T = plan.totalAssignments
        let gathered = Tensor.empty(shape: [max(T, 1), D], dtype: .f32)
        if T == 0 { return gathered }

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pGather)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        enc.setBuffer(plan.assignTok.buffer, offset: 0, index: 1)
        enc.setBuffer(gathered.buffer, offset: 0, index: 2)
        var dims = SIMD2<UInt32>(UInt32(T), UInt32(D))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)
        enc.dispatchThreads(MTLSize(width: D, height: T, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        return gathered
    }

    /// Scatter weighted expert outputs back into a dense [N, D] tensor.
    /// `y` is written in place: `y[n] += sum_t weights[t] * outs[t]` for
    /// every t whose source token was n.
    public static func scatter(y: Tensor, outs: Tensor, plan: Plan,
                               in cmd: MTLCommandBuffer) {
        precondition(y.dtype == .f32 && y.shape.count == 2)
        precondition(outs.dtype == .f32 && outs.shape.count == 2)
        let N = y.shape[0], D = y.shape[1]
        precondition(outs.shape[1] == D)

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pScatter)
        enc.setBuffer(y.buffer, offset: y.offset, index: 0)
        enc.setBuffer(plan.tokSlotStart.buffer, offset: 0, index: 1)
        enc.setBuffer(plan.tokSlotIdx.buffer, offset: 0, index: 2)
        enc.setBuffer(plan.weights.buffer, offset: 0, index: 3)
        enc.setBuffer(outs.buffer, offset: outs.offset, index: 4)
        var dims = SIMD2<UInt32>(UInt32(N), UInt32(D))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 5)
        enc.dispatchThreads(MTLSize(width: D, height: N, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }
}
