import Foundation
import Metal

/// GPU-side construction of MLA's [B, S, K] topk index table. Replaces
/// the host-side `AttentionIndices` path so the caller can build the
/// table without committing the command buffer to read intermediate
/// indexer output back to host.
///
/// Typical call sequence in MLA:
///   1. allocate `topkT = Tensor.empty([B, S, kWin + kComp], .i32)`
///   2. `AttnIndicesGPU.window(into: topkT, ...)`
///   3. either `copyCompressed(into:from:...)` (indexer case) or
///      `compressedDeterministic(into:...)` (ratio==128 case)
public enum AttnIndicesGPU {
    private static let pWindow      = Device.shared.makePipeline("attn_window_indices_i32")
    private static let pCompressed  = Device.shared.makePipeline("attn_compressed_indices_i32")
    private static let pCopy        = Device.shared.makePipeline("attn_copy_compressed_i32")

    /// Fill window indices into the first `kWin` slots of each row of `out`.
    public static func window(into out: Tensor, B: Int, S: Int, K: Int,
                               kWin: Int, startPos: Int,
                               in cmd: MTLCommandBuffer) {
        if kWin == 0 || B == 0 || S == 0 { return }
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pWindow)
        enc.setBuffer(out.buffer, offset: out.offset, index: 0)
        var dims = SIMD4<UInt32>(UInt32(B), UInt32(S), UInt32(K), UInt32(kWin))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 1)
        var sp = Int32(startPos)
        enc.setBytes(&sp, length: MemoryLayout<Int32>.size, index: 2)
        enc.dispatchThreads(MTLSize(width: kWin, height: S, depth: B),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
        enc.endEncoding()
    }

    /// Fill deterministic compressed indices into out[:, :, kWin:kWin+kComp].
    public static func compressedDeterministic(into out: Tensor,
                                                B: Int, S: Int, K: Int,
                                                kWin: Int, kComp: Int,
                                                ratio: Int, offset: Int,
                                                startPos: Int,
                                                in cmd: MTLCommandBuffer) {
        if kComp == 0 || B == 0 || S == 0 { return }
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pCompressed)
        enc.setBuffer(out.buffer, offset: out.offset, index: 0)
        var dims = SIMD4<UInt32>(UInt32(B), UInt32(S), UInt32(K), UInt32(kComp))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 1)
        var params = SIMD4<Int32>(Int32(kWin), Int32(startPos), Int32(ratio), Int32(offset))
        enc.setBytes(&params, length: MemoryLayout.size(ofValue: params), index: 2)
        enc.dispatchThreads(MTLSize(width: kComp, height: S, depth: B),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
        enc.endEncoding()
    }

    /// Copy indexer-emitted topk into out[:, :, kWin:kWin+kComp].
    public static func copyCompressed(into out: Tensor, from comp: Tensor,
                                       B: Int, S: Int, K: Int,
                                       kWin: Int, kComp: Int,
                                       in cmd: MTLCommandBuffer) {
        if kComp == 0 || B == 0 || S == 0 { return }
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pCopy)
        enc.setBuffer(out.buffer, offset: out.offset, index: 0)
        enc.setBuffer(comp.buffer, offset: comp.offset, index: 1)
        var dims = SIMD4<UInt32>(UInt32(B), UInt32(S), UInt32(K), UInt32(kComp))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 2)
        var kw = UInt32(kWin)
        enc.setBytes(&kw, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: kComp, height: S, depth: B),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
        enc.endEncoding()
    }
}
