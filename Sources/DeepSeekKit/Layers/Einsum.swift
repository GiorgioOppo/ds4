import Foundation
import Metal

/// Grouped einsum operations specialised for V4 attention paths.
public enum Einsum {
    private static let pBshdBtd = Device.shared.makePipeline("einsum_bshd_btd_to_bsht_f32")
    private static let pBsgdGrd = Device.shared.makePipeline("einsum_bsgd_grd_to_bsgr_f32")
    private static let pBsgdGrdBF16wo = Device.shared.makePipeline("einsum_bsgd_grd_to_bsgr_bf16wo")
    private static let pBsgdGrdFP8wo  = Device.shared.makePipeline("einsum_bsgd_grd_to_bsgr_fp8wo")

    /// `q`: [B, S, H, D]. `kv`: [B, T, D]. Returns [B, S, H, T].
    /// Used by Indexer.
    public static func bshdBtd(q: Tensor, kv: Tensor, in cmd: MTLCommandBuffer) -> Tensor {
        precondition(q.dtype == .f32 && q.shape.count == 4)
        precondition(kv.dtype == .f32 && kv.shape.count == 3)
        let B = q.shape[0], S = q.shape[1], H = q.shape[2], D = q.shape[3]
        precondition(kv.shape[0] == B && kv.shape[2] == D)
        let T = kv.shape[1]

        let out = Tensor.empty(shape: [B, S, H, T], dtype: .f32)
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pBshdBtd)
        enc.setBuffer(q.buffer, offset: q.offset, index: 0)
        enc.setBuffer(kv.buffer, offset: kv.offset, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        var dims = SIMD4<UInt32>(UInt32(B), UInt32(S), UInt32(H), UInt32(D))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)
        var tt = UInt32(T)
        enc.setBytes(&tt, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: T, height: S * H, depth: B),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
        enc.endEncoding()
        return out
    }

    /// `o`: [B, S, G, D] f32. `woA`: [G, R, D] in f32 / bf16 / fp8_e4m3.
    /// When `woA.dtype == .fp8E4M3` the caller MUST pass `woAScale`:
    /// [G*R/128, D/128] UE8M0 (1 byte per scale, 128×128 2D blocks).
    /// Returns [B, S, G, R] f32. Used by MLA grouped output projection.
    public static func bsgdGrd(o: Tensor, woA: Tensor, woAScale: Tensor? = nil,
                                 in cmd: MTLCommandBuffer) -> Tensor {
        precondition(o.dtype == .f32 && o.shape.count == 4)
        if !(woA.dtype == .f32 || woA.dtype == .bf16 || woA.dtype == .fp8E4M3)
            || woA.shape.count != 3 {
            let msg = """
            Einsum.bsgdGrd: unsupported woA tensor.
              expected: dtype=f32|bf16|fp8E4M3, rank=3
              got:      dtype=\(woA.dtype) shape=\(woA.shape)
            \n
            """
            FileHandle.standardError.write(Data(msg.utf8))
        }
        precondition((woA.dtype == .f32 || woA.dtype == .bf16 || woA.dtype == .fp8E4M3)
                     && woA.shape.count == 3,
                     "Einsum.bsgdGrd: woA must be f32, bf16 or fp8E4M3; got \(woA.dtype)")
        let B = o.shape[0], S = o.shape[1], G = o.shape[2], D = o.shape[3]
        precondition(woA.shape[0] == G && woA.shape[2] == D)
        let R = woA.shape[1]

        let out = Tensor.empty(shape: [B, S, G, R], dtype: .f32)
        let enc = cmd.makeComputeCommandEncoder()!

        switch woA.dtype {
        case .f32:
            enc.setComputePipelineState(pBsgdGrd)
            enc.setBuffer(o.buffer,   offset: o.offset,   index: 0)
            enc.setBuffer(woA.buffer, offset: woA.offset, index: 1)
            enc.setBuffer(out.buffer, offset: 0,          index: 2)
            var dims = SIMD4<UInt32>(UInt32(B), UInt32(S), UInt32(G), UInt32(D))
            enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)
            var rr = UInt32(R)
            enc.setBytes(&rr, length: 4, index: 4)
        case .bf16:
            enc.setComputePipelineState(pBsgdGrdBF16wo)
            enc.setBuffer(o.buffer,   offset: o.offset,   index: 0)
            enc.setBuffer(woA.buffer, offset: woA.offset, index: 1)
            enc.setBuffer(out.buffer, offset: 0,          index: 2)
            var dims = SIMD4<UInt32>(UInt32(B), UInt32(S), UInt32(G), UInt32(D))
            enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)
            var rr = UInt32(R)
            enc.setBytes(&rr, length: 4, index: 4)
        case .fp8E4M3:
            guard let scale = woAScale else {
                fatalError("Einsum.bsgdGrd: FP8 woA requires woAScale (UE8M0 [G*R/128, D/128])")
            }
            enc.setComputePipelineState(pBsgdGrdFP8wo)
            enc.setBuffer(o.buffer,     offset: o.offset,     index: 0)
            enc.setBuffer(woA.buffer,   offset: woA.offset,   index: 1)
            enc.setBuffer(scale.buffer, offset: scale.offset, index: 2)
            enc.setBuffer(out.buffer,   offset: 0,            index: 3)
            var dims = SIMD4<UInt32>(UInt32(B), UInt32(S), UInt32(G), UInt32(D))
            enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 4)
            var rr = UInt32(R)
            enc.setBytes(&rr, length: 4, index: 5)
        default:
            fatalError("unreachable")
        }

        enc.dispatchThreads(MTLSize(width: R, height: S * G, depth: B),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
        enc.endEncoding()
        return out
    }

    public static func referenceBshdBtdCPU(q: [Float], kv: [Float],
                                            B: Int, S: Int, H: Int, D: Int, T: Int) -> [Float] {
        var out = [Float](repeating: 0, count: B * S * H * T)
        for b in 0..<B {
            for s in 0..<S {
                for h in 0..<H {
                    for t in 0..<T {
                        var acc: Float = 0
                        for d in 0..<D {
                            acc += q[((b * S + s) * H + h) * D + d] * kv[(b * T + t) * D + d]
                        }
                        out[((b * S + s) * H + h) * T + t] = acc
                    }
                }
            }
        }
        return out
    }

    public static func referenceBsgdGrdCPU(o: [Float], woA: [Float],
                                            B: Int, S: Int, G: Int, D: Int, R: Int) -> [Float] {
        var out = [Float](repeating: 0, count: B * S * G * R)
        for b in 0..<B {
            for s in 0..<S {
                for g in 0..<G {
                    for r in 0..<R {
                        var acc: Float = 0
                        for d in 0..<D {
                            acc += o[((b * S + s) * G + g) * D + d] * woA[(g * R + r) * D + d]
                        }
                        out[((b * S + s) * G + g) * R + r] = acc
                    }
                }
            }
        }
        return out
    }
}
