import Foundation
import Metal
import DS4Core

extension StreamingDecoder {
    /// Reusable per-token staging for the batched prefill: ONE set per chunk,
    /// rewritten at every layer (layer i's phase B completes before layer i+1's
    /// phase A touches them) — instead of 3·n fresh Metal buffers per LAYER
    /// (43 × 512 × 3 ≈ 66k allocations per chunk).
    struct PrefillStage {
        let cur: [GPUTensor]     // n × nEmbd        (attn-normed FFN input)
        let attn: [GPUTensor]    // n × nHC·nEmbd    (post-attention residual)
        let split: [GPUTensor]   // n × 24           (HC split)
        let ids: [GPUTensor]     // n × k Int32      (remapped ids, padded to k)
        let rw: [GPUTensor]      // n × k Float      (route weights, 0-padded)
        /// Extra buffers for the mul_mm_id path (DS4_PREFILL_MM), rewritten per
        /// group: token-major activation matrix, group-local ids/weights, the
        /// expert-major map (htpe/hids) and the mid/down6 outputs.
        struct MMBuffers {
            let curMat: GPUTensor    // n × nEmbd f32 (chunk-global rows)
            let idsMat: GPUTensor    // n × k Int32   (group-local rows)
            let wMat: GPUTensor      // n × k f32     (group-local rows)
            let htpe: GPUTensor      // maxUnion u32
            let hids: GPUTensor      // maxUnion × n Int32
            let mid: GPUTensor       // n × k × expertFfn f16
            let down6: GPUTensor     // n × k × nEmbd f32
            // Batched SHARED-expert FFN (Q8_0 path only): token-major gate/up/
            // mid intermediates and the per-token shared output rows.
            let sGate: GPUTensor     // n × sharedFfn f32
            let sUp: GPUTensor       // n × sharedFfn f32
            let sMid: GPUTensor      // n × sharedFfn f32
            let sOut: GPUTensor      // n × nEmbd f32
            let ones: GPUTensor      // n × f32 = 1 (unit route weights for the rows-swiglu)
        }
        let mm: MMBuffers?

        /// Allocate one zeroed Metal slab and expose `n` fixed-size logical
        /// rows as GPUTensor views. Before this helper PrefillStage allocated
        /// five MTLBuffers per prompt token (2,560 buffers at chunk=512), which
        /// made buffer creation and Objective-C lifetime management measurable
        /// prefill work. Views keep the same hazard-tracked shared buffer while
        /// preserving every call site's existing GPUTensor API.
        private static func rowViews(_ rt: MetalRuntime, n: Int,
                                     rowBytes: Int, rowCount: Int) throws -> [GPUTensor] {
            let slab = try GPUTensor.zerosBytes(rt, byteLength: n * rowBytes)
            return (0..<n).map {
                slab.subview(byteOffset: $0 * rowBytes, byteLength: rowBytes,
                             count: rowCount)
            }
        }

        init(_ rt: MetalRuntime, n: Int, d: DSV4Dims, mmPath: Bool, maxUnion: Int) throws {
            cur = try Self.rowViews(rt, n: n, rowBytes: d.nEmbd * 4,
                                    rowCount: d.nEmbd)
            attn = try Self.rowViews(rt, n: n, rowBytes: d.nHC * d.nEmbd * 4,
                                     rowCount: d.nHC * d.nEmbd)
            split = try Self.rowViews(rt, n: n, rowBytes: 24 * 4, rowCount: 24)
            ids = try Self.rowViews(rt, n: n, rowBytes: d.k * 4, rowCount: d.k)
            rw = try Self.rowViews(rt, n: n, rowBytes: d.k * 4, rowCount: d.k)
            if mmPath {
                let onesBuf = try GPUTensor.zeros(rt, floatCount: n)
                let op = (onesBuf.buffer.contents() + onesBuf.byteOffset)
                    .bindMemory(to: Float.self, capacity: n)
                for i in 0..<n { op[i] = 1 }
                mm = MMBuffers(
                    curMat: try .zeros(rt, floatCount: n * d.nEmbd),
                    idsMat: try .zerosBytes(rt, byteLength: n * d.k * 4),
                    wMat: try .zeros(rt, floatCount: n * d.k),
                    htpe: try .zerosBytes(rt, byteLength: max(1, maxUnion) * 4),
                    hids: try .zerosBytes(rt, byteLength: max(1, maxUnion) * n * 4),
                    mid: try .zerosBytes(rt, byteLength: n * d.k * d.expertFfn * 2),
                    down6: try .zeros(rt, floatCount: n * d.k * d.nEmbd),
                    sGate: try .zeros(rt, floatCount: n * d.sharedFfn),
                    sUp: try .zeros(rt, floatCount: n * d.sharedFfn),
                    sMid: try .zeros(rt, floatCount: n * d.sharedFfn),
                    sOut: try .zeros(rt, floatCount: n * d.nEmbd),
                    ones: onesBuf)
            } else {
                mm = nil
            }
        }
    }
}
