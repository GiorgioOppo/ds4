import Foundation
import Metal

/// In-place softmax along an arbitrary axis. Reduces and normalises along
/// `axis`, broadcasting across all other dims. Mirrors `tensor.softmax(dim=axis)`.
public enum SoftmaxAxis {
    private static let pipeline = Device.shared.makePipeline("softmax_axis_f32")

    public static func apply(_ x: Tensor, axis: Int, in cmd: MTLCommandBuffer) {
        precondition(x.dtype == .f32)
        precondition(axis >= 0 && axis < x.shape.count, "axis out of range")

        let outer = x.shape[..<axis].reduce(1, *)
        let axisSize = x.shape[axis]
        let inner = x.shape[(axis + 1)...].reduce(1, *)

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        var dims = SIMD3<UInt32>(UInt32(outer), UInt32(axisSize), UInt32(inner))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 1)

        // Threadgroup sizing: punta a 256 thread (8 simdgroup × 32 su
        // Apple GPU) ma cap a `maxTotalThreadsPerThreadgroup` e
        // arrotonda al simdWidth. Per asse molto piccolo (<256)
        // potremmo ridurre, ma 256 è già un sweet spot empirico
        // perché il kernel è sweep-based (loop su AXIS internamente).
        let simdWidth = pipeline.threadExecutionWidth
        let maxTG = pipeline.maxTotalThreadsPerThreadgroup
        let tgWidth = min(maxTG, max(simdWidth, 256))

        // Shared memory: con il nuovo kernel simdgroup-based bastano
        // `nWarps` slot (un float per simdgroup) per il cross-simd
        // reduce. Prima erano `tgWidth` slot per la reduction tree
        // — risparmio 8× di threadgroup memory.
        let nWarps = (tgWidth + simdWidth - 1) / simdWidth
        enc.setThreadgroupMemoryLength(nWarps * MemoryLayout<Float>.size,
                                        index: 0)
        enc.dispatchThreadgroups(MTLSize(width: outer, height: inner, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Pure-Swift reference: softmax along `axis` of an N-D tensor stored
    /// row-major in `x` with shape `shape`.
    public static func referenceCPU(_ x: [Float], shape: [Int], axis: Int) -> [Float] {
        let outer = shape[..<axis].reduce(1, *)
        let axisSize = shape[axis]
        let inner = shape[(axis + 1)...].reduce(1, *)
        var out = x
        for o in 0..<outer {
            for n in 0..<inner {
                let base = o * axisSize * inner + n
                var m = -Float.infinity
                for i in 0..<axisSize { m = max(m, out[base + i * inner]) }
                var s: Float = 0
                for i in 0..<axisSize {
                    let e = exp(out[base + i * inner] - m)
                    out[base + i * inner] = e
                    s += e
                }
                for i in 0..<axisSize { out[base + i * inner] /= s }
            }
        }
        return out
    }
}
