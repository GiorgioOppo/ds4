import Foundation
import Metal

/// DEMO: porting di `softmax_f32` come subclass di `MetalKernel`.
///
/// Riferimento al kernel sorgente:
/// `Sources/DeepSeekKit/Kernels/softmax.metal` — `kernel void
/// softmax_f32(device float* x [[buffer(0)]], constant uint& N
/// [[buffer(1)]], ...)`.
///
/// È un kernel **in-place**: il buffer in `buffers[0]` è insieme
/// input e output. Threadgroup per riga, 256 thread per
/// threadgroup, ogni thread elabora `N/256` colonne.
public final class DemoSoftmaxKernel: MetalKernel {
    public init() { super.init(functionName: "softmax_f32") }

    public override func threadgroupSize(for problem: KernelProblem) -> MTLSize {
        MTLSize(width: 256, height: 1, depth: 1)
    }

    public override func gridSize(for problem: KernelProblem) -> MTLSize {
        // dims = [rows, cols] — una threadgroup per riga.
        MTLSize(width: problem.dims[0], height: 1, depth: 1)
    }

    public override func bind(problem: KernelProblem,
                              buffers: [MTLBuffer?],
                              offsets: [Int],
                              to enc: MTLComputeCommandEncoder) {
        enc.setBuffer(buffers[0], offset: offsets[0], index: 0)   // x (in-place)
        var cols = UInt32(problem.dims[1])
        enc.setBytes(&cols, length: MemoryLayout<UInt32>.size, index: 1)
    }
}
