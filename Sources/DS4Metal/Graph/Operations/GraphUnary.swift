import Foundation
import Metal

extension GraphContext {
    /// Encode-form elementwise unary (single row). See MetalRuntime.UnaryOp.
    public func unary(_ x: GPUTensor, op: MetalRuntime.UnaryOp, out: GPUTensor, width: Int,
                      minV: Float = 0, maxV: Float = 0, scale: Float = 0, bias: Float = 0) throws {
        let args = MetalRuntime.unaryArgs(width: width, rows: 1, minV: minV, maxV: maxV, scale: scale, bias: bias, val: 0)
        let pso = try rt.unaryPipeline(op: op)
        var nthMax = pso.maxTotalThreadsPerThreadgroup; if nthMax > 256 { nthMax = 256 }
        var nth = width; if nth > nthMax { nth = nthMax }; if nth == 0 { nth = 1 }
        let nk0 = (width + nth - 1) / nth
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(x.buffer, offset: x.byteOffset, index: 1)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 2)
        e.dispatchThreadgroups(MTLSize(width: nk0, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }
}

