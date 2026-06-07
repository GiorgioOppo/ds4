import Foundation
import Metal

// Stage B: the tensor-op layer. GraphContext owns one MTLCommandBuffer + compute
// encoder for a whole sequence of dispatches (encode many, commit once), mirroring
// the C ds4_gpu_command_buffer / ds4_gpu_finish_command_buffer pattern. Buffers
// are hazard-tracked, so chained dispatches that read a previous dispatch's output
// serialize automatically. The encode* methods are the GPUTensor analogues of the
// Stage A kernel wrappers (which allocate+commit+readback each call).

public final class GraphContext {
    public let rt: MetalRuntime
    private var cb: MTLCommandBuffer?
    private var enc: MTLComputeCommandEncoder?

    public init(_ rt: MetalRuntime) { self.rt = rt }

    public func begin() throws {
        guard let c = rt.queue.makeCommandBuffer(), let e = c.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        cb = c; enc = e
    }

    /// Flush: end encoding, commit, wait. After this the GPUTensor outputs are readable.
    public func commit() {
        enc?.endEncoding()
        cb?.commit()
        cb?.waitUntilCompleted()
        enc = nil; cb = nil
    }

    var encoder: MTLComputeCommandEncoder { enc! }

    // MARK: - tensor-ops (encode into the shared encoder; no commit)

    /// RMSNorm rows: out = normalize(x) [* weight]. n multiple of 4.
    public func rmsNorm(_ x: GPUTensor, weight: GPUTensor?, out: GPUTensor,
                        rows: Int, n: Int, eps: Float) throws {
        precondition(n % 4 == 0)
        let args = MetalRuntime.rmsNormArgs(n: n, rows: rows, eps: eps)
        let kernel = weight != nil ? "kernel_rms_norm_mul_f32_4" : "kernel_rms_norm_f32_4"
        let pso = try rt.pipeline(kernel)
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(x.buffer, offset: 0, index: 1)
        e.setBuffer((weight ?? x).buffer, offset: 0, index: 2)
        e.setBuffer(x.buffer, offset: 0, index: 3)
        e.setBuffer(out.buffer, offset: 0, index: 4)
        e.setThreadgroupMemoryLength(32 * 4, index: 0)
        let nth = MetalRuntime.rmsNormThreads(n)
        e.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }

    /// Dense F32 matvec: out[outDim] = W[outDim x inDim] * x[inDim].
    public func matmulF32(weight: GPUTensor, x: GPUTensor, out: GPUTensor,
                          inDim: Int, outDim: Int) throws {
        precondition(inDim % 4 == 0)
        let nsg = Int16(min(8, (inDim + 127) / 128))
        let nr0 = 2
        let rowBytes = inDim * 4
        let args = MetalRuntime.mulMVArgs(ne00: inDim, ne01: outDim, nb00: 4, nb01: UInt64(rowBytes),
                                          nb02: UInt64(rowBytes * outDim), ne10: inDim, ne11: 1,
                                          nb10: 4, nb11: UInt64(inDim * 4), ne0: outDim, ne1: 1, nr0: Int32(nr0))
        let pso = try rt.mulMVPipeline("kernel_mul_mv_f32_f32_4", nsg: nsg)
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(weight.buffer, offset: weight.byteOffset, index: 1)   // byteOffset != 0 for no-copy mmap weights
        e.setBuffer(x.buffer, offset: 0, index: 2)
        e.setBuffer(out.buffer, offset: 0, index: 3)
        e.setThreadgroupMemoryLength(32 * 2 * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (outDim + nr0 - 1) / nr0, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: Int(nsg), depth: 1))
    }

    /// Dense F16 matvec: out[outDim] = W(F16)[outDim x inDim] * x[inDim].
    public func matmulF16(weight: GPUTensor, x: GPUTensor, out: GPUTensor, inDim: Int, outDim: Int) throws {
        precondition(inDim % 4 == 0)
        let nsg = Int16(min(8, (inDim + 127) / 128))
        var nr0 = 2, smem = 32 * 2 * 4
        if (outDim == 512 || outDim == 1024) && inDim >= 4096 { nr0 = 4; smem = 32 * 4 * 4 }
        let rowBytes = inDim * 2
        let args = MetalRuntime.mulMVArgs(ne00: inDim, ne01: outDim, nb00: 2, nb01: UInt64(rowBytes),
                                          nb02: UInt64(rowBytes * outDim), ne10: inDim, ne11: 1,
                                          nb10: 4, nb11: UInt64(inDim * 4), ne0: outDim, ne1: 1, nr0: Int32(nr0))
        let pso = try rt.mulMVPipeline("kernel_mul_mv_f16_f32_4", nsg: nsg)
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(weight.buffer, offset: weight.byteOffset, index: 1)   // byteOffset != 0 for no-copy mmap weights
        e.setBuffer(x.buffer, offset: 0, index: 2)
        e.setBuffer(out.buffer, offset: 0, index: 3)
        e.setThreadgroupMemoryLength(smem, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (outDim + nr0 - 1) / nr0, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: Int(nsg), depth: 1))
    }

    /// Dense Q8_0 matvec: out[outDim] = W(Q8_0)[outDim x inDim] * x[inDim].
    public func matmulQ8_0(weight: GPUTensor, x: GPUTensor, out: GPUTensor,
                           inDim: Int, outDim: Int) throws {
        precondition(inDim % 32 == 0)
        let nsg: Int16 = 4, nr0 = 2
        let rowBytes = (inDim / 32) * 34
        let args = MetalRuntime.mulMVArgs(ne00: inDim, ne01: outDim, nb00: 34, nb01: UInt64(rowBytes),
                                          nb02: UInt64(rowBytes * outDim), ne10: inDim, ne11: 1,
                                          nb10: 4, nb11: UInt64(inDim * 4), ne0: outDim, ne1: 1, nr0: Int32(nr0))
        let pso = try rt.mulMVPipeline("kernel_mul_mv_q8_0_f32", nsg: nsg)
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(weight.buffer, offset: weight.byteOffset, index: 1)   // byteOffset != 0 for no-copy mmap weights
        e.setBuffer(x.buffer, offset: 0, index: 2)
        e.setBuffer(out.buffer, offset: 0, index: 3)
        e.setThreadgroupMemoryLength(32 * 2 * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (outDim + nr0 - 1) / nr0, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: Int(nsg), depth: 1))
    }

    /// Elementwise add: out = a + b (width x rows).
    public func add(_ a: GPUTensor, _ b: GPUTensor, out: GPUTensor, width: Int, rows: Int = 1) throws {
        let args = MetalRuntime.binArgs(width: width, rows: rows, rhsWidth: width)
        let pso = try rt.binPipeline(op: .add)
        var nthMax = pso.maxTotalThreadsPerThreadgroup; if nthMax > 256 { nthMax = 256 }
        var nth = 1; while 2 * nth < width && nth < nthMax { nth *= 2 }
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(a.buffer, offset: 0, index: 1)
        e.setBuffer(b.buffer, offset: 0, index: 2)
        e.setBuffer(out.buffer, offset: 0, index: 3)
        e.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }

    /// SwiGLU single row: out = silu(gate)*up (optionally clamped/scaled).
    public func swiglu(gate: GPUTensor, up: GPUTensor, out: GPUTensor, n: Int,
                       alpha: Float = 1, limit: Float = 0) throws {
        let args = MetalRuntime.gluArgs(n: n, alpha: alpha, limit: limit)
        let pso = try rt.pipeline("kernel_swiglu_f32")
        var nth = pso.maxTotalThreadsPerThreadgroup
        let want = n > 1 ? n / 2 : 1
        if nth > want { nth = want }; if nth == 0 { nth = 1 }
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(gate.buffer, offset: 0, index: 1)
        e.setBuffer(up.buffer, offset: 0, index: 2)
        e.setBuffer(out.buffer, offset: 0, index: 3)
        e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }
}
