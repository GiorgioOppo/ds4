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

    /// Profiling (DS4_PROFILE_ROUTE): set to `[:]` to time sub-phases. `phase(name)`
    /// then flushes the buffer and accumulates the wall-clock under `name`. nil = off.
    public var phaseTimes: [String: Double]?
    private var phaseStart = Date()

    /// DS4_Q8_NSG: simdgroups-per-threadgroup for the dense Q8_0 matvec (and the
    /// grouped attn-output low-rank matvec, which shares the same K-split kernel).
    /// Default 4 — exactly the reference `ds4_gpu_make_q8_0_mv_dispatch()` config.
    ///
    /// This is a *pure work-partition* lever, not a math change: the kernel splits the
    /// reduction (K) dimension across NSG simdgroups and `helper_mv_reduce_and_write`
    /// sums across exactly NSG of them; the threadgroup-memory size (NW*NR0*4) and the
    /// grid width ((outDim+NR0-1)/NR0) are both independent of NSG. So any value in
    /// 1...8 produces identical results — only occupancy / DRAM-latency hiding changes.
    /// Exposed because the optimal NSG is hardware-specific (M1 Pro vs M2/M3) and can
    /// only be found by sweeping on-device; we can't benchmark here. Read once.
    static let q8NSG: Int16 = {
        if let v = ProcessInfo.processInfo.environment["DS4_Q8_NSG"].flatMap(Int.init),
           v >= 1, v <= 8 {
            return Int16(v)
        }
        return 4
    }()

    public init(_ rt: MetalRuntime) { self.rt = rt }

    public func begin() throws {
        guard let c = rt.queue.makeCommandBuffer(), let e = c.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        cb = c; enc = e
        phaseStart = Date()
    }

    /// Mark the end of a named sub-phase: when `phaseTimes` is enabled, commit+wait,
    /// record the elapsed (CPU-encode + GPU-exec) time under `name`, and start a
    /// fresh command buffer. A no-op when profiling is off. MUST be called outside
    /// any pushDebugGroup/popDebugGroup pair (it breaks the command buffer).
    public func phase(_ name: String) throws {
        guard phaseTimes != nil else { return }
        enc?.endEncoding(); cb?.commit(); cb?.waitUntilCompleted()
        phaseTimes![name, default: 0] += Date().timeIntervalSince(phaseStart)
        guard let c = rt.queue.makeCommandBuffer(), let e = c.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        cb = c; enc = e
        phaseStart = Date()
    }

    /// Flush: end encoding, commit, wait. After this the GPUTensor outputs are readable.
    public func commit() {
        enc?.endEncoding()
        cb?.commit()
        cb?.waitUntilCompleted()
        enc = nil; cb = nil
    }

    /// End encoding and commit WITHOUT waiting — pair with waitCompleted().
    /// Lets CPU work (e.g. the expert-gather SSD I/O) overlap the GPU execution.
    /// No further encodes are allowed on this context after this call.
    public func commitAsync() {
        enc?.endEncoding()
        cb?.commit()
        enc = nil
    }

    /// Wait for a commitAsync()'d command buffer; outputs are readable after this.
    /// Safe to call more than once (idempotent).
    public func waitCompleted() {
        cb?.waitUntilCompleted()
        cb = nil
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
        let nsg = GraphContext.q8NSG, nr0 = 2   // nsg tunable via DS4_Q8_NSG (default 4 = reference)
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
