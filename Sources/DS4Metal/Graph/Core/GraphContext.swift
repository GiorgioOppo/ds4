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
    /// This is a work-partition lever, but it splits the K reduction across a
    /// different number of partial sums. All values compute the same operation,
    /// yet Float32 association can change the last bits. Therefore the exact GUI
    /// tuner leaves it fixed; only the opt-in numeric process profile sweeps it.
    /// Threadgroup-memory size and grid width remain independent of NSG, so the
    /// useful effect is occupancy / DRAM-latency hiding on different Apple GPUs.
    ///
    /// Re-read from the env at every DECODER creation (StreamingDecoder.init calls
    /// `refreshQ8NSG`), NOT at every dispatch: a model reload — e.g. the Settings
    /// auto-tune — can sweep it without restarting the process, while the hot
    /// encode path keeps a plain read (ProcessInfo.environment allocates a fresh
    /// dictionary per call — far too costly per-matvec).
    /// nonisolated(unsafe): written only single-threaded at decoder creation,
    /// before any decode thread encodes; readers see a stable value per load.
    nonisolated(unsafe) static var q8NSG: Int16 = GraphContext.readQ8NSG()

    static func readQ8NSG() -> Int16 {
        if let v = ProcessInfo.processInfo.environment["DS4_Q8_NSG"].flatMap(Int.init),
           v >= 1, v <= 8 {
            return Int16(v)
        }
        return 4
    }

    /// Re-arm `q8NSG` from the live environment (called on decoder creation).
    static func refreshQ8NSG() {
        q8NSG = readQ8NSG()
        moeNSG = readMoENSG()
        denseQ4NSG = readDenseQ4NSG()
        fusedCompressorProj = ProcessInfo.processInfo.environment["DS4_FUSED_COMP_PROJ"] != "0"
        adaptiveSplitK = ProcessInfo.processInfo.environment["DS4_ADAPTIVE_SPLITK"] != "0"
        vectorCopies = ProcessInfo.processInfo.environment["DS4_VECTOR_COPY"] == "1"
        fusedFlashKVStage = ProcessInfo.processInfo.environment["DS4_FLASH_KV_STAGE"] == "1"
    }

    /// DS4_ADAPTIVE_SPLITK (default on; `=0` restores the fixed dispatch):
    /// adaptive split-K depth for the decode flash-attention. Historically the
    /// vec kernel always dispatched nwg=32 workgroups in depth; at short
    /// context most of them exit their chunk loop immediately but still write
    /// a full DV=512 partial (+S/M) that the reduce reads back — dead traffic
    /// that scales with nHead*nwg, not with the context. The adaptive dispatch
    /// keeps every ACTIVE workgroup on exactly the same chunks (bit-for-bit
    /// partials) and drops only the empty ones; see flashAttnCore. Same
    /// refresh discipline as q8NSG (re-read per decoder creation).
    nonisolated(unsafe) static var adaptiveSplitK =
        ProcessInfo.processInfo.environment["DS4_ADAPTIVE_SPLITK"] != "0"

    /// DS4_VECTOR_COPY (`=1` opt-in): use
    /// packed four-element kernels for contiguous F32/F16 conversion.  This is
    /// an exact layout specialization: only index reconstruction and dispatch
    /// width change. The M1 Pro end-to-end A/B was decode-neutral/slightly
    /// negative, so the measured generic path remains the default.
    nonisolated(unsafe) static var vectorCopies =
        ProcessInfo.processInfo.environment["DS4_VECTOR_COPY"] == "1"

    /// DS4_FLASH_KV_STAGE (`=1` opt-in): gather the raw circular KV window and
    /// compressed F32 rows into FlashAttention's contiguous F16 scratch in one
    /// dispatch; a partial final block is padded by the same dispatch.  Kept
    /// opt-in because the order-balanced M1 Pro A/B found a small prefill gain
    /// but decode within run-order noise. Re-read once per decoder creation with
    /// the other tuning knobs.
    nonisolated(unsafe) static var fusedFlashKVStage =
        ProcessInfo.processInfo.environment["DS4_FLASH_KV_STAGE"] == "1"

    /// DS4_MOE_NSG: simdgroups-per-threadgroup for the MoE id-kernels — the
    /// routed FFN (pair_swiglu/sum6, ~100 ms/token measured on M1 Pro, the
    /// single biggest compute item) AND the resident dense-Q4 matvecs, which
    /// reuse the same row kernel. Unlike DS4_Q8_NSG (a K-split), here NSG
    /// partitions OUTPUT ROWS across simdgroups: each row is computed by
    /// exactly one simdgroup regardless of NSG, so any value in 1...8 is
    /// bit-identical — only occupancy changes. The iq2_xxs codebook staging
    /// is bounds-guarded and its threadgroup memory is NSG-independent
    /// (fixed 256×8+128 table). Default 4 = the historical dispatch.
    /// Same refresh discipline as q8NSG (re-read per decoder creation).
    nonisolated(unsafe) static var moeNSG: Int16 = GraphContext.readMoENSG()

    static func readMoENSG() -> Int16 {
        if let v = ProcessInfo.processInfo.environment["DS4_MOE_NSG"].flatMap(Int.init),
           v >= 1, v <= 8 {
            return Int16(v)
        }
        return 4
    }

    /// Dense resident Q4 projections have very different output geometry and
    /// memory behavior from the routed IQ2/Q2 expert kernels. Keep their row
    /// occupancy independently tunable instead of forcing both workloads to
    /// share DS4_MOE_NSG. Unset inherits MOE_NSG for backwards compatibility.
    nonisolated(unsafe) static var denseQ4NSG: Int16 = GraphContext.readDenseQ4NSG()

    static func readDenseQ4NSG() -> Int16 {
        if let v = ProcessInfo.processInfo.environment["DS4_DENSE_Q4_NSG"].flatMap(Int.init),
           v >= 1, v <= 8 {
            return Int16(v)
        }
        return readMoENSG()
    }

    /// Pair the attention/indexer compressor KV+gate F16 projections. Both use
    /// the same activation and shape, so one dispatch can share activation
    /// loads while preserving each matrix's original reduction order.
    nonisolated(unsafe) static var fusedCompressorProj =
        ProcessInfo.processInfo.environment["DS4_FUSED_COMP_PROJ"] != "0"

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

    /// Errore dell'ULTIMO command buffer committato (nil = ok). Un fault GPU
    /// completa il cb "con errore" senza alcun throw: i chiamanti che
    /// serializzano il risultato altrove (expert shard verticale) devono
    /// controllarlo dopo commit() — altrimenti spedirebbero byte stantii.
    public private(set) var lastError: Error?

    /// Flush: end encoding, commit, wait. After this the GPUTensor outputs are readable.
    public func commit() {
        enc?.endEncoding()
        cb?.commit()
        cb?.waitUntilCompleted()
        lastError = cb?.error
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

    /// Coalesced buffer-to-buffer copies via ONE blit encoder in the middle of
    /// the command buffer (the compute encoder is closed and reopened around
    /// it; encoder boundaries order the copies against the surrounding
    /// dispatches). Used by the batched prefill route phase to snapshot each
    /// token's scratch (FFN inputs + router selection) into per-token buffers
    /// BEFORE the next token's dispatches overwrite it — GPU-side, no commit.
    public func blitCopies(_ copies: [(src: GPUTensor, srcOff: Int, dst: GPUTensor, dstOff: Int, bytes: Int)]) throws {
        enc?.endEncoding()
        guard let b = cb?.makeBlitCommandEncoder() else { throw MetalError.bufferAlloc }
        for c in copies {
            b.copy(from: c.src.buffer, sourceOffset: c.src.byteOffset + c.srcOff,
                   to: c.dst.buffer, destinationOffset: c.dst.byteOffset + c.dstOff, size: c.bytes)
        }
        b.endEncoding()
        guard let e = cb?.makeComputeCommandEncoder() else { throw MetalError.bufferAlloc }
        enc = e
    }

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
        e.setBuffer(x.buffer, offset: x.byteOffset, index: 1)
        let w = weight ?? x
        e.setBuffer(w.buffer, offset: w.byteOffset, index: 2)
        e.setBuffer(x.buffer, offset: x.byteOffset, index: 3)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 4)
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
        e.setBuffer(x.buffer, offset: x.byteOffset, index: 2)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 3)
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
        e.setBuffer(x.buffer, offset: x.byteOffset, index: 2)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 3)
        e.setThreadgroupMemoryLength(smem, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (outDim + nr0 - 1) / nr0, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: Int(nsg), depth: 1))
    }

    /// Two same-shaped dense F16 matvecs sharing one activation stream and one
    /// dispatch. Each output keeps the scalar operation/reduction order of
    /// `matmulF16`, so this is suitable for numerical A/B parity.
    public func matmulF16Pair(weightA: GPUTensor, weightB: GPUTensor, x: GPUTensor,
                              outA: GPUTensor, outB: GPUTensor,
                              inDim: Int, outDim: Int) throws {
        precondition(inDim % 4 == 0)
        let nsg = Int16(min(8, (inDim + 127) / 128))
        var nr0 = 2, smem = 32 * 2 * 4
        if (outDim == 512 || outDim == 1024) && inDim >= 4096 {
            nr0 = 4; smem = 32 * 4 * 4
        }
        let rowBytes = inDim * 2
        let args = MetalRuntime.mulMVArgs(ne00: inDim, ne01: outDim, nb00: 2,
                                          nb01: UInt64(rowBytes), nb02: UInt64(rowBytes * outDim),
                                          ne10: inDim, ne11: 1, nb10: 4,
                                          nb11: UInt64(inDim * 4), ne0: outDim, ne1: 1,
                                          nr0: Int32(nr0))
        let pso = try rt.mulMVPipeline("kernel_mul_mv_f16_f32_pair_4", nsg: nsg)
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(weightA.buffer, offset: weightA.byteOffset, index: 1)
        e.setBuffer(weightB.buffer, offset: weightB.byteOffset, index: 2)
        e.setBuffer(x.buffer, offset: x.byteOffset, index: 3)
        e.setBuffer(outA.buffer, offset: outA.byteOffset, index: 4)
        e.setBuffer(outB.buffer, offset: outB.byteOffset, index: 5)
        e.setThreadgroupMemoryLength(smem, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (outDim + nr0 - 1) / nr0,
                                       height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32,
                                                              height: Int(nsg), depth: 1))
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
        e.setBuffer(x.buffer, offset: x.byteOffset, index: 2)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 3)
        e.setThreadgroupMemoryLength(32 * 2 * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (outDim + nr0 - 1) / nr0, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: Int(nsg), depth: 1))
    }

    /// Paired Q8_0 matvec for the compressor KV+gate projections. Both dot
    /// products retain the standalone kernel's accumulation order while one
    /// dispatch shares activation loads.
    public func matmulQ8_0Pair(weightA: GPUTensor, weightB: GPUTensor, x: GPUTensor,
                               outA: GPUTensor, outB: GPUTensor,
                               inDim: Int, outDim: Int) throws {
        precondition(inDim % 32 == 0)
        let nsg = GraphContext.q8NSG, nr0 = 2
        let rowBytes = (inDim / 32) * 34
        let args = MetalRuntime.mulMVArgs(ne00: inDim, ne01: outDim, nb00: 34,
                                          nb01: UInt64(rowBytes), nb02: UInt64(rowBytes * outDim),
                                          ne10: inDim, ne11: 1, nb10: 4,
                                          nb11: UInt64(inDim * 4), ne0: outDim, ne1: 1,
                                          nr0: Int32(nr0))
        let pso = try rt.mulMVPipeline("kernel_mul_mv_q8_0_f32_pair", nsg: nsg)
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(weightA.buffer, offset: weightA.byteOffset, index: 1)
        e.setBuffer(weightB.buffer, offset: weightB.byteOffset, index: 2)
        e.setBuffer(x.buffer, offset: x.byteOffset, index: 3)
        e.setBuffer(outA.buffer, offset: outA.byteOffset, index: 4)
        e.setBuffer(outB.buffer, offset: outB.byteOffset, index: 5)
        e.setThreadgroupMemoryLength(32 * 2 * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: (outDim + nr0 - 1) / nr0, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: Int(nsg), depth: 1))
    }

    /// Dense Q4_K matvec specialized for a single resident matrix. It invokes
    /// the same Q4_K row/dequant implementation as the MoE path, so arithmetic
    /// and reduction order remain unchanged while the synthetic expert id is
    /// removed.
    public func matmulQ4_K(weight: GPUTensor, x: GPUTensor, out: GPUTensor,
                           inDim: Int, outDim: Int) throws {
        precondition(inDim % 256 == 0)
        let nsg = GraphContext.denseQ4NSG, nr0 = 2
        let rowBytes = (inDim / 256) * 144
        let args = MetalRuntime.mulMVArgs(ne00: inDim, ne01: outDim, nb00: 144,
                                          nb01: UInt64(rowBytes),
                                          nb02: UInt64(rowBytes * outDim),
                                          ne10: inDim, ne11: 1,
                                          nb10: 4, nb11: UInt64(inDim * 4),
                                          ne0: outDim, ne1: 1, nr0: Int32(nr0))
        let pso = try rt.mulMVPipeline("kernel_mul_mv_q4_K_f32", nsg: nsg)
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(weight.buffer, offset: weight.byteOffset, index: 1)
        e.setBuffer(x.buffer, offset: x.byteOffset, index: 2)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 3)
        e.setThreadgroupMemoryLength(256, index: 0)
        e.dispatchThreadgroups(
            MTLSize(width: (outDim + Int(nsg) * nr0 - 1) / (Int(nsg) * nr0),
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: Int(nsg), depth: 1))
    }

    /// Decode router probability transform in one vectorized pass.
    public func routerProbabilities(logits: GPUTensor, probabilities: GPUTensor,
                                    width: Int) throws {
        precondition(width % 4 == 0)
        let args = MetalRuntime.unaryArgs(width: width, rows: 1,
                                          minV: 0, maxV: 0, scale: 0, bias: 0, val: 0)
        let pso = try rt.pipeline("kernel_dsv4_softplus_sqrt_f32_4")
        let vectors = width / 4
        let nth = min(vectors, pso.maxTotalThreadsPerThreadgroup)
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(logits.buffer, offset: logits.byteOffset, index: 1)
        e.setBuffer(probabilities.buffer, offset: probabilities.byteOffset, index: 2)
        e.dispatchThreadgroups(MTLSize(width: (vectors + nth - 1) / nth, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
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
        e.setBuffer(a.buffer, offset: a.byteOffset, index: 1)
        e.setBuffer(b.buffer, offset: b.byteOffset, index: 2)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 3)
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
        e.setBuffer(gate.buffer, offset: gate.byteOffset, index: 1)
        e.setBuffer(up.buffer, offset: up.byteOffset, index: 2)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 3)
        e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }
}
