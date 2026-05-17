import Foundation
import Metal

/// Linear layer dispatching to BF16 / FP8 / FP4 / INT8 GEMM based on weight dtype.
/// Mirrors `linear()` in `Reference/inference/model.py` lines 108–120.
///
/// FP8 path:  act_quant(x) → fp8_gemm(x_q, x_s, w, w_s) → f32 out
/// FP4 path:  act_quant(x) → fp4_gemm(x_q, x_s, w, w_s) → f32 out
/// INT8 path: direct int8_w8a16 gemm with raw f32/bf16 activations, f32 out
/// Dense BF16/F32 path: direct GEMM, f32 out
public final class Linear {
    public let inFeatures: Int
    public let outFeatures: Int
    public let weight: Tensor
    public let scale: Tensor?

    private static let pBF16ToF32     = Device.shared.makePipeline("gemm_bf16_to_f32")
    private static let pBF16ToF32SG   = Device.shared.makePipeline("gemm_bf16_to_f32_sg")
    private static let pF32BF16       = Device.shared.makePipeline("gemm_f32_bf16_to_f32")
    private static let pF32BF16SG     = Device.shared.makePipeline("gemm_f32_bf16_to_f32_sg")
    private static let pF32           = Device.shared.makePipeline("gemm_f32_to_f32")
    private static let pFP8           = Device.shared.makePipeline("gemm_fp8_to_f32")
    private static let pFP4           = Device.shared.makePipeline("gemm_fp8_fp4_to_f32")
    private static let pInt8F32       = Device.shared.makePipeline("gemm_int8_w8a16_to_f32")
    private static let pInt8BF16      = Device.shared.makePipeline("gemm_int8_w8a16_bf16_to_f32")
    private static let pInt8F32SG     = Device.shared.makePipeline("gemm_int8_w8a16_to_f32_sg")
    private static let pInt8BF16SG    = Device.shared.makePipeline("gemm_int8_w8a16_bf16_to_f32_sg")
    private static let pInt4F32       = Device.shared.makePipeline("gemm_int4_w4a16_to_f32")
    private static let pInt4BF16      = Device.shared.makePipeline("gemm_int4_w4a16_bf16_to_f32")
    private static let pInt2F32       = Device.shared.makePipeline("gemm_int2_w2a16_to_f32")
    private static let pInt2BF16      = Device.shared.makePipeline("gemm_int2_w2a16_bf16_to_f32")
    // W8A8 path (int8 activations × int8 weights → f32). Pipeline statics
    // lazy-cached come tutti gli altri; usate solo quando
    // `useW8A8Activations == true`.
    private static let pInt8W8A8      = Device.shared.makePipeline("gemm_int8_w8a8_to_f32")
    private static let pInt8W8A8SG    = Device.shared.makePipeline("gemm_int8_w8a8_to_f32_sg")

    /// simdgroup_matrix GEMM produces 32×32 C blocks via 8×8 tiles; the
    /// K dimension is reduced in steps of 8. M, N must be multiples of 32
    /// and K a multiple of 8 — caller falls back to the legacy tiled
    /// kernel otherwise.
    private static func canUseSG(M: Int, N: Int, K: Int) -> Bool {
        return M >= 32 && N >= 32 && K >= 8
            && M % 32 == 0 && N % 32 == 0 && K % 8 == 0
    }

    /// When true (default), the GEMM output is round-tripped through BF16
    /// in-place before being returned. This mirrors the reference's
    /// `torch_dtype: bfloat16` propagation: every Linear in V4-Flash writes
    /// its output as BF16 (the FP32 accumulator is cast back to BF16
    /// before downstream ops read it). Set to `false` only for the LM
    /// head, where the logits feed straight into argmax / softmax and
    /// losing the low 16 mantissa bits is undesirable.
    public let castOutputToBF16: Bool

    /// Opt-in W8A8 (activations quantizzate a int8) per layer con peso
    /// int8. Quando true e `weight.dtype == .i8`, `int8Forward` quantizza
    /// l'input a int8 (via `ActQuant(format: .int8)`) e dispatcha sul
    /// kernel `gemm_int8_w8a8_*` invece del path standard W8A16.
    ///
    /// Trade-off: ~2× throughput memory-bound (metà bandwidth per leggere
    /// l'input), in cambio di un quantization-noise aggiuntivo sui token
    /// activations. Default off — opt-in per layer dove la perdita è
    /// accettabile (in genere i layer interni; LM head / output projection
    /// vanno lasciati a W8A16).
    public let useW8A8Activations: Bool

    public init(inFeatures: Int, outFeatures: Int, weight: Tensor, scale: Tensor?,
                castOutputToBF16: Bool = false,
                useW8A8Activations: Bool = false) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.weight = weight
        self.scale = scale
        self.castOutputToBF16 = castOutputToBF16
        self.useW8A8Activations = useW8A8Activations
    }

    /// `x`: [M, K] f32 or bf16. Output: [M, N] f32. When `castOutputToBF16`
    /// is true (default), the returned tensor is round-tripped through
    /// BF16 in-place so callers see the same precision the reference's
    /// BF16 activations would carry.
    public func callAsFunction(_ x: Tensor, in cmd: MTLCommandBuffer) -> Tensor {
        let M = x.shape.dropLast().reduce(1, *)
        let outShape = Array(x.shape.dropLast()) + [outFeatures]
        let y = Tensor.empty(shape: outShape, dtype: .f32)

        switch weight.dtype {
        case .fp4E2M1:
            guard let s = scale else { fatalError("FP4 Linear needs scale") }
            fp4Forward(x: x, y: y, M: M, scale: s, in: cmd)
        case .fp8E4M3:
            guard let s = scale else { fatalError("FP8 Linear needs scale") }
            fp8Forward(x: x, y: y, M: M, scale: s, in: cmd)
        case .i8:
            guard let s = scale else { fatalError("INT8 Linear needs F16 group scale") }
            int8Forward(x: x, y: y, M: M, scale: s, in: cmd)
        case .i4:
            guard let s = scale else { fatalError("INT4 Linear needs F16 group scale") }
            int4Forward(x: x, y: y, M: M, scale: s, in: cmd)
        case .i2:
            guard let s = scale else { fatalError("INT2 Linear needs F16 group scale") }
            int2Forward(x: x, y: y, M: M, scale: s, in: cmd)
        case .bf16:
            denseForward(x: x, y: y, M: M, pipelineForFloatX: Self.pF32BF16,
                         pipelineForBFloatX: Self.pBF16ToF32, in: cmd)
        case .f32:
            precondition(x.dtype == .f32, "f32 weight requires f32 input")
            dispatchGEMM(pipeline: Self.pF32, x: x, y: y, M: M, in: cmd)
        default:
            fatalError("Linear: unsupported weight dtype \(weight.dtype)")
        }
        if castOutputToBF16 {
            Elementwise.bf16RoundTripInplace(y, in: cmd)
        }
        return y
    }

    private func denseForward(x: Tensor, y: Tensor, M: Int,
                              pipelineForFloatX: MTLComputePipelineState,
                              pipelineForBFloatX: MTLComputePipelineState,
                              in cmd: MTLCommandBuffer) {
        // Pick simdgroup_matrix variant when M/N/K align; otherwise fall
        // back to the legacy tile-and-reduce kernel.
        let useSG = Self.canUseSG(M: M, N: outFeatures, K: inFeatures)
        let pipeline: MTLComputePipelineState
        switch x.dtype {
        case .f32:
            pipeline = useSG ? Self.pF32BF16SG : pipelineForFloatX
        case .bf16:
            pipeline = useSG ? Self.pBF16ToF32SG : pipelineForBFloatX
        default:
            fatalError("Linear dense: input dtype \(x.dtype) not supported")
        }
        dispatchGEMM(pipeline: pipeline, x: x, y: y, M: M, useSG: useSG, in: cmd)
    }

    private func dispatchGEMM(pipeline: MTLComputePipelineState,
                              x: Tensor, y: Tensor, M: Int,
                              useSG: Bool = false,
                              in cmd: MTLCommandBuffer) {
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        enc.setBuffer(y.buffer, offset: 0, index: 2)
        var dims = SIMD3<UInt32>(UInt32(M), UInt32(outFeatures), UInt32(inFeatures))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)

        if useSG {
            // simdgroup_matrix: one simdgroup (32 threads) per 32×32 C tile.
            let tg = MTLSize(width: 32, height: 1, depth: 1)
            let gx = outFeatures / 32
            let gy = M / 32
            enc.dispatchThreadgroups(MTLSize(width: gx, height: gy, depth: 1),
                                     threadsPerThreadgroup: tg)
        } else {
            let tg = MTLSize(width: 16, height: 16, depth: 1)
            let gx = (outFeatures + 15) / 16
            let gy = (M + 15) / 16
            enc.dispatchThreadgroups(MTLSize(width: gx, height: gy, depth: 1),
                                     threadsPerThreadgroup: tg)
        }
        enc.endEncoding()
    }

    private func fp8Forward(x: Tensor, y: Tensor, M: Int, scale wScale: Tensor,
                            in cmd: MTLCommandBuffer) {
        precondition(x.dtype == .f32, "FP8 Linear: input must be f32 for now")
        // Quantize activations to FP8 with per-128 block scales (NOT inplace).
        let aq = ActQuant(format: .fp8)
        let act = aq.quant(x.reshape([M, inFeatures]), inplace: false, in: cmd)
        guard let qbytes = act.qbytes else { fatalError("ActQuant did not produce qbytes") }

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(Self.pFP8)
        enc.setBuffer(qbytes.buffer, offset: 0, index: 0)
        enc.setBuffer(act.scales.buffer, offset: 0, index: 1)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 2)
        enc.setBuffer(wScale.buffer, offset: wScale.offset, index: 3)
        enc.setBuffer(y.buffer, offset: 0, index: 4)
        var dims = SIMD3<UInt32>(UInt32(M), UInt32(outFeatures), UInt32(inFeatures))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 5)

        let grid = MTLSize(width: outFeatures, height: M, depth: 1)
        enc.dispatchThreads(grid,
                            threadsPerThreadgroup: Self.pFP8.tunedThreadgroup(forGrid: grid))
        enc.endEncoding()
    }

    private func fp4Forward(x: Tensor, y: Tensor, M: Int, scale wScale: Tensor,
                            in cmd: MTLCommandBuffer) {
        precondition(x.dtype == .f32, "FP4 Linear: input must be f32 for now")
        // FP4 weights pair with FP8 activations (per the reference); use FP8 act_quant.
        let aq = ActQuant(format: .fp8)
        let act = aq.quant(x.reshape([M, inFeatures]), inplace: false, in: cmd)
        guard let qbytes = act.qbytes else { fatalError("ActQuant did not produce qbytes") }

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(Self.pFP4)
        enc.setBuffer(qbytes.buffer, offset: 0, index: 0)
        enc.setBuffer(act.scales.buffer, offset: 0, index: 1)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 2)
        enc.setBuffer(wScale.buffer, offset: wScale.offset, index: 3)
        enc.setBuffer(y.buffer, offset: 0, index: 4)
        var dims = SIMD3<UInt32>(UInt32(M), UInt32(outFeatures), UInt32(inFeatures))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 5)

        let grid = MTLSize(width: outFeatures, height: M, depth: 1)
        enc.dispatchThreads(grid,
                            threadsPerThreadgroup: Self.pFP4.tunedThreadgroup(forGrid: grid))
        enc.endEncoding()
    }

    /// W8A16 INT8 forward. Weight is `[N, K]` int8 with per-row/per-128
    /// F16 group scales `[N, K/128]`. Activations pass through as F32 or
    /// BF16 — no activation quantization. K must be a multiple of 128;
    /// this is guaranteed by `Int8Quant.shouldQuantizeToInt8` at convert
    /// time, but we also check here so a shape mismatch fails loudly
    /// rather than silently miscomputing.
    private func int8Forward(x: Tensor, y: Tensor, M: Int, scale wScale: Tensor,
                              in cmd: MTLCommandBuffer) {
        precondition(inFeatures % 128 == 0,
                     "INT8 Linear: inFeatures must be a multiple of 128 (got \(inFeatures))")
        // Opt-in W8A8: quantizza l'attivazione a int8 e usa il kernel
        // int8×int8. Vedi `useW8A8Activations` per il trade-off.
        if useW8A8Activations {
            int8W8A8Forward(x: x, y: y, M: M, scale: wScale, in: cmd)
            return
        }
        // simdgroup_matrix INT8 path: 32×32 C blocks. K alignment is already
        // guaranteed (inFeatures % 128 == 0 → K % 8 == 0).
        let useSG = M >= 32 && outFeatures >= 32
                 && M % 32 == 0 && outFeatures % 32 == 0
        let pipeline: MTLComputePipelineState
        switch x.dtype {
        case .f32:  pipeline = useSG ? Self.pInt8F32SG  : Self.pInt8F32
        case .bf16: pipeline = useSG ? Self.pInt8BF16SG : Self.pInt8BF16
        default:
            fatalError("INT8 Linear: input dtype \(x.dtype) not supported (need f32 or bf16)")
        }

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        enc.setBuffer(wScale.buffer, offset: wScale.offset, index: 2)
        enc.setBuffer(y.buffer, offset: 0, index: 3)
        var dims = SIMD3<UInt32>(UInt32(M), UInt32(outFeatures), UInt32(inFeatures))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 4)

        if useSG {
            let tg = MTLSize(width: 32, height: 1, depth: 1)
            let gx = outFeatures / 32
            let gy = M / 32
            enc.dispatchThreadgroups(MTLSize(width: gx, height: gy, depth: 1),
                                     threadsPerThreadgroup: tg)
        } else {
            let grid = MTLSize(width: outFeatures, height: M, depth: 1)
            enc.dispatchThreads(grid,
                                threadsPerThreadgroup: pipeline.tunedThreadgroup(forGrid: grid))
        }
        enc.endEncoding()
    }

    /// W8A8 INT8 forward (int8 activations × int8 weights → f32). Solo
    /// quando `useW8A8Activations == true`. Quantizza l'input via
    /// `ActQuant(format: .int8)` con block size 128 (deve coincidere col
    /// K-grouping dei pesi `kInt8GroupK = 128`) e dispatcha sul kernel
    /// naive o simdgroup secondo l'allineamento di M/N/K.
    ///
    /// Activations dtype: solo f32 supportato per ora (l'`act_quant_int8`
    /// kernel legge un buffer float). BF16 in input richiederebbe un
    /// kernel variante — TODO se la perf si rivela utile.
    private func int8W8A8Forward(x: Tensor, y: Tensor, M: Int, scale wScale: Tensor,
                                  in cmd: MTLCommandBuffer) {
        precondition(x.dtype == .f32,
                     "INT8 W8A8 Linear: input must be f32 (BF16 input non supportato — TODO)")
        precondition(inFeatures % Quant.actBlockSizeINT8 == 0,
                     "INT8 W8A8 Linear: inFeatures (\(inFeatures)) deve essere multiplo di \(Quant.actBlockSizeINT8)")

        // 1) Quantizza le activations a int8 + scales f32 [M, K/128].
        let aq = ActQuant(format: .int8)
        let act = aq.quant(x.reshape([M, inFeatures]), inplace: false, in: cmd)
        guard let qbytes = act.qbytes else {
            fatalError("ActQuant(.int8) did not produce qbytes")
        }

        // 2) Sceglie SG vs naive con gli stessi vincoli di W8A16.
        let useSG = M >= 32 && outFeatures >= 32
                 && M % 32 == 0 && outFeatures % 32 == 0
        let pipeline = useSG ? Self.pInt8W8A8SG : Self.pInt8W8A8

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(qbytes.buffer, offset: 0, index: 0)                      // A_q   int8
        enc.setBuffer(act.scales.buffer, offset: 0, index: 1)                  // A_sc  f32
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 2)          // W     int8
        enc.setBuffer(wScale.buffer, offset: wScale.offset, index: 3)          // W_sc  f16
        enc.setBuffer(y.buffer, offset: 0, index: 4)                           // C     f32
        var dims = SIMD3<UInt32>(UInt32(M), UInt32(outFeatures), UInt32(inFeatures))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 5)

        if useSG {
            let tg = MTLSize(width: 32, height: 1, depth: 1)
            let gx = outFeatures / 32
            let gy = M / 32
            enc.dispatchThreadgroups(MTLSize(width: gx, height: gy, depth: 1),
                                     threadsPerThreadgroup: tg)
        } else {
            let grid = MTLSize(width: outFeatures, height: M, depth: 1)
            enc.dispatchThreads(grid,
                                threadsPerThreadgroup: pipeline.tunedThreadgroup(forGrid: grid))
        }
        enc.endEncoding()
    }

    /// W4A16 INT4 forward. Weight is `[N, K/2]` packed nibbles with per-
    /// row/per-128 F16 group scales `[N, K/128]`. Same K%128 requirement
    /// as the INT8 path; the kernel is naive (one thread per output) — a
    /// simdgroup_matrix variant can be added later for prefill speed.
    private func int4Forward(x: Tensor, y: Tensor, M: Int, scale wScale: Tensor,
                              in cmd: MTLCommandBuffer) {
        precondition(inFeatures % 128 == 0,
                     "INT4 Linear: inFeatures must be a multiple of 128 (got \(inFeatures))")
        let pipeline: MTLComputePipelineState
        switch x.dtype {
        case .f32:  pipeline = Self.pInt4F32
        case .bf16: pipeline = Self.pInt4BF16
        default:
            fatalError("INT4 Linear: input dtype \(x.dtype) not supported (need f32 or bf16)")
        }

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        enc.setBuffer(wScale.buffer, offset: wScale.offset, index: 2)
        enc.setBuffer(y.buffer, offset: 0, index: 3)
        var dims = SIMD3<UInt32>(UInt32(M), UInt32(outFeatures), UInt32(inFeatures))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 4)

        let grid = MTLSize(width: outFeatures, height: M, depth: 1)
        enc.dispatchThreads(grid,
                            threadsPerThreadgroup: pipeline.tunedThreadgroup(forGrid: grid))
        enc.endEncoding()
    }

    /// W2A16 INT2 forward. Weight is `[N, K/4]` packed 2-bit values
    /// (LSB-first) with per-row/per-128 F16 group scales. K%128 enforced
    /// at quant time and rechecked here. Same naive dispatch as INT4.
    private func int2Forward(x: Tensor, y: Tensor, M: Int, scale wScale: Tensor,
                              in cmd: MTLCommandBuffer) {
        precondition(inFeatures % 128 == 0,
                     "INT2 Linear: inFeatures must be a multiple of 128 (got \(inFeatures))")
        let pipeline: MTLComputePipelineState
        switch x.dtype {
        case .f32:  pipeline = Self.pInt2F32
        case .bf16: pipeline = Self.pInt2BF16
        default:
            fatalError("INT2 Linear: input dtype \(x.dtype) not supported (need f32 or bf16)")
        }

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        enc.setBuffer(wScale.buffer, offset: wScale.offset, index: 2)
        enc.setBuffer(y.buffer, offset: 0, index: 3)
        var dims = SIMD3<UInt32>(UInt32(M), UInt32(outFeatures), UInt32(inFeatures))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 4)

        let grid = MTLSize(width: outFeatures, height: M, depth: 1)
        enc.dispatchThreads(grid,
                            threadsPerThreadgroup: pipeline.tunedThreadgroup(forGrid: grid))
        enc.endEncoding()
    }
}
