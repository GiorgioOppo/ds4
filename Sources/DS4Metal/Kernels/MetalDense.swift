import Foundation
import Metal

// Phase 9 (first kernel group): the dense Q8_0 matrix-vector multiply, the
// workhorse decode op (shared experts, output-side projections). Faithful port
// of the n_tok==1 path of ds4_gpu_matmul_q8_0_legacy_tensor in ds4_metal.m:
// same kernel (kernel_mul_mv_q8_0_f32), same args struct layout
// (ds4_metal_args_mul_mv, 112 bytes), same function constant (FC_mul_mv_nsg=600
// -> nsg=4), same nr0=2, threadgroup memory (32*2*float), grid and threadgroup
// dimensions.

extension MetalRuntime {
    /// Create a pipeline with the mul_mv NSG function constant (index 600),
    /// mirroring ds4_gpu_get_mul_mv_pipeline.
    public func mulMVPipeline(_ name: String, nsg: Int16) throws -> MTLComputePipelineState {
        let cacheKey = "\(name)_nsg=\(nsg)"
        if let p = mulMVPipelineCache[cacheKey] { return p }
        let constants = MTLFunctionConstantValues()
        var n = nsg
        constants.setConstantValue(&n, type: .short, index: 600)
        let fn = try library.makeFunction(name: name, constantValues: constants)
        let pso = try device.makeComputePipelineState(function: fn)
        mulMVPipelineCache[cacheKey] = pso
        return pso
    }

    /// Dense Q8_0 matrix-vector multiply: out[outDim] = W[outDim x inDim] * x[inDim].
    /// `weight` holds outDim rows of Q8_0 blocks (each 34 bytes per 32 elements).
    public func matmulQ8_0(weight: [UInt8], activation: [Float],
                           inDim: Int, outDim: Int) throws -> [Float] {
        precondition(inDim % 32 == 0, "Q8_0 in_dim must be a multiple of 32")
        let rowBytes = (inDim / 32) * 34
        precondition(weight.count >= rowBytes * outDim)
        precondition(activation.count >= inDim)

        let nsg: Int16 = 4
        let nr0 = 2
        let args = Self.mulMVArgs(ne00: inDim, ne01: outDim,
                                  nb00: 34, nb01: UInt64(rowBytes),
                                  nb02: UInt64(rowBytes * outDim),
                                  ne10: inDim, ne11: 1,
                                  nb10: 4, nb11: UInt64(inDim * 4),
                                  ne0: outDim, ne1: 1, nr0: Int32(nr0))

        guard let wbuf = device.makeBuffer(bytes: weight, length: rowBytes * outDim, options: .storageModeShared),
              let xbuf = device.makeBuffer(bytes: activation, length: inDim * MemoryLayout<Float>.size, options: .storageModeShared),
              let obuf = device.makeBuffer(length: outDim * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }

        let pso = try mulMVPipeline("kernel_mul_mv_q8_0_f32", nsg: nsg)
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(wbuf, offset: 0, index: 1)
        enc.setBuffer(xbuf, offset: 0, index: 2)
        enc.setBuffer(obuf, offset: 0, index: 3)
        enc.setThreadgroupMemoryLength(32 * 2 * MemoryLayout<Float>.size, index: 0)
        let groups = MTLSize(width: (outDim + nr0 - 1) / nr0, height: 1, depth: 1)
        let tpt = MTLSize(width: 32, height: Int(nsg), depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tpt)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = obuf.contents().bindMemory(to: Float.self, capacity: outDim)
        return Array(UnsafeBufferPointer(start: p, count: outDim))
    }

    /// Dense F16 matrix-vector multiply: out[outDim] = W[outDim x inDim] * x[inDim].
    /// `weight` holds outDim rows of inDim half (uint16) values. Faithful port of
    /// the n_tok==1 path of ds4_gpu_matmul_f16_tensor (kernel kernel_mul_mv_f16_f32_4,
    /// nsg = min(8,(inDim+127)/128), nr0=2, with the out_dim 512/1024 & inDim>=4096
    /// nr0=4 special case).
    public func matmulF16(weight: [UInt16], activation: [Float],
                          inDim: Int, outDim: Int) throws -> [Float] {
        precondition(inDim % 4 == 0, "F16 _4 matvec needs in_dim multiple of 4")
        precondition(weight.count >= inDim * outDim)
        precondition(activation.count >= inDim)

        let nsg = Int16(min(8, (inDim + 127) / 128))
        var nr0 = 2
        var smem = 32 * 2 * MemoryLayout<Float>.size
        if (outDim == 512 || outDim == 1024) && inDim >= 4096 {
            nr0 = 4
            smem = 32 * 4 * MemoryLayout<Float>.size
        }
        let rowBytes = inDim * 2
        let args = Self.mulMVArgs(ne00: inDim, ne01: outDim,
                                  nb00: 2, nb01: UInt64(rowBytes),
                                  nb02: UInt64(rowBytes * outDim),
                                  ne10: inDim, ne11: 1,
                                  nb10: 4, nb11: UInt64(inDim * 4),
                                  ne0: outDim, ne1: 1, nr0: Int32(nr0))

        guard let wbuf = weight.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: rowBytes * outDim, options: .storageModeShared) }),
              let xbuf = device.makeBuffer(bytes: activation, length: inDim * MemoryLayout<Float>.size, options: .storageModeShared),
              let obuf = device.makeBuffer(length: outDim * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }

        let pso = try mulMVPipeline("kernel_mul_mv_f16_f32_4", nsg: nsg)
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(wbuf, offset: 0, index: 1)
        enc.setBuffer(xbuf, offset: 0, index: 2)
        enc.setBuffer(obuf, offset: 0, index: 3)
        enc.setThreadgroupMemoryLength(smem, index: 0)
        let groups = MTLSize(width: (outDim + nr0 - 1) / nr0, height: 1, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: 32, height: Int(nsg), depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = obuf.contents().bindMemory(to: Float.self, capacity: outDim)
        return Array(UnsafeBufferPointer(start: p, count: outDim))
    }

    /// Paired F16 matvec (gate+up in one dispatch): out_a = Wa*x, out_b = Wb*x.
    /// Faithful port of ds4_gpu_matmul_f16_pair_tensor (kernel
    /// kernel_mul_mv_f16_f32_pair_4, same 112-byte mul_mv args, nsg =
    /// min(8,(inDim+127)/128), nr0=2; weights at buffer index 1/2, x at 3,
    /// out_a/out_b at 4/5).
    public func matmulF16Pair(weightA: [UInt16], weightB: [UInt16], activation: [Float],
                              inDim: Int, outDim: Int) throws -> (a: [Float], b: [Float]) {
        precondition(inDim % 4 == 0)
        precondition(weightA.count >= inDim * outDim && weightB.count >= inDim * outDim)
        precondition(activation.count >= inDim)

        let nsg = Int16(min(8, (inDim + 127) / 128))
        let nr0 = 2
        let smem = 32 * 2 * MemoryLayout<Float>.size
        let rowBytes = inDim * 2
        let args = Self.mulMVArgs(ne00: inDim, ne01: outDim,
                                  nb00: 2, nb01: UInt64(rowBytes),
                                  nb02: UInt64(rowBytes * outDim),
                                  ne10: inDim, ne11: 1,
                                  nb10: 4, nb11: UInt64(inDim * 4),
                                  ne0: outDim, ne1: 1, nr0: Int32(nr0))

        guard let wabuf = weightA.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: rowBytes * outDim, options: .storageModeShared) }),
              let wbbuf = weightB.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: rowBytes * outDim, options: .storageModeShared) }),
              let xbuf = device.makeBuffer(bytes: activation, length: inDim * 4, options: .storageModeShared),
              let oabuf = device.makeBuffer(length: outDim * 4, options: .storageModeShared),
              let obbuf = device.makeBuffer(length: outDim * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }

        let pso = try mulMVPipeline("kernel_mul_mv_f16_f32_pair_4", nsg: nsg)
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(wabuf, offset: 0, index: 1)
        enc.setBuffer(wbbuf, offset: 0, index: 2)
        enc.setBuffer(xbuf, offset: 0, index: 3)
        enc.setBuffer(oabuf, offset: 0, index: 4)
        enc.setBuffer(obbuf, offset: 0, index: 5)
        enc.setThreadgroupMemoryLength(smem, index: 0)
        let groups = MTLSize(width: (outDim + nr0 - 1) / nr0, height: 1, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: 32, height: Int(nsg), depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let pa = oabuf.contents().bindMemory(to: Float.self, capacity: outDim)
        let pb = obbuf.contents().bindMemory(to: Float.self, capacity: outDim)
        return (Array(UnsafeBufferPointer(start: pa, count: outDim)),
                Array(UnsafeBufferPointer(start: pb, count: outDim)))
    }

    /// Dense F32 matrix-vector multiply: out[outDim] = W[outDim x inDim] * x[inDim].
    /// `weight` holds outDim rows of inDim F32 values. Faithful port of the
    /// n_tok==1 plain path (ds4_gpu_make_plain_mv_dispatch, f32_weights=1):
    /// kernel_mul_mv_f32_f32_4, nsg = min(8,(inDim+127)/128), nr0=2, smem 256.
    /// Used by the output head (4K -> vocab logits).
    public func matmulF32(weight: [Float], activation: [Float],
                          inDim: Int, outDim: Int) throws -> [Float] {
        precondition(inDim % 4 == 0, "F32 _4 matvec needs in_dim multiple of 4")
        precondition(weight.count >= inDim * outDim)
        precondition(activation.count >= inDim)

        let nsg = Int16(min(8, (inDim + 127) / 128))
        let nr0 = 2
        let smem = 32 * 2 * MemoryLayout<Float>.size
        let rowBytes = inDim * 4
        let args = Self.mulMVArgs(ne00: inDim, ne01: outDim,
                                  nb00: 4, nb01: UInt64(rowBytes),
                                  nb02: UInt64(rowBytes * outDim),
                                  ne10: inDim, ne11: 1,
                                  nb10: 4, nb11: UInt64(inDim * 4),
                                  ne0: outDim, ne1: 1, nr0: Int32(nr0))

        guard let wbuf = device.makeBuffer(bytes: weight, length: rowBytes * outDim, options: .storageModeShared),
              let xbuf = device.makeBuffer(bytes: activation, length: inDim * 4, options: .storageModeShared),
              let obuf = device.makeBuffer(length: outDim * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }

        let pso = try mulMVPipeline("kernel_mul_mv_f32_f32_4", nsg: nsg)
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(wbuf, offset: 0, index: 1)
        enc.setBuffer(xbuf, offset: 0, index: 2)
        enc.setBuffer(obuf, offset: 0, index: 3)
        enc.setThreadgroupMemoryLength(smem, index: 0)
        let groups = MTLSize(width: (outDim + nr0 - 1) / nr0, height: 1, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: 32, height: Int(nsg), depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = obuf.contents().bindMemory(to: Float.self, capacity: outDim)
        return Array(UnsafeBufferPointer(start: p, count: outDim))
    }

    /// Build the 112-byte ds4_metal_args_mul_mv buffer with explicit offsets
    /// matching the C/Metal natural layout (int32 + uint64 + int16 fields).
    static func mulMVArgs(ne00: Int, ne01: Int, nb00: UInt64, nb01: UInt64, nb02: UInt64,
                          ne10: Int, ne11: Int, nb10: UInt64, nb11: UInt64,
                          ne0: Int, ne1: Int, nr0: Int32) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 112)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func i16(_ off: Int, _ v: Int16) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<2 { b[off+k] = $0[k] } } }
        i32(0, Int32(ne00)); i32(4, Int32(ne01)); i32(8, 1)               // ne00, ne01, ne02
        u64(16, nb00); u64(24, nb01); u64(32, nb02); u64(40, nb02)        // nb00..nb03 (nb03==nb02)
        i32(48, Int32(ne10)); i32(52, Int32(ne11)); i32(56, 1)           // ne10, ne11, ne12
        u64(64, nb10); u64(72, nb11); u64(80, nb11); u64(88, nb11)       // nb10..nb13
        i32(96, Int32(ne0)); i32(100, Int32(ne1)); i32(104, nr0)         // ne0, ne1, nr0
        i16(108, 1); i16(110, 1)                                          // r2, r3
        return b
    }
}
