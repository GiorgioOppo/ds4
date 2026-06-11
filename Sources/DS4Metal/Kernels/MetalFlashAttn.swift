import Foundation
import Metal
import DS4Core

// Phase 9 / Stage A3: decode FlashAttention (the MLA dk=dv=512 path). Faithful
// port of the ds4_gpu_attention_decode_heads vec+reduce dispatch using the real
// metal/flash_attn.metal kernels kernel_flash_attn_ext_vec_f16_dk512_dv512 and
// kernel_flash_attn_ext_vec_reduce. K and V are the same 512-dim latent buffer
// (MLA). This wrapper covers the no-kv-pad case (n_keys % 32 == 0); the partial
// last-block pad kernel is a separate variant.

extension MetalRuntime {
    func flashVecPipeline(nsg: Int32, nwg: Int32, hasSinks: Bool, hasKvpad kvpad: Bool = false) throws -> MTLComputePipelineState {
        let key = "flash_vec_nsg=\(nsg)_nwg=\(nwg)_sinks=\(hasSinks)_kvpad=\(kvpad)"
        if let p = mulMVPipelineCache[key] { return p }
        let c = MTLFunctionConstantValues()
        var hasMask = true, hs = hasSinks, hasBias = false, hasScap = false, hasKvpad = kvpad
        var ns10: Int32 = 512, ns20: Int32 = 512, nsgv = nsg, nwgv = nwg
        c.setConstantValue(&hasMask, type: .bool, index: 400)
        c.setConstantValue(&hs, type: .bool, index: 401)
        c.setConstantValue(&hasBias, type: .bool, index: 402)
        c.setConstantValue(&hasScap, type: .bool, index: 403)
        c.setConstantValue(&hasKvpad, type: .bool, index: 404)
        c.setConstantValue(&ns10, type: .int, index: 420)
        c.setConstantValue(&ns20, type: .int, index: 421)
        c.setConstantValue(&nsgv, type: .int, index: 422)
        c.setConstantValue(&nwgv, type: .int, index: 423)
        let fn = try library.makeFunction(name: "kernel_flash_attn_ext_vec_f16_dk512_dv512", constantValues: c)
        let pso = try device.makeComputePipelineState(function: fn)
        mulMVPipelineCache[key] = pso
        return pso
    }

    func flashPadPipeline(ncpsg: Int32) throws -> MTLComputePipelineState {
        let key = "flash_pad_ncpsg=\(ncpsg)"
        if let p = mulMVPipelineCache[key] { return p }
        let c = MTLFunctionConstantValues()
        var hasMask = true, n = ncpsg
        c.setConstantValue(&hasMask, type: .bool, index: 100)
        c.setConstantValue(&n, type: .int, index: 125)
        let fn = try library.makeFunction(name: "kernel_flash_attn_ext_pad", constantValues: c)
        let pso = try device.makeComputePipelineState(function: fn)
        mulMVPipelineCache[key] = pso
        return pso
    }

    /// 104-byte ds4_gpu_flash_attn_pad_args (decode, dk=dv=512, single plane).
    static func flashPadArgs(nKeys: Int, headDim: Int) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 104)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        let rowF16 = UInt64(headDim) * 2
        let plane = UInt64(nKeys) * rowF16
        let maskBytes = UInt64(nKeys) * 2
        i32(0, Int32(nKeys)); i32(4, 1); i32(8, 1)                       // ne11, ne_12_2, ne_12_3
        u64(16, rowF16); u64(24, plane); u64(32, plane)                  // nb11, nb12, nb13
        u64(40, rowF16); u64(48, plane); u64(56, plane)                  // nb21, nb22, nb23
        i32(64, 1); i32(68, 1); i32(72, 1)                              // ne31, ne32, ne33
        u64(80, maskBytes); u64(88, maskBytes); u64(96, maskBytes)      // nb31, nb32, nb33
        return b
    }

    func flashReducePipeline(dv: Int32, nwg: Int32) throws -> MTLComputePipelineState {
        let key = "flash_reduce_dv=\(dv)_nwg=\(nwg)"
        if let p = mulMVPipelineCache[key] { return p }
        let c = MTLFunctionConstantValues()
        var dvv = dv, nwgv = nwg
        c.setConstantValue(&dvv, type: .int, index: 500)
        c.setConstantValue(&nwgv, type: .int, index: 501)
        let fn = try library.makeFunction(name: "kernel_flash_attn_ext_vec_reduce", constantValues: c)
        let pso = try device.makeComputePipelineState(function: fn)
        mulMVPipelineCache[key] = pso
        return pso
    }

    /// Decode attention for one query position. `q` is nHead*512 F32, `kv` is the
    /// nKeys*512 F32 latent (used as both K and V). Returns nHead*512 F32. No mask
    /// (all keys visible). Requires nKeys % 32 == 0.
    public func flashAttnDecode(q: [Float], kv: [Float], nHead: Int, nKeys: Int) throws -> [Float] {
        let headDim = 512
        precondition(nKeys % 32 == 0, "this wrapper covers the no-pad case")
        precondition(q.count >= nHead * headDim && kv.count >= nKeys * headDim)

        let ncpsg = 32, nwg = 32
        var nsg = 1
        while 2 * nwg * nsg * ncpsg < nKeys && nsg < 4 { nsg *= 2 }

        // K/V as F16 (same values the cpy kernel would produce).
        var kvF16 = [UInt16](repeating: 0, count: nKeys * headDim)
        for i in 0..<kvF16.count { kvF16[i] = Half.bits(kv[i]) }   // portable f32→f16 (arch-safe)

        let rowBytes = headDim * 4
        let rowBytesF16 = headDim * 2
        let maskBytes = nKeys * 2
        let padBytes = 2 * ncpsg * rowBytesF16 + ncpsg * 2
        let tmpBytes = nHead * headDim * nwg * 4 + nHead * (2 * nwg) * 4

        let scale = 1.0 / Float(headDim).squareRoot()
        let args = Self.flashVecArgs(nHead: nHead, nKeys: nKeys, headDim: headDim, scale: scale)

        guard let qbuf = device.makeBuffer(bytes: q, length: nHead * headDim * 4, options: .storageModeShared),
              let kvbuf = device.makeBuffer(bytes: kvF16, length: nKeys * headDim * 2, options: .storageModeShared),
              let maskbuf = device.makeBuffer(length: maskBytes, options: .storageModeShared),
              let sinksbuf = device.makeBuffer(length: max(4, nHead * 4), options: .storageModeShared),
              let padbuf = device.makeBuffer(length: padBytes, options: .storageModeShared),
              let tmpbuf = device.makeBuffer(length: tmpBytes, options: .storageModeShared),
              let headsbuf = device.makeBuffer(length: nHead * headDim * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        memset(maskbuf.contents(), 0, maskBytes)

        let vec = try flashVecPipeline(nsg: Int32(nsg), nwg: Int32(nwg), hasSinks: false)
        let reduce = try flashReducePipeline(dv: Int32(headDim), nwg: Int32(nwg))

        let alignUp = { (v: Int, a: Int) in (v + a - 1) & ~(a - 1) }
        let sharedElems = (alignUp(headDim, 128) + 4 * ncpsg + 2 * alignUp(headDim, 128)) * nsg
        let sharedBytes = alignUp(sharedElems * 2, 16)

        guard let cb = queue.makeCommandBuffer() else { throw MetalError.bufferAlloc }
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(vec)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(qbuf, offset: 0, index: 1)
        enc.setBuffer(kvbuf, offset: 0, index: 2)
        enc.setBuffer(kvbuf, offset: 0, index: 3)
        enc.setBuffer(maskbuf, offset: 0, index: 4)
        enc.setBuffer(sinksbuf, offset: 0, index: 5)
        enc.setBuffer(padbuf, offset: 0, index: 6)
        enc.setBuffer(tmpbuf, offset: 0, index: 7)
        enc.setThreadgroupMemoryLength(sharedBytes, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: nHead, depth: nwg),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: nsg, depth: 1))
        enc.endEncoding()

        var reduceArgs = Int32(nHead)
        let enc2 = cb.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(reduce)
        enc2.setBytes(&reduceArgs, length: 4, index: 0)
        enc2.setBuffer(tmpbuf, offset: 0, index: 1)
        enc2.setBuffer(headsbuf, offset: 0, index: 2)
        enc2.dispatchThreadgroups(MTLSize(width: nHead, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32 * nwg, height: 1, depth: 1))
        enc2.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = headsbuf.contents().bindMemory(to: Float.self, capacity: nHead * headDim)
        return Array(UnsafeBufferPointer(start: p, count: nHead * headDim))
    }

    /// 192-byte ds4_gpu_flash_attn_vec_args (decode, dk=dv=512, no bias/scap).
    static func flashVecArgs(nHead: Int, nKeys: Int, headDim: Int, scale: Float) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 192)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func f32(_ off: Int, _ v: Float) { withUnsafeBytes(of: v.bitPattern.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        let rowBytes = UInt64(headDim) * 4
        let rowF16 = UInt64(headDim) * 2
        let kvPlane = UInt64(nKeys) * rowF16
        let maskBytes = UInt64(nKeys) * 2
        i32(0, 1); i32(4, Int32(nHead)); i32(8, 1)                        // ne01, ne02, ne03
        u64(16, UInt64(nHead) * rowBytes); u64(24, rowBytes); u64(32, UInt64(nHead) * rowBytes) // nb01, nb02, nb03
        i32(40, Int32(nKeys)); i32(44, 1); i32(48, 1); i32(52, Int32(headDim)) // ne11, ne_12_2, ne_12_3, ns10
        u64(56, rowF16); u64(64, kvPlane); u64(72, kvPlane)               // nb11, nb12, nb13
        i32(80, Int32(headDim))                                           // ns20
        u64(88, rowF16); u64(96, kvPlane); u64(104, kvPlane)             // nb21, nb22, nb23
        i32(112, 1); i32(116, 1); i32(120, 1)                            // ne31, ne32, ne33
        u64(128, maskBytes); u64(136, maskBytes); u64(144, maskBytes)   // nb31, nb32, nb33
        i32(152, Int32(nHead)); i32(156, 1); i32(160, 1)                // ne1, ne2, ne3
        f32(164, scale); f32(168, 0); f32(172, 0); f32(176, 0)          // scale, max_bias, m0, m1
        i32(180, 0); f32(184, 0)                                        // n_head_log2, logit_softcap
        return b
    }
}
