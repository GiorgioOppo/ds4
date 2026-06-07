import Foundation
import Metal

// Phase 9 / Stage A1: dense matrix-matrix multiply for PREFILL (multi-token).
// Faithful port of the portable mul_mm path in ds4_gpu_matmul_{q8_0,f16}_tensor
// (n_tok>1) dispatching the real metal/dense.metal kernels kernel_mul_mm_q8_0_f32
// / kernel_mul_mm_f16_f32 with function constants FC_MUL_MM bc_inp (700) and
// bc_out (701). Output is n_tok rows of out_dim floats. The MPP/NAX tensor-ops
// fast path is an optional optimization and is intentionally not ported.

extension MetalRuntime {
    public func mulMMPipeline(_ name: String, bcInp: Bool, bcOut: Bool) throws -> MTLComputePipelineState {
        let key = "\(name)_bci=\(bcInp)_bco=\(bcOut)"
        if let p = mulMVPipelineCache[key] { return p }
        let constants = MTLFunctionConstantValues()
        var bi = bcInp, bo = bcOut
        constants.setConstantValue(&bi, type: .bool, index: 700)
        constants.setConstantValue(&bo, type: .bool, index: 701)
        let fn = try library.makeFunction(name: name, constantValues: constants)
        let pso = try device.makeComputePipelineState(function: fn)
        mulMVPipelineCache[key] = pso
        return pso
    }

    /// Q8_0 prefill matmul: out[n_tok][out_dim] = x[n_tok][in_dim] * W^T.
    /// `weight` is out_dim rows of Q8_0 blocks (34B/32 elems).
    public func matmulMMQ8_0(weight: [UInt8], activation: [Float],
                             inDim: Int, outDim: Int, nTok: Int) throws -> [Float] {
        precondition(inDim % 32 == 0)
        let rowBytes = (inDim / 32) * 34
        return try mulMM(kernel: "kernel_mul_mm_q8_0_f32", weightBytes: weight,
                         rowBytes: rowBytes, activation: activation,
                         inDim: inDim, outDim: outDim, nTok: nTok)
    }

    /// F16 prefill matmul: out[n_tok][out_dim] = x[n_tok][in_dim] * W^T.
    public func matmulMMF16(weight: [UInt16], activation: [Float],
                            inDim: Int, outDim: Int, nTok: Int) throws -> [Float] {
        let rowBytes = inDim * 2
        let bytes = weight.withUnsafeBytes { Array($0) }
        return try mulMM(kernel: "kernel_mul_mm_f16_f32", weightBytes: bytes,
                         rowBytes: rowBytes, activation: activation,
                         inDim: inDim, outDim: outDim, nTok: nTok)
    }

    private func mulMM(kernel: String, weightBytes: [UInt8], rowBytes: Int,
                       activation: [Float], inDim: Int, outDim: Int, nTok: Int) throws -> [Float] {
        precondition(weightBytes.count >= rowBytes * outDim)
        precondition(activation.count >= inDim * nTok)
        let bcInp = (inDim % 32) != 0
        let bcOut = (outDim % 64) != 0 || (nTok % 32) != 0
        let args = Self.mulMMArgs(inDim: inDim, outDim: outDim, nTok: nTok, rowBytes: UInt64(rowBytes))

        guard let wbuf = device.makeBuffer(bytes: weightBytes, length: rowBytes * outDim, options: .storageModeShared),
              let xbuf = device.makeBuffer(bytes: activation, length: inDim * nTok * 4, options: .storageModeShared),
              let obuf = device.makeBuffer(length: outDim * nTok * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try mulMMPipeline(kernel, bcInp: bcInp, bcOut: bcOut)
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(wbuf, offset: 0, index: 1)
        enc.setBuffer(xbuf, offset: 0, index: 2)
        enc.setBuffer(obuf, offset: 0, index: 3)
        enc.setThreadgroupMemoryLength(bcOut ? 8192 : 6144, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: (nTok + 31) / 32, height: (outDim + 63) / 64, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = obuf.contents().bindMemory(to: Float.self, capacity: outDim * nTok)
        return Array(UnsafeBufferPointer(start: p, count: outDim * nTok))
    }

    /// 88-byte ds4_gpu_mul_mm_args (matches ds4_gpu_make_mm_args).
    static func mulMMArgs(inDim: Int, outDim: Int, nTok: Int, rowBytes: UInt64) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 88)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func i16(_ off: Int, _ v: Int16) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<2 { b[off+k] = $0[k] } } }
        let xstride = UInt64(inDim) * 4
        i32(0, Int32(inDim)); i32(4, 1)                                   // ne00, ne02
        u64(8, rowBytes); u64(16, rowBytes * UInt64(outDim)); u64(24, rowBytes * UInt64(outDim)) // nb01, nb02, nb03
        i32(32, 1)                                                        // ne12
        u64(40, 4); u64(48, xstride); u64(56, xstride * UInt64(nTok)); u64(64, xstride * UInt64(nTok)) // nb10..nb13
        i32(72, Int32(outDim)); i32(76, Int32(nTok))                      // ne0, ne1
        i16(80, 1); i16(82, 1)                                            // r2, r3
        return b
    }
}
