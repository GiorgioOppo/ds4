import Metal

extension GraphContext {
    /// One-token output-B matvec followed by HC=4. `split` contains the usual
    /// 24 F32 values (pre[4], post[4], comb[16]); both the projected row and HC
    /// output remain materialized. Quantized bytes and reduction order match
    /// matmulQ4_K/matmulQ8_0 followed by hcExpand4.
    ///
    /// Returns false without encoding anything when disabled or when shape,
    /// bounds, alignment, aliasing or pipeline support requires the reference
    /// path. `enabled` is an explicit A/B override; nil uses the per-load knob
    /// (Q8 defaults on, Q4 requires DS4_FUSED_ATTN_OUT_HC=1).
    @discardableResult
    public func attentionOutputHCFused(
        weight: GPUTensor, x: GPUTensor, blockOut: GPUTensor,
        residual: GPUTensor, split: GPUTensor, out: GPUTensor,
        inDim: Int, outDim: Int, q4: Bool, nHC: Int = 4,
        enabled: Bool? = nil
    ) throws -> Bool {
        guard enabled ?? (q4 ? Self.fusedAttentionOutputHCQ4 : Self.fusedAttentionOutputHC),
              nHC == 4, inDim > 0, outDim > 0,
              inDim <= Int(Int32.max) - 16,
              outDim <= Int(Int32.max) - 16,
              inDim % (q4 ? 256 : 32) == 0, outDim % 2 == 0 else {
            return false
        }
        let nsg = q4 ? Self.denseQ4NSG : Self.q8NSG
        guard (1...8).contains(Int(nsg)) else { return false }
        let blockBytes = q4 ? 144 : 34
        let rowBytes = (inDim / (q4 ? 256 : 32)) * blockBytes
        // Dimensions are bounded by Int32 above, so these products fit Int64.
        let weightBytes = rowBytes * outDim
        let rowOutputBytes = outDim * MemoryLayout<Float>.stride
        let hcBytes = rowOutputBytes * 4
        let reads: [(GPUTensor, Int, Int)] = [
            (weight, weightBytes, q4 ? 8 : 2),
            (x, inDim * 4, q4 ? 16 : 4),
            (residual, hcBytes, 4), (split, 24 * 4, 4),
        ]
        let writes: [(GPUTensor, Int, Int)] = [
            (blockOut, rowOutputBytes, 4), (out, hcBytes, 4),
        ]
        for (tensor, bytes, alignment) in reads + writes {
            guard tensor.buffer.device.registryID == rt.device.registryID,
                  tensor.buffer.hazardTrackingMode != .untracked,
                  Self.attentionOutputViewFits(tensor, bytes: bytes, alignment: alignment) else {
                return false
            }
        }
        for (destination, bytes, _) in writes {
            for (source, sourceBytes, _) in reads {
                guard !Self.attentionOutputViewsOverlap(destination, bytes, source, sourceBytes) else {
                    return false
                }
            }
        }
        guard !Self.attentionOutputViewsOverlap(blockOut, rowOutputBytes, out, hcBytes) else {
            return false
        }

        let name = q4 ? "kernel_dsv4_attn_out_q4_K_hc_expand4"
                      : "kernel_dsv4_q8_hc_expand4_q8_0"
        // A library/device that lacks the specialization keeps the ordinary
        // graph path. No binding or dispatch has happened before this check.
        guard let pso = try? rt.mulMVPipeline(name, nsg: nsg),
              pso.threadExecutionWidth == 32,
              pso.maxTotalThreadsPerThreadgroup >= 32 * Int(nsg) else {
            return false
        }
        let smem = 256 // Q8: 32 lanes × 2 rows × F32; Q4 helper ignores it.
        guard rt.device.maxThreadgroupMemoryLength >= smem,
              pso.staticThreadgroupMemoryLength <= rt.device.maxThreadgroupMemoryLength - smem else {
            return false
        }
        let mv = MetalRuntime.mulMVArgs(
            ne00: inDim, ne01: outDim, nb00: UInt64(blockBytes),
            nb01: UInt64(rowBytes), nb02: UInt64(weightBytes),
            ne10: inDim, ne11: 1, nb10: 4, nb11: UInt64(inDim * 4),
            ne0: outDim, ne1: 1, nr0: 2)
        let hc = MetalRuntime.hcExpandArgs(nEmbd: outDim, nHC: 4, nTokens: 1, hasAdd: false)
        let e = encoder
        e.setComputePipelineState(pso)
        mv.withUnsafeBytes { e.setBytes($0.baseAddress!, length: mv.count, index: 0) }
        hc.withUnsafeBytes { e.setBytes($0.baseAddress!, length: hc.count, index: 1) }
        e.setBuffer(weight.buffer, offset: weight.byteOffset, index: 2)
        e.setBuffer(x.buffer, offset: x.byteOffset, index: 3)
        e.setBuffer(blockOut.buffer, offset: blockOut.byteOffset, index: 4)
        e.setBuffer(residual.buffer, offset: residual.byteOffset, index: 5)
        e.setBuffer(split.buffer, offset: split.byteOffset + 4 * 4, index: 6)
        e.setBuffer(split.buffer, offset: split.byteOffset + 8 * 4, index: 7)
        e.setBuffer(out.buffer, offset: out.byteOffset, index: 8)
        e.setThreadgroupMemoryLength(smem, index: 0)
        let rowsPerGroup = q4 ? Int(nsg) * 2 : 2
        e.dispatchThreadgroups(
            MTLSize(width: (outDim + rowsPerGroup - 1) / rowsPerGroup, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: Int(nsg), depth: 1))
        return true
    }

    private static func attentionOutputViewFits(_ tensor: GPUTensor, bytes: Int, alignment: Int) -> Bool {
        tensor.byteOffset >= 0 && tensor.byteLength >= bytes &&
            tensor.byteOffset <= tensor.buffer.length &&
            tensor.byteLength <= tensor.buffer.length - tensor.byteOffset &&
            tensor.byteOffset % alignment == 0
    }

    private static func attentionOutputViewsOverlap(
        _ a: GPUTensor, _ aBytes: Int, _ b: GPUTensor, _ bBytes: Int
    ) -> Bool {
        // Subtraction avoids overflow when constructing the ends of GPU or
        // CPU address ranges. Also handles distinct Metal wrappers of the same
        // shared allocation, beyond the usual same-buffer subviews.
        func overlap(_ aStart: UInt64, _ bStart: UInt64) -> Bool {
            aStart <= bStart ? bStart - aStart < UInt64(aBytes)
                            : aStart - bStart < UInt64(bBytes)
        }
        if a.buffer === b.buffer {
            return overlap(UInt64(a.byteOffset), UInt64(b.byteOffset))
        }
        let ag = a.buffer.gpuAddress, bg = b.buffer.gpuAddress
        if ag != 0, bg != 0,
           ag <= UInt64.max - UInt64(a.byteOffset), bg <= UInt64.max - UInt64(b.byteOffset),
           overlap(ag + UInt64(a.byteOffset), bg + UInt64(b.byteOffset)) { return true }
        let aShared = a.buffer.storageMode == .shared || a.buffer.storageMode == .managed
        let bShared = b.buffer.storageMode == .shared || b.buffer.storageMode == .managed
        if aShared && bShared {
            let ap = UInt64(UInt(bitPattern: a.buffer.contents().advanced(by: a.byteOffset)))
            let bp = UInt64(UInt(bitPattern: b.buffer.contents().advanced(by: b.byteOffset)))
            return overlap(ap, bp)
        }
        return false
    }
}
