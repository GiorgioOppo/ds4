import Foundation
import Metal

// Phase 9: get-rows (table gather), used for token embeddings. Faithful port of
// the embedding path of ds4_gpu (kernel_get_rows_f16: half table -> f32 row),
// ds4_gpu_get_rows_args (88 bytes), grid (ceil(nEmbd/nth),1,1).

extension MetalRuntime {
    /// Gather row `id` from an F16 table of shape [nVocab][nEmbd] into an F32 row.
    public func getRowsF16(table: [UInt16], id: Int, nEmbd: Int, nVocab: Int) throws -> [Float] {
        precondition(table.count >= nVocab * nEmbd)
        precondition(id >= 0 && id < nVocab)

        let srcRowBytes = UInt64(nEmbd) * 2
        let dstRowBytes = UInt64(nEmbd) * 4
        let args = Self.getRowsArgs(ne00t: nEmbd, ne00: nEmbd,
                                    nb01: srcRowBytes, nb02: UInt64(nVocab) * srcRowBytes,
                                    ne10: 1, nb1: dstRowBytes)

        guard let tbuf = table.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: nVocab * nEmbd * 2, options: .storageModeShared) }),
              let obuf = device.makeBuffer(length: nEmbd * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        var idv = Int32(id)

        let pso = try pipeline("kernel_get_rows_f16")
        var nth = pso.maxTotalThreadsPerThreadgroup
        if nth > nEmbd { nth = nEmbd }
        if nth == 0 { nth = 1 }
        let nw0 = (nEmbd + nth - 1) / nth

        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(tbuf, offset: 0, index: 1)
        enc.setBytes(&idv, length: 4, index: 2)
        enc.setBuffer(obuf, offset: 0, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: nw0, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = obuf.contents().bindMemory(to: Float.self, capacity: nEmbd)
        return Array(UnsafeBufferPointer(start: p, count: nEmbd))
    }

    /// Gather row `id` from an F32 table of shape [nVocab][nEmbd] into an F32 row.
    public func getRowsF32(table: [Float], id: Int, nEmbd: Int, nVocab: Int) throws -> [Float] {
        precondition(table.count >= nVocab * nEmbd)
        precondition(id >= 0 && id < nVocab)
        let srcRowBytes = UInt64(nEmbd) * 4
        let dstRowBytes = UInt64(nEmbd) * 4
        let args = Self.getRowsArgs(ne00t: nEmbd, ne00: nEmbd,
                                    nb01: srcRowBytes, nb02: UInt64(nVocab) * srcRowBytes,
                                    ne10: 1, nb1: dstRowBytes)
        guard let tbuf = device.makeBuffer(bytes: table, length: nVocab * nEmbd * 4, options: .storageModeShared),
              let obuf = device.makeBuffer(length: nEmbd * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        var idv = Int32(id)
        let pso = try pipeline("kernel_get_rows_f32")
        var nth = pso.maxTotalThreadsPerThreadgroup
        if nth > nEmbd { nth = nEmbd }
        if nth == 0 { nth = 1 }
        let nw0 = (nEmbd + nth - 1) / nth
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(tbuf, offset: 0, index: 1)
        enc.setBytes(&idv, length: 4, index: 2)
        enc.setBuffer(obuf, offset: 0, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: nw0, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let p = obuf.contents().bindMemory(to: Float.self, capacity: nEmbd)
        return Array(UnsafeBufferPointer(start: p, count: nEmbd))
    }

    /// Gather row `id` from an I32 table of shape [nVocab][nEmbd].
    public func getRowsI32(table: [Int32], id: Int, nEmbd: Int, nVocab: Int) throws -> [Int32] {
        precondition(table.count >= nVocab * nEmbd)
        precondition(id >= 0 && id < nVocab)
        let srcRowBytes = UInt64(nEmbd) * 4
        let args = Self.getRowsArgs(ne00t: nEmbd, ne00: nEmbd,
                                    nb01: srcRowBytes, nb02: UInt64(nVocab) * srcRowBytes,
                                    ne10: 1, nb1: srcRowBytes)
        guard let tbuf = device.makeBuffer(bytes: table, length: nVocab * nEmbd * 4, options: .storageModeShared),
              let obuf = device.makeBuffer(length: nEmbd * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        var idv = Int32(id)
        let pso = try pipeline("kernel_get_rows_i32")
        var nth = pso.maxTotalThreadsPerThreadgroup
        if nth > nEmbd { nth = nEmbd }
        if nth == 0 { nth = 1 }
        let nw0 = (nEmbd + nth - 1) / nth
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(tbuf, offset: 0, index: 1)
        enc.setBytes(&idv, length: 4, index: 2)
        enc.setBuffer(obuf, offset: 0, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: nw0, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let p = obuf.contents().bindMemory(to: Int32.self, capacity: nEmbd)
        return Array(UnsafeBufferPointer(start: p, count: nEmbd))
    }

    /// 88-byte ds4_gpu_get_rows_args (single-table gather; ne11/ne12 implicitly 1).
    static func getRowsArgs(ne00t: Int, ne00: Int, nb01: UInt64, nb02: UInt64,
                            ne10: Int, nb1: UInt64) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 88)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        i32(0, Int32(ne00t)); i32(4, Int32(ne00))
        u64(8, nb01); u64(16, nb02); u64(24, nb02)        // nb01, nb02, nb03
        i32(32, Int32(ne10))
        u64(40, 4); u64(48, 4); u64(56, 4)                // nb10, nb11, nb12 (int32 ids)
        u64(64, nb1); u64(72, nb1); u64(80, nb1)          // nb1, nb2, nb3
        return b
    }
}
