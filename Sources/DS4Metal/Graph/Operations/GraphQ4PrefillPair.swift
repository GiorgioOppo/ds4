import Foundation
import Metal

extension GraphContext {
    /// C b8507c9f's measured Q-A/KV pair scope. An unsupported shape/device
    /// must keep the generic GEMMs, including partial final token tiles.
    public static func q4PrefillPairEligible(deviceName: String, inDim: Int,
                                              qOutDim: Int, kvOutDim: Int,
                                              nTok: Int) -> Bool {
        guard inDim == 4096, qOutDim == 1024, kvOutDim == 512,
              (32...256).contains(nTok), nTok % 32 == 0,
              deviceName.hasPrefix("Apple M") else { return false }
        let suffix = deviceName.dropFirst("Apple M".count)
        guard let generation = suffix.first, "1234".contains(generation) else { return false }
        return suffix.count == 1 || suffix.dropFirst().first == " "
    }

    /// Encode Q-A and KV from one F16 activation copy. False means no dispatch
    /// or writes occurred: callers may immediately encode the generic GEMMs.
    /// Scratch belongs to one decoder's prefill stage and is used only on its
    /// serial, hazard-tracked queue; it must not be shared across independent queues.
    public func encodeQ4PrefillPair(weightQA: GPUTensor, weightKV: GPUTensor,
                                     act: GPUTensor, qOut: GPUTensor, kvOut: GPUTensor,
                                     rhsF16: GPUTensor, inDim: Int, qOutDim: Int,
                                     kvOutDim: Int, nTok: Int,
                                     enabled: Bool = true) throws -> Bool {
        guard enabled, Self.q4PrefillPairEligible(deviceName: rt.device.name,
            inDim: inDim, qOutDim: qOutDim, kvOutDim: kvOutDim, nTok: nTok) else { return false }
        // Exact dimensions above bound every product below (at most 4 MiB).
        let rowBytes = (inDim / 256) * 144
        let readers = [(weightQA, qOutDim * rowBytes, 2),
                       (weightKV, kvOutDim * rowBytes, 2), (act, nTok * inDim * 4, 4)]
        let writers = [(qOut, nTok * qOutDim * 4, 4),
                       (kvOut, nTok * kvOutDim * 4, 4), (rhsF16, nTok * inDim * 2, 16)]
        func valid(_ view: (GPUTensor, Int, Int)) -> Bool {
            let (t, bytes, alignment) = view
            return t.byteOffset >= 0 && t.byteOffset % alignment == 0
                && t.byteLength >= bytes && t.buffer.length >= bytes
                && t.byteOffset <= t.buffer.length - bytes
                && t.buffer.hazardTrackingMode != .untracked
                && t.buffer.device.registryID == rt.device.registryID
        }
        guard (readers + writers).allSatisfy(valid) else { return false }
        func overlaps(_ lhs: (GPUTensor, Int, Int), _ rhs: (GPUTensor, Int, Int)) -> Bool {
            let (a, an, _) = lhs, (b, bn, _) = rhs
            if a.buffer === b.buffer {
                return a.byteOffset < b.byteOffset + bn && b.byteOffset < a.byteOffset + an
            }
            // Distinct no-copy MTLBuffers can still wrap overlapping mmap pages.
            if a.buffer.storageMode != .private && b.buffer.storageMode != .private {
                let (ap, ao) = UInt(bitPattern: a.buffer.contents()).addingReportingOverflow(UInt(a.byteOffset))
                let (bp, bo) = UInt(bitPattern: b.buffer.contents()).addingReportingOverflow(UInt(b.byteOffset))
                let (ae, ax) = ap.addingReportingOverflow(UInt(an))
                let (be, bx) = bp.addingReportingOverflow(UInt(bn))
                return ao || bo || ax || bx || (ap < be && bp < ae)
            }
            return false
        }
        for (index, writer) in writers.enumerated() {
            guard !readers.contains(where: { overlaps(writer, $0) }),
                  !writers.dropFirst(index + 1).contains(where: { overlaps(writer, $0) }) else { return false }
        }
        // Resolve everything that can fail before the copy writes scratch.
        guard let copy = try? rt.pipeline("kernel_dsv4_q4_prefill_rhs_f16"),
              let mm = try? rt.pipeline("kernel_dsv4_q4_prefill_f16_rhs_m32_k64"),
              copy.threadExecutionWidth == 32, copy.maxTotalThreadsPerThreadgroup >= 32,
              mm.threadExecutionWidth == 32, mm.maxTotalThreadsPerThreadgroup >= 128,
              mm.staticThreadgroupMemoryLength <= rt.device.maxThreadgroupMemoryLength,
              8192 <= rt.device.maxThreadgroupMemoryLength - mm.staticThreadgroupMemoryLength else { return false }
        let qArgs = Self.q4PrefillPairMMArgs(inDim: inDim, outDim: qOutDim, nTok: nTok)
        let kvArgs = Self.q4PrefillPairMMArgs(inDim: inDim, outDim: kvOutDim, nTok: nTok)
        var count = UInt32(nTok * inDim)
        let e = encoder
        e.setComputePipelineState(copy)
        e.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 0)
        e.setBuffer(act.buffer, offset: act.byteOffset, index: 1)
        e.setBuffer(rhsF16.buffer, offset: rhsF16.byteOffset, index: 2)
        e.dispatchThreads(MTLSize(width: Int(count) / 4, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: min(256, copy.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        e.memoryBarrier(scope: .buffers)
        e.setComputePipelineState(mm)
        e.setThreadgroupMemoryLength(8192, index: 0)
        e.setBuffer(rhsF16.buffer, offset: rhsF16.byteOffset, index: 2)
        for (weight, out, args, width) in [(weightQA, qOut, qArgs, qOutDim),
                                         (weightKV, kvOut, kvArgs, kvOutDim)] {
            args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
            e.setBuffer(weight.buffer, offset: weight.byteOffset, index: 1)
            e.setBuffer(out.buffer, offset: out.byteOffset, index: 3)
            e.dispatchThreadgroups(MTLSize(width: nTok / 32, height: width / 32, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
        return true
    }

    static func q4PrefillPairMMArgs(inDim: Int, outDim: Int, nTok: Int) -> [UInt8] {
        var args = MetalRuntime.mulMMArgs(inDim: inDim, outDim: outDim, nTok: nTok,
                                          rowBytes: UInt64((inDim / 256) * 144))
        for (offset, value) in [(40, 2), (48, inDim * 2),
                                (56, nTok * inDim * 2), (64, nTok * inDim * 2)] {
            var v = UInt64(value).littleEndian
            withUnsafeBytes(of: &v) { args.replaceSubrange(offset..<(offset + 8), with: $0) }
        }
        return args
    }
}
