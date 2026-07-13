import Foundation
import Metal

// Phase 9: RMSNorm over activation rows. Faithful port of ds4_gpu_rms_norm
// (plain) and the weighted fuse path. Uses the unchanged metal/norm.metal
// kernels kernel_rms_norm_f32_4 (F=1, plain) and kernel_rms_norm_mul_f32_4
// (F=2, normalized * weight), the ds4_metal_args_norm struct (144 bytes), and
// the same threadgroup sizing (pow2 up to n/4) and 32-float shmem.

extension MetalRuntime {
    /// RMSNorm of `rows` rows of `n` floats. If `weight` (length n) is given,
    /// fuses the learned weight multiply (kernel_rms_norm_mul_f32_4); otherwise
    /// plain normalize (kernel_rms_norm_f32_4). `n` must be a multiple of 4.
    public func rmsNorm(_ x: [Float], rows: Int, n: Int, eps: Float,
                        weight: [Float]? = nil) throws -> [Float] {
        precondition(n % 4 == 0, "RMSNorm float4 kernel needs n multiple of 4")
        precondition(x.count >= rows * n)
        if let w = weight { precondition(w.count >= n) }

        let args = Self.rmsNormArgs(n: n, rows: rows, eps: eps)
        let byteLen = rows * n * MemoryLayout<Float>.size

        guard let xbuf = device.makeBuffer(bytes: x, length: byteLen, options: .storageModeShared),
              let obuf = device.makeBuffer(length: byteLen, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        // src1_0 holds the (broadcast) weight row for F=2; unused for F=1.
        let wbuf: MTLBuffer
        let kernelName: String
        if let w = weight {
            guard let b = device.makeBuffer(bytes: w, length: n * MemoryLayout<Float>.size, options: .storageModeShared) else {
                throw MetalError.bufferAlloc
            }
            wbuf = b
            kernelName = "kernel_rms_norm_mul_f32_4"
        } else {
            wbuf = xbuf
            kernelName = "kernel_rms_norm_f32_4"
        }

        let pso = try pipeline(kernelName)
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(xbuf, offset: 0, index: 1)   // src0
        enc.setBuffer(wbuf, offset: 0, index: 2)   // src1_0 (weight or x)
        enc.setBuffer(xbuf, offset: 0, index: 3)   // src1_1 (unused)
        enc.setBuffer(obuf, offset: 0, index: 4)   // dst
        enc.setThreadgroupMemoryLength(32 * MemoryLayout<Float>.size, index: 0)
        let nth = Self.rmsNormThreads(n)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = obuf.contents().bindMemory(to: Float.self, capacity: rows * n)
        return Array(UnsafeBufferPointer(start: p, count: rows * n))
    }

    /// Port of ds4_gpu_rms_norm_threads: pow2 from 32 up to n/4, capped 1024.
    static func rmsNormThreads(_ n: Int) -> Int {
        let ne00t = n / 4
        var nth = 32
        while nth < ne00t && nth < 1024 { nth *= 2 }
        if nth > ne00t { nth = ne00t }
        return nth > 0 ? nth : 1
    }

    /// Build the 144-byte ds4_metal_args_norm buffer (make_rms_norm_args), with
    /// explicit offsets matching the C/Metal natural layout.
    static func rmsNormArgs(n: Int, rows: Int, eps: Float) -> [UInt8] {
        let rowBytes = UInt64(n) * 4
        let plane = rowBytes * UInt64(rows)
        var b = [UInt8](repeating: 0, count: 144)
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func f32(_ off: Int, _ v: Float) { withUnsafeBytes(of: v.bitPattern.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        i32(0, Int32(n)); i32(4, Int32(n / 4))             // ne00, ne00_t
        u64(8, rowBytes); u64(16, plane); u64(24, plane)   // nb1, nb2, nb3
        f32(32, eps)                                       // eps
        i32(36, Int32(rows)); i32(40, 1); i32(44, 1)       // nef1 = {rows,1,1}
        i32(48, 1); i32(52, 1); i32(56, 1)                 // nef2 = {1,1,1}
        i32(60, 1); i32(64, 1); i32(68, 1)                 // nef3 = {1,1,1}
        u64(72, rowBytes); u64(80, rowBytes); u64(88, rowBytes)  // nbf1
        u64(96, plane); u64(104, rowBytes); u64(112, rowBytes)   // nbf2
        u64(120, plane); u64(128, rowBytes); u64(136, rowBytes)  // nbf3
        return b
    }
}
