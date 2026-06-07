import Foundation
import Metal

// Phase 9 / Stage A4: compressed-KV kernels. Faithful ports dispatching the real
// metal/dsv4_kv.metal kernels:
//   - kernel_dsv4_fp8_kv_quantize_f32: E4M3FN quantize-dequantize of the non-RoPE
//     part of each KV row (RoPE tail copied unchanged).
//   - kernel_dsv4_ratio4_shift_f32: ratio-4 recurrent state window shift.
// The E4M3FN/E2M1FN LUTs live in the .metal file itself (compiled in).

extension MetalRuntime {
    /// In-place E4M3FN quantize of the first (headDim - nRot) elements of each of
    /// `nTok` rows; the last nRot (RoPE) elements are copied unchanged.
    public func fp8KVQuantize(_ x: [Float], nTok: Int, headDim: Int, nRot: Int) throws -> [Float] {
        precondition(x.count >= nTok * headDim)
        precondition(nRot <= headDim)
        if nRot == headDim { return Array(x.prefix(nTok * headDim)) }
        let args = Self.fp8KVQuantizeArgs(headDim: headDim, nTok: nTok, nRot: nRot)
        guard let xbuf = device.makeBuffer(bytes: x, length: nTok * headDim * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_fp8_kv_quantize_f32")
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: args.count, index: 0) }
        enc.setBuffer(xbuf, offset: 0, index: 1)
        enc.setBuffer(xbuf, offset: 0, index: 2)
        enc.setThreadgroupMemoryLength(64 * 4, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: nTok, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let p = xbuf.contents().bindMemory(to: Float.self, capacity: nTok * headDim)
        return Array(UnsafeBufferPointer(start: p, count: nTok * headDim))
    }

    /// Ratio-4 state window shift: for gid < 4*width, state[gid] = state[4*width+gid].
    /// `stateKv`/`stateScore` hold at least 8*width floats. Returns both shifted.
    public func ratio4Shift(stateKv: [Float], stateScore: [Float], width: Int) throws -> (kv: [Float], score: [Float]) {
        let n = 4 * width
        precondition(stateKv.count >= 2 * n && stateScore.count >= 2 * n)
        var wArg = UInt32(width)
        guard let kvbuf = device.makeBuffer(bytes: stateKv, length: stateKv.count * 4, options: .storageModeShared),
              let scbuf = device.makeBuffer(bytes: stateScore, length: stateScore.count * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_ratio4_shift_f32")
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        enc.setComputePipelineState(pso)
        enc.setBytes(&wArg, length: 4, index: 0)
        enc.setBuffer(kvbuf, offset: 0, index: 1)
        enc.setBuffer(scbuf, offset: 0, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let pk = kvbuf.contents().bindMemory(to: Float.self, capacity: stateKv.count)
        let ps = scbuf.contents().bindMemory(to: Float.self, capacity: stateScore.count)
        return (Array(UnsafeBufferPointer(start: pk, count: stateKv.count)),
                Array(UnsafeBufferPointer(start: ps, count: stateScore.count)))
    }

    /// 128-wide Hadamard rotation + E2M1FN FP4 simulation, in-place per row.
    /// `x` is nRows x 128 F32. Returns the rotated/quantized rows.
    public func indexerHadamardFP4(_ x: [Float], nRows: Int) throws -> [Float] {
        let headDim = 128
        precondition(x.count >= nRows * headDim)
        var args = [UInt8](repeating: 0, count: 16)
        withUnsafeBytes(of: UInt32(nRows).littleEndian) { for k in 0..<4 { args[k] = $0[k] } }
        withUnsafeBytes(of: UInt32(headDim).littleEndian) { for k in 0..<4 { args[4+k] = $0[k] } }
        withUnsafeBytes(of: UInt64(headDim * 4).littleEndian) { for k in 0..<8 { args[8+k] = $0[k] } }
        guard let xbuf = device.makeBuffer(bytes: x, length: nRows * headDim * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_indexer_hadamard_fp4_f32")
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw MetalError.bufferAlloc }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: 16, index: 0) }
        enc.setBuffer(xbuf, offset: 0, index: 1)
        enc.setThreadgroupMemoryLength(256 * 4, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: nRows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let p = xbuf.contents().bindMemory(to: Float.self, capacity: nRows * headDim)
        return Array(UnsafeBufferPointer(start: p, count: nRows * headDim))
    }

    /// Fused decode KV finalizer: FP8 (E4M3) the non-RoPE part of `kv` in place,
    /// and write the row (F16-rounded) into raw_cache[rawRow]. Returns (kv, raw).
    public func kvFP8Store(kv: [Float], rawCache: [Float], headDim: Int, nRot: Int,
                           rawRow: Int, rawCap: Int) throws -> (kv: [Float], raw: [Float]) {
        precondition(kv.count >= headDim && rawCache.count >= rawCap * headDim)
        var args = [UInt8](repeating: 0, count: 12)
        withUnsafeBytes(of: Int32(headDim).littleEndian) { for k in 0..<4 { args[k] = $0[k] } }
        withUnsafeBytes(of: Int32(nRot).littleEndian) { for k in 0..<4 { args[4+k] = $0[k] } }
        withUnsafeBytes(of: Int32(rawRow).littleEndian) { for k in 0..<4 { args[8+k] = $0[k] } }
        guard let kvbuf = device.makeBuffer(bytes: kv, length: headDim * 4, options: .storageModeShared),
              let rawbuf = device.makeBuffer(bytes: rawCache, length: rawCap * headDim * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_kv_fp8_store_f32")
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw MetalError.bufferAlloc }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: 12, index: 0) }
        enc.setBuffer(kvbuf, offset: 0, index: 1)
        enc.setBuffer(rawbuf, offset: 0, index: 2)
        enc.setThreadgroupMemoryLength(64 * 4, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let pk = kvbuf.contents().bindMemory(to: Float.self, capacity: headDim)
        let pr = rawbuf.contents().bindMemory(to: Float.self, capacity: rawCap * headDim)
        return (Array(UnsafeBufferPointer(start: pk, count: headDim)),
                Array(UnsafeBufferPointer(start: pr, count: rawCap * headDim)))
    }

    /// One-token compressor frontier update (ape_type=0 / F32 APE):
    /// state_kv[dst]=kv[g]; state_score[dst]=score[g]+ape[pos%ratio*width+g],
    /// dst_row = ratio==4 ? ratio+pos%ratio : pos%ratio.
    public func compressorStoreOne(kv: [Float], score: [Float], ape: [Float],
                                   stateKv: [Float], stateScore: [Float],
                                   width: Int, ratio: Int, pos: Int) throws -> (kv: [Float], score: [Float]) {
        precondition(kv.count >= width && score.count >= width)
        let stateRows = ratio == 4 ? 2 * ratio : ratio
        precondition(stateKv.count >= stateRows * width && stateScore.count >= stateRows * width)
        precondition(ape.count >= ratio * width)
        var args = [UInt8](repeating: 0, count: 16)
        withUnsafeBytes(of: UInt32(width).littleEndian) { for k in 0..<4 { args[k] = $0[k] } }
        withUnsafeBytes(of: UInt32(ratio).littleEndian) { for k in 0..<4 { args[4+k] = $0[k] } }
        withUnsafeBytes(of: UInt32(pos).littleEndian) { for k in 0..<4 { args[8+k] = $0[k] } }
        withUnsafeBytes(of: UInt32(0).littleEndian) { for k in 0..<4 { args[12+k] = $0[k] } } // ape_type=0
        guard let kvbuf = device.makeBuffer(bytes: kv, length: width * 4, options: .storageModeShared),
              let scbuf = device.makeBuffer(bytes: score, length: width * 4, options: .storageModeShared),
              let apebuf = device.makeBuffer(bytes: ape, length: ape.count * 4, options: .storageModeShared),
              let skbuf = device.makeBuffer(bytes: stateKv, length: stateKv.count * 4, options: .storageModeShared),
              let ssbuf = device.makeBuffer(bytes: stateScore, length: stateScore.count * 4, options: .storageModeShared) else {
            throw MetalError.bufferAlloc
        }
        let pso = try pipeline("kernel_dsv4_compressor_store_one")
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw MetalError.bufferAlloc }
        enc.setComputePipelineState(pso)
        args.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: 16, index: 0) }
        enc.setBuffer(kvbuf, offset: 0, index: 1)
        enc.setBuffer(scbuf, offset: 0, index: 2)
        enc.setBuffer(apebuf, offset: 0, index: 3)
        enc.setBuffer(skbuf, offset: 0, index: 4)
        enc.setBuffer(ssbuf, offset: 0, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: (width + 255) / 256, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let pk = skbuf.contents().bindMemory(to: Float.self, capacity: stateKv.count)
        let ps = ssbuf.contents().bindMemory(to: Float.self, capacity: stateScore.count)
        return (Array(UnsafeBufferPointer(start: pk, count: stateKv.count)),
                Array(UnsafeBufferPointer(start: ps, count: stateScore.count)))
    }

    static let e2m1Values: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    static func e2m1Dequant(_ x: Float) -> Float {
        let sign: Float = x < 0 ? -1 : 1
        let ax = min(abs(x), 6.0)
        var best = 0; var bd = abs(ax - e2m1Values[0])
        for i in 1..<8 {
            let d = abs(ax - e2m1Values[i])
            if d < bd || (d == bd && (i & 1) == 0 && (best & 1) != 0) { best = i; bd = d }
        }
        return sign * e2m1Values[best]
    }

    // CPU references (mirror the .metal E4M3FN LUT) for validation.
    static func e4m3Value(_ i: Int) -> Float {
        let scale: [Float] = [0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
        let exp = (i >> 3) & 0x0f, mant = i & 0x07
        return exp == 0 ? Float(mant) * 0.001953125 : (1.0 + Float(mant) * 0.125) * scale[exp]
    }
    static func e4m3Dequant(_ x: Float) -> Float {
        let sign: Float = x < 0 ? -1 : 1
        let ax = min(abs(x), 448.0)
        var lo = 0, hi = 126
        while lo < hi { let mid = (lo + hi + 1) >> 1; if e4m3Value(mid) <= ax { lo = mid } else { hi = mid - 1 } }
        var best = lo
        if best < 126 {
            let bd = abs(ax - e4m3Value(best)), nd = abs(ax - e4m3Value(best + 1))
            if nd < bd || (nd == bd && ((best + 1) & 1) == 0 && (best & 1) != 0) { best += 1 }
        }
        return sign * e4m3Value(best)
    }

    /// 104-byte ds4_gpu_dsv4_fp8_kv_quantize_args.
    static func fp8KVQuantizeArgs(headDim: Int, nTok: Int, nRot: Int) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 104)
        func i64(_ off: Int, _ v: Int64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func u64(_ off: Int, _ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<8 { b[off+k] = $0[k] } } }
        func i32(_ off: Int, _ v: Int32) { withUnsafeBytes(of: v.littleEndian) { for k in 0..<4 { b[off+k] = $0[k] } } }
        let rowBytes = UInt64(headDim) * 4
        let plane = UInt64(nTok) * rowBytes
        i64(0, Int64(headDim)); i64(8, Int64(nTok)); i64(16, 1); i64(24, 1)  // ne00..ne03
        u64(32, 4); u64(40, rowBytes); u64(48, plane); u64(56, plane)        // nb00..nb03
        u64(64, 4); u64(72, rowBytes); u64(80, plane); u64(88, plane)        // nb0..nb3
        i32(96, Int32(nRot))                                                 // n_rot
        return b
    }
}
