import Foundation

// INT4 W4A16 weight quantization. Mirrors Int8Quant.swift; the only
// numerical differences are the quantization range and the packed-byte
// storage layout.
//
// Symmetric round-to-nearest, range [-8, 7] (full 4-bit two's complement).
// We don't reserve -8 like INT8 reserves -128, because losing one of 16
// codes is much more wasteful in 4-bit than losing one of 256 in 8-bit;
// the asymmetry (max(|q|)=8 on the negative side vs 7 on the positive)
// is small noise compared to RTN error at this bit-width.
//
// Scale layout: per row × per 128 input-feature columns. Scale dtype F16
// (10-bit mantissa, sufficient for the magnitudes we see). The 128-
// element block matches the INT8 block size so MoE expert shapes that
// quant cleanly to INT8 also quant cleanly to INT4.
//
//   weight: [N, K/2]    I4 packed (1 byte per 2 elements)
//   scale:  [N, K/128]  F16
//
// Packing convention (also baked into Kernels/int4_gemm.metal):
//   byte[k/2] = (high_nibble = col 2k+1) << 4 | (low_nibble = col 2k)
//
// Scale = max_abs / 7 (positive max). Quantization clamps to [-8, 7]
// so the rare value that would have rounded to -8 simply stays -8 (no
// scale headroom loss vs the symmetric [-7, 7] choice).

/// Same 128 as INT8. Baked into the Metal kernel constant `INT4_BLOCK_K`.
public let kInt4GroupK: Int = 128

/// Returns true if `renamedName` is a `Linear` weight that should be
/// quantized to INT4. Same whitelist as INT8 — INT4 only changes the
/// per-tensor encoding, not which tensors are quantized.
public func shouldQuantizeToInt4(_ renamedName: String, lastDim: Int) -> Bool {
    guard renamedName.hasSuffix(".weight") else { return false }
    guard lastDim % kInt4GroupK == 0 else { return false }
    let stripped = String(renamedName.dropLast(".weight".count))
    guard let leaf = stripped.split(separator: ".").map(String.init).last else { return false }
    let linearLeaves: Set<String> = [
        "wq", "wq_a", "wq_b",
        "wkv", "wkv_a", "wkv_b",
        "wo", "wo_b",
        "wgate",
        "w1", "w2", "w3",
    ]
    return linearLeaves.contains(leaf)
}

@inline(__always)
private func floatToF16Local(_ f: Float) -> UInt16 {
    // Duplicated from Int8Quant.swift / main.swift so this file is
    // self-contained (and unit-testable in isolation).
    let bits = f.bitPattern
    let sign = (bits >> 31) & 1
    let exp = (bits >> 23) & 0xFF
    let mant = bits & 0x7FFFFF
    if exp == 0 { return UInt16(truncatingIfNeeded: sign << 15) }
    if exp == 0xFF {
        let m: UInt32 = mant != 0 ? 0x200 : 0
        return UInt16(truncatingIfNeeded: (sign << 15) | (0x1F << 10) | m)
    }
    let unbiased = Int(exp) - 127
    if unbiased > 15 { return UInt16(truncatingIfNeeded: (sign << 15) | (0x1F << 10)) }
    if unbiased < -14 {
        let shift = -14 - unbiased + 13
        if shift > 24 { return UInt16(truncatingIfNeeded: sign << 15) }
        let full = (mant | 0x800000) >> (shift - 1)
        let halfMant = (full + 1) >> 1
        return UInt16(truncatingIfNeeded: (sign << 15) | halfMant)
    }
    let halfExp = UInt32(unbiased + 15)
    let halfMant = (mant + 0x1000) >> 13
    if halfMant >= 0x400 {
        if halfExp + 1 >= 0x1F {
            return UInt16(truncatingIfNeeded: (sign << 15) | (0x1F << 10))
        }
        return UInt16(truncatingIfNeeded: (sign << 15) | ((halfExp + 1) << 10))
    }
    return UInt16(truncatingIfNeeded: (sign << 15) | (halfExp << 10) | halfMant)
}

@inline(__always)
private func bf16ToFloat(_ b: UInt16) -> Float {
    return Float(bitPattern: UInt32(b) << 16)
}

/// F16 → Float. Mirrors the inverse of `floatToF16Local` above.
/// Used by `quantizeInt8ToInt4` to read back the source
/// checkpoint's per-block scale companion.
@inline(__always)
internal func f16ToFloatLocal(_ h: UInt16) -> Float {
    let sign = UInt32(h >> 15) & 0x1
    let exp = UInt32(h >> 10) & 0x1F
    let mant = UInt32(h) & 0x3FF
    var f: UInt32
    if exp == 0 {
        if mant == 0 {
            f = sign << 31
        } else {
            var e: UInt32 = 1
            var m = mant
            while m & 0x400 == 0 { m <<= 1; e += 1 }
            m &= 0x3FF
            f = (sign << 31) | ((127 - 15 - e + 1) << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        f = (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    }
    return Float(bitPattern: f)
}

/// Pack two signed-4-bit values into one byte: low nibble = a, high = b.
/// Caller has already clamped to [-8, 7]; we mask to 4 bits to clear the
/// sign-extension bits.
@inline(__always)
private func packI4(_ a: Int8, _ b: Int8) -> UInt8 {
    let lo = UInt8(truncatingIfNeeded: a) & 0x0F
    let hi = UInt8(truncatingIfNeeded: b) & 0x0F
    return (hi << 4) | lo
}

/// Symmetric INT4 RTN over a single row of `inDim` floats. Writes
/// `inDim/2` packed bytes to `outRow` and `blocksIn` F16 scales to
/// `scaleRow`.
@inline(__always)
private func quantizeRowFromFloatI4(rowPtr: UnsafePointer<Float>,
                                     outRow: UnsafeMutablePointer<UInt8>,
                                     scaleRow: UnsafeMutablePointer<UInt16>,
                                     inDim: Int) {
    let blocksIn = inDim / kInt4GroupK
    for sb in 0..<blocksIn {
        let base = sb * kInt4GroupK
        var maxAbs: Float = 0
        for k in 0..<kInt4GroupK {
            let v = abs(rowPtr[base + k])
            if v > maxAbs { maxAbs = v }
        }
        let s: Float
        let invS: Float
        if maxAbs == 0 {
            s = 0; invS = 0
        } else {
            s = maxAbs / 7.0
            invS = 7.0 / maxAbs
        }
        scaleRow[sb] = floatToF16Local(s)

        for k in stride(from: 0, to: kInt4GroupK, by: 2) {
            let q0 = (rowPtr[base + k    ] * invS).rounded(.toNearestOrEven)
            let q1 = (rowPtr[base + k + 1] * invS).rounded(.toNearestOrEven)
            let c0 = Int8(min(max(q0, -8), 7))
            let c1 = Int8(min(max(q1, -8), 7))
            outRow[(base + k) / 2] = packI4(c0, c1)
        }
    }
}

/// Quantize a `[outDim, inDim]` BF16 weight tensor at `(srcURL, srcOffset)`
/// into INT4 (packed) + F16 group scales. Returns the two buffers as
/// `Data` blobs in the safetensors order the writer expects.
public func quantizeBF16ToInt4(srcURL: URL, srcOffset: Int,
                                outDim: Int, inDim: Int) throws -> (weight: Data, scale: Data) {
    precondition(inDim % kInt4GroupK == 0,
                 "INT4 quant requires inDim % \(kInt4GroupK) == 0")
    let blocksIn = inDim / kInt4GroupK

    guard let fh = FileHandle(forReadingAtPath: srcURL.path) else {
        throw NSError(domain: "quantizeBF16ToInt4", code: 1)
    }
    defer { try? fh.close() }
    try fh.seek(toOffset: UInt64(srcOffset))
    let byteLen = outDim * inDim * 2
    guard let bf16Bytes = try fh.read(upToCount: byteLen),
          bf16Bytes.count == byteLen else {
        throw NSError(domain: "quantizeBF16ToInt4", code: 2)
    }

    let packedRowBytes = inDim / 2
    var weight = Data(count: outDim * packedRowBytes)
    var scale = Data(count: outDim * blocksIn * 2)

    weight.withUnsafeMutableBytes { wRaw in
        scale.withUnsafeMutableBytes { sRaw in
            bf16Bytes.withUnsafeBytes { bRaw in
                let wPtr = wRaw.bindMemory(to: UInt8.self).baseAddress!
                let sPtr = sRaw.bindMemory(to: UInt16.self).baseAddress!
                let bPtr = bRaw.bindMemory(to: UInt16.self).baseAddress!
                DispatchQueue.concurrentPerform(iterations: outDim) { i in
                    let rowIn = bPtr.advanced(by: i * inDim)
                    let outRow = wPtr.advanced(by: i * packedRowBytes)
                    let scaleRow = sPtr.advanced(by: i * blocksIn)
                    var rowF = [Float](repeating: 0, count: inDim)
                    rowF.withUnsafeMutableBufferPointer { buf in
                        for k in 0..<inDim {
                            buf.baseAddress![k] = bf16ToFloat(rowIn[k])
                        }
                        quantizeRowFromFloatI4(rowPtr: buf.baseAddress!,
                                                outRow: outRow,
                                                scaleRow: scaleRow,
                                                inDim: inDim)
                    }
                }
            }
        }
    }
    return (weight, scale)
}

/// Read a previously-quantized INT8 W8A16 checkpoint
/// (signed-int8 weight + F16 per-128 group scale) and re-quantize
/// it to INT4. Used by the converter for the INT8 → INT4 transcode
/// path so users with an existing INT8 model don't need to re-
/// download the FP8/FP4-native HF weights to halve their disk
/// footprint.
///
/// Block geometry matches both source and destination (128
/// elements per group → same `[N, K/128]` scale shape). Per row we:
///   1. Multiply each int8 value by its F16 block scale → float.
///   2. Re-find max-abs across the 128 floats.
///   3. New scale = max_abs / 7; quantize each to [-8, 7]; pack
///      two-per-byte. (`quantizeRowFromFloatI4`)
///
/// Note this is a double-quantization (8-bit → 4-bit on already-
/// noisy weights) so RTN noise compounds. Expect ~10-20% extra
/// perplexity vs INT4 quantized directly from BF16 / FP8 source.
public func quantizeInt8ToInt4(weightURL: URL, weightOffset: Int,
                                scaleURL: URL, scaleOffset: Int,
                                outDim: Int, inDim: Int) throws -> (weight: Data, scale: Data) {
    precondition(inDim % kInt4GroupK == 0)
    let blocksIn = inDim / kInt4GroupK

    guard let sf = FileHandle(forReadingAtPath: scaleURL.path) else {
        throw NSError(domain: "quantizeInt8ToInt4", code: 1)
    }
    defer { try? sf.close() }
    try sf.seek(toOffset: UInt64(scaleOffset))
    let inScaleBytes = outDim * blocksIn * 2
    guard let sBytes = try sf.read(upToCount: inScaleBytes),
          sBytes.count == inScaleBytes else {
        throw NSError(domain: "quantizeInt8ToInt4", code: 2)
    }

    guard let wf = FileHandle(forReadingAtPath: weightURL.path) else {
        throw NSError(domain: "quantizeInt8ToInt4", code: 3)
    }
    defer { try? wf.close() }
    try wf.seek(toOffset: UInt64(weightOffset))
    let weightLen = outDim * inDim
    guard let wBytes = try wf.read(upToCount: weightLen),
          wBytes.count == weightLen else {
        throw NSError(domain: "quantizeInt8ToInt4", code: 4)
    }

    let packedRowBytes = inDim / 2
    var weight = Data(count: outDim * packedRowBytes)
    var scale = Data(count: outDim * blocksIn * 2)
    weight.withUnsafeMutableBytes { wOutRaw in
        scale.withUnsafeMutableBytes { sOutRaw in
            wBytes.withUnsafeBytes { wInRaw in
                sBytes.withUnsafeBytes { sInRaw in
                    let wOut = wOutRaw.bindMemory(to: UInt8.self).baseAddress!
                    let sOut = sOutRaw.bindMemory(to: UInt16.self).baseAddress!
                    let wIn = wInRaw.bindMemory(to: Int8.self).baseAddress!
                    let sIn = sInRaw.bindMemory(to: UInt16.self).baseAddress!
                    DispatchQueue.concurrentPerform(iterations: outDim) { i in
                        let rowIn = wIn.advanced(by: i * inDim)
                        let scaleRow = sIn.advanced(by: i * blocksIn)
                        var rowF = [Float](repeating: 0, count: inDim)
                        rowF.withUnsafeMutableBufferPointer { buf in
                            for sb in 0..<blocksIn {
                                let s = f16ToFloatLocal(scaleRow[sb])
                                let base = sb * kInt4GroupK
                                for k in 0..<kInt4GroupK {
                                    buf.baseAddress![base + k] = Float(rowIn[base + k]) * s
                                }
                            }
                            quantizeRowFromFloatI4(
                                rowPtr: buf.baseAddress!,
                                outRow: wOut.advanced(by: i * packedRowBytes),
                                scaleRow: sOut.advanced(by: i * blocksIn),
                                inDim: inDim)
                        }
                    }
                }
            }
        }
    }
    return (weight, scale)
}

/// FP8-E4M3 weight + per-128-block E8M0 scale → INT4 + F16. Mirrors
/// `quantizeFP8ToInt8` line-for-line; the only difference is the per-
/// row quantizer.
public func quantizeFP8ToInt4(weightURL: URL, weightOffset: Int,
                              scaleURL: URL, scaleOffset: Int,
                              outDim: Int, inDim: Int,
                              e4m3LUT: [Float],
                              e8m0LUT: [Float]) throws -> (weight: Data, scale: Data) {
    precondition(outDim % 128 == 0 && inDim % 128 == 0)
    precondition(inDim % kInt4GroupK == 0)
    let blocksOut = outDim / 128
    let blocksIn = inDim / 128

    guard let sf = FileHandle(forReadingAtPath: scaleURL.path) else {
        throw NSError(domain: "quantizeFP8ToInt4", code: 1)
    }
    defer { try? sf.close() }
    try sf.seek(toOffset: UInt64(scaleOffset))
    guard let fp8ScaleBytes = try sf.read(upToCount: blocksOut * blocksIn) else {
        throw NSError(domain: "quantizeFP8ToInt4", code: 2)
    }
    guard let wf = FileHandle(forReadingAtPath: weightURL.path) else {
        throw NSError(domain: "quantizeFP8ToInt4", code: 3)
    }
    defer { try? wf.close() }
    try wf.seek(toOffset: UInt64(weightOffset))
    let weightLen = outDim * inDim
    guard let weightBytes = try wf.read(upToCount: weightLen),
          weightBytes.count == weightLen else {
        throw NSError(domain: "quantizeFP8ToInt4", code: 4)
    }

    let packedRowBytes = inDim / 2
    var weight = Data(count: outDim * packedRowBytes)
    var scale = Data(count: outDim * blocksIn * 2)
    weight.withUnsafeMutableBytes { wOutRaw in
        scale.withUnsafeMutableBytes { sOutRaw in
            weightBytes.withUnsafeBytes { wInRaw in
                fp8ScaleBytes.withUnsafeBytes { sInRaw in
                    e4m3LUT.withUnsafeBufferPointer { e4m3 in
                        e8m0LUT.withUnsafeBufferPointer { e8m0 in
                            let wOut = wOutRaw.bindMemory(to: UInt8.self).baseAddress!
                            let sOut = sOutRaw.bindMemory(to: UInt16.self).baseAddress!
                            let wIn = wInRaw.bindMemory(to: UInt8.self).baseAddress!
                            let sIn = sInRaw.bindMemory(to: UInt8.self).baseAddress!
                            let e4m3Ptr = e4m3.baseAddress!
                            let e8m0Ptr = e8m0.baseAddress!
                            DispatchQueue.concurrentPerform(iterations: outDim) { i in
                                let bo = i / 128
                                let rowIn = wIn.advanced(by: i * inDim)
                                var rowF = [Float](repeating: 0, count: inDim)
                                rowF.withUnsafeMutableBufferPointer { buf in
                                    for sb in 0..<blocksIn {
                                        let fp8Scale = e8m0Ptr[Int(sIn[bo * blocksIn + sb])]
                                        let jBase = sb * 128
                                        for k in 0..<128 {
                                            let w = e4m3Ptr[Int(rowIn[jBase + k])]
                                            buf.baseAddress![jBase + k] = w * fp8Scale
                                        }
                                    }
                                    quantizeRowFromFloatI4(
                                        rowPtr: buf.baseAddress!,
                                        outRow: wOut.advanced(by: i * packedRowBytes),
                                        scaleRow: sOut.advanced(by: i * blocksIn),
                                        inDim: inDim)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return (weight, scale)
}

/// FP4-E2M1 weight (packed two-per-byte) + per-32-block E8M0 scale → INT4.
public func quantizeFP4ToInt4(weightURL: URL, weightOffset: Int,
                              scaleURL: URL, scaleOffset: Int,
                              outDim: Int, inDim: Int,
                              e2m1LUT: [Float],
                              e8m0LUT: [Float]) throws -> (weight: Data, scale: Data) {
    precondition(inDim % 32 == 0)
    precondition(inDim % kInt4GroupK == 0)
    let fp4BlocksIn = inDim / 32
    let int4BlocksIn = inDim / kInt4GroupK

    guard let sf = FileHandle(forReadingAtPath: scaleURL.path) else {
        throw NSError(domain: "quantizeFP4ToInt4", code: 1)
    }
    defer { try? sf.close() }
    try sf.seek(toOffset: UInt64(scaleOffset))
    guard let fp4ScaleBytes = try sf.read(upToCount: outDim * fp4BlocksIn) else {
        throw NSError(domain: "quantizeFP4ToInt4", code: 2)
    }
    guard let wf = FileHandle(forReadingAtPath: weightURL.path) else {
        throw NSError(domain: "quantizeFP4ToInt4", code: 3)
    }
    defer { try? wf.close() }
    try wf.seek(toOffset: UInt64(weightOffset))
    let fp4PackedRowBytes = inDim / 2
    let weightLen = outDim * fp4PackedRowBytes
    guard let weightBytes = try wf.read(upToCount: weightLen),
          weightBytes.count == weightLen else {
        throw NSError(domain: "quantizeFP4ToInt4", code: 4)
    }

    let outPackedRowBytes = inDim / 2
    var weight = Data(count: outDim * outPackedRowBytes)
    var scale = Data(count: outDim * int4BlocksIn * 2)
    weight.withUnsafeMutableBytes { wOutRaw in
        scale.withUnsafeMutableBytes { sOutRaw in
            weightBytes.withUnsafeBytes { wInRaw in
                fp4ScaleBytes.withUnsafeBytes { sInRaw in
                    e2m1LUT.withUnsafeBufferPointer { e2m1 in
                        e8m0LUT.withUnsafeBufferPointer { e8m0 in
                            let wOut = wOutRaw.bindMemory(to: UInt8.self).baseAddress!
                            let sOut = sOutRaw.bindMemory(to: UInt16.self).baseAddress!
                            let wIn = wInRaw.bindMemory(to: UInt8.self).baseAddress!
                            let sIn = sInRaw.bindMemory(to: UInt8.self).baseAddress!
                            let e2m1Ptr = e2m1.baseAddress!
                            let e8m0Ptr = e8m0.baseAddress!
                            DispatchQueue.concurrentPerform(iterations: outDim) { i in
                                let rowIn = wIn.advanced(by: i * fp4PackedRowBytes)
                                let scaleRow = sIn.advanced(by: i * fp4BlocksIn)
                                var rowF = [Float](repeating: 0, count: inDim)
                                rowF.withUnsafeMutableBufferPointer { buf in
                                    for sb in 0..<fp4BlocksIn {
                                        let s = e8m0Ptr[Int(scaleRow[sb])]
                                        let inBase = sb * 16
                                        let outBase = sb * 32
                                        for k in 0..<16 {
                                            let byte = rowIn[inBase + k]
                                            let vLow  = e2m1Ptr[Int(byte & 0xF)] * s
                                            let vHigh = e2m1Ptr[Int(byte >> 4)] * s
                                            buf.baseAddress![outBase + 2 * k]     = vLow
                                            buf.baseAddress![outBase + 2 * k + 1] = vHigh
                                        }
                                    }
                                    quantizeRowFromFloatI4(
                                        rowPtr: buf.baseAddress!,
                                        outRow: wOut.advanced(by: i * outPackedRowBytes),
                                        scaleRow: sOut.advanced(by: i * int4BlocksIn),
                                        inDim: inDim)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return (weight, scale)
}

/// F32 source variant. Same contract as quantizeBF16ToInt4.
public func quantizeF32ToInt4(srcURL: URL, srcOffset: Int,
                               outDim: Int, inDim: Int) throws -> (weight: Data, scale: Data) {
    precondition(inDim % kInt4GroupK == 0)
    let blocksIn = inDim / kInt4GroupK

    guard let fh = FileHandle(forReadingAtPath: srcURL.path) else {
        throw NSError(domain: "quantizeF32ToInt4", code: 1)
    }
    defer { try? fh.close() }
    try fh.seek(toOffset: UInt64(srcOffset))
    let byteLen = outDim * inDim * 4
    guard let f32Bytes = try fh.read(upToCount: byteLen),
          f32Bytes.count == byteLen else {
        throw NSError(domain: "quantizeF32ToInt4", code: 2)
    }

    let packedRowBytes = inDim / 2
    var weight = Data(count: outDim * packedRowBytes)
    var scale = Data(count: outDim * blocksIn * 2)
    weight.withUnsafeMutableBytes { wRaw in
        scale.withUnsafeMutableBytes { sRaw in
            f32Bytes.withUnsafeBytes { fRaw in
                let wPtr = wRaw.bindMemory(to: UInt8.self).baseAddress!
                let sPtr = sRaw.bindMemory(to: UInt16.self).baseAddress!
                let fPtr = fRaw.bindMemory(to: Float.self).baseAddress!
                DispatchQueue.concurrentPerform(iterations: outDim) { i in
                    let rowIn = fPtr.advanced(by: i * inDim)
                    let outRow = wPtr.advanced(by: i * packedRowBytes)
                    let scaleRow = sPtr.advanced(by: i * blocksIn)
                    quantizeRowFromFloatI4(rowPtr: rowIn,
                                            outRow: outRow,
                                            scaleRow: scaleRow,
                                            inDim: inDim)
                }
            }
        }
    }
    return (weight, scale)
}
