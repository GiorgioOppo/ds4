import Foundation

// INT2 W2A16 weight quantization. Same column-grouping (128 elements
// per scale block, F16 scales) as INT4/INT8; the only differences are:
//   - 2-bit two's-complement values, range [-2, 1]
//   - four-per-byte packing (LSB first: bits[1:0]=4k, [3:2]=4k+1,
//     [5:4]=4k+2, [7:6]=4k+3)
//   - scale = max_abs / 2 (positive max is 1; -2 reaches -max_abs)
//
// Accuracy at this bit-width is poor for plain RTN; INT2 is intended
// for experimentation / memory-bound runs, not production. Tolerances
// in Int2GemmTests reflect this.
//
//   weight: [N, K/4]    I2 packed (1 byte per 4 elements)
//   scale:  [N, K/128]  F16

public let kInt2GroupK: Int = 128

public func shouldQuantizeToInt2(_ renamedName: String, lastDim: Int) -> Bool {
    guard renamedName.hasSuffix(".weight") else { return false }
    guard lastDim % kInt2GroupK == 0 else { return false }
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

/// Pack four signed-2-bit values into one byte (LSB first).
/// Caller has already clamped to [-2, 1]; we mask to 2 bits.
@inline(__always)
private func packI2(_ a: Int8, _ b: Int8, _ c: Int8, _ d: Int8) -> UInt8 {
    let v0 = UInt8(truncatingIfNeeded: a) & 0x3
    let v1 = UInt8(truncatingIfNeeded: b) & 0x3
    let v2 = UInt8(truncatingIfNeeded: c) & 0x3
    let v3 = UInt8(truncatingIfNeeded: d) & 0x3
    return v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
}

@inline(__always)
private func quantizeRowFromFloatI2(rowPtr: UnsafePointer<Float>,
                                     outRow: UnsafeMutablePointer<UInt8>,
                                     scaleRow: UnsafeMutablePointer<UInt16>,
                                     inDim: Int) {
    let blocksIn = inDim / kInt2GroupK
    for sb in 0..<blocksIn {
        let base = sb * kInt2GroupK
        var maxAbs: Float = 0
        for k in 0..<kInt2GroupK {
            let v = abs(rowPtr[base + k])
            if v > maxAbs { maxAbs = v }
        }
        // Scale = max_abs / 2 because the negative side reaches -2*s.
        // Positives clip at 1*s = max_abs/2 — accepted asymmetry; the
        // alternative (max_abs/1.5) leaves -2 unreachable for half the
        // weights.
        let s: Float
        let invS: Float
        if maxAbs == 0 {
            s = 0; invS = 0
        } else {
            s = maxAbs / 2.0
            invS = 2.0 / maxAbs
        }
        scaleRow[sb] = floatToF16Local(s)
        for k in stride(from: 0, to: kInt2GroupK, by: 4) {
            let q0 = (rowPtr[base + k    ] * invS).rounded(.toNearestOrEven)
            let q1 = (rowPtr[base + k + 1] * invS).rounded(.toNearestOrEven)
            let q2 = (rowPtr[base + k + 2] * invS).rounded(.toNearestOrEven)
            let q3 = (rowPtr[base + k + 3] * invS).rounded(.toNearestOrEven)
            let c0 = Int8(min(max(q0, -2), 1))
            let c1 = Int8(min(max(q1, -2), 1))
            let c2 = Int8(min(max(q2, -2), 1))
            let c3 = Int8(min(max(q3, -2), 1))
            outRow[(base + k) / 4] = packI2(c0, c1, c2, c3)
        }
    }
}

public func quantizeBF16ToInt2(srcURL: URL, srcOffset: Int,
                                outDim: Int, inDim: Int) throws -> (weight: Data, scale: Data) {
    precondition(inDim % kInt2GroupK == 0)
    let blocksIn = inDim / kInt2GroupK

    guard let fh = FileHandle(forReadingAtPath: srcURL.path) else {
        throw NSError(domain: "quantizeBF16ToInt2", code: 1)
    }
    defer { try? fh.close() }
    try fh.seek(toOffset: UInt64(srcOffset))
    let byteLen = outDim * inDim * 2
    guard let bf16Bytes = try fh.read(upToCount: byteLen),
          bf16Bytes.count == byteLen else {
        throw NSError(domain: "quantizeBF16ToInt2", code: 2)
    }

    let packedRowBytes = inDim / 4
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
                        quantizeRowFromFloatI2(rowPtr: buf.baseAddress!,
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

public func quantizeF32ToInt2(srcURL: URL, srcOffset: Int,
                               outDim: Int, inDim: Int) throws -> (weight: Data, scale: Data) {
    precondition(inDim % kInt2GroupK == 0)
    let blocksIn = inDim / kInt2GroupK

    guard let fh = FileHandle(forReadingAtPath: srcURL.path) else {
        throw NSError(domain: "quantizeF32ToInt2", code: 1)
    }
    defer { try? fh.close() }
    try fh.seek(toOffset: UInt64(srcOffset))
    let byteLen = outDim * inDim * 4
    guard let f32Bytes = try fh.read(upToCount: byteLen),
          f32Bytes.count == byteLen else {
        throw NSError(domain: "quantizeF32ToInt2", code: 2)
    }

    let packedRowBytes = inDim / 4
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
                    quantizeRowFromFloatI2(rowPtr: rowIn,
                                            outRow: outRow,
                                            scaleRow: scaleRow,
                                            inDim: inDim)
                }
            }
        }
    }
    return (weight, scale)
}

/// Read a previously-quantized INT8 W8A16 checkpoint
/// (signed-int8 weight + F16 per-128 group scale) and re-quantize
/// to INT2. Sister of `quantizeInt8ToInt4` for the more aggressive
/// 2-bit target. The double-quantization noise is even more
/// pronounced here — accuracy hit is severe; treat as a
/// memory-bound experiment, not a production path.
public func quantizeInt8ToInt2(weightURL: URL, weightOffset: Int,
                                scaleURL: URL, scaleOffset: Int,
                                outDim: Int, inDim: Int) throws -> (weight: Data, scale: Data) {
    precondition(inDim % kInt2GroupK == 0)
    let blocksIn = inDim / kInt2GroupK

    guard let sf = FileHandle(forReadingAtPath: scaleURL.path) else {
        throw NSError(domain: "quantizeInt8ToInt2", code: 1)
    }
    defer { try? sf.close() }
    try sf.seek(toOffset: UInt64(scaleOffset))
    let inScaleBytes = outDim * blocksIn * 2
    guard let sBytes = try sf.read(upToCount: inScaleBytes),
          sBytes.count == inScaleBytes else {
        throw NSError(domain: "quantizeInt8ToInt2", code: 2)
    }

    guard let wf = FileHandle(forReadingAtPath: weightURL.path) else {
        throw NSError(domain: "quantizeInt8ToInt2", code: 3)
    }
    defer { try? wf.close() }
    try wf.seek(toOffset: UInt64(weightOffset))
    let weightLen = outDim * inDim
    guard let wBytes = try wf.read(upToCount: weightLen),
          wBytes.count == weightLen else {
        throw NSError(domain: "quantizeInt8ToInt2", code: 4)
    }

    let packedRowBytes = inDim / 4
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
                                let base = sb * kInt2GroupK
                                for k in 0..<kInt2GroupK {
                                    buf.baseAddress![base + k] = Float(rowIn[base + k]) * s
                                }
                            }
                            quantizeRowFromFloatI2(
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

public func quantizeFP8ToInt2(weightURL: URL, weightOffset: Int,
                              scaleURL: URL, scaleOffset: Int,
                              outDim: Int, inDim: Int,
                              e4m3LUT: [Float],
                              e8m0LUT: [Float]) throws -> (weight: Data, scale: Data) {
    precondition(outDim % 128 == 0 && inDim % 128 == 0)
    precondition(inDim % kInt2GroupK == 0)
    let blocksOut = outDim / 128
    let blocksIn = inDim / 128

    guard let sf = FileHandle(forReadingAtPath: scaleURL.path) else {
        throw NSError(domain: "quantizeFP8ToInt2", code: 1)
    }
    defer { try? sf.close() }
    try sf.seek(toOffset: UInt64(scaleOffset))
    guard let fp8ScaleBytes = try sf.read(upToCount: blocksOut * blocksIn) else {
        throw NSError(domain: "quantizeFP8ToInt2", code: 2)
    }
    guard let wf = FileHandle(forReadingAtPath: weightURL.path) else {
        throw NSError(domain: "quantizeFP8ToInt2", code: 3)
    }
    defer { try? wf.close() }
    try wf.seek(toOffset: UInt64(weightOffset))
    let weightLen = outDim * inDim
    guard let weightBytes = try wf.read(upToCount: weightLen),
          weightBytes.count == weightLen else {
        throw NSError(domain: "quantizeFP8ToInt2", code: 4)
    }

    let packedRowBytes = inDim / 4
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
                                    quantizeRowFromFloatI2(
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

public func quantizeFP4ToInt2(weightURL: URL, weightOffset: Int,
                              scaleURL: URL, scaleOffset: Int,
                              outDim: Int, inDim: Int,
                              e2m1LUT: [Float],
                              e8m0LUT: [Float]) throws -> (weight: Data, scale: Data) {
    precondition(inDim % 32 == 0)
    precondition(inDim % kInt2GroupK == 0)
    let fp4BlocksIn = inDim / 32
    let int2BlocksIn = inDim / kInt2GroupK

    guard let sf = FileHandle(forReadingAtPath: scaleURL.path) else {
        throw NSError(domain: "quantizeFP4ToInt2", code: 1)
    }
    defer { try? sf.close() }
    try sf.seek(toOffset: UInt64(scaleOffset))
    guard let fp4ScaleBytes = try sf.read(upToCount: outDim * fp4BlocksIn) else {
        throw NSError(domain: "quantizeFP4ToInt2", code: 2)
    }
    guard let wf = FileHandle(forReadingAtPath: weightURL.path) else {
        throw NSError(domain: "quantizeFP4ToInt2", code: 3)
    }
    defer { try? wf.close() }
    try wf.seek(toOffset: UInt64(weightOffset))
    let fp4PackedRowBytes = inDim / 2
    let weightLen = outDim * fp4PackedRowBytes
    guard let weightBytes = try wf.read(upToCount: weightLen),
          weightBytes.count == weightLen else {
        throw NSError(domain: "quantizeFP4ToInt2", code: 4)
    }

    let outPackedRowBytes = inDim / 4
    var weight = Data(count: outDim * outPackedRowBytes)
    var scale = Data(count: outDim * int2BlocksIn * 2)
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
                                    quantizeRowFromFloatI2(
                                        rowPtr: buf.baseAddress!,
                                        outRow: wOut.advanced(by: i * outPackedRowBytes),
                                        scaleRow: sOut.advanced(by: i * int2BlocksIn),
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
