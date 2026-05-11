import Foundation

// INT8 W8A16 weight quantization.
//
// Symmetric round-to-nearest, range [-127, 127] (the `-128` slot stays
// unused so that `q * s` is exactly representable in both directions and we
// don't have to special-case `abs(min)`). Scale layout: per row, per group
// of K=128 input-features columns. Scale dtype is F16 (10-bit mantissa is
// enough for the magnitudes of typical LLM weights, BF16's 7-bit mantissa
// would lose precision on small-magnitude weights).
//
// Output layout matches the existing `<base>.weight` + `<base>.scale`
// safetensors convention so the runtime loader (`Assembly.swift:loadLinear`)
// picks both up automatically.
//
//   weight: [N, K]      I8   (1 byte/elem)
//   scale:  [N, K/128]  F16  (2 bytes/group)
//
// All quantizers below run in parallel across rows via
// `DispatchQueue.concurrentPerform`, matching the structure of
// `fuseFP8ToNative` / `fuseFP4ToNative` in `main.swift`.

/// INT8 quantization is row-grouped over the input-feature axis with a
/// hard group size of 128. The same constant is baked into the Metal kernel
/// (`gemm_int8_w8a16_to_f32` in `int8_gemm.metal`).
public let kInt8GroupK: Int = 128

/// Returns true if `renamedName` is a `Linear` weight that should be
/// quantized to INT8. We restrict to leaves the model actually consumes via
/// `Linear`; everything else (embeddings, RMSNorm gains, attn_sink, hyper-
/// connection scalars, biases, gate matrices that need full precision)
/// passes through as BF16.
///
/// `lastDim` is the K dimension of the weight (last shape entry, which is
/// the input-feature count for `Linear` weights in this codebase). The
/// `kInt8GroupK`-alignment check guarantees the grouped-scale layout fits
/// cleanly with no per-tensor padding logic.
public func shouldQuantizeToInt8(_ renamedName: String, lastDim: Int) -> Bool {
    guard renamedName.hasSuffix(".weight") else { return false }
    guard lastDim % kInt8GroupK == 0 else { return false }

    // Leaf of the renamed name (excluding the trailing `.weight`).
    let stripped = String(renamedName.dropLast(".weight".count))
    let parts = stripped.split(separator: ".").map(String.init)
    guard let leaf = parts.last else { return false }

    // Whitelist of leaves we know are `Linear` weights consumed via
    // `Linear.callAsFunction`. Mirrors the `loadLinear` call sites in
    // Sources/DeepSeekKit/Assembly.swift.
    //
    // Intentionally excluded:
    // - `wo_a`: its weight is read directly by `Einsum.bsgdGrd` in
    //   MLA.forward (Attention.swift:305-306), which requires F32.
    //   Quantizing it to INT8 would break inference.
    // - `gate`, `weights_proj`: small routing/scoring matrices used by
    //   `MoE.Gate` and `Indexer`. RTN noise on these directly perturbs
    //   the routing/topk selection, so we keep them BF16.
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
    // Same algorithm as `floatToF16` in main.swift; duplicated here so this
    // file is self-contained and can be unit-tested in isolation.
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

/// Symmetric INT8 RTN over a row of `inDim` floats. Writes `inDim` bytes
/// to `outRow` and `blocksIn` F16 scales to `scaleRow`. The reciprocal of
/// the absolute-max in each group is computed once and reused for the
/// inner multiply (avoids a divide in the hot loop).
@inline(__always)
private func quantizeRowFromFloat(rowPtr: UnsafePointer<Float>,
                                   outRow: UnsafeMutablePointer<Int8>,
                                   scaleRow: UnsafeMutablePointer<UInt16>,
                                   inDim: Int) {
    let blocksIn = inDim / kInt8GroupK
    for sb in 0..<blocksIn {
        let base = sb * kInt8GroupK
        var maxAbs: Float = 0
        for k in 0..<kInt8GroupK {
            let v = abs(rowPtr[base + k])
            if v > maxAbs { maxAbs = v }
        }
        let s: Float
        let invS: Float
        if maxAbs == 0 {
            // Encode an all-zero group as scale=0 so dequant produces zeros.
            s = 0
            invS = 0
        } else {
            s = maxAbs / 127.0
            invS = 127.0 / maxAbs
        }
        scaleRow[sb] = floatToF16Local(s)
        for k in 0..<kInt8GroupK {
            let q = rowPtr[base + k] * invS
            // Round to nearest even, then clamp to [-127, 127].
            let r = q.rounded(.toNearestOrEven)
            let clamped = min(max(r, -127), 127)
            outRow[base + k] = Int8(clamped)
        }
    }
}

/// Quantize a `[outDim, inDim]` BF16 weight tensor stored at
/// `(srcURL, srcOffset)` into INT8 + F16 group scales.
/// Returns `(weight: [outDim*inDim] I8, scale: [outDim*blocksIn] F16)`.
public func quantizeBF16ToInt8(srcURL: URL, srcOffset: Int,
                               outDim: Int, inDim: Int) throws -> (weight: Data, scale: Data) {
    precondition(inDim % kInt8GroupK == 0,
                 "INT8 quant requires inDim % \(kInt8GroupK) == 0")
    let blocksIn = inDim / kInt8GroupK

    guard let fh = FileHandle(forReadingAtPath: srcURL.path) else {
        throw NSError(domain: "quantizeBF16ToInt8", code: 1)
    }
    defer { try? fh.close() }
    try fh.seek(toOffset: UInt64(srcOffset))
    let byteLen = outDim * inDim * 2
    guard let bf16Bytes = try fh.read(upToCount: byteLen),
          bf16Bytes.count == byteLen else {
        throw NSError(domain: "quantizeBF16ToInt8", code: 2)
    }

    var weight = Data(count: outDim * inDim)
    var scale = Data(count: outDim * blocksIn * 2)

    weight.withUnsafeMutableBytes { wRaw in
        scale.withUnsafeMutableBytes { sRaw in
            bf16Bytes.withUnsafeBytes { bRaw in
                let wPtr = wRaw.bindMemory(to: Int8.self).baseAddress!
                let sPtr = sRaw.bindMemory(to: UInt16.self).baseAddress!
                let bPtr = bRaw.bindMemory(to: UInt16.self).baseAddress!
                DispatchQueue.concurrentPerform(iterations: outDim) { i in
                    let rowIn = bPtr.advanced(by: i * inDim)
                    let outRow = wPtr.advanced(by: i * inDim)
                    let scaleRow = sPtr.advanced(by: i * blocksIn)
                    // Materialize the row as Float once (cheaper than
                    // re-converting BF16 in both the max-abs and quantize
                    // passes).
                    var rowF = [Float](repeating: 0, count: inDim)
                    rowF.withUnsafeMutableBufferPointer { buf in
                        for k in 0..<inDim {
                            buf.baseAddress![k] = bf16ToFloat(rowIn[k])
                        }
                        quantizeRowFromFloat(rowPtr: buf.baseAddress!,
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

/// Quantize a `[outDim, inDim]` F32 weight tensor stored at
/// `(srcURL, srcOffset)` into INT8 + F16 group scales. Same contract as
/// `quantizeBF16ToInt8`.
public func quantizeF32ToInt8(srcURL: URL, srcOffset: Int,
                              outDim: Int, inDim: Int) throws -> (weight: Data, scale: Data) {
    precondition(inDim % kInt8GroupK == 0,
                 "INT8 quant requires inDim % \(kInt8GroupK) == 0")
    let blocksIn = inDim / kInt8GroupK

    guard let fh = FileHandle(forReadingAtPath: srcURL.path) else {
        throw NSError(domain: "quantizeF32ToInt8", code: 1)
    }
    defer { try? fh.close() }
    try fh.seek(toOffset: UInt64(srcOffset))
    let byteLen = outDim * inDim * 4
    guard let f32Bytes = try fh.read(upToCount: byteLen),
          f32Bytes.count == byteLen else {
        throw NSError(domain: "quantizeF32ToInt8", code: 2)
    }

    var weight = Data(count: outDim * inDim)
    var scale = Data(count: outDim * blocksIn * 2)
    weight.withUnsafeMutableBytes { wRaw in
        scale.withUnsafeMutableBytes { sRaw in
            f32Bytes.withUnsafeBytes { fRaw in
                let wPtr = wRaw.bindMemory(to: Int8.self).baseAddress!
                let sPtr = sRaw.bindMemory(to: UInt16.self).baseAddress!
                let fPtr = fRaw.bindMemory(to: Float.self).baseAddress!
                DispatchQueue.concurrentPerform(iterations: outDim) { i in
                    let rowIn = fPtr.advanced(by: i * inDim)
                    let outRow = wPtr.advanced(by: i * inDim)
                    let scaleRow = sPtr.advanced(by: i * blocksIn)
                    quantizeRowFromFloat(rowPtr: rowIn,
                                          outRow: outRow,
                                          scaleRow: scaleRow,
                                          inDim: inDim)
                }
            }
        }
    }
    return (weight, scale)
}

/// Quantize a `[outDim, inDim]` FP8-E4M3 weight + per-128-block E8M0 scale
/// directly into INT8 + F16 group scales, without staging through BF16.
/// `e4m3LUT` and `e8m0LUT` are the dequant lookup tables already populated
/// in `main.swift`.
public func quantizeFP8ToInt8(weightURL: URL, weightOffset: Int,
                              scaleURL: URL, scaleOffset: Int,
                              outDim: Int, inDim: Int,
                              e4m3LUT: [Float],
                              e8m0LUT: [Float]) throws -> (weight: Data, scale: Data) {
    precondition(outDim % 128 == 0 && inDim % 128 == 0,
                 "FP8 weight dims must be 128-aligned")
    precondition(inDim % kInt8GroupK == 0)
    let blocksOut = outDim / 128
    let blocksIn = inDim / 128   // also the INT8 group count per row.

    guard let sf = FileHandle(forReadingAtPath: scaleURL.path) else {
        throw NSError(domain: "quantizeFP8ToInt8", code: 1)
    }
    defer { try? sf.close() }
    try sf.seek(toOffset: UInt64(scaleOffset))
    guard let fp8ScaleBytes = try sf.read(upToCount: blocksOut * blocksIn) else {
        throw NSError(domain: "quantizeFP8ToInt8", code: 2)
    }

    guard let wf = FileHandle(forReadingAtPath: weightURL.path) else {
        throw NSError(domain: "quantizeFP8ToInt8", code: 3)
    }
    defer { try? wf.close() }
    try wf.seek(toOffset: UInt64(weightOffset))
    let weightLen = outDim * inDim
    guard let weightBytes = try wf.read(upToCount: weightLen),
          weightBytes.count == weightLen else {
        throw NSError(domain: "quantizeFP8ToInt8", code: 4)
    }

    var weight = Data(count: outDim * inDim)
    var scale = Data(count: outDim * blocksIn * 2)
    weight.withUnsafeMutableBytes { wOutRaw in
        scale.withUnsafeMutableBytes { sOutRaw in
            weightBytes.withUnsafeBytes { wInRaw in
                fp8ScaleBytes.withUnsafeBytes { sInRaw in
                    e4m3LUT.withUnsafeBufferPointer { e4m3 in
                        e8m0LUT.withUnsafeBufferPointer { e8m0 in
                            let wOut = wOutRaw.bindMemory(to: Int8.self).baseAddress!
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
                                    quantizeRowFromFloat(
                                        rowPtr: buf.baseAddress!,
                                        outRow: wOut.advanced(by: i * inDim),
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

/// Quantize a `[outDim, inDim]` FP4-E2M1 weight (packed two-per-byte) +
/// per-32-block E8M0 scale into INT8 + F16 group scales (group=128).
/// The FP4 scale is at 32-element granularity while INT8 uses 128, so each
/// INT8 group folds 4 FP4 scale blocks.
public func quantizeFP4ToInt8(weightURL: URL, weightOffset: Int,
                              scaleURL: URL, scaleOffset: Int,
                              outDim: Int, inDim: Int,
                              e2m1LUT: [Float],
                              e8m0LUT: [Float]) throws -> (weight: Data, scale: Data) {
    precondition(inDim % 32 == 0, "FP4 weight inDim must be 32-aligned")
    precondition(inDim % kInt8GroupK == 0)
    let fp4BlocksIn = inDim / 32
    let int8BlocksIn = inDim / kInt8GroupK

    guard let sf = FileHandle(forReadingAtPath: scaleURL.path) else {
        throw NSError(domain: "quantizeFP4ToInt8", code: 1)
    }
    defer { try? sf.close() }
    try sf.seek(toOffset: UInt64(scaleOffset))
    guard let fp4ScaleBytes = try sf.read(upToCount: outDim * fp4BlocksIn) else {
        throw NSError(domain: "quantizeFP4ToInt8", code: 2)
    }

    guard let wf = FileHandle(forReadingAtPath: weightURL.path) else {
        throw NSError(domain: "quantizeFP4ToInt8", code: 3)
    }
    defer { try? wf.close() }
    try wf.seek(toOffset: UInt64(weightOffset))
    let packedRowBytes = inDim / 2
    let weightLen = outDim * packedRowBytes
    guard let weightBytes = try wf.read(upToCount: weightLen),
          weightBytes.count == weightLen else {
        throw NSError(domain: "quantizeFP4ToInt8", code: 4)
    }

    var weight = Data(count: outDim * inDim)
    var scale = Data(count: outDim * int8BlocksIn * 2)
    weight.withUnsafeMutableBytes { wOutRaw in
        scale.withUnsafeMutableBytes { sOutRaw in
            weightBytes.withUnsafeBytes { wInRaw in
                fp4ScaleBytes.withUnsafeBytes { sInRaw in
                    e2m1LUT.withUnsafeBufferPointer { e2m1 in
                        e8m0LUT.withUnsafeBufferPointer { e8m0 in
                            let wOut = wOutRaw.bindMemory(to: Int8.self).baseAddress!
                            let sOut = sOutRaw.bindMemory(to: UInt16.self).baseAddress!
                            let wIn = wInRaw.bindMemory(to: UInt8.self).baseAddress!
                            let sIn = sInRaw.bindMemory(to: UInt8.self).baseAddress!
                            let e2m1Ptr = e2m1.baseAddress!
                            let e8m0Ptr = e8m0.baseAddress!
                            DispatchQueue.concurrentPerform(iterations: outDim) { i in
                                let rowIn = wIn.advanced(by: i * packedRowBytes)
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
                                    quantizeRowFromFloat(
                                        rowPtr: buf.baseAddress!,
                                        outRow: wOut.advanced(by: i * inDim),
                                        scaleRow: sOut.advanced(by: i * int8BlocksIn),
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
