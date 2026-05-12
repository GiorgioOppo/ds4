import Foundation

// FP8 / FP4 weight + per-block E8M0 scale → dense BF16 or F16 in
// one pass. Used by `Converter.runQuantize` for the "fuse"
// path (target=bf16/f16, source FP8 or FP4). Both functions
// parallelise across output rows via DispatchQueue.concurrentPerform.

/// Weight: `[out, in]` FP8-E4M3. Scale: `[out/128, in/128]` E8M0.
///   fused[i, j] = deqE4M3(W[i,j]) * deqE8M0(S[i/128, j/128])
/// Output: `[out, in]` in `target` (BF16 or F16). Rows are
/// dequantized in parallel; the input weight is slurped once so
/// workers index it without serializing on a shared FileHandle.
public func fuseFP8ToNative(weightURL: URL, weightOffset: Int,
                             scaleURL: URL, scaleOffset: Int,
                             outDim: Int, inDim: Int,
                             target: ConversionTarget) throws -> Data {
    precondition(outDim % 128 == 0 && inDim % 128 == 0,
                 "FP8 weight dims must be 128-aligned")
    let blocksOut = outDim / 128
    let blocksIn = inDim / 128

    guard let sf = FileHandle(forReadingAtPath: scaleURL.path) else {
        throw NSError(domain: "fuseFP8", code: 1)
    }
    defer { try? sf.close() }
    try sf.seek(toOffset: UInt64(scaleOffset))
    guard let scaleBytes = try sf.read(upToCount: blocksOut * blocksIn) else {
        throw NSError(domain: "fuseFP8", code: 2)
    }

    guard let wf = FileHandle(forReadingAtPath: weightURL.path) else {
        throw NSError(domain: "fuseFP8", code: 3)
    }
    defer { try? wf.close() }
    try wf.seek(toOffset: UInt64(weightOffset))
    let weightLen = outDim * inDim
    guard let weightBytes = try wf.read(upToCount: weightLen),
          weightBytes.count == weightLen else {
        throw NSError(domain: "fuseFP8", code: 4)
    }

    var out = Data(count: outDim * inDim * 2)
    out.withUnsafeMutableBytes { outRaw in
        let outPtr = outRaw.bindMemory(to: UInt16.self).baseAddress!
        scaleBytes.withUnsafeBytes { scaleRaw in
            let scalePtr = scaleRaw.bindMemory(to: UInt8.self).baseAddress!
            weightBytes.withUnsafeBytes { wRaw in
                let wPtr = wRaw.bindMemory(to: UInt8.self).baseAddress!
                e4m3LUT.withUnsafeBufferPointer { e4m3 in
                    e8m0LUT.withUnsafeBufferPointer { e8m0 in
                        let e4m3Ptr = e4m3.baseAddress!
                        let e8m0Ptr = e8m0.baseAddress!
                        DispatchQueue.concurrentPerform(iterations: outDim) { i in
                            let bo = i / 128
                            let rowIn = wPtr.advanced(by: i * inDim)
                            let rowOut = outPtr.advanced(by: i * inDim)
                            for sb in 0..<blocksIn {
                                let s = e8m0Ptr[Int(scalePtr[bo * blocksIn + sb])]
                                let jBase = sb * 128
                                for k in 0..<128 {
                                    let w = e4m3Ptr[Int(rowIn[jBase + k])]
                                    rowOut[jBase + k] = packNative(w * s, target)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return out
}

/// Weight: `[out, in/2]` packed FP4-E2M1 (low nibble = element 2*i,
/// high nibble = element 2*i+1). Scale: `[out, in/32]` E8M0.
///   fused[i, j] = deqE2M1(W_nibble(i, j)) * deqE8M0(S[i, j/32])
/// Output: `[out, in]` in `target` — 4× the input weight bytes.
public func fuseFP4ToNative(weightURL: URL, weightOffset: Int,
                             scaleURL: URL, scaleOffset: Int,
                             outDim: Int, inDim: Int,
                             target: ConversionTarget) throws -> Data {
    precondition(inDim % 32 == 0, "FP4 weight inDim must be 32-aligned")
    precondition(inDim % 2 == 0, "FP4 weight inDim must be even (packed pairs)")
    let blocksIn = inDim / 32

    guard let sf = FileHandle(forReadingAtPath: scaleURL.path) else {
        throw NSError(domain: "fuseFP4", code: 1)
    }
    defer { try? sf.close() }
    try sf.seek(toOffset: UInt64(scaleOffset))
    guard let scaleBytes = try sf.read(upToCount: outDim * blocksIn) else {
        throw NSError(domain: "fuseFP4", code: 2)
    }

    guard let wf = FileHandle(forReadingAtPath: weightURL.path) else {
        throw NSError(domain: "fuseFP4", code: 3)
    }
    defer { try? wf.close() }
    try wf.seek(toOffset: UInt64(weightOffset))
    let packedRowBytes = inDim / 2
    let weightLen = outDim * packedRowBytes
    guard let weightBytes = try wf.read(upToCount: weightLen),
          weightBytes.count == weightLen else {
        throw NSError(domain: "fuseFP4", code: 4)
    }

    var out = Data(count: outDim * inDim * 2)
    out.withUnsafeMutableBytes { outRaw in
        let outPtr = outRaw.bindMemory(to: UInt16.self).baseAddress!
        scaleBytes.withUnsafeBytes { scaleRaw in
            let scalePtr = scaleRaw.bindMemory(to: UInt8.self).baseAddress!
            weightBytes.withUnsafeBytes { wRaw in
                let wPtr = wRaw.bindMemory(to: UInt8.self).baseAddress!
                e2m1LUT.withUnsafeBufferPointer { e2m1 in
                    e8m0LUT.withUnsafeBufferPointer { e8m0 in
                        let e2m1Ptr = e2m1.baseAddress!
                        let e8m0Ptr = e8m0.baseAddress!
                        DispatchQueue.concurrentPerform(iterations: outDim) { i in
                            let rowIn = wPtr.advanced(by: i * packedRowBytes)
                            let rowOut = outPtr.advanced(by: i * inDim)
                            let scaleRow = scalePtr.advanced(by: i * blocksIn)
                            for sb in 0..<blocksIn {
                                let s = e8m0Ptr[Int(scaleRow[sb])]
                                let inBase = sb * 16
                                let outBase = sb * 32
                                for k in 0..<16 {
                                    let byte = rowIn[inBase + k]
                                    let vLow  = e2m1Ptr[Int(byte & 0xF)] * s
                                    let vHigh = e2m1Ptr[Int(byte >> 4)] * s
                                    rowOut[outBase + 2 * k]     = packNative(vLow, target)
                                    rowOut[outBase + 2 * k + 1] = packNative(vHigh, target)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return out
}
