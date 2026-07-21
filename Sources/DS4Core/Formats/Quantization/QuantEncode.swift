import Foundation

/// CPU quantization ENCODERS for the GGUF writer — the Swift port of
/// `gguf-tools/quants.c` from the MIT-licensed ds4 repository (itself derived
/// from the GGML/llama.cpp quantizers). Byte-layout compatibility with those
/// references is the contract here: the unit tests pin every output format
/// byte-for-byte against fixtures generated with the compiled C reference
/// (built with `-ffp-contract=off` so both sides do plain float32 ops).
///
/// Implemented output targets — the DS4/GLM recipe formats only:
/// `q8_0` (8), `q2_K` (10), `q4_K` (12), `q8_K` (15), `iq2_xxs` (16).
/// K-quant formats have a reference and an imatrix-weighted variant;
/// `iq2_xxs` always requires an importance vector (the caller provides the
/// synthetic column-energy fallback when no measured imatrix exists).
public enum QuantEncode {
    static let blockK = 256
    static let groupMaxEps: Float = 1e-15

    /// GGUF raw type ids this encoder can emit.
    public static func canQuantize(_ type: UInt32) -> Bool {
        type == 8 || type == 10 || type == 12 || type == 15 || type == 16
    }

    public static func requiresImatrix(_ type: UInt32) -> Bool { type == 16 }

    /// Bytes of one row of `columns` elements (columns must be a multiple of
    /// the type's block size).
    public static func rowSize(type: UInt32, columns: Int) -> Int {
        guard let info = GGUF.typeInfo(type) else { return 0 }
        return columns / Int(info.blockElems) * Int(info.blockBytes)
    }

    /// Port of `ds4q_quantize_chunk`: quantize `rows` full rows starting at
    /// element offset `start` (a multiple of `columns`). The imatrix, when
    /// given, holds one importance value per COLUMN and is shared by all
    /// rows. Returns the bytes written, 0 for unsupported types.
    @discardableResult
    public static func quantizeChunk(type: UInt32,
                                     src: UnsafePointer<Float>,
                                     dst: UnsafeMutableRawPointer,
                                     start: Int, rows: Int, columns: Int,
                                     imatrix: UnsafePointer<Float>?) -> Int {
        switch type {
        case 8: return quantizeQ8_0(src: src, dst: dst, start: start,
                                    rows: rows, columns: columns)
        case 10: return quantizeK(type: 10, src: src, dst: dst, start: start,
                                  rows: rows, columns: columns, imatrix: imatrix)
        case 12: return quantizeK(type: 12, src: src, dst: dst, start: start,
                                  rows: rows, columns: columns, imatrix: imatrix)
        case 15: return quantizeK(type: 15, src: src, dst: dst, start: start,
                                  rows: rows, columns: columns, imatrix: imatrix)
        case 16:
            guard imatrix != nil else { return 0 }
            return quantizeK(type: 16, src: src, dst: dst, start: start,
                             rows: rows, columns: columns, imatrix: imatrix)
        default: return 0
        }
    }

    // MARK: - Bit-exact float helpers

    /// Port of `ds4q_f32_to_f16` (round-to-nearest-even, bit-twiddled): the
    /// exact conversion the C reference uses, kept verbatim so encoded
    /// scales match byte-for-byte.
    static func f16Bits(_ f: Float) -> UInt16 {
        let scaleToInf = Float(bitPattern: 0x7780_0000)   // 0x1.0p+112
        let scaleToZero = Float(bitPattern: 0x0880_0000)  // 0x1.0p-110
        var base = (abs(f) * scaleToInf) * scaleToZero

        let w = f.bitPattern
        let shl1w = w &+ w
        let sign = w & 0x8000_0000
        var bias = shl1w & 0xFF00_0000
        if bias < 0x7100_0000 { bias = 0x7100_0000 }

        base = Float(bitPattern: (bias >> 1) &+ 0x0780_0000) + base
        let out = base.bitPattern
        let expBits = (out >> 13) & 0x0000_7C00
        let mantissaBits = out & 0x0000_0FFF
        let nonsign = expBits &+ mantissaBits
        return UInt16(truncatingIfNeeded: (sign >> 16)
            | (shl1w > 0xFF00_0000 ? 0x7E00 : nonsign))
    }

    static func f16ToF32(_ bits: UInt16) -> Float {
        Float(Float16(bitPattern: bits))
    }

    /// Port of `ds4q_nearest_int`: the 12582912.0 magic-number round.
    @inline(__always)
    static func nearestInt(_ value: Float) -> Int {
        let biased = value + 12_582_912.0
        let bits = Int32(bitPattern: biased.bitPattern)
        return Int((bits & 0x007f_ffff) - 0x0040_0000)
    }

    // MARK: - Shared K-quant search helpers (qkx2 / qkx3 / qp)

    /// Port of `ds4q_make_qkx2_quants`.
    static func makeQKX2(n: Int, nmax: Int, x: UnsafePointer<Float>,
                         weights: UnsafePointer<Float>,
                         L: UnsafeMutablePointer<UInt8>,
                         theMin: inout Float,
                         Laux: UnsafeMutablePointer<UInt8>,
                         rmin: Float, rdelta: Float, nstep: Int,
                         useMad: Bool) -> Float {
        var minV = x[0]
        var maxV = x[0]
        var sumW = weights[0]
        var sumX = sumW * x[0]
        for i in 1..<n {
            if x[i] < minV { minV = x[i] }
            if x[i] > maxV { maxV = x[i] }
            let w = weights[i]
            sumW += w
            sumX += w * x[i]
        }
        if minV > 0 { minV = 0 }
        if maxV == minV {
            for i in 0..<n { L[i] = 0 }
            theMin = -minV
            return 0
        }
        var iscale = Float(nmax) / (maxV - minV)
        var scale = 1 / iscale
        var bestError: Float = 0
        for i in 0..<n {
            let l = nearestInt(iscale * (x[i] - minV))
            L[i] = UInt8(max(0, min(nmax, l)))
            var diff = scale * Float(L[i]) + minV - x[i]
            diff = useMad ? abs(diff) : diff * diff
            bestError += weights[i] * diff
        }
        if nstep < 1 {
            theMin = -minV
            return scale
        }
        for is_ in 0...nstep {
            iscale = (rmin + rdelta * Float(is_) + Float(nmax)) / (maxV - minV)
            var sumL: Float = 0, sumL2: Float = 0, sumXL: Float = 0
            for i in 0..<n {
                var l = nearestInt(iscale * (x[i] - minV))
                l = max(0, min(nmax, l))
                Laux[i] = UInt8(l)
                let w = weights[i]
                sumL += w * Float(l)
                sumL2 += w * Float(l) * Float(l)
                sumXL += w * Float(l) * x[i]
            }
            let D = sumW * sumL2 - sumL * sumL
            if D > 0 {
                var thisScale = (sumW * sumXL - sumX * sumL) / D
                var thisMin = (sumL2 * sumX - sumL * sumXL) / D
                if thisMin > 0 {
                    thisMin = 0
                    thisScale = sumXL / sumL2
                }
                var curError: Float = 0
                for i in 0..<n {
                    var diff = thisScale * Float(Laux[i]) + thisMin - x[i]
                    diff = useMad ? abs(diff) : diff * diff
                    curError += weights[i] * diff
                }
                if curError < bestError {
                    for i in 0..<n { L[i] = Laux[i] }
                    bestError = curError
                    scale = thisScale
                    minV = thisMin
                }
            }
        }
        theMin = -minV
        return scale
    }

    /// Port of `ds4q_make_qkx3_quants` (nil weights fall back to x²).
    static func makeQKX3(n: Int, nmax: Int, x: UnsafePointer<Float>,
                         weights: UnsafePointer<Float>?,
                         L: UnsafeMutablePointer<UInt8>,
                         theMin: inout Float,
                         Laux: UnsafeMutablePointer<UInt8>,
                         rmin: Float, rdelta: Float, nstep: Int,
                         useMad: Bool) -> Float {
        @inline(__always) func w(_ i: Int) -> Float {
            weights.map { $0[i] } ?? x[i] * x[i]
        }
        var minV = x[0]
        var maxV = x[0]
        var sumW = w(0)
        var sumX = sumW * x[0]
        for i in 1..<n {
            if x[i] < minV { minV = x[i] }
            if x[i] > maxV { maxV = x[i] }
            let wi = w(i)
            sumW += wi
            sumX += wi * x[i]
        }
        if minV > 0 { minV = 0 }
        if maxV <= minV {
            for i in 0..<n { L[i] = 0 }
            theMin = -minV
            return 0
        }
        var iscale = Float(nmax) / (maxV - minV)
        var scale = 1 / iscale
        var bestMad: Float = 0
        for i in 0..<n {
            let l = nearestInt(iscale * (x[i] - minV))
            L[i] = UInt8(max(0, min(nmax, l)))
            var diff = scale * Float(L[i]) + minV - x[i]
            diff = useMad ? abs(diff) : diff * diff
            bestMad += w(i) * diff
        }
        if nstep < 1 {
            theMin = -minV
            return scale
        }
        for is_ in 0...nstep {
            iscale = (rmin + rdelta * Float(is_) + Float(nmax)) / (maxV - minV)
            var sumL: Float = 0, sumL2: Float = 0, sumXL: Float = 0
            for i in 0..<n {
                var l = nearestInt(iscale * (x[i] - minV))
                l = max(0, min(nmax, l))
                Laux[i] = UInt8(l)
                let wi = w(i)
                sumL += wi * Float(l)
                sumL2 += wi * Float(l) * Float(l)
                sumXL += wi * Float(l) * x[i]
            }
            let D = sumW * sumL2 - sumL * sumL
            if D > 0 {
                var thisScale = (sumW * sumXL - sumX * sumL) / D
                var thisMin = (sumL2 * sumX - sumL * sumXL) / D
                if thisMin > 0 {
                    thisMin = 0
                    thisScale = sumXL / sumL2
                }
                var mad: Float = 0
                for i in 0..<n {
                    var diff = thisScale * Float(Laux[i]) + thisMin - x[i]
                    diff = useMad ? abs(diff) : diff * diff
                    mad += w(i) * diff
                }
                if mad < bestMad {
                    for i in 0..<n { L[i] = Laux[i] }
                    bestMad = mad
                    scale = thisScale
                    minV = thisMin
                }
            }
        }
        theMin = -minV
        return scale
    }

    /// Port of `ds4q_make_qp_quants` (non-negative values, iterative refit).
    static func makeQP(n: Int, nmax: Int, x: UnsafePointer<Float>,
                       L: UnsafeMutablePointer<UInt8>,
                       quantWeights: UnsafePointer<Float>) -> Float {
        var maxV: Float = 0
        for i in 0..<n { maxV = max(maxV, x[i]) }
        if maxV < groupMaxEps {
            for i in 0..<n { L[i] = 0 }
            return 0
        }
        var iscale = Float(nmax) / maxV
        for i in 0..<n {
            L[i] = UInt8(truncatingIfNeeded: nearestInt(iscale * x[i]))
        }
        let scale = 1 / iscale
        var bestMSE: Float = 0
        for i in 0..<n {
            let diff = x[i] - scale * Float(L[i])
            bestMSE += quantWeights[i] * diff * diff
        }
        for is_ in -4...4 where is_ != 0 {
            let iscaleIs = (0.1 * Float(is_) + Float(nmax)) / maxV
            let scaleIs = 1 / iscaleIs
            var mse: Float = 0
            for i in 0..<n {
                var l = nearestInt(iscaleIs * x[i])
                l = min(nmax, l)
                let diff = x[i] - scaleIs * Float(l)
                mse += quantWeights[i] * diff * diff
            }
            if mse < bestMSE {
                bestMSE = mse
                iscale = iscaleIs
            }
        }
        var sumlx: Float = 0
        var suml2: Float = 0
        for i in 0..<n {
            var l = nearestInt(iscale * x[i])
            l = min(nmax, l)
            // The C stores the (possibly negative) int into uint8_t with
            // wrapping and keeps accumulating on the int — mirror both.
            L[i] = UInt8(truncatingIfNeeded: l)
            let w = quantWeights[i]
            sumlx += w * x[i] * Float(l)
            suml2 += w * Float(l) * Float(l)
        }
        for _ in 0..<5 {
            var nChanged = 0
            for i in 0..<n {
                let w = quantWeights[i]
                var slx = sumlx - w * x[i] * Float(L[i])
                var sl2 = suml2 - w * Float(L[i]) * Float(L[i])
                if slx > 0 && sl2 > 0 {
                    var newL = nearestInt(x[i] * sl2 / slx)
                    newL = min(nmax, newL)
                    if newL != Int(L[i]) {
                        slx += w * x[i] * Float(newL)
                        sl2 += w * Float(newL) * Float(newL)
                        if slx * slx * suml2 > sumlx * sumlx * sl2 {
                            L[i] = UInt8(truncatingIfNeeded: newL)
                            sumlx = slx
                            suml2 = sl2
                            nChanged += 1
                        }
                    }
                }
            }
            if nChanged == 0 { break }
        }
        return suml2 > 0 ? sumlx / suml2 : 0
    }

    static func getScaleMinK4(_ j: Int, _ q: UnsafePointer<UInt8>) -> (d: UInt8, m: UInt8) {
        if j < 4 {
            return (q[j] & 63, q[j + 4] & 63)
        }
        return ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }

    // MARK: - q8_0

    private static func quantizeQ8_0(src: UnsafePointer<Float>,
                                     dst: UnsafeMutableRawPointer,
                                     start: Int, rows: Int, columns: Int) -> Int {
        let qk = 32
        let rowBytes = rowSize(type: 8, columns: columns)
        var out = dst.advanced(by: (start / columns) * rowBytes)
            .assumingMemoryBound(to: UInt8.self)
        let nblocks = rows * (columns / qk)
        for b in 0..<nblocks {
            let x = src + start + b * qk
            var amax: Float = 0
            for j in 0..<qk { amax = max(amax, abs(x[j])) }
            let d = amax / 127.0
            let id: Float = d != 0 ? 1.0 / d : 0.0
            let hd = f16Bits(d)
            out[0] = UInt8(hd & 0xff)
            out[1] = UInt8(hd >> 8)
            for j in 0..<qk {
                out[2 + j] = UInt8(bitPattern: Int8((x[j] * id).rounded(.toNearestOrAwayFromZero)))
            }
            out += 2 + qk
        }
        return rows * rowBytes
    }

    // MARK: - q8_K

    static func writeQ8KBlock(_ x: UnsafePointer<Float>,
                              _ y: UnsafeMutablePointer<UInt8>) {
        let dOff = 0, qsOff = 4, bsumsOff = 260, blockBytes = 292
        var maxV: Float = 0
        var amax: Float = 0
        for j in 0..<blockK {
            let ax = abs(x[j])
            if ax > amax { amax = ax; maxV = x[j] }
        }
        if amax == 0 {
            for j in 0..<blockBytes { y[j] = 0 }
            return
        }
        let iscale = -127.0 / maxV
        var qs = [Int8](repeating: 0, count: blockK)
        for j in 0..<blockK {
            // lrintf: round to nearest even under the default FP environment.
            var v = Int((iscale * x[j]).rounded(.toNearestOrEven))
            if v > 127 { v = 127 }
            if v < -128 { v = -128 }
            qs[j] = Int8(v)
        }
        let d = 1.0 / iscale
        withUnsafeBytes(of: d.bitPattern.littleEndian) {
            for (i, byte) in $0.enumerated() { y[dOff + i] = byte }
        }
        for j in 0..<blockK { y[qsOff + j] = UInt8(bitPattern: qs[j]) }
        for j in 0..<(blockK / 16) {
            var sum: Int = 0
            for i in 0..<16 { sum += Int(qs[j * 16 + i]) }
            let bits = UInt16(bitPattern: Int16(sum)).littleEndian
            y[bsumsOff + j * 2] = UInt8(bits & 0xff)
            y[bsumsOff + j * 2 + 1] = UInt8(bits >> 8)
        }
    }

    // MARK: - q4_K

    static func writeQ4KBlockRef(_ x: UnsafePointer<Float>,
                                 _ y: UnsafeMutablePointer<UInt8>) {
        let scalesOff = 4, qsOff = 16
        var L = [UInt8](repeating: 0, count: blockK)
        var Laux = [UInt8](repeating: 0, count: 32)
        var weights = [Float](repeating: 0, count: 32)
        var mins = [Float](repeating: 0, count: blockK / 32)
        var scales = [Float](repeating: 0, count: blockK / 32)

        var maxScale: Float = 0
        var maxMin: Float = 0
        for j in 0..<(blockK / 32) {
            var sumX2: Float = 0
            for l in 0..<32 { sumX2 += x[32 * j + l] * x[32 * j + l] }
            let avX = (sumX2 / 32).squareRoot()
            for l in 0..<32 { weights[l] = avX + abs(x[32 * j + l]) }
            scales[j] = L.withUnsafeMutableBufferPointer { lp in
                Laux.withUnsafeMutableBufferPointer { auxp in
                    makeQKX2(n: 32, nmax: 15, x: x + 32 * j, weights: weights,
                             L: lp.baseAddress! + 32 * j, theMin: &mins[j],
                             Laux: auxp.baseAddress!, rmin: -1.0, rdelta: 0.1,
                             nstep: 20, useMad: false)
                }
            }
            if scales[j] > maxScale { maxScale = scales[j] }
            if mins[j] > maxMin { maxMin = mins[j] }
        }

        let invScale: Float = maxScale > 0 ? 63.0 / maxScale : 0
        let invMin: Float = maxMin > 0 ? 63.0 / maxMin : 0
        for j in 0..<(blockK / 32) {
            var ls = UInt8(truncatingIfNeeded: nearestInt(invScale * scales[j]))
            var lm = UInt8(truncatingIfNeeded: nearestInt(invMin * mins[j]))
            ls = min(63, ls)
            lm = min(63, lm)
            if j < 4 {
                y[scalesOff + j] = ls
                y[scalesOff + j + 4] = lm
            } else {
                y[scalesOff + j + 4] = (ls & 0xF) | ((lm & 0xF) << 4)
                y[scalesOff + j - 4] |= (ls >> 4) << 6
                y[scalesOff + j] |= (lm >> 4) << 6
            }
        }

        let d = f16Bits(maxScale / 63.0)
        let dmin = f16Bits(maxMin / 63.0)
        y[0] = UInt8(d & 0xff); y[1] = UInt8(d >> 8)
        y[2] = UInt8(dmin & 0xff); y[3] = UInt8(dmin >> 8)

        finishQ4K(x: x, y: y, L: &L, d: d, dmin: dmin,
                  scalesOff: scalesOff, qsOff: qsOff)
    }

    static func writeQ4KBlockWeighted(_ x: UnsafePointer<Float>,
                                      _ y: UnsafeMutablePointer<UInt8>,
                                      _ quantWeights: UnsafePointer<Float>) {
        let scalesOff = 4, qsOff = 16
        var L = [UInt8](repeating: 0, count: blockK)
        var Laux = [UInt8](repeating: 0, count: 32)
        var Ls = [UInt8](repeating: 0, count: blockK / 32)
        var Lm = [UInt8](repeating: 0, count: blockK / 32)
        var weights = [Float](repeating: 0, count: 32)
        var sw = [Float](repeating: 0, count: blockK / 32)
        var mins = [Float](repeating: 0, count: blockK / 32)
        var scales = [Float](repeating: 0, count: blockK / 32)

        var sumX2: Float = 0
        for l in 0..<blockK { sumX2 += x[l] * x[l] }
        let sigma2 = 2 * sumX2 / Float(blockK)

        for j in 0..<(blockK / 32) {
            for l in 0..<32 {
                weights[l] = quantWeights[32 * j + l]
                    * (sigma2 + x[32 * j + l] * x[32 * j + l]).squareRoot()
            }
            var sumw: Float = 0
            for l in 0..<32 { sumw += weights[l] }
            sw[j] = sumw
            scales[j] = L.withUnsafeMutableBufferPointer { lp in
                Laux.withUnsafeMutableBufferPointer { auxp in
                    makeQKX3(n: 32, nmax: 15, x: x + 32 * j, weights: weights,
                             L: lp.baseAddress! + 32 * j, theMin: &mins[j],
                             Laux: auxp.baseAddress!, rmin: -0.9, rdelta: 0.05,
                             nstep: 36, useMad: false)
                }
            }
        }

        let dBlock = makeQP(n: blockK / 32, nmax: 63, x: scales, L: &Ls,
                            quantWeights: sw)
        let mBlock = makeQP(n: blockK / 32, nmax: 63, x: mins, L: &Lm,
                            quantWeights: sw)
        for j in 0..<(blockK / 32) {
            let ls = Ls[j]
            let lm = Lm[j]
            if j < 4 {
                y[scalesOff + j] = ls
                y[scalesOff + j + 4] = lm
            } else {
                y[scalesOff + j + 4] = (ls & 0xF) | ((lm & 0xF) << 4)
                y[scalesOff + j - 4] |= (ls >> 4) << 6
                y[scalesOff + j] |= (lm >> 4) << 6
            }
        }

        let d = f16Bits(dBlock)
        let dmin = f16Bits(mBlock)
        y[0] = UInt8(d & 0xff); y[1] = UInt8(d >> 8)
        y[2] = UInt8(dmin & 0xff); y[3] = UInt8(dmin >> 8)

        finishQ4K(x: x, y: y, L: &L, d: d, dmin: dmin,
                  scalesOff: scalesOff, qsOff: qsOff)
    }

    /// Shared q4_K tail: re-derive L from the packed scales, pack nibbles.
    private static func finishQ4K(x: UnsafePointer<Float>,
                                  y: UnsafeMutablePointer<UInt8>,
                                  L: inout [UInt8], d: UInt16, dmin: UInt16,
                                  scalesOff: Int, qsOff: Int) {
        for j in 0..<(blockK / 32) {
            let (sc, m) = getScaleMinK4(j, y + scalesOff)
            let dd = f16ToF32(d) * Float(sc)
            if dd == 0 { continue }
            let dm = f16ToF32(dmin) * Float(m)
            for ii in 0..<32 {
                var l = nearestInt((x[32 * j + ii] + dm) / dd)
                l = max(0, min(15, l))
                L[32 * j + ii] = UInt8(l)
            }
        }
        var q = qsOff
        var j = 0
        while j < blockK {
            for l in 0..<32 { y[q + l] = L[j + l] | (L[j + l + 32] << 4) }
            q += 32
            j += 64
        }
    }

    // MARK: - q2_K

    static func writeQ2KBlockRef(_ x: UnsafePointer<Float>,
                                 _ y: UnsafeMutablePointer<UInt8>) {
        let scalesOff = 0, qsOff = 16, dOff = 80, dminOff = 82
        let q4scale: Float = 15.0
        var L = [UInt8](repeating: 0, count: blockK)
        var Laux = [UInt8](repeating: 0, count: 16)
        var weights = [Float](repeating: 0, count: 16)
        var mins = [Float](repeating: 0, count: blockK / 16)
        var scales = [Float](repeating: 0, count: blockK / 16)

        var maxScale: Float = 0
        var maxMin: Float = 0
        for j in 0..<(blockK / 16) {
            for l in 0..<16 { weights[l] = abs(x[16 * j + l]) }
            scales[j] = L.withUnsafeMutableBufferPointer { lp in
                Laux.withUnsafeMutableBufferPointer { auxp in
                    makeQKX2(n: 16, nmax: 3, x: x + 16 * j, weights: weights,
                             L: lp.baseAddress! + 16 * j, theMin: &mins[j],
                             Laux: auxp.baseAddress!, rmin: -0.5, rdelta: 0.1,
                             nstep: 15, useMad: true)
                }
            }
            if scales[j] > maxScale { maxScale = scales[j] }
            if mins[j] > maxMin { maxMin = mins[j] }
        }

        var hd: UInt16
        var hmin: UInt16
        if maxScale > 0 {
            let iscale = q4scale / maxScale
            for j in 0..<(blockK / 16) {
                y[scalesOff + j] = UInt8(truncatingIfNeeded: nearestInt(iscale * scales[j]))
            }
            hd = f16Bits(maxScale / q4scale)
        } else {
            for j in 0..<(blockK / 16) { y[scalesOff + j] = 0 }
            hd = f16Bits(0)
        }
        if maxMin > 0 {
            let iscale = q4scale / maxMin
            for j in 0..<(blockK / 16) {
                y[scalesOff + j] |= UInt8(truncatingIfNeeded: nearestInt(iscale * mins[j])) << 4
            }
            hmin = f16Bits(maxMin / q4scale)
        } else {
            hmin = f16Bits(0)
        }
        y[dOff] = UInt8(hd & 0xff); y[dOff + 1] = UInt8(hd >> 8)
        y[dminOff] = UInt8(hmin & 0xff); y[dminOff + 1] = UInt8(hmin >> 8)

        finishQ2K(x: x, y: y, L: &L, hd: hd, hmin: hmin,
                  scalesOff: scalesOff, qsOff: qsOff)
    }

    static func writeQ2KBlockWeighted(_ x: UnsafePointer<Float>,
                                      _ y: UnsafeMutablePointer<UInt8>,
                                      _ quantWeights: UnsafePointer<Float>) {
        let scalesOff = 0, qsOff = 16, dOff = 80, dminOff = 82
        var L = [UInt8](repeating: 0, count: blockK)
        var Laux = [UInt8](repeating: 0, count: 16)
        var mins = [Float](repeating: 0, count: blockK / 16)
        var scales = [Float](repeating: 0, count: blockK / 16)
        var sw = [Float](repeating: 0, count: blockK / 16)
        var weight = [Float](repeating: 0, count: 16)
        var Ls = [UInt8](repeating: 0, count: blockK / 16)
        var Lm = [UInt8](repeating: 0, count: blockK / 16)

        var sumx2: Float = 0
        for j in 0..<blockK { sumx2 += x[j] * x[j] }
        let sigma2 = sumx2 / Float(blockK)
        for j in 0..<(blockK / 16) {
            for l in 0..<16 {
                weight[l] = quantWeights[16 * j + l]
                    * (sigma2 + x[16 * j + l] * x[16 * j + l]).squareRoot()
            }
            // Faithful port note: the C sums `weight[l]` over QK_K/16 = 16
            // entries here, which equals the full 16-element group.
            for l in 0..<(blockK / 16) { sw[j] += weight[l] }
            scales[j] = L.withUnsafeMutableBufferPointer { lp in
                Laux.withUnsafeMutableBufferPointer { auxp in
                    makeQKX3(n: 16, nmax: 3, x: x + 16 * j, weights: weight,
                             L: lp.baseAddress! + 16 * j, theMin: &mins[j],
                             Laux: auxp.baseAddress!, rmin: -0.9, rdelta: 0.05,
                             nstep: 36, useMad: false)
                }
            }
        }

        let dm = makeQP(n: blockK / 16, nmax: 15, x: scales, L: &Ls,
                        quantWeights: sw)
        let mm = makeQP(n: blockK / 16, nmax: 15, x: mins, L: &Lm,
                        quantWeights: sw)
        let hd = f16Bits(dm)
        let hmin = f16Bits(mm)
        y[dOff] = UInt8(hd & 0xff); y[dOff + 1] = UInt8(hd >> 8)
        y[dminOff] = UInt8(hmin & 0xff); y[dminOff + 1] = UInt8(hmin >> 8)

        for j in 0..<(blockK / 16) { y[scalesOff + j] = Ls[j] | (Lm[j] << 4) }

        finishQ2K(x: x, y: y, L: &L, hd: hd, hmin: hmin,
                  scalesOff: scalesOff, qsOff: qsOff)
    }

    /// Shared q2_K tail: re-derive L from packed scales, pack 2-bit quads.
    private static func finishQ2K(x: UnsafePointer<Float>,
                                  y: UnsafeMutablePointer<UInt8>,
                                  L: inout [UInt8], hd: UInt16, hmin: UInt16,
                                  scalesOff: Int, qsOff: Int) {
        for j in 0..<(blockK / 16) {
            let d = f16ToF32(hd) * Float(y[scalesOff + j] & 0xF)
            if d == 0 { continue }
            let dm = f16ToF32(hmin) * Float(y[scalesOff + j] >> 4)
            for ii in 0..<16 {
                var l = nearestInt((x[16 * j + ii] + dm) / d)
                l = max(0, min(3, l))
                L[16 * j + ii] = UInt8(l)
            }
        }
        var j = 0
        while j < blockK {
            for l in 0..<32 {
                y[qsOff + j / 4 + l] = L[j + l] | (L[j + l + 32] << 2)
                    | (L[j + l + 64] << 4) | (L[j + l + 96] << 6)
            }
            j += 128
        }
    }

    // MARK: - Block dispatch shared by the K formats

    private static func quantizeK(type: UInt32, src: UnsafePointer<Float>,
                                  dst: UnsafeMutableRawPointer,
                                  start: Int, rows: Int, columns: Int,
                                  imatrix: UnsafePointer<Float>?) -> Int {
        let blockBytes = Int(GGUF.typeInfo(type)!.blockBytes)
        let rowBytes = rowSize(type: type, columns: columns)
        let out = dst.advanced(by: (start / columns) * rowBytes)
            .assumingMemoryBound(to: UInt8.self)
        let blocksPerRow = columns / blockK
        for row in 0..<rows {
            let xrow = src + start + row * columns
            for b in 0..<blocksPerRow {
                let block = out + row * rowBytes + b * blockBytes
                let x = xrow + b * blockK
                switch type {
                case 10:
                    if let imatrix {
                        writeQ2KBlockWeighted(x, block, imatrix + b * blockK)
                    } else {
                        writeQ2KBlockRef(x, block)
                    }
                case 12:
                    if let imatrix {
                        writeQ4KBlockWeighted(x, block, imatrix + b * blockK)
                    } else {
                        writeQ4KBlockRef(x, block)
                    }
                case 15:
                    writeQ8KBlock(x, block)
                case 16:
                    writeIQ2XXSBlock(x, block, imatrix! + b * blockK)
                default:
                    break
                }
            }
        }
        return rows * rowBytes
    }
}
