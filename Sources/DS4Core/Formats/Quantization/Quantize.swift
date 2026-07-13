import Foundation

// CPU (re)quantization — faithful port of the ggml reference quantizers used by
// the GGUF ecosystem. Used by DS4_DENSE_Q4: the two giant Q8_0 attention
// projections (q_b, output_b — 71 of the ~145 MB/layer of dense weights) are
// requantized to Q4_K at model load and kept RESIDENT, halving their bytes.
// The engine already has validated Q4_K matvec kernels (the MoE experts run on
// them); this makes the same format available to dense weights.
//
// Block layouts (identical to ggml):
//   Q8_0  = blocks of 32:  [f16 d][32 × int8]                      → 34 B
//   Q4_K  = super-blocks of 256: [f16 d][f16 dmin][12 B scales][128 B qs] → 144 B
public enum Quantize {
    // MARK: F16 → Q8_0

    /// Quantize an F16 row-major tensor to the standard GGML Q8_0 layout.
    /// Blocks are independent and matrix rows in DS4 are multiples of 32.
    public static func quantizeF16Q8_0(_ src: UnsafeRawPointer, count: Int,
                                      into dst: UnsafeMutableRawPointer) {
        precondition(count % 32 == 0)
        for b in 0..<(count / 32) {
            let x = src + b * 64
            var amax: Float = 0
            for i in 0..<32 {
                amax = max(amax, abs(Half.float(x.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self))))
            }
            let d = amax / 127
            (dst + b * 34).storeBytes(of: Half.bits(d), as: UInt16.self)
            let q = (dst + b * 34 + 2).assumingMemoryBound(to: Int8.self)
            if d == 0 {
                for i in 0..<32 { q[i] = 0 }
            } else {
                let inv = 1 / d
                for i in 0..<32 {
                    let v = Half.float(x.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self))
                    q[i] = Int8(max(-127, min(127, Int((v * inv).rounded(.toNearestOrEven)))))
                }
            }
        }
    }

    // MARK: Q8_0 → F32

    /// Dequantize `count` elements of Q8_0 (count % 32 == 0).
    public static func dequantQ8_0(_ src: UnsafeRawPointer, count: Int,
                                   into dst: UnsafeMutablePointer<Float>) {
        let nb = count / 32
        for b in 0..<nb {
            let base = src + b * 34
            let d = Half.float(base.loadUnaligned(as: UInt16.self))
            let q = (base + 2).assumingMemoryBound(to: Int8.self)
            for i in 0..<32 {
                dst[b * 32 + i] = d * Float(q[i])
            }
        }
    }

    // MARK: F32 → Q4_K (ggml quantize_row_q4_K_ref)

    /// Quantize `count` elements (count % 256 == 0) to Q4_K. `dst` must hold
    /// count/256 × 144 bytes. Super-blocks are independent → parallelized.
    public static func quantizeQ4_K(_ x: UnsafePointer<Float>, count: Int,
                                    into dst: UnsafeMutableRawPointer) {
        let nsb = count / 256
        // Single superblock: call straight through. The streaming requant
        // (DenseStreamer.requantQ4) converts superblock-by-superblock from
        // INSIDE its own concurrentPerform — routing each 256-element call
        // through a nested dispatch fan-out added ~4k dispatch round-trips
        // per MB for zero extra parallelism.
        if nsb == 1 { return quantizeSuperblockQ4_K(x, into: dst) }
        // nonisolated(unsafe): every iteration reads/writes a DISJOINT 256-elem
        // superblock (x + sb*256 → dst + sb*144) — no shared mutable state.
        nonisolated(unsafe) let src = x
        nonisolated(unsafe) let out = dst
        DispatchQueue.concurrentPerform(iterations: nsb) { sb in
            quantizeSuperblockQ4_K(src + sb * 256, into: out + sb * 144)
        }
    }

    /// Dequantize `count` elements of Q4_K (validation / parity tests).
    public static func dequantQ4_K(_ src: UnsafeRawPointer, count: Int,
                                   into dst: UnsafeMutablePointer<Float>) {
        let nsb = count / 256
        for sb in 0..<nsb {
            let base = src + sb * 144
            let d = Half.float(base.loadUnaligned(as: UInt16.self))
            let dmin = Half.float((base + 2).loadUnaligned(as: UInt16.self))
            let scales = (base + 4).assumingMemoryBound(to: UInt8.self)
            let qs = (base + 16).assumingMemoryBound(to: UInt8.self)
            for j in 0..<8 {
                let (sc, m) = scaleMinK4(j, scales)
                let dj = d * Float(sc), mj = dmin * Float(m)
                // sub-block j covers elements 32j..<32j+32; qs pack low/high
                // nibbles of element pairs (l, l+32) per 64-element chunk.
                let chunk = j / 2            // 64-elem chunk index
                let hi = (j % 2) == 1        // second half of the chunk → high nibble
                for i in 0..<32 {
                    let qb = qs[chunk * 32 + i]
                    let q = hi ? (qb >> 4) : (qb & 0xF)
                    dst[sb * 256 + j * 32 + i] = dj * Float(q) - mj
                }
            }
        }
    }

    // MARK: internals

    /// ggml get_scale_min_k4: unpack the 6-bit (scale, min) of sub-block j.
    @inline(__always)
    static func scaleMinK4(_ j: Int, _ q: UnsafePointer<UInt8>) -> (UInt8, UInt8) {
        if j < 4 {
            return (q[j] & 63, q[j + 4] & 63)
        }
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
        return (d, m)
    }

    /// ggml nearest_int (round half to even).
    @inline(__always)
    private static func nearestInt(_ x: Float) -> Int {
        Int(x.rounded(.toNearestOrEven))
    }

    private static func quantizeSuperblockQ4_K(_ x: UnsafePointer<Float>,
                                               into out: UnsafeMutableRawPointer) {
        // Scratch on the STACK instead of six heap arrays per call: this runs
        // once per 256-element superblock over GIGABYTES of weights in the
        // DENSE_Q4 requant, where the per-call malloc/ARC traffic dominated
        // the profile — worst in unoptimized builds, where it stretched the
        // one-time requant from minutes to hours (looking like a hang).
        // Same operations in the same order → byte-identical output.
        withUnsafeTemporaryAllocation(of: UInt8.self, capacity: 256 + 32 + 12) { ub in
            withUnsafeTemporaryAllocation(of: Float.self, capacity: 32 + 8 + 8) { fb in
                let L = ub.baseAddress!            // 256: 4-bit codes
                let Laux = L + 256                 // 32: candidate codes per step
                let sb = Laux + 32                 // 12: packed 6-bit scales/mins
                let weights = fb.baseAddress!      // 32
                let scales = weights + 32          // 8
                let mins = scales + 8              // 8

                var maxScale: Float = 0, maxMin: Float = 0
                for j in 0..<8 {
                    let xj = x + j * 32
                    var sumX2: Float = 0
                    for i in 0..<32 { sumX2 += xj[i] * xj[i] }
                    let avX = (sumX2 / 32).squareRoot()
                    for i in 0..<32 { weights[i] = avX + abs(xj[i]) }
                    var theMin: Float = 0
                    scales[j] = makeQKX2Quants(n: 32, nmax: 15, x: xj, weights: weights,
                                               L: L + j * 32, theMin: &theMin, Laux: Laux,
                                               rmin: -1, rdelta: 0.1, nstep: 20, useMad: false)
                    mins[j] = theMin
                    if scales[j] > maxScale { maxScale = scales[j] }
                    if mins[j] > maxMin { maxMin = mins[j] }
                }

                for i in 0..<12 { sb[i] = 0 }      // temporary allocation is uninitialized; sb uses |=
                let invScale: Float = maxScale > 0 ? 63 / maxScale : 0
                let invMin: Float = maxMin > 0 ? 63 / maxMin : 0
                for j in 0..<8 {
                    let ls = UInt8(min(63, nearestInt(invScale * scales[j])))
                    let lm = UInt8(min(63, nearestInt(invMin * mins[j])))
                    if j < 4 {
                        sb[j] = ls
                        sb[j + 4] = lm
                    } else {
                        sb[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4)
                        sb[j - 4] |= (ls >> 4) << 6
                        sb[j] |= (lm >> 4) << 6
                    }
                }
                let d = Half.bits(maxScale / 63)
                let dmin = Half.bits(maxMin / 63)
                out.storeBytes(of: d, toByteOffset: 0, as: UInt16.self)
                out.storeBytes(of: dmin, toByteOffset: 2, as: UInt16.self)
                for i in 0..<12 { out.storeBytes(of: sb[i], toByteOffset: 4 + i, as: UInt8.self) }

                // Re-derive (scale, min) from the PACKED 6-bit values (like the
                // ggml reference), then quantize each sub-block to 4 bits.
                let dF = Half.float(d), dminF = Half.float(dmin)
                for j in 0..<8 {
                    let (sc, m) = scaleMinK4(j, sb)
                    let dj = dF * Float(sc)
                    if dj == 0 { continue }
                    let mj = dminF * Float(m)
                    for i in 0..<32 {
                        let l = nearestInt((x[j * 32 + i] + mj) / dj)
                        L[j * 32 + i] = UInt8(max(0, min(15, l)))
                    }
                }
                // Pack nibbles: per 64-element chunk, qs[l] = L[l] | (L[l+32] << 4).
                var qoff = 16
                var base = 0
                while base < 256 {
                    for l in 0..<32 {
                        out.storeBytes(of: L[base + l] | (L[base + l + 32] << 4),
                                       toByteOffset: qoff + l, as: UInt8.self)
                    }
                    qoff += 32
                    base += 64
                }
            }
        }
    }

    /// ggml make_qkx2_quants: weighted grid search for the best (scale, min)
    /// of one 32-element sub-block quantized to 0...nmax.
    private static func makeQKX2Quants(n: Int, nmax: Int,
                                       x: UnsafePointer<Float>, weights: UnsafePointer<Float>,
                                       L: UnsafeMutablePointer<UInt8>, theMin: inout Float,
                                       Laux: UnsafeMutablePointer<UInt8>,
                                       rmin: Float, rdelta: Float, nstep: Int, useMad: Bool) -> Float {
        var minV = x[0], maxV = x[0]
        var sumW = weights[0], sumX = weights[0] * x[0]
        for i in 1..<n {
            if x[i] < minV { minV = x[i] }
            if x[i] > maxV { maxV = x[i] }
            sumW += weights[i]
            sumX += weights[i] * x[i]
        }
        if minV > 0 { minV = 0 }
        if maxV == minV {
            for i in 0..<n { L[i] = 0 }
            theMin = -minV
            return 0
        }
        var iscale = Float(nmax) / (maxV - minV)
        var scale = 1 / iscale
        var bestMad: Float = 0
        for i in 0..<n {
            let l = max(0, min(nmax, nearestInt(iscale * (x[i] - minV))))
            L[i] = UInt8(l)
            var diff = scale * Float(l) + minV - x[i]
            diff = useMad ? abs(diff) : diff * diff
            bestMad += weights[i] * diff
        }
        if nstep < 1 {
            theMin = -minV
            return scale
        }
        for iStep in 0...nstep {
            iscale = (rmin + rdelta * Float(iStep) + Float(nmax)) / (maxV - minV)
            var sumL: Float = 0, sumL2: Float = 0, sumXL: Float = 0
            for i in 0..<n {
                let l = max(0, min(nmax, nearestInt(iscale * (x[i] - minV))))
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
                var mad: Float = 0
                for i in 0..<n {
                    var diff = thisScale * Float(Laux[i]) + thisMin - x[i]
                    diff = useMad ? abs(diff) : diff * diff
                    mad += weights[i] * diff
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
}
