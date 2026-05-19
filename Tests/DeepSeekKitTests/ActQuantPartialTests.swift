import XCTest
import Metal
@testable import DeepSeekKit

/// Verifies `ActQuant.partialInplaceQuant` matches the V4 reference's
/// `act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)` semantics:
/// in-place FP8 E4M3 round-trip on a column slice, leaving the tail
/// columns byte-identical to the input.
final class ActQuantPartialTests: XCTestCase {

    /// Round-trip on [M=2, total=128] with block=32, slicing colStart=0..96.
    /// Verifies (a) the unquantized tail [96, 128) is untouched bit-for-bit,
    /// (b) the quantized prefix is the result of f32 → e4m3 → f32 with the
    /// per-block absmax scale, (c) no NaN or Inf produced.
    func testPartialInplaceRoundTrip() throws {
        let M = 2
        let total = 128
        let blockSize = 32
        let colStart = 0
        let colEnd = 96   // leaves the last 32 columns untouched

        var data = randomArray(count: M * total, seed: 17)
        let original = data
        let xT = data.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [M, total], dtype: .f32)
        }

        let cmd = Device.shared.queue.makeCommandBuffer()!
        ActQuant.partialInplaceQuant(xT,
                                     colStart: colStart, colEnd: colEnd,
                                     blockSize: blockSize, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let out = xT.toFloatArray()

        // Tail [colEnd, total) must be byte-identical to the input.
        for r in 0..<M {
            for c in colEnd..<total {
                let i = r * total + c
                XCTAssertEqual(out[i].bitPattern, original[i].bitPattern,
                               "tail must be untouched at row=\(r) col=\(c)")
            }
        }

        // Prefix must equal `f32 → e4m3 → f32 * blockScale` per block.
        for r in 0..<M {
            for blk in 0..<((colEnd - colStart) / blockSize) {
                let lo = colStart + blk * blockSize
                let hi = lo + blockSize
                // Per-block absmax of the ORIGINAL inputs.
                var amax: Float = 0
                for c in lo..<hi {
                    amax = max(amax, abs(original[r * total + c]))
                }
                amax = max(amax, 1e-4)
                let scale = roundPow2Scale(amax: amax, maxInv: 1.0 / 448.0)

                for c in lo..<hi {
                    let v = original[r * total + c]
                    let clipped = max(-448.0, min(448.0, v / scale))
                    let qb = f32ToE4M3(clipped)
                    let expected = e4m3ToF32(qb) * scale
                    let got = out[r * total + c]
                    XCTAssertEqual(got, expected, accuracy: 1e-6,
                                   "row=\(r) col=\(c) block=\(blk)")
                    XCTAssertFalse(got.isNaN, "row=\(r) col=\(c)")
                    XCTAssertFalse(got.isInfinite, "row=\(r) col=\(c)")
                }
            }
        }
    }

    /// Sanity check: with block=64 and 448-column span (the V4 KV-nope
    /// shape: head_dim=512, rope=64, so nope=448), the kernel completes
    /// without trapping.
    func testKVNopeShape() throws {
        let M = 5
        let total = 512        // head_dim
        let blockSize = 64
        let span = 448         // head_dim - rope_head_dim
        let data = randomArray(count: M * total, seed: 99)
        let xT = data.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [M, total], dtype: .f32)
        }
        let cmd = Device.shared.queue.makeCommandBuffer()!
        ActQuant.partialInplaceQuant(xT,
                                     colStart: 0, colEnd: span,
                                     blockSize: blockSize, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let out = xT.toFloatArray()
        for v in out {
            XCTAssertFalse(v.isNaN, "no NaN")
            XCTAssertFalse(v.isInfinite, "no Inf")
        }
    }

    // ---- helpers ---------------------------------------------------------

    private func randomArray(count: Int, seed: UInt64) -> [Float] {
        var state = seed | 1
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let frac = Float(Double(state >> 11) / Double(1 << 53))
            out[i] = (frac - 0.5) * 4   // range ~[-2, +2]
        }
        return out
    }

    /// Mirrors Metal's `round_pow2_scale(amax, max_inv) =
    /// fast_pow2_exp(fast_log2_ceil(amax * max_inv))`.
    private func roundPow2Scale(amax: Float, maxInv: Float) -> Float {
        let v = amax * maxInv
        let bits = v.bitPattern
        let exp = Int((bits >> 23) & 0xFF)
        let mant = bits & 0x7FFFFF
        let log2ceil = exp - 127 + (mant != 0 ? 1 : 0)
        return Float(bitPattern: UInt32(log2ceil + 127) << 23)
    }

    /// Round f32 → e4m3 byte, matching Metal's `f32_to_e4m3` (RTNE through
    /// half-precision then re-bias to 4-bit exp / 3-bit mantissa).
    /// Routes through `floatToF16Local` (pure-Swift IEEE-754 F32→F16,
    /// shared with the INT4/INT8 quantizers) instead of `Float16(x)`,
    /// which is unavailable on x86_64 macOS. Accuracy is good to a
    /// fraction of an ULP at the FP8 magnitudes we care about.
    private func f32ToE4M3(_ x: Float) -> UInt8 {
        let bits = floatToF16Local(x)
        let sign = UInt8((bits >> 15) & 1)
        var exp16 = Int((bits >> 10) & 0x1F)
        var mant10 = Int(bits & 0x3FF)

        if exp16 == 0 && mant10 == 0 { return sign << 7 }

        var newExp = exp16 - 15 + 7

        if newExp <= 0 {
            if exp16 == 0 { return sign << 7 }
            // Half normal → e4m3 subnormal.
            let shift = 1 - newExp
            let full = 1024 + mant10
            var result = full >> (shift + 7)
            let roundBit = (full >> (shift + 6)) & 1
            let stickyMask = (1 << (shift + 6)) - 1
            let sticky = full & stickyMask
            if roundBit != 0 && (sticky != 0 || (result & 1) != 0) { result += 1 }
            if result >= 8 {
                return (sign << 7) | (1 << 3)
            }
            return (sign << 7) | UInt8(result)
        }

        var mant3 = mant10 >> 7
        let roundBit = (mant10 >> 6) & 1
        let sticky = mant10 & 0x3F
        if roundBit != 0 && (sticky != 0 || (mant3 & 1) != 0) {
            mant3 += 1
            if mant3 == 8 { mant3 = 0; newExp += 1 }
        }
        if newExp >= 16 || (newExp == 15 && mant3 == 7) {
            return (sign << 7) | (0xE << 3) | 6   // closest finite (448)
        }
        return (sign << 7) | UInt8((newExp << 3) | mant3)
    }

    /// Inverse of the above. Mirrors Metal's `e4m3_to_f32`.
    private func e4m3ToF32(_ b: UInt8) -> Float {
        let sign = UInt32(b >> 7) & 1
        let exp = UInt32(b >> 3) & 0xF
        let mant = UInt32(b) & 0x7
        if exp == 0 && mant == 0 { return sign == 1 ? -0.0 : 0.0 }
        if exp == 0xF && mant == 0x7 { return .nan }
        if exp == 0 {
            let v = Float(mant) * 0x1p-9
            return sign == 1 ? -v : v
        }
        let bits = (sign << 31) | ((exp + 120) << 23) | (mant << 20)
        return Float(bitPattern: bits)
    }
}
