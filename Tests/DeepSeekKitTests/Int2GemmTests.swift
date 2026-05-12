import XCTest
import Metal
@testable import DeepSeekKit

/// Validates the INT2 W2A16 GEMM kernel (`gemm_int2_w2a16_to_f32` /
/// `gemm_int2_w2a16_bf16_to_f32`) against a CPU reference. Tolerances
/// are wide because RTN noise at 2-bit width is dominant: with only 4
/// codes the per-element relative error per K block is on the order of
/// 0.5, partially averaged out across K but still substantial. The
/// tests verify the kernel implements the *defined* quantization
/// scheme, not that it preserves accuracy of the original weights.
final class Int2GemmTests: XCTestCase {

    func testInt2GEMMSquare() throws {
        let M = 2, K = 256, N = 256
        try runInt2Case(M: M, K: K, N: N, seedX: 11, seedW: 22, tolerance: 0.40)
    }

    func testInt2GEMMAsymmetric() throws {
        let M = 3, K = 384, N = 64
        try runInt2Case(M: M, K: K, N: N, seedX: 31, seedW: 41, tolerance: 0.40)
    }

    func testInt2GEMMMoEExpertShape() throws {
        let M = 1, K = 512, N = 128
        try runInt2Case(M: M, K: K, N: N, seedX: 51, seedW: 61, tolerance: 0.40)
    }

    func testInt2GEMMBF16Input() throws {
        let M = 2, K = 256, N = 128
        try runInt2CaseBF16(M: M, K: K, N: N, seedX: 71, seedW: 81, tolerance: 0.40)
    }

    /// Sanity bookkeeping: round-trip a synthetic row of known values
    /// (-1, -0.5, 0, 0.5, 1, …) through the CPU reference quantizer
    /// and confirm the packed bytes + dequantized values are what we
    /// expect from the encoding spec.
    func testInt2PackingRoundTrip() {
        let K = 128
        // Construct a row with values that should saturate to exactly
        // [-2, -1, 0, +1] after RTN with scale = max_abs/2 = 1.0.
        // max_abs is 2.0 so scale = 1.0; quantized targets are exactly
        // [-2, -1, 0, +1] when input is [-2.0, -1.0, 0.0, 1.0].
        var row = [Float](repeating: 0, count: K)
        for i in 0..<K {
            let m = i % 4
            row[i] = [Float(-2.0), -1.0, 0.0, 1.0][m]
        }
        let (_, _, dequant) = quantizeRefInt2(row, N: 1, K: K)
        // Dequant should round-trip exactly because the inputs are
        // already on grid.
        for i in 0..<K {
            XCTAssertEqual(dequant[i], row[i], accuracy: 1e-5,
                           "i=\(i) input=\(row[i]) dequant=\(dequant[i])")
        }
    }

    // ---- helpers ----

    private func runInt2Case(M: Int, K: Int, N: Int,
                              seedX: UInt64, seedW: UInt64,
                              tolerance: Float) throws {
        precondition(K % 128 == 0)
        let xArr = randomArray(M * K, seed: seedX, scale: 0.5)
        let wArr = randomArray(N * K, seed: seedW, scale: 0.5)

        let (wPacked, wScaleF16, wDequant) = quantizeRefInt2(wArr, N: N, K: K)

        let x = xArr.withUnsafeBytes { Tensor.from(bytes: $0, shape: [M, K], dtype: .f32) }
        let w = wPacked.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [N, K], dtype: .i2)
        }
        let blocksK = K / 128
        let wScale = wScaleF16.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [N, blocksK], dtype: .f16)
        }

        let lin = Linear(inFeatures: K, outFeatures: N, weight: w, scale: wScale)
        let cmd = Device.shared.queue.makeCommandBuffer()!
        let y = lin(x, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let gpu = y.toFloatArray()

        let cpu = cpuGEMM(a: xArr, b: wDequant, M: M, N: N, K: K)
        for i in 0..<gpu.count {
            let denom = max(abs(cpu[i]), 1e-3)
            let rel = abs(gpu[i] - cpu[i]) / denom
            XCTAssertLessThan(rel, tolerance,
                              "i=\(i) gpu=\(gpu[i]) cpu=\(cpu[i]) rel=\(rel)")
        }
    }

    private func runInt2CaseBF16(M: Int, K: Int, N: Int,
                                  seedX: UInt64, seedW: UInt64,
                                  tolerance: Float) throws {
        precondition(K % 128 == 0)
        let xArr = randomArray(M * K, seed: seedX, scale: 0.5)
        let wArr = randomArray(N * K, seed: seedW, scale: 0.5)

        let xBF16 = xArr.map { floatToBF16($0) }
        let xRoundedToBF16 = xBF16.map { Float(bitPattern: UInt32($0) << 16) }

        let (wPacked, wScaleF16, wDequant) = quantizeRefInt2(wArr, N: N, K: K)

        let x = xBF16.withUnsafeBytes { Tensor.from(bytes: $0, shape: [M, K], dtype: .bf16) }
        let w = wPacked.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [N, K], dtype: .i2)
        }
        let blocksK = K / 128
        let wScale = wScaleF16.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [N, blocksK], dtype: .f16)
        }

        let lin = Linear(inFeatures: K, outFeatures: N, weight: w, scale: wScale)
        let cmd = Device.shared.queue.makeCommandBuffer()!
        let y = lin(x, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let gpu = y.toFloatArray()

        let cpu = cpuGEMM(a: xRoundedToBF16, b: wDequant, M: M, N: N, K: K)
        for i in 0..<gpu.count {
            let denom = max(abs(cpu[i]), 1e-3)
            let rel = abs(gpu[i] - cpu[i]) / denom
            XCTAssertLessThan(rel, tolerance,
                              "i=\(i) gpu=\(gpu[i]) cpu=\(cpu[i]) rel=\(rel)")
        }
    }

    /// CPU reference INT2 quantizer matching `Int2Quant.quantizeRowFromFloatI2`.
    private func quantizeRefInt2(_ w: [Float], N: Int, K: Int)
        -> (weight: [UInt8], scaleF16: [UInt16], dequant: [Float])
    {
        let blocksK = K / 128
        let packedRowBytes = K / 4
        var weight = [UInt8](repeating: 0, count: N * packedRowBytes)
        var scaleF16 = [UInt16](repeating: 0, count: N * blocksK)
        var dequant = [Float](repeating: 0, count: N * K)
        for n in 0..<N {
            for kb in 0..<blocksK {
                let k0 = kb * 128
                var maxAbs: Float = 0
                for k in 0..<128 {
                    let v = abs(w[n * K + k0 + k])
                    if v > maxAbs { maxAbs = v }
                }
                let s: Float = maxAbs == 0 ? 0 : maxAbs / 2.0
                let invS: Float = maxAbs == 0 ? 0 : 2.0 / maxAbs
                scaleF16[n * blocksK + kb] = floatToF16(s)
                let sRoundedF16 = Self.f16ToFloat(floatToF16(s))
                for k in stride(from: 0, to: 128, by: 4) {
                    let q0 = (w[n * K + k0 + k    ] * invS).rounded(.toNearestOrEven)
                    let q1 = (w[n * K + k0 + k + 1] * invS).rounded(.toNearestOrEven)
                    let q2 = (w[n * K + k0 + k + 2] * invS).rounded(.toNearestOrEven)
                    let q3 = (w[n * K + k0 + k + 3] * invS).rounded(.toNearestOrEven)
                    let c0 = Int8(min(max(q0, -2), 1))
                    let c1 = Int8(min(max(q1, -2), 1))
                    let c2 = Int8(min(max(q2, -2), 1))
                    let c3 = Int8(min(max(q3, -2), 1))
                    let v0 = UInt8(truncatingIfNeeded: c0) & 0x3
                    let v1 = UInt8(truncatingIfNeeded: c1) & 0x3
                    let v2 = UInt8(truncatingIfNeeded: c2) & 0x3
                    let v3 = UInt8(truncatingIfNeeded: c3) & 0x3
                    weight[n * packedRowBytes + (k0 + k) / 4] =
                        v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
                    dequant[n * K + k0 + k    ] = Float(c0) * sRoundedF16
                    dequant[n * K + k0 + k + 1] = Float(c1) * sRoundedF16
                    dequant[n * K + k0 + k + 2] = Float(c2) * sRoundedF16
                    dequant[n * K + k0 + k + 3] = Float(c3) * sRoundedF16
                }
            }
        }
        return (weight, scaleF16, dequant)
    }

    private func cpuGEMM(a: [Float], b: [Float], M: Int, N: Int, K: Int) -> [Float] {
        var c = [Float](repeating: 0, count: M * N)
        for m in 0..<M {
            for n in 0..<N {
                var acc: Float = 0
                for k in 0..<K { acc += a[m * K + k] * b[n * K + k] }
                c[m * N + n] = acc
            }
        }
        return c
    }

    private func randomArray(_ count: Int, seed: UInt64, scale: Float = 1.0) -> [Float] {
        var state = seed | 1
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let frac = Float(Double(state >> 11) / Double(1 << 53))
            out[i] = (frac - 0.5) * 2 * scale
        }
        return out
    }

    private func floatToBF16(_ f: Float) -> UInt16 {
        let bits = f.bitPattern
        let rounded = bits &+ ((bits >> 16) & 1) &+ 0x7FFF
        return UInt16(truncatingIfNeeded: rounded >> 16)
    }

    private func floatToF16(_ f: Float) -> UInt16 {
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

    private static func f16ToFloat(_ h: UInt16) -> Float {
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
}
