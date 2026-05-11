import XCTest
import Metal
@testable import DeepSeekKit

/// Validates the INT8 W8A16 GEMM kernel (`gemm_int8_w8a16_to_f32` /
/// `gemm_int8_w8a16_bf16_to_f32`) against a CPU reference. Mirrors the
/// structure of `LinearTests.testFP8GEMM`.
final class Int8GemmTests: XCTestCase {

    /// f32 activations × int8 weights with per-row, per-128-group F16 scales
    /// matches a CPU dequant-and-multiply reference within tight tolerance.
    /// INT8 RTN of a uniform [-0.5, 0.5] distribution has worst-case
    /// quantization error of `s/2 ≈ |max|/254`, which after K=256
    /// accumulations gives < 1% relative error.
    func testInt8GEMMSquare() throws {
        let M = 2, K = 256, N = 256
        try runInt8Case(M: M, K: K, N: N, seedX: 11, seedW: 22, tolerance: 0.02)
    }

    /// Asymmetric shape that exercises a non-square output. K must remain
    /// a multiple of 128.
    func testInt8GEMMAsymmetric() throws {
        let M = 3, K = 384, N = 64
        try runInt8Case(M: M, K: K, N: N, seedX: 31, seedW: 41, tolerance: 0.02)
    }

    /// Shape typical of MoE expert weights (stacked along N dimension).
    /// The FP4 path historically had issues on this shape; INT8 should not.
    func testInt8GEMMMoEExpertShape() throws {
        let M = 1, K = 512, N = 128
        try runInt8Case(M: M, K: K, N: N, seedX: 51, seedW: 61, tolerance: 0.02)
    }

    /// BF16 activation input variant. Same accuracy contract but BF16's
    /// 7-bit mantissa loosens the achievable tolerance vs F32.
    func testInt8GEMMBF16Input() throws {
        let M = 2, K = 256, N = 128
        try runInt8CaseBF16(M: M, K: K, N: N, seedX: 71, seedW: 81, tolerance: 0.05)
    }

    // ---- helpers ----

    private func runInt8Case(M: Int, K: Int, N: Int,
                              seedX: UInt64, seedW: UInt64,
                              tolerance: Float) throws {
        precondition(K % 128 == 0)
        let xArr = randomArray(M * K, seed: seedX, scale: 0.5)
        let wArr = randomArray(N * K, seed: seedW, scale: 0.5)

        let (wInt8, wScaleF16, wDequant) = quantizeRefInt8(wArr, N: N, K: K)

        let x = xArr.withUnsafeBytes { Tensor.from(bytes: $0, shape: [M, K], dtype: .f32) }
        let w = wInt8.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [N, K], dtype: .i8)
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
        var maxRel: Float = 0
        for i in 0..<gpu.count {
            let denom = max(abs(cpu[i]), 1e-3)
            let rel = abs(gpu[i] - cpu[i]) / denom
            maxRel = max(maxRel, rel)
            XCTAssertLessThan(rel, tolerance,
                              "i=\(i) gpu=\(gpu[i]) cpu=\(cpu[i]) rel=\(rel)")
        }
        _ = maxRel
    }

    private func runInt8CaseBF16(M: Int, K: Int, N: Int,
                                  seedX: UInt64, seedW: UInt64,
                                  tolerance: Float) throws {
        precondition(K % 128 == 0)
        let xArr = randomArray(M * K, seed: seedX, scale: 0.5)
        let wArr = randomArray(N * K, seed: seedW, scale: 0.5)

        // BF16-quantize the activations (round to BF16).
        let xBF16 = xArr.map { floatToBF16($0) }
        let xRoundedToBF16 = xBF16.map { Float(bitPattern: UInt32($0) << 16) }

        let (wInt8, wScaleF16, wDequant) = quantizeRefInt8(wArr, N: N, K: K)

        let x = xBF16.withUnsafeBytes { Tensor.from(bytes: $0, shape: [M, K], dtype: .bf16) }
        let w = wInt8.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [N, K], dtype: .i8)
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

        // Reference uses the BF16-rounded activations.
        let cpu = cpuGEMM(a: xRoundedToBF16, b: wDequant, M: M, N: N, K: K)
        for i in 0..<gpu.count {
            let denom = max(abs(cpu[i]), 1e-3)
            let rel = abs(gpu[i] - cpu[i]) / denom
            XCTAssertLessThan(rel, tolerance,
                              "i=\(i) gpu=\(gpu[i]) cpu=\(cpu[i]) rel=\(rel)")
        }
    }

    /// Reference symmetric INT8 RTN quantization matching `Int8Quant.swift`.
    /// Returns the INT8 weight bytes, the F16 group scale bytes, and the
    /// dequantized FP32 reference matrix (`int8 * scale`).
    private func quantizeRefInt8(_ w: [Float], N: Int, K: Int)
        -> (weight: [Int8], scaleF16: [UInt16], dequant: [Float])
    {
        let blocksK = K / 128
        var weight = [Int8](repeating: 0, count: N * K)
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
                let s: Float = maxAbs == 0 ? 0 : maxAbs / 127.0
                let invS: Float = maxAbs == 0 ? 0 : 127.0 / maxAbs
                scaleF16[n * blocksK + kb] = floatToF16(s)
                // Use the F16-rounded scale for the dequant reference so
                // we match the GPU exactly.
                let sRoundedF16 = Self.f16ToFloat(floatToF16(s))
                for k in 0..<128 {
                    let q = w[n * K + k0 + k] * invS
                    let r = q.rounded(.toNearestOrEven)
                    let clamped = min(max(r, -127), 127)
                    let qi = Int8(clamped)
                    weight[n * K + k0 + k] = qi
                    dequant[n * K + k0 + k] = Float(qi) * sRoundedF16
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
