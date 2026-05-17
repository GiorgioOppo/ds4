import XCTest
import Metal
@testable import DeepSeekKit

/// Verifica il path W8A8: `Linear(useW8A8Activations: true)` con peso
/// int8 quantizza l'input a int8 e dispatcha `gemm_int8_w8a8_*`,
/// producendo un output vicino al riferimento CPU di
/// `dequant(int8_act) @ dequant(int8_weight)`.
///
/// La tolleranza è più larga rispetto a W8A16 perché entrambi i lati
/// pagano l'errore RTN int8 (≈ scale/2 per elemento, accumulato su K).
final class Int8W8A8GemmTests: XCTestCase {

    private func requireMetal() throws {
        try XCTSkipUnless(MTLCreateSystemDefaultDevice() != nil,
                          "Metal not available")
    }

    /// Shape "naive" (M < 32 → niente SG variant).
    func testW8A8NaiveSmall() throws {
        try requireMetal()
        try runW8A8Case(M: 2, K: 256, N: 64,
                         seedX: 11, seedW: 22, tolerance: 0.06)
    }

    /// Shape SG-eligible (M, N multipli di 32, K multiplo di 128).
    func testW8A8SimdgroupSquare() throws {
        try requireMetal()
        try runW8A8Case(M: 32, K: 256, N: 64,
                         seedX: 31, seedW: 41, tolerance: 0.06)
    }

    /// Asymmetric prefill-style shape.
    func testW8A8AsymmetricPrefill() throws {
        try requireMetal()
        try runW8A8Case(M: 64, K: 384, N: 128,
                         seedX: 51, seedW: 61, tolerance: 0.06)
    }

    // MARK: - runner

    private func runW8A8Case(M: Int, K: Int, N: Int,
                              seedX: UInt64, seedW: UInt64,
                              tolerance: Float) throws {
        precondition(K % 128 == 0)
        let xArr = randomArray(M * K, seed: seedX, scale: 0.5)
        let wArr = randomArray(N * K, seed: seedW, scale: 0.5)

        let (wInt8, wScaleF16, wDequant) = quantizeRefInt8(wArr, N: N, K: K)
        // Calcola anche il dequant di riferimento delle attivazioni:
        // sappiamo come il kernel `act_quant_int8` quantizza (amax/127,
        // simmetrico). Lo facciamo CPU-side per il riferimento.
        let xDequant = referenceActQuantInt8(xArr, M: M, K: K)

        let x = xArr.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [M, K], dtype: .f32)
        }
        let w = wInt8.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [N, K], dtype: .i8)
        }
        let blocksK = K / 128
        let wScale = wScaleF16.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [N, blocksK], dtype: .f16)
        }

        // Opt-in W8A8.
        let lin = Linear(inFeatures: K, outFeatures: N, weight: w, scale: wScale,
                          useW8A8Activations: true)
        let cmd = Device.shared.queue.makeCommandBuffer()!
        let y = lin(x, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let gpu = y.toFloatArray()

        // CPU reference: GEMM su attivazioni e pesi entrambi
        // dequantizzati.
        let cpu = cpuGEMM(a: xDequant, b: wDequant, M: M, N: N, K: K)

        var maxRel: Float = 0
        for i in 0..<gpu.count {
            let denom = max(abs(cpu[i]), 1e-3)
            let rel = abs(gpu[i] - cpu[i]) / denom
            if rel > maxRel { maxRel = rel }
            XCTAssertLessThan(rel, tolerance,
                              "M=\(M) N=\(N) K=\(K) i=\(i) gpu=\(gpu[i]) cpu=\(cpu[i]) rel=\(rel)")
        }
        // Diagnostic: log the max relative error so a regression on
        // the tolerance shows up as a visible drift.
        print("[W8A8 M=\(M) N=\(N) K=\(K)] maxRel=\(maxRel)")
    }

    // MARK: - reference helpers

    /// Replica della quantizzazione `act_quant_int8` su CPU.
    /// Per-row, per-128-block, scale = amax/127 (floored a 1e-5/127).
    /// Restituisce la matrice dequantizzata: `Float(int8) * scale`.
    private func referenceActQuantInt8(_ x: [Float], M: Int, K: Int) -> [Float] {
        let blockSize = 128
        let blocks = K / blockSize
        var out = [Float](repeating: 0, count: M * K)
        for r in 0..<M {
            for b in 0..<blocks {
                let blockStart = b * blockSize
                var amax: Float = 0
                for k in 0..<blockSize {
                    let v = abs(x[r * K + blockStart + k])
                    if v > amax { amax = v }
                }
                amax = max(amax, 1e-5)
                let scale = amax / 127.0
                let invScale = 127.0 / amax
                for k in 0..<blockSize {
                    let q = x[r * K + blockStart + k] * invScale
                    let rr = q.rounded(.toNearestOrEven)
                    let clamped = max(-127.0, min(127.0, rr))
                    out[r * K + blockStart + k] = Float(Int(clamped)) * scale
                }
            }
        }
        return out
    }

    /// Identico al pattern di `Int8GemmTests.quantizeRefInt8`.
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
                let sF16 = floatToF16(s)
                scaleF16[n * blocksK + kb] = sF16
                let sRoundedF16 = Self.f16ToFloat(sF16)
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
        let sign = UInt32(h >> 15) & 1
        let exp = UInt32(h >> 10) & 0x1F
        let mant = UInt32(h & 0x3FF)
        if exp == 0 { return Float(bitPattern: sign << 31) * (Float(mant) / 1024.0) * Float(bitPattern: (UInt32(127 - 14) << 23)) }
        if exp == 0x1F { return Float.infinity }
        let bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13)
        return Float(bitPattern: bits)
    }
}
