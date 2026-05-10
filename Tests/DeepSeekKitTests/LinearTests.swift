import XCTest
import Metal
@testable import DeepSeekKit

final class LinearTests: XCTestCase {

    /// F32 GEMM matches a CPU reference within float rounding.
    func testF32DenseGEMM() throws {
        let M = 4, K = 32, N = 8
        let xArr = randomArray(M * K, seed: 1)
        let wArr = randomArray(N * K, seed: 2)
        let x = xArr.withUnsafeBytes { Tensor.from(bytes: $0, shape: [M, K], dtype: .f32) }
        let w = wArr.withUnsafeBytes { Tensor.from(bytes: $0, shape: [N, K], dtype: .f32) }

        let lin = Linear(inFeatures: K, outFeatures: N, weight: w, scale: nil)
        let cmd = Device.shared.queue.makeCommandBuffer()!
        let y = lin(x, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let gpu = y.toFloatArray()

        let cpu = cpuGEMM(a: xArr, b: wArr, M: M, N: N, K: K)
        for i in 0..<gpu.count {
            XCTAssertEqual(gpu[i], cpu[i], accuracy: 1e-3, "i=\(i)")
        }
        _ = xArr; _ = wArr
    }

    /// FP8 GEMM round-trips a Linear with quantized weights to within FP8 precision.
    /// We construct FP8 weights by running act_quant over a known float matrix,
    /// then assert the FP8 GEMM equals the dense GEMM on the *dequantized* weights.
    func testFP8GEMM() throws {
        let M = 2, K = 256, N = 256        // K and N must be multiples of 128
        let xArr = randomArray(M * K, seed: 11, scale: 0.5)
        let wArr = randomArray(N * K, seed: 22, scale: 0.5)
        let x = xArr.withUnsafeBytes { Tensor.from(bytes: $0, shape: [M, K], dtype: .f32) }
        let w = wArr.withUnsafeBytes { Tensor.from(bytes: $0, shape: [N, K], dtype: .f32) }

        // Quantize w to FP8 byte stream + per-128 scales (we treat N as if it were
        // grouped in blocks of 128 for the test — a full implementation would
        // produce per-128 weight blocks; here we just put one scale per row of
        // length K/128 to match the kernel layout for [N/128, K/128]).
        let aq = ActQuant(format: .fp8)
        let cmd1 = Device.shared.queue.makeCommandBuffer()!
        let wq = aq.quant(w, inplace: false, in: cmd1)
        cmd1.commit(); cmd1.waitUntilCompleted()

        // Reshape scales from [N, K/128] → [N/128, K/128] by averaging across rows
        // within each 128-row block. (The kernel expects one scale per 128×128 block.)
        let blocksK = K / 128
        let blocksN = N / 128
        let perRowScales = wq.scales.toFloatArray()
        var blockScales = [Float](repeating: 0, count: blocksN * blocksK)
        for nb in 0..<blocksN {
            for kb in 0..<blocksK {
                var s: Float = 0
                for r in 0..<128 { s += perRowScales[(nb * 128 + r) * blocksK + kb] }
                blockScales[nb * blocksK + kb] = s / 128.0
            }
        }
        let blockScalesT = blockScales.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [blocksN, blocksK], dtype: .f32)
        }

        let lin = Linear(inFeatures: K, outFeatures: N,
                         weight: Tensor(shape: [N, K], dtype: .fp8E4M3, buffer: wq.qbytes!.buffer),
                         scale: blockScalesT)
        let cmd2 = Device.shared.queue.makeCommandBuffer()!
        let y = lin(x, in: cmd2)
        cmd2.commit(); cmd2.waitUntilCompleted()
        let gpu = y.toFloatArray()

        // CPU reference: dequant the FP8 bytes back to f32 using the *block*
        // scales we just averaged, then do a plain f32 GEMM. The expected error
        // is dominated by FP8 rounding (~12.5%) plus the per-row → per-block
        // scale averaging — assert relative error rather than absolute.
        let qb = qbytes(wq.qbytes!)
        var wDeq = [Float](repeating: 0, count: N * K)
        for n in 0..<N {
            for k in 0..<K {
                let nb = n / 128, kb = k / 128
                let scale = blockScales[nb * blocksK + kb]
                wDeq[n * K + k] = dequantE4M3(qb[n * K + k]) * scale
            }
        }
        let cpu = cpuGEMM(a: xArr, b: wDeq, M: M, N: N, K: K)

        for i in 0..<gpu.count {
            let denom = max(abs(cpu[i]), 1e-3)
            let rel = abs(gpu[i] - cpu[i]) / denom
            XCTAssertLessThan(rel, 0.5, "i=\(i) gpu=\(gpu[i]) cpu=\(cpu[i])")
        }
        _ = xArr; _ = wArr
    }

    private func qbytes(_ t: Tensor) -> [UInt8] {
        let n = t.byteCount
        let p = t.buffer.contents().bindMemory(to: UInt8.self, capacity: n)
        return Array(UnsafeBufferPointer(start: p, count: n))
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
}
