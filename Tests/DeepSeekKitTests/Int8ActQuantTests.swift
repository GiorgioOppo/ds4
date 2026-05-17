import XCTest
import Metal
@testable import DeepSeekKit

/// Verifica il kernel `act_quant_int8` esposto via `ActQuant(format: .int8)`.
/// Per ogni blocco di 128 elementi: lo scale dev'essere `amax/127`,
/// e la dequantizzazione (qbyte * scale) deve riprodurre l'input
/// con errore RTN-bounded (≤ scale/2 per elemento).
final class Int8ActQuantTests: XCTestCase {

    private func requireMetal() throws {
        try XCTSkipUnless(MTLCreateSystemDefaultDevice() != nil,
                          "Metal not available")
    }

    func testRoundTripUniformDistribution() throws {
        try requireMetal()
        let M = 4, N = 256
        let blockSize = Quant.actBlockSizeINT8        // 128
        let blocks = N / blockSize
        let xArr = randomArray(M * N, seed: 7, scale: 1.0)

        let x = xArr.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [M, N], dtype: .f32)
        }

        let aq = ActQuant(format: .int8)
        let cmd = Device.shared.queue.makeCommandBuffer()!
        let out = aq.quant(x, inplace: false, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        guard let qbytes = out.qbytes else {
            XCTFail("ActQuant(.int8) did not produce qbytes")
            return
        }
        // Sanity di shape.
        XCTAssertEqual(qbytes.shape, [M, N])
        XCTAssertEqual(out.scales.shape, [M, blocks])
        XCTAssertEqual(qbytes.dtype, .i8)
        XCTAssertEqual(out.scales.dtype, .f32)

        // Leggi qbytes (int8) e scales (f32) sul lato host.
        let qPtr = qbytes.buffer.contents()
            .bindMemory(to: Int8.self, capacity: M * N)
        let sPtr = out.scales.buffer.contents()
            .bindMemory(to: Float.self, capacity: M * blocks)

        for r in 0..<M {
            for b in 0..<blocks {
                let blockStart = b * blockSize
                // Calcola absmax CPU del blocco originale.
                var amax: Float = 0
                for k in 0..<blockSize {
                    let v = abs(xArr[r * N + blockStart + k])
                    if v > amax { amax = v }
                }
                amax = max(amax, 1e-5)
                let expectedScale = amax / 127.0
                let gotScale = sPtr[r * blocks + b]
                // Lo scale dev'essere identico (entrambi calcolati come
                // amax/127 con max-floor 1e-5).
                XCTAssertEqual(gotScale, expectedScale, accuracy: 1e-6,
                                "scale mismatch at row=\(r) block=\(b)")

                // Errore di dequant per elemento ≤ scale/2.
                for k in 0..<blockSize {
                    let orig = xArr[r * N + blockStart + k]
                    let q = qPtr[r * N + blockStart + k]
                    let dequant = Float(q) * gotScale
                    let err = abs(dequant - orig)
                    XCTAssertLessThanOrEqual(err, gotScale * 0.5 + 1e-6,
                                              "dequant error at row=\(r) k=\(blockStart+k)")
                }
            }
        }
    }

    func testZeroInputProducesZeroOutput() throws {
        try requireMetal()
        let M = 2, N = 128
        let xArr = [Float](repeating: 0, count: M * N)
        let x = xArr.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [M, N], dtype: .f32)
        }

        let aq = ActQuant(format: .int8)
        let cmd = Device.shared.queue.makeCommandBuffer()!
        let out = aq.quant(x, inplace: false, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let qPtr = out.qbytes!.buffer.contents()
            .bindMemory(to: Int8.self, capacity: M * N)
        for i in 0..<(M * N) {
            // Con tutti zeri, amax cade al floor 1e-5 → scale = 1e-5/127.
            // Round di 0/scale = 0 → qbyte == 0.
            XCTAssertEqual(qPtr[i], 0)
        }
    }

    // MARK: - helpers

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
