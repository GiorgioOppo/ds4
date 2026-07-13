import XCTest
@testable import DS4Core

/// CPU Q4_K requantizer (DS4_DENSE_Q4). Superblocks are independent by
/// construction, so the single-superblock fast path of `quantizeQ4_K` (used by
/// the streaming requant, one 256-element call per superblock) must produce
/// the same bytes as the parallel multi-superblock path.
final class QuantizeTests: XCTestCase {

    /// Deterministic pseudo-weights in a plausible range (like dequantized Q8).
    private func makeInput(count: Int, seed: UInt64) -> [Float] {
        var rng = SplitMix64(seed: seed)
        return (0..<count).map { _ in Float.random(in: -0.25...0.25, using: &rng) }
    }

    func testSingleSuperblockPathMatchesParallelPath() {
        let nsb = 8
        let x = makeInput(count: nsb * 256, seed: 0xD54)
        var whole = [UInt8](repeating: 0, count: nsb * 144)
        var perBlock = [UInt8](repeating: 0, count: nsb * 144)
        x.withUnsafeBufferPointer { xp in
            whole.withUnsafeMutableBytes { wp in
                Quantize.quantizeQ4_K(xp.baseAddress!, count: nsb * 256,
                                      into: wp.baseAddress!)
            }
            perBlock.withUnsafeMutableBytes { pp in
                for sb in 0..<nsb {
                    Quantize.quantizeQ4_K(xp.baseAddress! + sb * 256, count: 256,
                                          into: pp.baseAddress! + sb * 144)
                }
            }
        }
        XCTAssertEqual(whole, perBlock,
                       "one N-superblock call and N single-superblock calls must be byte-identical")
    }

    func testQ4KRoundtripError() {
        let nsb = 4
        let x = makeInput(count: nsb * 256, seed: 0xBEEF)
        var q4 = [UInt8](repeating: 0, count: nsb * 144)
        var back = [Float](repeating: 0, count: nsb * 256)
        x.withUnsafeBufferPointer { xp in
            q4.withUnsafeMutableBytes { qp in
                Quantize.quantizeQ4_K(xp.baseAddress!, count: nsb * 256, into: qp.baseAddress!)
            }
        }
        q4.withUnsafeBytes { qp in
            back.withUnsafeMutableBufferPointer { bp in
                Quantize.dequantQ4_K(qp.baseAddress!, count: nsb * 256, into: bp.baseAddress!)
            }
        }
        // 4-bit grid over a ±0.25 range: per-element error stays well under one
        // coarse step (~0.5/15 ≈ 0.033); allow slack for the min/scale packing.
        for i in 0..<x.count {
            XCTAssertLessThan(abs(x[i] - back[i]), 0.05,
                              "roundtrip error too large at \(i): \(x[i]) → \(back[i])")
        }
    }

    func testF16ToQ8Roundtrip() {
        let values: [Float] = (0..<64).map { i in
            i == 0 ? 0 : sin(Float(i) * 0.37) * 0.4
        }
        let half = values.map(Half.bits)
        var q8 = [UInt8](repeating: 0, count: 68)
        var back = [Float](repeating: 0, count: 64)
        half.withUnsafeBytes { src in
            q8.withUnsafeMutableBytes {
                Quantize.quantizeF16Q8_0(src.baseAddress!, count: 64, into: $0.baseAddress!)
            }
        }
        q8.withUnsafeBytes { Quantize.dequantQ8_0($0.baseAddress!, count: 64, into: &back) }
        for i in values.indices {
            // One Q8 step per independent 32-value block, plus F16 source rounding.
            XCTAssertLessThan(abs(values[i] - back[i]), 0.004, "Q8 error at \(i)")
        }
    }
}

private struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state = state &+ 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
