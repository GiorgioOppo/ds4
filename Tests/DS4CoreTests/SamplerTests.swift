import XCTest
import CDS4
@testable import DS4Core

/// Cross-checks the Swift sampler against the public C ds4_sample_logits over a
/// grid of (temperature, top_k, top_p, min_p, seed). Uses distinct logits so
/// the C qsort and the Swift sort agree on ordering. Asserts both the chosen
/// token and the resulting RNG state match (so the RNG consumption is identical).
final class SamplerTests: XCTestCase {

    /// Distinct, scattered logits: a bijective permutation of evenly spaced values.
    private func makeLogits(_ n: Int) -> [Float] {
        var out = [Float](repeating: 0, count: n)
        let stride = 1009 % n   // 1009 is coprime with n=2003 -> bijection
        var idx = 0
        for i in 0..<n {
            out[idx] = (Float(i) - Float(n) / 2) * 0.02
            idx = (idx + stride) % n
        }
        return out
    }

    func testRNGMatchesC() {
        // The RNG is the backbone; verify a long sequence matches indirectly via
        // sampling-state equality below, and the f32 distribution here.
        var s: UInt64 = 12345
        for _ in 0..<1000 {
            let f = Sampler.rngF32(&s)
            XCTAssertGreaterThanOrEqual(f, 0)
            XCTAssertLessThan(f, 1)
        }
    }

    func testSamplerMatchesC() {
        let n = 2003
        let logits = makeLogits(n)
        let temps: [Float] = [0.0, 0.5, 0.7, 1.0, 1.5]
        let topKs: [Int32] = [0, 1, 8, 40, 256, 1024, 5000]
        let topPs: [Float] = [0.5, 0.9, 0.95, 1.0]
        let minPs: [Float] = [0.0, 0.01, 0.05, 0.2]
        let seeds: [UInt64] = [1, 42, 0xDEAD_BEEF, 0x9E37_79B9_7F4A_7C15, 0]

        var cases = 0
        for temp in temps {
            for topK in topKs {
                for topP in topPs {
                    for minP in minPs {
                        for seed in seeds {
                            var rngC = seed
                            let tokC = logits.withUnsafeBufferPointer {
                                ds4_sample_logits($0.baseAddress, Int32(n), temp, topK, topP, minP, &rngC)
                            }
                            var rngS = seed
                            let tokS = Sampler.sample(logits, temperature: temp,
                                                      topK: Int(topK), topP: topP, minP: minP, rng: &rngS)
                            let label = "temp=\(temp) topK=\(topK) topP=\(topP) minP=\(minP) seed=\(seed)"
                            XCTAssertEqual(Int(tokC), tokS, "token mismatch: \(label)")
                            XCTAssertEqual(rngC, rngS, "rng state mismatch: \(label)")
                            cases += 1
                        }
                    }
                }
            }
        }
        XCTAssertGreaterThan(cases, 1000)
    }

    func testArgmaxMatchesC() {
        // temperature 0 must select the global argmax with no RNG consumption.
        let logits = makeLogits(2003)
        var rng: UInt64 = 7
        let tok = Sampler.sample(logits, temperature: 0, topK: 0, topP: 1, minP: 0, rng: &rng)
        XCTAssertEqual(tok, Sampler.argmax(logits))
        XCTAssertEqual(rng, 7) // unchanged
    }
}
