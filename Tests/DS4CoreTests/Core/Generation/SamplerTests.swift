import XCTest
@testable import DS4Core

/// Swift-only checks of the sampler: the RNG distribution, greedy argmax, and
/// self-consistency invariants of the top-k / top-p / min-p path. (The original
/// bit-for-bit cross-check against the C ds4_sample_logits was dropped when the
/// C engine was removed from the project.)
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

    func testRNGF32Range() {
        var s: UInt64 = 12345
        for _ in 0..<1000 {
            let f = Sampler.rngF32(&s)
            XCTAssertGreaterThanOrEqual(f, 0)
            XCTAssertLessThan(f, 1)
        }
    }

    /// The RNG must be deterministic for a given seed and advance its state.
    func testRNGDeterministic() {
        var a: UInt64 = 42, b: UInt64 = 42
        for _ in 0..<100 { XCTAssertEqual(Sampler.rngNext(&a), Sampler.rngNext(&b)) }
        var s: UInt64 = 7
        let before = s
        _ = Sampler.rngNext(&s)
        XCTAssertNotEqual(s, before)
    }

    func testArgmax() {
        // temperature 0 must select the global argmax with no RNG consumption.
        let logits = makeLogits(2003)
        var rng: UInt64 = 7
        let tok = Sampler.sample(logits, temperature: 0, topK: 0, topP: 1, minP: 0, rng: &rng)
        XCTAssertEqual(tok, Sampler.argmax(logits))
        XCTAssertEqual(rng, 7) // unchanged
    }

    func testRepetitionPenaltyMovesArgmax() {
        // With a clear winner at index 5, penalizing it should change the argmax
        // (it stops being the top token) — this is what breaks a repeat loop.
        var logits = [Float](repeating: 0, count: 100)
        logits[5] = 10; logits[9] = 8
        XCTAssertEqual(Sampler.applyRepetitionPenalty(logits, recent: [], penalty: 1.5).count, 100)
        let penalized = Sampler.applyRepetitionPenalty(logits, recent: [5][...], penalty: 2.0)
        XCTAssertEqual(penalized[5], 5, accuracy: 1e-5)   // 10/2
        XCTAssertEqual(Sampler.argmax(penalized), 9)      // 8 now wins over 5

        // penalty 1.0 or empty window leaves logits untouched.
        XCTAssertEqual(Sampler.applyRepetitionPenalty(logits, recent: [5][...], penalty: 1.0), logits)
        XCTAssertEqual(Sampler.applyRepetitionPenalty(logits, recent: [], penalty: 2.0), logits)

        // Negative logits are multiplied (pushed further down), not divided.
        var neg = [Float](repeating: 0, count: 4); neg[2] = -4
        XCTAssertEqual(Sampler.applyRepetitionPenalty(neg, recent: [2][...], penalty: 2.0)[2], -8, accuracy: 1e-5)
    }

    func testRepetitionPenaltyBreaksGreedyLoop() {
        // Greedy (temp 0) keeps picking the winner; once it's in `recent`, the
        // penalty lets a different token win.
        var logits = [Float](repeating: 0, count: 50)
        logits[7] = 5; logits[3] = 4
        var rng: UInt64 = 1
        let plain = Sampler.sample(logits, temperature: 0, topK: 0, topP: 1, minP: 0, rng: &rng)
        XCTAssertEqual(plain, 7)
        let penalized = Sampler.sample(logits, temperature: 0, topK: 0, topP: 1, minP: 0,
                                       repetitionPenalty: 2.0, recent: [7, 7][...], rng: &rng)
        XCTAssertEqual(penalized, 3)
    }

    /// Self-consistency invariants across a grid of sampling parameters: the
    /// sampler must always return a valid in-range index, and top_k==1 must
    /// collapse to the argmax (the single most likely token).
    func testSamplerInvariants() {
        let n = 2003
        let logits = makeLogits(n)
        let argmax = Sampler.argmax(logits)
        let temps: [Float] = [0.0, 0.5, 0.7, 1.0, 1.5]
        let topKs: [Int] = [0, 1, 8, 40, 256, 1024, 5000]
        let topPs: [Float] = [0.5, 0.9, 0.95, 1.0]
        let minPs: [Float] = [0.0, 0.01, 0.05, 0.2]
        let seeds: [UInt64] = [1, 42, 0xDEAD_BEEF, 0x9E37_79B9_7F4A_7C15, 0]

        var cases = 0
        for temp in temps {
            for topK in topKs {
                for topP in topPs {
                    for minP in minPs {
                        for seed in seeds {
                            var rng = seed
                            let tok = Sampler.sample(logits, temperature: temp, topK: topK,
                                                     topP: topP, minP: minP, rng: &rng)
                            let label = "temp=\(temp) topK=\(topK) topP=\(topP) minP=\(minP) seed=\(seed)"
                            XCTAssertGreaterThanOrEqual(tok, 0, "index < 0: \(label)")
                            XCTAssertLessThan(tok, n, "index >= n: \(label)")
                            if temp == 0 || topK == 1 {
                                XCTAssertEqual(tok, argmax, "expected argmax: \(label)")
                            }
                            cases += 1
                        }
                    }
                }
            }
        }
        XCTAssertGreaterThan(cases, 1000)
    }
}
