import XCTest
@testable import DS4Metal

/// The GLM chat sampler's contracts: greedy collapse (temperature 0 or
/// top-K 1), llama.cpp-convention repetition penalty inside the recent
/// window, top-K candidate restriction, and deterministic behavior under an
/// injected RNG. Pure functions — no device.
final class GLM52SamplerTests: XCTestCase {
    private struct SeededGenerator: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state
        }
    }

    func testGreedyCollapse() {
        var rng = SeededGenerator(state: 7)
        let logits: [Float] = [0.1, 3.0, 2.9, -1]
        XCTAssertEqual(GLM52Sampler.sample(
            logits: logits, temperature: 0, topK: 40,
            repetitionPenalty: 1, recentTokens: [][...], using: &rng), 1)
        XCTAssertEqual(GLM52Sampler.sample(
            logits: logits, temperature: 0.8, topK: 1,
            repetitionPenalty: 1, recentTokens: [][...], using: &rng), 1)
        XCTAssertNil(GLM52Sampler.sample(
            logits: [], temperature: 0, topK: 1,
            repetitionPenalty: 1, recentTokens: [][...], using: &rng))
    }

    func testRepetitionPenaltyDemotesRecentTokens() {
        var rng = SeededGenerator(state: 9)
        // Token 1 leads, but it is recent and the penalty drops it below 2.
        let logits: [Float] = [0.1, 3.0, 2.9]
        let recent: [Int32] = [1]
        XCTAssertEqual(GLM52Sampler.sample(
            logits: logits, temperature: 0, topK: 40,
            repetitionPenalty: 1.2, recentTokens: recent[...],
            using: &rng), 2)
        // Negative logits move AWAY from zero (llama.cpp convention).
        let negative: [Float] = [-1.0, -0.5]
        XCTAssertEqual(GLM52Sampler.sample(
            logits: negative, temperature: 0, topK: 40,
            repetitionPenalty: 3.0, recentTokens: [Int32(1)][...],
            using: &rng), 0)
    }

    func testTopKRestrictsCandidates() {
        var rng = SeededGenerator(state: 11)
        let logits: [Float] = [5, 4.9, -10, -10, -10]
        for _ in 0..<200 {
            let token = GLM52Sampler.sample(
                logits: logits, temperature: 1.0, topK: 2,
                repetitionPenalty: 1, recentTokens: [][...], using: &rng)
            XCTAssertTrue(token == 0 || token == 1,
                          "top-2 sampling escaped the candidate set: "
                          + "\(String(describing: token))")
        }
    }

    func testSamplingIsDeterministicUnderSeededRNG() {
        let logits: [Float] = [1.0, 0.9, 0.8, 0.2]
        var first = SeededGenerator(state: 42)
        var second = SeededGenerator(state: 42)
        for _ in 0..<50 {
            XCTAssertEqual(
                GLM52Sampler.sample(logits: logits, temperature: 0.7,
                                    topK: 3, repetitionPenalty: 1.1,
                                    recentTokens: [Int32(0)][...],
                                    using: &first),
                GLM52Sampler.sample(logits: logits, temperature: 0.7,
                                    topK: 3, repetitionPenalty: 1.1,
                                    recentTokens: [Int32(0)][...],
                                    using: &second))
        }
    }
}
