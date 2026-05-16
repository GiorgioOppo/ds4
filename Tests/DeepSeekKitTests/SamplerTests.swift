import XCTest
@testable import DeepSeekKit

/// Unit tests for the host-side sampling pipeline. The argmax path is
/// exercised indirectly through the existing greedy decode tests; here
/// we focus on the new filters (min-p, tail-free, typical, freq/presence
/// penalties, mirostat). All tests work on a small synthetic vocabulary
/// so the entire pipeline finishes in microseconds.
final class SamplerTests: XCTestCase {

    /// Build a `[1, V]` f32 logits Tensor from a host-side array.
    private func logitsTensor(_ values: [Float]) -> Tensor {
        return values.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [1, values.count], dtype: .f32)
        }
    }

    // MARK: - Greedy shortcut

    /// All filters at neutral defaults → argmax of the input logits.
    func testGreedyShortcutWithAllDisabled() {
        let logits = logitsTensor([0.1, 0.2, 9.0, -1.0, 0.0])
        var opts = SamplingOptions(temperature: 0)
        let picked = Sampler.sample(logits, history: [], options: &opts)
        XCTAssertEqual(picked, 2)
    }

    // MARK: - Min-P

    /// With a steep distribution and `minP = 0.25`, only the top token
    /// survives. `temperature = 1` keeps the sampler stochastic but on a
    /// degenerate set the choice is forced.
    func testMinPFiltersTail() {
        let logits = logitsTensor([5.0, 1.0, 0.0, -1.0, -2.0])
        // Top prob ≈ softmax(5)/total ≈ 0.97; second ≈ 0.018. With
        // minP=0.25 only the top token has prob >= 0.25 * 0.97 = 0.24.
        var opts = SamplingOptions(temperature: 1.0, minP: 0.25, rngState: 42)
        for _ in 0..<20 {
            let p = Sampler.sample(logits, history: [], options: &opts)
            XCTAssertEqual(p, 0, "min-P should have collapsed the support to token 0")
        }
    }

    /// `minP = 0` keeps the original behaviour (every token candidatable).
    /// We can't assert a specific draw, but we can assert that we see
    /// more than one distinct outcome over many samples on a flat
    /// distribution.
    func testMinPZeroPreservesSupport() {
        let logits = logitsTensor([1.0, 1.0, 1.0, 1.0])
        var opts = SamplingOptions(temperature: 1.0, minP: 0.0, rngState: 7)
        var seen = Set<Int>()
        for _ in 0..<200 {
            seen.insert(Sampler.sample(logits, history: [], options: &opts))
        }
        XCTAssertGreaterThan(seen.count, 1, "uniform logits + minP=0 should produce multiple outcomes")
    }

    // MARK: - Frequency / presence penalty

    /// Token visited 3× should drop by 3 × frequencyPenalty in logit
    /// space; with a large penalty it gets pushed out of the argmax.
    func testFrequencyPenaltyScalesWithCount() {
        let logits = logitsTensor([5.0, 4.0, 0.0, 0.0])
        // Without penalty, argmax is token 0.
        var opts = SamplingOptions(temperature: 0)
        XCTAssertEqual(Sampler.sample(logits, history: [], options: &opts), 0)
        // Penalise token 0 three times: 5.0 - 3 × 1.0 = 2.0, < 4.0.
        var opts2 = SamplingOptions(temperature: 0, frequencyPenalty: 1.0)
        let picked = Sampler.sample(logits, history: [0, 0, 0], options: &opts2)
        XCTAssertEqual(picked, 1, "frequency penalty should have demoted token 0 below token 1")
    }

    /// Presence penalty is binary in count: once is enough.
    func testPresencePenaltyIsBinary() {
        let logits = logitsTensor([5.0, 4.0, 0.0, 0.0])
        // 5.0 − 2.0 = 3.0, still > 4.0 - 0 = 4.0? No, 3 < 4. So token 1 wins.
        var opts = SamplingOptions(temperature: 0, presencePenalty: 2.0)
        let picked1 = Sampler.sample(logits, history: [0], options: &opts)
        let picked5 = Sampler.sample(logits, history: [0, 0, 0, 0, 0], options: &opts)
        XCTAssertEqual(picked1, 1)
        XCTAssertEqual(picked5, 1, "presence penalty must apply once regardless of count")
    }

    // MARK: - Tail-free

    /// Tail-free with a tight `z` should mask away the long flat tail of
    /// a Zipf-like distribution.
    func testTailFreeMasksFlatTail() {
        // Sharp head + flat tail.
        var logits: [Float] = [5.0, 3.0, 1.0]
        for _ in 0..<20 { logits.append(0.0) }
        var opts = SamplingOptions(temperature: 1.0, tailFree: 0.5, rngState: 99)
        var seen = Set<Int>()
        for _ in 0..<200 {
            seen.insert(Sampler.sample(logitsTensor(logits), history: [], options: &opts))
        }
        // We expect tokens 0..2 to dominate the support, never the tail.
        for i in 3..<logits.count {
            XCTAssertFalse(seen.contains(i), "tail-free should have removed token \(i)")
        }
    }

    // MARK: - Typical sampling

    /// Locally-typical sampling with `p` small keeps the tokens whose
    /// surprise is closest to the entropy. On a delta-like distribution
    /// the entropy is near zero, so the argmax token (lowest surprise)
    /// is kept.
    func testTypicalSamplingKeepsNearEntropyToken() {
        let logits = logitsTensor([10.0, 0.0, 0.0, 0.0])
        var opts = SamplingOptions(temperature: 1.0, typical: 0.2, rngState: 31)
        for _ in 0..<50 {
            XCTAssertEqual(Sampler.sample(logits, history: [], options: &opts), 0)
        }
    }

    // MARK: - Mirostat

    /// After a few hundred decode steps mu converges around the
    /// observed average surprise. We can't test full convergence
    /// against a real LM, but on a deterministic prompt we expect
    /// |mu - 2τ| to shrink as the update rule is applied (initial
    /// mu defaults to 10 ≈ 2τ ≈ 10 when tau=5).
    func testMirostatMuUpdatesEachStep() {
        let logits = logitsTensor([2.0, 1.0, 0.5, 0.0, -0.5])
        var opts = SamplingOptions(temperature: 1.0,
                                   mirostatTau: 3.0, mirostatEta: 0.1,
                                   mirostatMu: 6.0, rngState: 123)
        let initialMu = opts.mirostatMu
        _ = Sampler.sample(logits, history: [], options: &opts)
        XCTAssertNotEqual(opts.mirostatMu, initialMu, "mu should update after a step")
    }

    /// Mirostat does not deadlock on a degenerate (zero-surprise)
    /// distribution: it should pick the dominant token without
    /// throwing or returning out-of-range ids.
    func testMirostatHandlesDegenerateDistribution() {
        let logits = logitsTensor([100.0, 0.0, 0.0, 0.0])
        var opts = SamplingOptions(temperature: 1.0, mirostatTau: 5.0, rngState: 5)
        let picked = Sampler.sample(logits, history: [], options: &opts)
        XCTAssertEqual(picked, 0)
    }

    // MARK: - Combined / safety

    /// All filters on at the same time should still pick a valid token
    /// without NaN or Inf in the output id.
    func testAllFiltersStackedNoCrash() {
        let logits = logitsTensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
        var opts = SamplingOptions(
            temperature: 0.8,
            topK: 5, topP: 0.9,
            minP: 0.05, tailFree: 0.95, typical: 0.95,
            repetitionPenalty: 1.05,
            frequencyPenalty: 0.1, presencePenalty: 0.1,
            rngState: 2024)
        let picked = Sampler.sample(logits, history: [0, 1, 1], options: &opts)
        XCTAssertGreaterThanOrEqual(picked, 0)
        XCTAssertLessThan(picked, 8)
    }

    /// Filter combinations that would otherwise mask the entire support
    /// must still return a valid token (defensive fallback).
    func testAggressiveFiltersFallback() {
        let logits = logitsTensor([5.0, 4.0, 3.0])
        var opts = SamplingOptions(
            temperature: 1.0,
            topK: 0, topP: 0.01, minP: 0.99, tailFree: 0.01, typical: 0.01,
            rngState: 13)
        let picked = Sampler.sample(logits, history: [], options: &opts)
        XCTAssertGreaterThanOrEqual(picked, 0)
        XCTAssertLessThan(picked, 3)
    }

    // MARK: - Reproducibility

    /// Identical seeds + identical logits must produce identical token
    /// streams. This is the key invariant the LCG state was added for.
    func testReproducibilityFromExplicitSeed() {
        let logits = logitsTensor([2.0, 1.5, 1.0, 0.5, 0.0])
        var optsA = SamplingOptions(temperature: 1.0, topK: 5, rngState: 0xDEAD_BEEF)
        var optsB = SamplingOptions(temperature: 1.0, topK: 5, rngState: 0xDEAD_BEEF)
        for _ in 0..<20 {
            let a = Sampler.sample(logits, history: [], options: &optsA)
            let b = Sampler.sample(logits, history: [], options: &optsB)
            XCTAssertEqual(a, b)
        }
    }
}
