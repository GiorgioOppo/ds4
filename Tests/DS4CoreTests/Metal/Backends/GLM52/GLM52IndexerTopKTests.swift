import DS4Core
import XCTest
@testable import DS4Metal

/// The multi-block GPU top-k must reproduce the CPU oracle
/// (`GLM52IndexerCPUReference.causalTopK`) on distinct scores — single-pass and
/// merge path — and must never select causal -infinity rows. Distinct scores
/// sidestep the one intended divergence: the bitonic network does not promise
/// the oracle's lowest-index tie order.
final class GLM52IndexerTopKTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    /// Distinct pseudo-random scores: a seeded shuffle of evenly spaced values,
    /// so every comparison in sort, merge and oracle is strict.
    private func distinctScores(count: Int, seed: UInt64) -> [Float] {
        var values = (0..<count).map { Float($0) * 0.125 - Float(count) * 0.0625 }
        var state = seed
        func next(_ bound: Int) -> Int {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int(state >> 33) % bound
        }
        for i in stride(from: count - 1, to: 0, by: -1) {
            values.swapAt(i, next(i + 1))
        }
        return values
    }

    func testSinglePassMatchesOracle() throws {
        let runtime = try makeRuntime()
        let rowCount = 96
        let topK = 8
        let scores = distinctScores(count: rowCount, seed: 11)

        let gpu = try runtime.glm52IndexerTopK(
            scores: scores, rowCount: rowCount, tokenCount: 1, topK: topK)
        let oracle = try GLM52IndexerCPUReference.causalTopK(
            scores: scores, queryPosition: rowCount - 1, topK: topK)

        XCTAssertEqual(gpu.map(Int.init), oracle)
    }

    func testMultiBlockMergeMatchesOraclePerToken() throws {
        let runtime = try makeRuntime()
        let rowCount = 4_096
        let tokenCount = 3
        let topK = 100
        var scores = [Float]()
        scores.reserveCapacity(rowCount * tokenCount)
        for token in 0..<tokenCount {
            scores.append(contentsOf: distinctScores(
                count: rowCount, seed: 900 + UInt64(token)))
        }

        let gpu = try runtime.glm52IndexerTopK(
            scores: scores, rowCount: rowCount,
            tokenCount: tokenCount, topK: topK)
        XCTAssertEqual(gpu.count, tokenCount * topK)

        for token in 0..<tokenCount {
            let row = Array(scores[token * rowCount..<(token + 1) * rowCount])
            let oracle = try GLM52IndexerCPUReference.causalTopK(
                scores: row, queryPosition: rowCount - 1, topK: topK)
            XCTAssertEqual(
                Array(gpu[token * topK..<(token + 1) * topK]).map(Int.init),
                oracle,
                "token \(token) diverges from the oracle")
        }
    }

    func testCausalInfinityRowsAreNeverSelected() throws {
        let runtime = try makeRuntime()
        let rowCount = 64
        let visible = 40
        let topK = 16
        var scores = distinctScores(count: visible, seed: 77)
        scores.append(contentsOf: [Float](
            repeating: -.infinity, count: rowCount - visible))

        let gpu = try runtime.glm52IndexerTopK(
            scores: scores, rowCount: rowCount, tokenCount: 1, topK: topK)
        XCTAssertTrue(gpu.allSatisfy { Int($0) < visible },
                      "selected a causal future row: \(gpu)")

        let oracle = try GLM52IndexerCPUReference.causalTopK(
            scores: Array(scores[0..<visible]),
            queryPosition: visible - 1,
            topK: topK)
        XCTAssertEqual(gpu.map(Int.init), oracle)
    }

    func testSpecializedTopK2048MatchesGenericAtBoundaries() throws {
        let runtime = try makeRuntime()
        for rowCount in [2_048, 2_049, 4_096, 4_097, 8_192, 8_193] {
            let scores = distinctScores(
                count: rowCount, seed: 2_048 + UInt64(rowCount))
            if rowCount > 8_192 {
                XCTAssertTrue(
                    try runtime.glm52SupportsFastIndexerTopK(
                        rowCount: rowCount),
                    "the test device cannot exercise the fast path at "
                        + "\(rowCount) rows")
            }

            let fast = try runtime.glm52IndexerTopK(
                scores: scores,
                rowCount: rowCount,
                tokenCount: 1,
                topK: 2_048,
                preferFastPath: true)
            let generic = try runtime.glm52IndexerTopK(
                scores: scores,
                rowCount: rowCount,
                tokenCount: 1,
                topK: 2_048,
                preferFastPath: false)
            XCTAssertEqual(
                fast, generic,
                "specialized top-2048 diverges at \(rowCount) rows")
        }
    }

    /// Opt-in microbenchmark used for the implementation A/B:
    /// `DS4_TOPK_AB_BENCH=1 swift test -c release
    ///   --filter GLM52IndexerTopKTests/testTopK2048ABBenchmark`
    func testTopK2048ABBenchmark() throws {
        guard ProcessInfo.processInfo.environment["DS4_TOPK_AB_BENCH"] == "1"
        else {
            throw XCTSkip("set DS4_TOPK_AB_BENCH=1 to run the top-k A/B")
        }
        let runtime = try makeRuntime()
        let topK = 2_048
        for rowCount in [4_096, 8_192, 8_193, 12_288, 16_384, 32_768] {
            let scores = distinctScores(
                count: rowCount, seed: 10 * UInt64(rowCount))
            let scoreBuffer = try runtime.glm52GraphBuffer(scores)
            let fastOutput = try runtime.glm52GraphOutputBuffer(floats: topK)
            let genericOutput = try runtime.glm52GraphOutputBuffer(floats: topK)
            let scratch = try runtime.glm52GraphOutputBuffer(
                floats: 2 * rowCount)
            let supportsFast = try runtime.glm52SupportsFastIndexerTopK(
                rowCount: rowCount,
                outputBytes: fastOutput.length,
                scratchBytes: scratch.length)

            func run(
                preferFastPath: Bool,
                iterations: Int
            ) throws -> Double {
                let start = Date()
                for _ in 0..<iterations {
                    guard let commandBuffer =
                            runtime.queue.makeCommandBuffer()
                    else { throw MetalError.bufferAlloc }
                    try runtime.glm52EncodeIndexerTopK(
                        into: commandBuffer,
                        scores: scoreBuffer,
                        rowCount: rowCount,
                        topK: topK,
                        output: preferFastPath
                            ? fastOutput : genericOutput,
                        sortScratch: scratch,
                        preferFastPath: preferFastPath)
                    try runtime.glm52GraphCommit(commandBuffer)
                }
                return Date().timeIntervalSince(start) / Double(iterations)
            }

            _ = try run(preferFastPath: true, iterations: 1)
            _ = try run(preferFastPath: false, iterations: 1)
            let iterations = 20
            var fastSamples = [Double]()
            var genericSamples = [Double]()
            for round in 0..<7 {
                if round.isMultiple(of: 2) {
                    fastSamples.append(try run(
                        preferFastPath: true, iterations: iterations))
                    genericSamples.append(try run(
                        preferFastPath: false, iterations: iterations))
                } else {
                    genericSamples.append(try run(
                        preferFastPath: false, iterations: iterations))
                    fastSamples.append(try run(
                        preferFastPath: true, iterations: iterations))
                }
            }
            fastSamples.sort()
            genericSamples.sort()
            let fastSeconds = fastSamples[fastSamples.count / 2]
            let genericSeconds = genericSamples[genericSamples.count / 2]

            let fast = fastOutput.contents()
                .bindMemory(to: UInt32.self, capacity: topK)
            let generic = genericOutput.contents()
                .bindMemory(to: UInt32.self, capacity: topK)
            XCTAssertEqual(
                Array(UnsafeBufferPointer(start: fast, count: topK)),
                Array(UnsafeBufferPointer(start: generic, count: topK)))
            print(String(
                format: "GLM52 top-k %d A/B (%@): fast %.3f ms, "
                    + "generic %.3f ms, speedup %.2fx",
                rowCount,
                supportsFast ? "compact" : "same generic",
                fastSeconds * 1_000,
                genericSeconds * 1_000,
                genericSeconds / fastSeconds))
        }
    }

    func testValidationRejectsBadGeometry() throws {
        let runtime = try makeRuntime()
        let scores = distinctScores(count: 32, seed: 5)

        XCTAssertThrowsError(try runtime.glm52IndexerTopK(
            scores: scores, rowCount: 33, tokenCount: 1, topK: 4))
        XCTAssertThrowsError(try runtime.glm52IndexerTopK(
            scores: scores, rowCount: 32, tokenCount: 1, topK: 0))
        XCTAssertThrowsError(try runtime.glm52IndexerTopK(
            scores: scores, rowCount: 32, tokenCount: 1, topK: 33))
        XCTAssertThrowsError(try runtime.glm52IndexerTopK(
            scores: [], rowCount: 0, tokenCount: 1, topK: 1))
    }
}
