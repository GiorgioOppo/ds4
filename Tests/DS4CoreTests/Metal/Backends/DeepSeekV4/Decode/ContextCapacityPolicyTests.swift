import XCTest
@testable import DS4Metal

/// Pure policy tests for long-context decode. They intentionally require no
/// Metal device: the expensive resources are allowed to follow the live
/// high-water mark, not the configured context capacity.
final class ContextCapacityPolicyTests: XCTestCase {

    func testIndexerScoringStartsOnlyPastTheLiveSparseBoundary() {
        let topK = 512
        let sparseThreshold = 1_024

        // Ratio-4 layers have 1,024 emitted rows through live key 4,099.
        // Activation is strict (`rows > threshold`), so scoring starts only
        // when live key 4,100 can produce row 1,025.
        XCTAssertFalse(StreamingDecoder.indexerScoringNeeded(
            liveKeys: 4_096, topK: topK, sparseThreshold: sparseThreshold
        ))
        XCTAssertFalse(StreamingDecoder.indexerScoringNeeded(
            liveKeys: 4_099, topK: topK, sparseThreshold: sparseThreshold
        ))
        XCTAssertTrue(StreamingDecoder.indexerScoringNeeded(
            liveKeys: 4_100, topK: topK, sparseThreshold: sparseThreshold
        ))
    }

    func testScratchRowsFollowLiveKeysAndSlidingWindow() {
        XCTAssertEqual(StreamingDecoder.scratchRowsNeeded(nKeys: 0, nSWA: 128), 8)
        XCTAssertEqual(StreamingDecoder.scratchRowsNeeded(nKeys: 1, nSWA: 128), 9)
        XCTAssertEqual(StreamingDecoder.scratchRowsNeeded(nKeys: 128, nSWA: 128), 168)
        XCTAssertEqual(StreamingDecoder.scratchRowsNeeded(nKeys: 129, nSWA: 128), 168)
        XCTAssertEqual(StreamingDecoder.scratchRowsNeeded(nKeys: 4_096, nSWA: 128), 1_160)
        XCTAssertEqual(StreamingDecoder.scratchRowsNeeded(nKeys: 4_100, nSWA: 128), 1_161)
    }

    func testScratchCapacityGrowsGeometricallyAndHonoursCap() {
        XCTAssertEqual(StreamingDecoder.grownScratchCapacity(
            current: 16, required: 16, maximum: 100
        ), 16)
        XCTAssertEqual(StreamingDecoder.grownScratchCapacity(
            current: 16, required: 17, maximum: 100
        ), 32)
        XCTAssertEqual(StreamingDecoder.grownScratchCapacity(
            current: 64, required: 65, maximum: 100
        ), 100)
        XCTAssertEqual(StreamingDecoder.grownScratchCapacity(
            current: 1, required: 90, maximum: 100
        ), 90)
    }
}
