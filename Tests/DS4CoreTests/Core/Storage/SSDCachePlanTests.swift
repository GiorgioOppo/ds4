import XCTest
@testable import DS4Core

/// Swift-only checks of the SSD streaming cache planning logic in
/// `SSDStreaming`. Expected values are derived directly from the documented
/// behavior (parse "<n>"/"<n>GB", 4/5 budget target, byte→expert conversion).
/// The original bit-for-bit cross-check against ds4_ssd.c was dropped with the
/// C engine.
final class SSDCachePlanTests: XCTestCase {
    private let gib = SSDStreaming.gib

    // MARK: parseGiBArg

    func testParseGiBArg() {
        XCTAssertEqual(SSDStreaming.parseGiBArg("10"), 10 * gib)
        XCTAssertEqual(SSDStreaming.parseGiBArg("10GB"), 10 * gib)
        XCTAssertEqual(SSDStreaming.parseGiBArg("10gb"), 10 * gib)
        XCTAssertEqual(SSDStreaming.parseGiBArg("10gB"), 10 * gib)
        XCTAssertEqual(SSDStreaming.parseGiBArg("32GB"), 32 * gib)
        XCTAssertEqual(SSDStreaming.parseGiBArg("007"), 7 * gib)

        // Rejected: empty, suffix-only, zero, non-digits, decimals, overflow.
        for bad in ["", "GB", "G", "B", "0", "abc", "1.5GB", " 10", "10 ",
                    "999999999999GB", "18446744073709551616"] {
            XCTAssertNil(SSDStreaming.parseGiBArg(bad), "should reject \"\(bad)\"")
        }
    }

    // MARK: parseCacheExpertsArg

    func testParseCacheExpertsArg() {
        // A "…GB" suffix is a byte budget; a bare integer is an expert count.
        XCTAssertEqual(SSDStreaming.parseCacheExpertsArg("32GB"), .bytes(32 * gib))
        XCTAssertEqual(SSDStreaming.parseCacheExpertsArg("1gb"), .bytes(gib))
        XCTAssertEqual(SSDStreaming.parseCacheExpertsArg("4854"), .experts(4854))
        XCTAssertEqual(SSDStreaming.parseCacheExpertsArg("4294967295"), .experts(UInt32.max))

        // Rejected: empty, zero, non-digits, > UInt32.max expert count.
        for bad in ["", "0", "GB", "abc", "4294967296"] {
            XCTAssertNil(SSDStreaming.parseCacheExpertsArg(bad), "should reject \"\(bad)\"")
        }
    }

    // MARK: cacheExpertsForByteBudget

    func testCacheExpertsForByteBudget() {
        XCTAssertEqual(SSDStreaming.cacheExpertsForByteBudget(bytes: 0, perExpertBytes: 100), 0)
        XCTAssertEqual(SSDStreaming.cacheExpertsForByteBudget(bytes: 100, perExpertBytes: 0), 0)
        XCTAssertEqual(SSDStreaming.cacheExpertsForByteBudget(bytes: 1000, perExpertBytes: 7), 142) // floor(1000/7)
        XCTAssertEqual(SSDStreaming.cacheExpertsForByteBudget(bytes: gib, perExpertBytes: gib), 1)
        // Result that would exceed UInt32.max is clamped to 0 (rejected).
        XCTAssertEqual(SSDStreaming.cacheExpertsForByteBudget(bytes: UInt64.max, perExpertBytes: 1), 0)
    }

    // MARK: autoCachePlan (targets 4/5 of the recommended budget)

    func testAutoCachePlanCleanCase() {
        // recommended 5 GiB -> modelTarget 4 GiB; nonRouted 1 GiB -> cache 3 GiB;
        // per-expert 1 GiB -> 3 experts (no cap).
        guard let p = SSDStreaming.autoCachePlan(recommendedBytes: 5 * gib, nonRoutedBytes: gib,
                                                 perExpertBytes: gib, maxModelExperts: 0) else {
            return XCTFail("plan should be non-nil")
        }
        XCTAssertEqual(p.modelTargetBytes, 4 * gib)
        XCTAssertEqual(p.cacheBytes, 3 * gib)
        XCTAssertEqual(p.cacheExperts, 3)
        XCTAssertEqual(p.effectiveCacheBytes, 3 * gib)
    }

    func testAutoCachePlanExpertCap() {
        // Same target/cache (4 experts available) but capped to 2 by maxModelExperts.
        guard let p = SSDStreaming.autoCachePlan(recommendedBytes: 5 * gib, nonRoutedBytes: 0,
                                                 perExpertBytes: gib, maxModelExperts: 2) else {
            return XCTFail("plan should be non-nil")
        }
        XCTAssertEqual(p.cacheExperts, 2)
    }

    func testAutoCachePlanFloorsToOneExpert() {
        // nonRouted >= modelTarget -> cacheBytes 0 -> floored to 1 expert.
        guard let p = SSDStreaming.autoCachePlan(recommendedBytes: 5 * gib, nonRoutedBytes: 10 * gib,
                                                 perExpertBytes: gib, maxModelExperts: 0) else {
            return XCTFail("plan should be non-nil")
        }
        XCTAssertEqual(p.cacheBytes, 0)
        XCTAssertEqual(p.cacheExperts, 1)
        XCTAssertEqual(p.effectiveCacheBytes, gib)
    }

    func testAutoCachePlanRejectsDegenerate() {
        XCTAssertNil(SSDStreaming.autoCachePlan(recommendedBytes: 0, nonRoutedBytes: 0,
                                                perExpertBytes: gib, maxModelExperts: 0))
        XCTAssertNil(SSDStreaming.autoCachePlan(recommendedBytes: 5 * gib, nonRoutedBytes: 0,
                                                perExpertBytes: 0, maxModelExperts: 0))
    }
}
