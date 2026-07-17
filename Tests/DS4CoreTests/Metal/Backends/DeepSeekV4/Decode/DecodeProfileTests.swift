import XCTest
@testable import DS4Metal

final class DecodeProfileTests: XCTestCase {
    func testMixedCacheReportSeparatesCacheableAndGlobalRates() {
        var profile = DecodeProfile()
        profile.forwards = 2
        profile.expertHits = 6
        profile.expertMisses = 2
        profile.expertHitBytes = 60 * 1_048_576
        profile.expertMissBytes = 20 * 1_048_576
        profile.expertBypasses = 4
        profile.expertBypassBytes = 200 * 1_048_576
        profile.expertPrefilled = 3
        profile.expertPrefilledBytes = 20 * 1_048_576
        profile.expertWarmed = 2
        profile.expertWarmedBytes = 12 * 1_048_576

        XCTAssertEqual(profile.expertCacheableHitRate ?? -1, 0.75, accuracy: 0.000_001)
        XCTAssertEqual(profile.expertGlobalHitRate ?? -1, 0.5, accuracy: 0.000_001)
        XCTAssertEqual(profile.expertCacheableByteHitRate ?? -1, 0.75, accuracy: 0.000_001)
        XCTAssertEqual(profile.expertGlobalByteHitRate ?? -1, 60.0 / 280.0, accuracy: 0.000_001)
        let report = profile.report()
        XCTAssertTrue(report.contains("6 hit / 2 miss  (75% sui cacheabili) — 3 slab da look-ahead, 10.0 MB/token I/O nascosti"))
        XCTAssertTrue(report.contains("cache bypass 4 selezioni, 100.0 MB/token — 50% hit globale"))
        XCTAssertTrue(report.contains("cache warm   2 slab iniziali, 6.0 MB/token I/O sincroni"))
        XCTAssertTrue(report.contains("cache byte    75% hit sui cacheabili / 21% globale"))
    }

    func testBypassOnlyReportStillShowsHonestZeroGlobalHitRate() {
        var profile = DecodeProfile()
        profile.forwards = 1
        profile.expertBypasses = 6

        let report = profile.report()
        XCTAssertFalse(report.contains("cache expert"))
        XCTAssertTrue(report.contains("cache bypass 6 selezioni — 0% hit globale"))
    }
}
