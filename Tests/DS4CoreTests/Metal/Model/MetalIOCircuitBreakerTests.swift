import XCTest
@testable import DS4Metal

final class MetalIOCircuitBreakerTests: XCTestCase {
    func testSmallBatchesAreAggregatedAndRequireTwoSlowWindows() {
        var breaker = MetalIOCircuitBreaker(minimumGBs: 1.5,
                                            windowBytes: 64 << 20,
                                            slowWindowsRequired: 2)
        let bytes = 8 << 20
        let slowSeconds = Double(bytes) / 1.0e9

        for _ in 0..<7 {
            guard case .keep = breaker.record(bytes: bytes, seconds: slowSeconds) else {
                return XCTFail("a sub-window sample must not trip the breaker")
            }
        }
        guard case let .slowWindow(_, count) = breaker.record(bytes: bytes, seconds: slowSeconds) else {
            return XCTFail("the first complete slow window must only warn")
        }
        XCTAssertEqual(count, 1)

        for _ in 0..<7 {
            guard case .keep = breaker.record(bytes: bytes, seconds: slowSeconds) else {
                return XCTFail("the second window must also be accumulated")
            }
        }
        guard case .disable = breaker.record(bytes: bytes, seconds: slowSeconds) else {
            return XCTFail("two consecutive slow windows must disable MetalIO")
        }
    }

    func testFastWindowClearsPreviousSlowWindow() {
        var breaker = MetalIOCircuitBreaker(minimumGBs: 1.5,
                                            windowBytes: 16 << 20,
                                            slowWindowsRequired: 2)
        let bytes = 16 << 20
        guard case .slowWindow = breaker.record(bytes: bytes, seconds: Double(bytes) / 1.0e9) else {
            return XCTFail("expected first slow window")
        }
        guard case .keep = breaker.record(bytes: bytes, seconds: Double(bytes) / 3.0e9) else {
            return XCTFail("a fast window must keep MetalIO enabled")
        }
        XCTAssertEqual(breaker.consecutiveSlowWindows, 0)
        guard case .slowWindow = breaker.record(bytes: bytes, seconds: Double(bytes) / 1.0e9) else {
            return XCTFail("a later slow window must restart at warning state")
        }
    }

    func testFastLargeBatchAbsorbsOneSlowSmallBatch() {
        var breaker = MetalIOCircuitBreaker(minimumGBs: 1.5,
                                            windowBytes: 64 << 20,
                                            slowWindowsRequired: 2)
        let fastBytes = 56 << 20
        let slowBytes = 8 << 20
        XCTAssertKeep(breaker.record(bytes: fastBytes, seconds: Double(fastBytes) / 4.0e9))
        XCTAssertKeep(breaker.record(bytes: slowBytes, seconds: Double(slowBytes) / 1.0e9))
        XCTAssertEqual(breaker.consecutiveSlowWindows, 0)
    }

    private func XCTAssertKeep(_ decision: MetalIOCircuitBreaker.Decision,
                               file: StaticString = #filePath, line: UInt = #line) {
        guard case .keep = decision else {
            return XCTFail("expected keep", file: file, line: line)
        }
    }
}
