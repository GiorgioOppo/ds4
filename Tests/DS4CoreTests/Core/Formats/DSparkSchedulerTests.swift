import XCTest
@testable import DS4Metal

final class DSparkSchedulerTests: XCTestCase {
    func testTailSkipProtectsShortGenerations() {
        var scheduler = DSparkAdaptiveScheduler(policy: .init(tailMinimumTokens: 10))
        XCTAssertEqual(scheduler.decision(remainingTokens: 8), .skipTail)
        XCTAssertEqual(scheduler.decision(remainingTokens: 10), .attempt)
        XCTAssertEqual(scheduler.snapshot.tailSkips, 1)
        XCTAssertEqual(scheduler.snapshot.attempts, 0)
    }

    func testColdLowConfidenceUsesLongBackoff() {
        var scheduler = DSparkAdaptiveScheduler(policy: .init(
            coldLowConfidenceSkipCycles: 7,
            coldLowConfidenceThreshold: 0.5,
            tailMinimumTokens: 0))
        scheduler.note(proposedDrafts: 0, acceptedDrafts: 0,
                       firstConfidence: 0.25)

        for remaining in stride(from: 6, through: 0, by: -1) {
            XCTAssertEqual(
                scheduler.decision(remainingTokens: 100),
                .skipBackoff(remaining: remaining))
        }
        XCTAssertEqual(scheduler.decision(remainingTokens: 100), .attempt)
        XCTAssertEqual(scheduler.snapshot.backoffSkips, 7)
        XCTAssertEqual(scheduler.snapshot.emptyProposals, 1)
    }

    func testRollingLowAcceptancePausesWithoutTimingInput() {
        var scheduler = DSparkAdaptiveScheduler(policy: .init(
            window: 4, skipCycles: 2, slowSkipCycles: 4,
            minimumAverageAccepted: 1.5, tailMinimumTokens: 0))
        for _ in 0..<4 {
            scheduler.note(proposedDrafts: 5, acceptedDrafts: 1,
                           firstConfidence: 0.9)
        }
        XCTAssertEqual(
            scheduler.decision(remainingTokens: 100),
            .skipBackoff(remaining: 1))
        XCTAssertEqual(
            scheduler.decision(remainingTokens: 100),
            .skipBackoff(remaining: 0))
        XCTAssertEqual(scheduler.decision(remainingTokens: 100), .attempt)
    }

    func testHealthyAcceptanceKeepsSchedulerOpen() {
        var scheduler = DSparkAdaptiveScheduler(policy: .init(
            window: 4, minimumAverageAccepted: 1.5,
            tailMinimumTokens: 0))
        for _ in 0..<4 {
            scheduler.note(proposedDrafts: 5, acceptedDrafts: 4,
                           firstConfidence: 0.9)
        }
        XCTAssertEqual(scheduler.decision(remainingTokens: 100), .attempt)
        XCTAssertEqual(scheduler.snapshot.accepted, 16)
        XCTAssertEqual(scheduler.snapshot.proposals, 20)
        XCTAssertEqual(scheduler.snapshot.acceptanceRate, 0.8, accuracy: 0.000_001)
    }

    func testEnvironmentPolicyMatchesUpstreamNames() {
        let policy = DSparkSchedulerPolicy.fromEnvironment([
            "DS4_DSPARK_SCHEDULER": "0",
            "DS4_DSPARK_SCHEDULER_WINDOW": "6",
            "DS4_DSPARK_SCHEDULER_MIN_AVG_MILLI": "2250",
            "DS4_DSPARK_SCHEDULER_TAIL_MIN_TOKENS": "12",
        ])
        XCTAssertFalse(policy.enabled)
        XCTAssertEqual(policy.window, 6)
        XCTAssertEqual(policy.minimumAverageAccepted, 2.25, accuracy: 0.000_001)
        XCTAssertEqual(policy.tailMinimumTokens, 12)
    }
}
