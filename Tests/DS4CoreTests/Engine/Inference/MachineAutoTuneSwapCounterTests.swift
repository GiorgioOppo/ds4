import XCTest
@testable import DS4Engine

final class MachineAutoTuneSwapCounterTests: XCTestCase {
    private let mib: UInt64 = 1_048_576

    func testSeparatesLoadChurnFromSteadyStatePolicyWindow() throws {
        let windows = MachineAutoTuneSwapWindows(
            beforeLoadBytes: 100 * mib,
            steadyStateStartBytes: 350 * mib,
            steadyStateEndBytes: 362 * mib
        )

        XCTAssertEqual(windows.load, .advanced(bytes: 250 * mib))
        XCTAssertEqual(windows.steadyState, .advanced(bytes: 12 * mib))
        XCTAssertEqual(try XCTUnwrap(windows.loadMiB), 250, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(windows.policySwapoutMiB), 12, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(windows.totalMiB), 262, accuracy: 1e-12)
    }

    func testUnchangedCounterProducesZeroForBothWindows() throws {
        let windows = MachineAutoTuneSwapWindows(
            beforeLoadBytes: 42,
            steadyStateStartBytes: 42,
            steadyStateEndBytes: 42
        )

        XCTAssertEqual(windows.load, .advanced(bytes: 0))
        XCTAssertEqual(windows.steadyState, .advanced(bytes: 0))
        XCTAssertEqual(try XCTUnwrap(windows.policySwapoutMiB), 0, accuracy: 0)
        XCTAssertEqual(try XCTUnwrap(windows.totalMiB), 0, accuracy: 0)
    }

    func testTwoGiBLoadChurnWithIdleSteadyWindowReportsZeroPolicySwap() throws {
        let initial = 64 * mib
        let afterLoad = initial + 2_048 * mib
        let windows = MachineAutoTuneSwapWindows(
            beforeLoadBytes: initial,
            steadyStateStartBytes: afterLoad,
            steadyStateEndBytes: afterLoad
        )

        XCTAssertEqual(try XCTUnwrap(windows.loadMiB), 2_048, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(windows.policySwapoutMiB), 0, accuracy: 0)
        XCTAssertEqual(try XCTUnwrap(windows.totalMiB), 2_048, accuracy: 1e-12)
    }

    func testCredibleCounterWrapIsMeasuredExactly() throws {
        let windows = MachineAutoTuneSwapWindows(
            beforeLoadBytes: 250,
            steadyStateStartBytes: 5,
            steadyStateEndBytes: 9,
            counterMaximum: 255
        )

        XCTAssertEqual(windows.load, .wrapped(bytes: 11))
        XCTAssertTrue(windows.load.didWrap)
        XCTAssertFalse(windows.load.didReset)
        XCTAssertEqual(windows.steadyState, .advanced(bytes: 4))
        XCTAssertEqual(
            try XCTUnwrap(windows.totalMiB),
            15 / MachineAutoTuneSwapWindows.bytesPerMiB,
            accuracy: 1e-15
        )
    }

    func testLoadCounterResetDoesNotInvalidateLaterSteadyWindow() throws {
        let windows = MachineAutoTuneSwapWindows(
            beforeLoadBytes: 1_000,
            steadyStateStartBytes: 10,
            steadyStateEndBytes: 30,
            counterMaximum: 2_000
        )

        XCTAssertTrue(windows.load.didReset)
        XCTAssertNil(windows.loadMiB)
        XCTAssertNil(windows.totalMiB)
        XCTAssertEqual(windows.steadyState, .advanced(bytes: 20))
        XCTAssertEqual(
            try XCTUnwrap(windows.policySwapoutMiB),
            20 / MachineAutoTuneSwapWindows.bytesPerMiB,
            accuracy: 1e-15
        )
    }

    func testSteadyCounterResetFailsClosedInsteadOfBecomingZero() {
        let windows = MachineAutoTuneSwapWindows(
            beforeLoadBytes: 100,
            steadyStateStartBytes: 1_000,
            steadyStateEndBytes: 10,
            counterMaximum: 2_000
        )

        XCTAssertEqual(windows.load, .advanced(bytes: 900))
        XCTAssertTrue(windows.steadyState.didReset)
        XCTAssertNil(windows.policySwapoutMiB)
        XCTAssertNil(windows.totalMiB)
    }

    func testOutOfDeclaredCounterRangeFailsClosed() {
        XCTAssertEqual(
            MachineAutoTuneSwapWindows.delta(
                from: 200, to: 210, counterMaximum: 205
            ),
            .reset
        )
        XCTAssertEqual(
            MachineAutoTuneSwapWindows.delta(
                from: 2, to: 1, counterMaximum: 2
            ),
            .reset
        )
    }

    func testDiagnosticTotalRejectsOverflowWithoutLosingSteadyMetric() throws {
        let windows = MachineAutoTuneSwapWindows(
            beforeLoadBytes: 0,
            steadyStateStartBytes: UInt64.max - 5,
            steadyStateEndBytes: 1
        )

        XCTAssertEqual(windows.load, .advanced(bytes: UInt64.max - 5))
        XCTAssertEqual(windows.steadyState, .wrapped(bytes: 7))
        XCTAssertNil(windows.totalMiB)
        XCTAssertEqual(
            try XCTUnwrap(windows.policySwapoutMiB),
            7 / MachineAutoTuneSwapWindows.bytesPerMiB,
            accuracy: 1e-15
        )
    }
}
