import XCTest
@testable import DS4Core

final class DS4RuntimeEnvironmentTests: XCTestCase {
    func testCanonicalNameIsUsedAsFallback() {
        let environment = ["DS4_PREFILL_BATCH": "1"]
        XCTAssertEqual(
            DS4RuntimeEnvironment.value(
                "DS4_PREFILL_BATCH",
                overrides: ["DS4_GLM_PREFILL_BATCH"],
                environment: environment),
            "1")
    }

    func testBackendSpecificNameOverridesCanonicalName() {
        let environment = [
            "DS4_PREFILL_BATCH": "1",
            "DS4_GLM_PREFILL_BATCH": "0",
        ]
        XCTAssertFalse(DS4RuntimeEnvironment.flag(
            "DS4_PREFILL_BATCH",
            overrides: ["DS4_GLM_PREFILL_BATCH"],
            default: true,
            environment: environment))
    }

    func testAbsentFlagKeepsBackendDefault() {
        XCTAssertTrue(DS4RuntimeEnvironment.flag(
            "DS4_MLOCK", default: true, environment: [:]))
        XCTAssertFalse(DS4RuntimeEnvironment.flag(
            "DS4_MLOCK", default: false, environment: [:]))
    }

    func testIntegerUsesTheSamePrecedence() {
        let environment = [
            "DS4_PREFILL_ROUTE_BATCH": "32",
            "DS4_GLM_PREFILL_ROUTE_BATCH": "16",
        ]
        XCTAssertEqual(DS4RuntimeEnvironment.integer(
            "DS4_PREFILL_ROUTE_BATCH",
            overrides: ["DS4_GLM_PREFILL_ROUTE_BATCH"],
            environment: environment), 16)
    }
}
