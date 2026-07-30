import XCTest
@testable import DS4Core

final class DS4RuntimeEnvironmentTests: XCTestCase {
    func testCanonicalNameIsUsedDirectly() {
        let environment = ["DS4_PREFILL_BATCH": "1"]
        XCTAssertEqual(
            DS4RuntimeEnvironment.value(
                "DS4_PREFILL_BATCH",
                overrides: ["DS4_GLM_PREFILL_BATCH"],
                environment: environment),
            "1")
    }

    func testCanonicalNameOverridesDeprecatedBackendAlias() {
        let environment = [
            "DS4_PREFILL_BATCH": "1",
            "DS4_GLM_PREFILL_BATCH": "0",
        ]
        XCTAssertTrue(DS4RuntimeEnvironment.flag(
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
            environment: environment), 32)
    }

    func testDeprecatedBackendAliasRemainsAFallback() {
        let environment = ["DS4_GLM_MTLIO": "0"]
        XCTAssertFalse(DS4RuntimeEnvironment.flag(
            "DS4_MTLIO",
            overrides: ["DS4_GLM_MTLIO"],
            default: true,
            environment: environment))
    }

    func testTypedSchemaKeepsBackendAliasesIsolated() {
        let environment = [
            "DS4_GLM_MTLIO": "0",
            "DS4_LAGUNA_MTLIO": "1",
        ]
        XCTAssertFalse(DS4RuntimeEnvironment.flag(
            .metalIO, backend: .glm52, default: true,
            environment: environment))
        XCTAssertTrue(DS4RuntimeEnvironment.flag(
            .metalIO, backend: .laguna, default: false,
            environment: environment))
    }

    func testCanonicalSchemaNamesAreBackendAgnosticAndUnique() {
        let names = DS4RuntimeKnob.allCases.map(\.rawValue)
        XCTAssertEqual(Set(names).count, names.count)
        XCTAssertTrue(names.allSatisfy {
            $0.hasPrefix("DS4_")
                && !$0.hasPrefix("DS4_GLM_")
                && !$0.hasPrefix("DS4_LAGUNA_")
        })
        XCTAssertEqual(
            DS4RuntimeKnob.metalIO.definition.valueKind, .boolean)
        XCTAssertEqual(
            DS4RuntimeKnob.prefillChunk.definition.area, .prefill)
    }
}
