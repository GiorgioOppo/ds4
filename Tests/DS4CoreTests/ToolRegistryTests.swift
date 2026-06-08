import XCTest
@testable import DS4Engine
import DS4Core

/// Pure-Swift checks of the built-in demo tools and the registry dispatch.
final class ToolRegistryTests: XCTestCase {

    func testCalculatorEvaluates() {
        let call = ToolCall(id: "c0", name: "calculator", argumentsJSON: #"{"expression":"2+3*4"}"#)
        let out = ToolRegistry.execute(call)
        XCTAssertEqual(out?.name, "calculator")
        XCTAssertEqual(out?.callId, "c0")
        XCTAssertTrue(out?.content.contains("14") ?? false, "got \(out?.content ?? "nil")")
    }

    func testCalculatorRejectsNonArithmetic() {
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("system('rm')").contains("error"))
    }

    func testCalculatorParenthesesAndUnary() {
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("-(2+3)*2").contains("-10"))
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("(1+2)*(3+4)").contains("21"))
    }

    func testCalculatorMalformedDoesNotCrash() {
        // Previously NSExpression would throw an uncatchable ObjC exception here.
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("2+").contains("error"))
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("()").contains("error"))
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("5/0").contains("error"))
    }

    func testCalculatorMissingArgument() {
        let call = ToolCall(id: "c1", name: "calculator", argumentsJSON: "{}")
        let out = ToolRegistry.execute(call)
        XCTAssertTrue(out?.content.contains("error") ?? false)
    }

    func testClockReturnsDatetime() {
        let out = ToolRegistry.execute(ToolCall(id: "c2", name: "now", argumentsJSON: "{}"))
        XCTAssertTrue(out?.content.contains("datetime") ?? false)
    }

    func testUnknownToolIsManual() {
        XCTAssertNil(ToolRegistry.execute(ToolCall(id: "c3", name: "get_weather", argumentsJSON: "{}")))
    }

    func testSpecsForEnabledSubset() {
        let specs = ToolRegistry.specs(enabled: ["calculator"])
        XCTAssertEqual(specs.map(\.name), ["calculator"])
    }
}
