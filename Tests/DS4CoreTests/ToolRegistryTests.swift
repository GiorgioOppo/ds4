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

    func testAddSubtractMultiply() {
        func run(_ name: String, _ args: String) -> String? {
            ToolRegistry.execute(ToolCall(id: "x", name: name, argumentsJSON: args))?.content
        }
        XCTAssertTrue(run("add", #"{"a":2,"b":3}"#)?.contains("5") ?? false)
        XCTAssertTrue(run("subtract", #"{"a":10,"b":4}"#)?.contains("6") ?? false)
        XCTAssertTrue(run("multiply", #"{"a":6,"b":7}"#)?.contains("42") ?? false)
        // Negatives and decimals.
        XCTAssertTrue(run("subtract", #"{"a":3,"b":8}"#)?.contains("-5") ?? false)
        XCTAssertTrue(run("multiply", #"{"a":1.5,"b":2}"#)?.contains("3") ?? false)
    }

    func testBinaryToolAcceptsQuotedNumbers() {
        let out = ToolRegistry.execute(ToolCall(id: "q", name: "add", argumentsJSON: #"{"a":"2","b":"40"}"#))
        XCTAssertTrue(out?.content.contains("42") ?? false)
    }

    func testBinaryToolRejectsMissingArgs() {
        let out = ToolRegistry.execute(ToolCall(id: "m", name: "add", argumentsJSON: #"{"a":2}"#))
        XCTAssertTrue(out?.content.contains("error") ?? false)
    }

    func testNewToolsAreDeclared() {
        let names = Set(ToolRegistry.builtins.map(\.spec.name))
        XCTAssertTrue(names.isSuperset(of: ["add", "subtract", "multiply"]))
    }

    func testSpecsForEnabledSubset() {
        let specs = ToolRegistry.specs(enabled: ["calculator"])
        XCTAssertEqual(specs.map(\.name), ["calculator"])
    }
}
