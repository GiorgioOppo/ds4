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
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("5 % 0").contains("error"))
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("unknown(2)").contains("error"))
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("sqrt(2").contains("error"))
    }

    func testCalculatorPowerAndModulo() {
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("2^10").contains("1024"))
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("2^3^2").contains("512"))   // right-assoc
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("2^-1").contains("0.5"))
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("-2^2").contains("-4"))     // -(2^2)
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("10 % 3").contains(#"result":1"#))
    }

    func testCalculatorFunctionsAndConstants() {
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("sqrt(144)").contains("12"))
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("abs(-5) + floor(1.9)").contains(#"result":6"#))
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("cos(0)").contains(#"result":1"#))
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("pi").contains("3.14159"))
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("ln(1)").contains(#"result":0"#))
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("round(log(1000))").contains(#"result":3"#))
        // Non-finite results are reported as errors, not garbage numbers.
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("sqrt(-1)").contains("error"))
        XCTAssertTrue(ToolRegistry.evaluateArithmetic("ln(0)").contains("error"))
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
        XCTAssertTrue(names.isSuperset(of: ["add", "subtract", "multiply",
                                            "web_search", "web_fetch",
                                            "project_tree", "project_find", "file_delete"]))
    }

    func testSpecsForEnabledSubset() {
        let specs = ToolRegistry.specs(enabled: ["calculator"])
        XCTAssertEqual(specs.map(\.name), ["calculator"])
    }

    /// Sub-agents may receive any built-in except the orchestration tools
    /// (no nested sub-agents), and the new tools are grantable.
    func testSubAgentGrantable() {
        let g = ToolRegistry.subAgentGrantable
        XCTAssertFalse(g.contains("subagent_run"))
        XCTAssertFalse(g.contains("subagent_search"))
        XCTAssertFalse(g.contains("agents_list"))
        XCTAssertTrue(g.isSuperset(of: ["project_tree", "project_find", "file_delete",
                                        "web_search", "web_fetch", "git"]))
    }

    /// Every tool a default agent declares must exist in the registry —
    /// catches typos when agent rosters and built-ins evolve separately.
    func testDefaultAgentToolsExist() {
        let names = Set(ToolRegistry.builtins.map(\.spec.name))
        for agent in AgentProfile.defaults {
            for tool in agent.toolNames {
                XCTAssertTrue(names.contains(tool),
                              "agent '\(agent.id)' references unknown tool '\(tool)'")
            }
        }
    }

    /// agents_list output carries the role hint the orchestrator picks by.
    func testAgentsListDescribesRoles() {
        let out = ToolRegistry.execute(ToolCall(id: "a", name: "agents_list", argumentsJSON: "{}"))?.content ?? ""
        XCTAssertTrue(out.contains("revisore"))
        XCTAssertTrue(out.contains("debug"))
        XCTAssertTrue(out.contains("role:"))
    }
}
