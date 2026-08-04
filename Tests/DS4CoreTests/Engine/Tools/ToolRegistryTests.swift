import XCTest
@testable import DS4Engine
import DS4Core

/// Pure-Swift checks of the built-in demo tools and the registry dispatch.
final class ToolRegistryTests: XCTestCase {

    private var allBuiltinsPolicy: ToolExecutionPolicy {
        ToolExecutionPolicy(allowedToolNames: Set(ToolRegistry.builtins.map(\.spec.name)))
    }

    func testCalculatorEvaluates() {
        let call = ToolCall(id: "c0", name: "calculator", argumentsJSON: #"{"expression":"2+3*4"}"#)
        let out = ToolRegistry.execute(call, policy: allBuiltinsPolicy)
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
        let out = ToolRegistry.execute(call, policy: allBuiltinsPolicy)
        XCTAssertTrue(out?.content.contains("error") ?? false)
    }

    func testClockReturnsDatetime() {
        let out = ToolRegistry.execute(ToolCall(id: "c2", name: "now", argumentsJSON: "{}"),
                                       policy: allBuiltinsPolicy)
        XCTAssertTrue(out?.content.contains("datetime") ?? false)
    }

    func testUnknownToolIsManual() {
        let policy = ToolExecutionPolicy(allowedToolNames: ["get_weather"])
        XCTAssertNil(ToolRegistry.execute(ToolCall(id: "c3", name: "get_weather", argumentsJSON: "{}"),
                                          policy: policy))
    }

    func testAddSubtractMultiply() {
        func run(_ name: String, _ args: String) -> String? {
            ToolRegistry.execute(ToolCall(id: "x", name: name, argumentsJSON: args),
                                 policy: allBuiltinsPolicy)?.content
        }
        XCTAssertTrue(run("add", #"{"a":2,"b":3}"#)?.contains("5") ?? false)
        XCTAssertTrue(run("subtract", #"{"a":10,"b":4}"#)?.contains("6") ?? false)
        XCTAssertTrue(run("multiply", #"{"a":6,"b":7}"#)?.contains("42") ?? false)
        // Negatives and decimals.
        XCTAssertTrue(run("subtract", #"{"a":3,"b":8}"#)?.contains("-5") ?? false)
        XCTAssertTrue(run("multiply", #"{"a":1.5,"b":2}"#)?.contains("3") ?? false)
    }

    func testBinaryToolAcceptsQuotedNumbers() {
        let out = ToolRegistry.execute(ToolCall(id: "q", name: "add", argumentsJSON: #"{"a":"2","b":"40"}"#),
                                       policy: allBuiltinsPolicy)
        XCTAssertTrue(out?.content.contains("42") ?? false)
    }

    func testBinaryToolRejectsMissingArgs() {
        let out = ToolRegistry.execute(ToolCall(id: "m", name: "add", argumentsJSON: #"{"a":2}"#),
                                       policy: allBuiltinsPolicy)
        XCTAssertTrue(out?.content.contains("error") ?? false)
    }

    func testNewToolsAreDeclared() {
        let names = Set(ToolRegistry.builtins.map(\.spec.name))
        XCTAssertTrue(names.isSuperset(of: ["add", "subtract", "multiply",
                                            "web_search", "web_fetch", "github_clone",
                                            "project_tree", "project_find", "project_inspect",
                                            "file_delete"]))
    }

    func testSpecsForEnabledSubset() {
        let specs = ToolRegistry.specs(enabled: ["calculator"])
        XCTAssertEqual(specs.map(\.name), ["calculator"])
    }

    /// Sub-agents may receive any built-in except the orchestration tools
    /// (no nested sub-agents) and github_clone (it replaces the shared active
    /// project), and the new tools are grantable.
    func testSubAgentGrantable() {
        let g = ToolRegistry.subAgentGrantable
        XCTAssertFalse(g.contains("subagent_run"))
        XCTAssertFalse(g.contains("subagent_search"))
        XCTAssertFalse(g.contains("agents_list"))
        XCTAssertFalse(g.contains("github_clone"))
        XCTAssertTrue(g.isSuperset(of: ["project_tree", "project_find", "project_inspect", "file_delete",
                                        "web_search", "web_fetch", "git"]))
    }

    func testSubAgentToolsCannotExceedParentDelegationScope() {
        let constrained = ToolRegistry.constrainSubAgentTools(
            ["project_read", "file_delete", "git", "subagent_run", "project_search"],
            allowedByParent: ["project_read", "project_search"])
        XCTAssertEqual(constrained, ["project_read", "project_search"])
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
        let out = ToolRegistry.execute(ToolCall(id: "a", name: "agents_list", argumentsJSON: "{}"),
                                       policy: allBuiltinsPolicy)?.content ?? ""
        XCTAssertTrue(out.contains("revisore"))
        XCTAssertTrue(out.contains("debug"))
        XCTAssertTrue(out.contains("role:"))
    }

    // MARK: Execution policy

    func testPolicyAllowsOnlyDeclaredBuiltin() {
        let policy = ToolExecutionPolicy(allowedToolNames: ["calculator"])
        let allowed = ToolRegistry.execute(
            ToolCall(id: "ok", name: "calculator", argumentsJSON: #"{"expression":"6*7"}"#),
            policy: policy
        )
        XCTAssertEqual(allowed?.content, #"{"result":42}"#)

        let denied = ToolRegistry.execute(
            ToolCall(id: "no", name: "file_delete", argumentsJSON: #"{"path":"important"}"#),
            policy: policy
        )
        XCTAssertEqual(denied?.callId, "no")
        XCTAssertEqual(denied?.name, "file_delete")
        XCTAssertTrue(denied?.content.contains(#""error":"tool_not_allowed""#) ?? false)
        XCTAssertTrue(denied?.content.contains(#""tool":"file_delete""#) ?? false)
    }

    func testEmptyPolicyDeniesByDefault() {
        let denied = ToolRegistry.execute(
            ToolCall(id: "deny", name: "calculator", argumentsJSON: #"{"expression":"1+1"}"#),
            policy: ToolExecutionPolicy(allowedToolNames: [])
        )
        XCTAssertTrue(denied?.content.contains("tool_not_allowed") ?? false)
        XCTAssertFalse(denied?.content.contains(#""result":2""#) ?? false)
    }

    func testDisallowedUnknownDoesNotFallThroughToManualEntry() {
        let denied = ToolRegistry.execute(
            ToolCall(id: "hallucinated", name: "unknown_mutation", argumentsJSON: "{}"),
            policy: ToolExecutionPolicy(allowedToolNames: [])
        )
        XCTAssertNotNil(denied)
        XCTAssertTrue(denied?.content.contains("tool_not_allowed") ?? false)
    }

    func testAutoPolicyRejectsMCPNameBeforeDispatch() async {
        let denied = await ToolRegistry.executeAuto(
            ToolCall(id: "mcp-denied", name: "mcp_files_delete", argumentsJSON: "{}"),
            policy: ToolExecutionPolicy(allowedToolNames: [])
        )
        XCTAssertEqual(denied?.callId, "mcp-denied")
        XCTAssertTrue(denied?.content.contains("tool_not_allowed") ?? false)
    }

    func testAllowedUnknownAutoToolStillFallsBackToManualEntry() async {
        let name = "manual_weather_for_policy_test"
        let output = await ToolRegistry.executeAuto(
            ToolCall(id: "manual", name: name, argumentsJSON: "{}"),
            policy: ToolExecutionPolicy(allowedToolNames: [name])
        )
        XCTAssertNil(output)
    }
}
