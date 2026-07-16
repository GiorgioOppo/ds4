import XCTest
@testable import DS4Engine

final class AgentProfileTests: XCTestCase {
    /// Every built-in role carries the same compact behavioral contract. This
    /// keeps language choice, multi-round work and prompt-injection boundaries
    /// consistent even for roles without tools.
    func testDefaultPromptsShareOperatingRules() {
        for agent in AgentProfile.defaults {
            let prompt = agent.systemPrompt
            XCTAssertTrue(prompt.contains("user's language"), agent.id)
            XCTAssertTrue(prompt.contains("untrusted data"), agent.id)
            XCTAssertTrue(prompt.contains("sequential tool/result rounds"), agent.id)
            XCTAssertTrue(prompt.contains("side effects only when required"), agent.id)
        }
    }

    func testDefaultPromptsStayCompactForPrefill() {
        for agent in AgentProfile.defaults {
            XCTAssertLessThanOrEqual(agent.systemPrompt.utf8.count, 1_800,
                                     "agent '\(agent.id)' prompt grew beyond its prefill budget")
        }
    }

    /// Reviewer is capability-level read-only. In particular the broad `git`
    /// tool cannot be exposed because it also supports add/commit/stash/tag.
    func testReviewerExposesOnlyReadOnlyTools() throws {
        let reviewer = try XCTUnwrap(AgentProfile.defaults.first { $0.id == "revisore" })
        let mutating: Set<String> = ["github_clone", "project_reload", "project_write", "project_edit",
                                     "file_write", "file_add", "file_modify", "file_delete", "git",
                                     "subagent_run"]
        XCTAssertTrue(Set(reviewer.toolNames).isDisjoint(with: mutating))
        XCTAssertTrue(reviewer.systemPrompt.contains("READ-ONLY"))
    }

    func testDebugDoesNotExposeBroadGitTool() throws {
        let debug = try XCTUnwrap(AgentProfile.defaults.first { $0.id == "debug" })
        XCTAssertFalse(debug.toolNames.contains("git"))
    }

    func testOrchestratorHasExplicitBoundedDelegationScope() throws {
        let orchestrator = try XCTUnwrap(AgentProfile.defaults.first { $0.id == "orchestratore" })
        let delegated = Set(try XCTUnwrap(orchestrator.delegatedToolNames))
        XCTAssertFalse(delegated.isEmpty)
        XCTAssertTrue(delegated.isSubset(of: ToolRegistry.subAgentGrantable))
        XCTAssertFalse(delegated.contains("git"))
        XCTAssertFalse(delegated.contains("file_delete"))
        XCTAssertFalse(delegated.contains("subagent_run"))
        XCTAssertFalse(delegated.contains("github_clone"))
    }

    func testLegacyProfileJSONDecodesWithDenyAllDelegation() throws {
        let legacy = #"{"id":"legacy","name":"Legacy","icon":"person","systemPrompt":"x","toolNames":["subagent_run"]}"#
        let profile = try JSONDecoder().decode(AgentProfile.self, from: Data(legacy.utf8))
        XCTAssertNil(profile.delegatedToolNames)
    }
}
