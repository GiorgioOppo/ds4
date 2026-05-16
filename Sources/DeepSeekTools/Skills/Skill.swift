import Foundation

/// A *skill* is a small reusable bundle of (system prompt addendum,
/// allowed-tool subset, suggested model defaults). Skills are looser
/// than agents — an agent has one identity; a skill is a hat the
/// agent puts on for a specific task.
///
/// Modelled after opencode's `tool/skill.ts` + `config/skills.ts`:
/// the model can invoke a skill the same way it invokes a tool,
/// and the host materialises the skill as "expand the system prompt
/// with this block, restrict tools to this subset" for the duration
/// of one tool round.
///
/// Persisted as JSON next to agents. The on-disk format is stable
/// (string-typed fields, no embedded executables), and the loader
/// is `MainActor` so it can be driven from SwiftUI without bridging.
public struct Skill: Codable, Sendable, Identifiable, Hashable {
    public let id: UUID
    public var name: String
    public var summary: String
    /// Body appended to the agent's system prompt while the skill is
    /// active. Plain text; the chat template wraps it.
    public var systemPromptAddendum: String
    /// Names of the tools (registry names, not MCP qualified names)
    /// the skill needs. `nil` = no restriction, `[]` = no tools.
    public var allowedToolNames: Set<String>?
    /// Suggested temperature override; `nil` keeps the agent's
    /// current temperature.
    public var temperature: Double?
    public var maxTokens: Int?
    public var createdAt: Date

    public init(id: UUID = UUID(),
                name: String,
                summary: String = "",
                systemPromptAddendum: String = "",
                allowedToolNames: Set<String>? = nil,
                temperature: Double? = nil,
                maxTokens: Int? = nil,
                createdAt: Date = .now) {
        self.id = id
        self.name = name
        self.summary = summary
        self.systemPromptAddendum = systemPromptAddendum
        self.allowedToolNames = allowedToolNames
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.createdAt = createdAt
    }
}

/// Built-in skill presets shipped with the app. Hosts can add custom
/// skills on top of these; the IDs are stable so the catalog can
/// evolve without breaking saved selections.
public enum BuiltInSkills {
    public static let all: [Skill] = [refactor, review, explain, testWriter]

    public static let refactor = Skill(
        id: UUID(uuidString: "00000000-0000-4000-8000-000000000001")!,
        name: "Refactor",
        summary: "Restructure code without changing behaviour. Keep diffs small and reviewable.",
        systemPromptAddendum:
            "You are refactoring existing code. Do not change observable behaviour. " +
            "Always read the file with 'read' before proposing changes. Prefer " +
            "'edit' over 'write'. Keep each change reviewable on its own.",
        allowedToolNames: ["read", "glob", "grep", "edit", "apply_patch"],
        temperature: 0.3
    )

    public static let review = Skill(
        id: UUID(uuidString: "00000000-0000-4000-8000-000000000002")!,
        name: "Review",
        summary: "Read-only audit. Surface bugs, security issues, design smells.",
        systemPromptAddendum:
            "You are reviewing code, not changing it. Only use read-only tools. " +
            "Be concrete: cite path:line. Flag issues by severity (critical / " +
            "important / nit). Don't speculate beyond what's in the codebase.",
        allowedToolNames: ["read", "glob", "grep", "repo_overview"],
        temperature: 0.2
    )

    public static let explain = Skill(
        id: UUID(uuidString: "00000000-0000-4000-8000-000000000003")!,
        name: "Explain",
        summary: "Walk through how a piece of code works.",
        systemPromptAddendum:
            "You are explaining code to a developer who is new to this codebase. " +
            "Read with 'read' first, then describe the data flow and key decisions. " +
            "Reference path:line. Avoid speculation.",
        allowedToolNames: ["read", "glob", "grep", "repo_overview"],
        temperature: 0.3
    )

    public static let testWriter = Skill(
        id: UUID(uuidString: "00000000-0000-4000-8000-000000000004")!,
        name: "Test writer",
        summary: "Write or extend the test suite for the target code.",
        systemPromptAddendum:
            "You are writing tests. First locate the existing test file with " +
            "'glob' or 'grep'. Match the project's test style. Keep tests fast, " +
            "deterministic, isolated. Run the suite via 'shell' if available.",
        allowedToolNames: ["read", "glob", "grep", "edit", "write", "shell"],
        temperature: 0.4
    )
}
