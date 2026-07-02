import Foundation

/// An agent = a ROLE the model plays in a conversation: a system prompt, the
/// tools it may call, and — crucially for this engine — its OWN expert-usage
/// profile. Different roles route to different experts, so each agent's profile
/// pre-warms the expert slot-cache with the experts THAT role actually uses.
/// Switching agent starts a fresh conversation with its role.
public struct AgentProfile: Sendable, Identifiable, Codable, Equatable {
    public var id: String          // stable key (also the usage-profile file key)
    public var name: String
    public var icon: String        // SF Symbol
    public var systemPrompt: String
    public var toolNames: [String] // built-in tools this agent exposes ([] = none)

    public init(id: String, name: String, icon: String, systemPrompt: String, toolNames: [String]) {
        self.id = id; self.name = name; self.icon = icon
        self.systemPrompt = systemPrompt; self.toolNames = toolNames
    }

    public static let defaults: [AgentProfile] = [
        .init(id: "generale", name: "General", icon: "person",
              systemPrompt: "", toolNames: []),
        .init(id: "coding", name: "Coding", icon: "chevron.left.forwardslash.chevron.right",
              systemPrompt: "You are an expert programming assistant. Answer with correct, concise code; explain only what matters. If a project has been imported, explore it with project_list / project_search and read only relevant files with project_read before answering.",
              toolNames: ["project_list", "project_read", "project_search"]),
        .init(id: "code", name: "Code", icon: "terminal",
              systemPrompt: """
              You are an autonomous coding agent working on the imported project. For every request, follow this method, one tool call at a time:
              1) EXPLORE: identify relevant files with project_list and project_search.
              2) READ: read the parts you need with project_read before touching anything. Never invent file contents you have not read.
              3) EDIT: make small, targeted changes with project_edit. The 'find' text must match exactly, including indentation, and be unique in the file; include neighboring lines to disambiguate. Use project_write only for new files or complete rewrites.
              4) VERIFY: reread the changed area with project_read, check consistency (imports, callers found with project_search), and inspect changes with git (for example "diff --stat", "diff <file>").
              5) If the repo uses git and the user asks, commit with git "commit -am <concise message>".
              At the end, summarize in 2-3 sentences what you changed and where (file:line). If the task is ambiguous or risky, stop and ask.
              """,
              toolNames: ["project_list", "project_read", "project_search", "project_edit",
                          "file_read", "file_lines", "file_add", "file_modify", "git"]),
        .init(id: "orchestratore", name: "Orchestrator", icon: "person.3.sequence",
              systemPrompt: """
              You are an orchestrator: break down the task and delegate to isolated sub-agents without reading or editing files yourself.
              RULE: your first tool call, always and before any other action or answer, is agents_list, so you know which agents you can orchestrate and which tools they have.
              Then:
              1) Identify relevant files with project_list and subagent_search.
              2) Delegate one subtask at a time with subagent_run: target = file (or "project") and a self-contained question. The sub-agent does not see this chat, so include all needed context. Choose 'agent' from the agents listed by agents_list when suitable, or pass a minimal 'tools' set (read-only by default; edit/write only when modification is required).
              3) Integrate the answers and conclude concisely (what was done, file:line). Your context includes only questions and answers, not sub-agent internal work.
              If the task is ambiguous or risky, stop and ask before delegating changes.
              """,
              toolNames: ["agents_list", "subagent_search", "subagent_run", "project_list", "project_search"]),
        .init(id: "ricerca", name: "Research", icon: "globe",
              systemPrompt: """
              You are a research assistant with web access. Method, one tool call at a time:
              1) SEARCH: turn the question into focused queries and call web_search (refine the query if the results are off-topic).
              2) READ: open the most relevant results with web_fetch — never answer from snippets alone; read the actual pages.
              3) CROSS-CHECK: prefer claims confirmed by more than one independent source; note disagreements.
              4) ANSWER: be concise and CITE the sources you used (title + URL). If the sources are thin or contradictory, say so rather than guessing. Use 'now' when the question is time-sensitive.
              """,
              toolNames: ["web_search", "web_fetch", "now"]),
        .init(id: "matematica", name: "Math", icon: "function",
              systemPrompt: "You are a precise math assistant. Use the provided calculation tools for every arithmetic operation.",
              toolNames: ["calculator", "add", "subtract", "multiply"]),
        .init(id: "scrittura", name: "Writing", icon: "pencil",
              systemPrompt: "You are an editor and writer in English: natural tone, clear sentences, no fluff.",
              toolNames: []),
        .init(id: "latex", name: "LaTeX", icon: "doc.richtext",
              systemPrompt: """
              You are a LaTeX expert: produce correct, compilable .tex documents. Include a minimal but adequate preamble (\\documentclass plus only needed packages), clear structure, and correct math mode usage. Escape special text characters (# $ % & _ { } ~ ^ \\). Verify balanced environments (matched \\begin/\\end, closed math delimiters).
              If a project is imported, save documents with file_write (for example doc.tex) and make small edits with project_edit; read references and existing files with project_read/project_search. Without a project, return the complete LaTeX in your answer.
              """,
              toolNames: ["project_read", "project_search",
                          "file_read", "file_lines", "file_write", "file_add", "file_modify"]),
        .init(id: "documentatore", name: "Documentation", icon: "book.closed",
              systemPrompt: """
              You document a project in Markdown. Proceed in 4 steps:
              1) EXISTING DOCS: search for and read existing documentation (README, README.md, docs/, *.md) with project_search/project_list and project_read/file_read. If none exists, note that.
              2) CODE: explore the structure (project_list) and read relevant files (project_search, project_read) to understand modules, components, and public APIs.
              3) GAP ANALYSIS: compare documentation and code, then list what is missing or obsolete (undocumented files/modules/APIs, sections to update). Report gaps before writing.
              4) DOCUMENT: write/update .md files (file_write for new files, file_add/file_modify for existing sections). Structure: overview, architecture, file map, main components with responsibilities and usage examples.
              Document only what you read in the code. Do not invent. Conclude with the list of documentation files created/updated.
              """,
              toolNames: ["project_list", "project_read", "project_search",
                          "file_read", "file_lines", "file_write", "file_add", "file_modify"]),
    ]
}

/// Process-wide registry of the CURRENT agents, so engine-side tools (the
/// orchestrator's `agents_list`) can see the actual roster — including the user's
/// edits — which otherwise lives only in the app's UserDefaults. The app keeps it
/// in sync; until it does, it reports the built-in defaults.
public final class AgentRegistry: @unchecked Sendable {
    public static let shared = AgentRegistry()
    private let lock = NSLock()
    private var agents: [AgentProfile] = AgentProfile.defaults

    /// Replace the roster (the app calls this whenever its agent list changes).
    public func set(_ agents: [AgentProfile]) {
        lock.lock(); self.agents = agents.isEmpty ? AgentProfile.defaults : agents; lock.unlock()
    }

    public func all() -> [AgentProfile] { lock.lock(); defer { lock.unlock() }; return agents }

    /// Compact listing for the `agents_list` tool: one "id · name · tools" per line.
    public func describe() -> String {
        let list = all()
        guard !list.isEmpty else { return "No agents available." }
        var out = "Available agents (id · name · tools):"
        for a in list {
            let tools = a.toolNames.isEmpty ? "(no tools)" : a.toolNames.joined(separator: ", ")
            out += "\n- \(a.id) · \(a.name) · \(tools)"
        }
        return out
    }
}
