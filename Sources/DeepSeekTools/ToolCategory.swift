import Foundation

/// Side-effect classification of a tool. Drives both the plan-mode
/// filter (only `.readOnly` survives in plan mode) and the default
/// permission policy (everything else needs explicit consent the
/// first time it's invoked in a session).
public enum ToolCategory: String, Codable, Sendable {
    /// Pure observation. Reads from disk, network, or memory without
    /// causing any user-visible change. Allowed by default in all modes.
    case readOnly

    /// Writes a file, mutates filesystem state in a way that's
    /// recoverable (e.g. atomic write, edit with diff visible to UI).
    case mutating

    /// Runs arbitrary code (`shell`, `bash`, `apply_patch` with
    /// post-hooks). Always requires confirmation in `.plan` mode;
    /// in `.build` mode, governed by the session permission policy.
    case dangerous

    /// Outbound network call without local FS side effects (`webfetch`,
    /// `websearch`). Allowed by default but counted separately so a
    /// network-off agent profile can deny en masse.
    case network

    /// Tool that only manipulates in-process planning state
    /// (`plan`, `task`, `todo`). No external effects. Always allowed.
    case planning
}
