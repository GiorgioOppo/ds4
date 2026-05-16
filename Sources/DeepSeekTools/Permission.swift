import Foundation

/// A user's resolution of one permission request. Cached per-session
/// by the registry when the user picks `.alwaysAllow`; never persisted
/// to disk by this module — durable defaults live in the host's
/// `PermissionStore` (see DeepSeekUI).
public enum PermissionDecision: Sendable, Equatable {
    /// Run this single call.
    case allowOnce
    /// Run this call and remember the decision for the rest of the
    /// session for the same `(tool, category)` pair.
    case alwaysAllow
    /// Reject this call. The registry surfaces a `denied` error to
    /// the model, so it can try a different approach.
    case deny
}

/// Caller-supplied policy. The registry forwards every gated call
/// through `decide`. Hosts that want non-interactive behaviour
/// (CLI mode, tests) can return a fixed decision; the GUI host
/// renders a modal sheet.
public protocol PermissionDelegate: Sendable {
    func decide(request: PermissionRequest) async -> PermissionDecision
}

/// What the model wants to do, in human-readable form. The UI shows
/// this verbatim — keep `summary` short and concrete (one line, no
/// trailing punctuation), put any longer context in `detail`.
public struct PermissionRequest: Sendable {
    public let tool: String
    public let category: ToolCategory
    public let summary: String
    public let detail: String?
    public let mode: AgentMode

    public init(tool: String,
                category: ToolCategory,
                summary: String,
                detail: String? = nil,
                mode: AgentMode) {
        self.tool = tool
        self.category = category
        self.summary = summary
        self.detail = detail
        self.mode = mode
    }
}

/// Auto-allow policy useful for tests, headless servers, and the
/// CLI's "trust me" flag. Returns `.alwaysAllow` for everything
/// except `.dangerous`, which is denied unless `allowDangerous` is
/// set. Network category is allowed iff `allowNetwork`.
public struct AutoPermissionDelegate: PermissionDelegate {
    public let allowDangerous: Bool
    public let allowNetwork: Bool

    public init(allowDangerous: Bool = false, allowNetwork: Bool = true) {
        self.allowDangerous = allowDangerous
        self.allowNetwork = allowNetwork
    }

    public func decide(request: PermissionRequest) async -> PermissionDecision {
        switch request.category {
        case .readOnly, .planning:
            return .alwaysAllow
        case .mutating:
            return .alwaysAllow
        case .dangerous:
            return allowDangerous ? .alwaysAllow : .deny
        case .network:
            return allowNetwork ? .alwaysAllow : .deny
        }
    }
}

/// Permission delegate that denies every gated call. Used as the
/// "panic" default if the host hasn't wired in a real one.
public struct DenyAllPermissionDelegate: PermissionDelegate {
    public init() {}
    public func decide(request: PermissionRequest) async -> PermissionDecision {
        .deny
    }
}
