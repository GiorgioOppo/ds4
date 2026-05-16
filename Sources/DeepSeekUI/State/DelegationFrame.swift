import Foundation

/// One in-flight sub-agent invocation, surfaced to the UI so the
/// user can watch the delegation chain unfold instead of staring
/// at a "running tools…" spinner for what may be tens of seconds.
///
/// `buffer` grows as the sub-agent's tokens arrive (every
/// `.token(text:, id:)` event from the inference stream appends
/// here on the main actor). The frame is removed from the stack
/// when its `runSubAgentToCompletionInner` returns, regardless
/// of how (success, error, depth cap).
///
/// Identity-by-id (random UUID at construction time) so the
/// SwiftUI ForEach over `activeDelegations[convID]` stays stable
/// even when two consecutive delegations happen to target the
/// same agent.
struct DelegationFrame: Identifiable, Hashable {
    let id: UUID
    /// Target agent's identifying metadata, cached at push time
    /// so the UI never has to look it up against an AgentLibrary
    /// that might have been mutated mid-delegation.
    let agentID: UUID
    let agentName: String
    let agentIconName: String
    /// Raw tint identifier — matches `AgentTint.color(for:)` keys.
    let agentTint: String
    /// The `task` argument as the host agent wrote it. Plain
    /// prose, suitable for direct display.
    let task: String
    /// Streaming reply buffer. Empty at push, updated on every
    /// token sample, finalised when the inner loop captures
    /// `.done`. The UI clamps line count when rendering so a
    /// chatty sub-agent doesn't push everything else off-screen.
    var buffer: String
    /// 1-indexed nesting level. 1 = direct delegation from the
    /// host; 2 = sub-agent delegated to a sub-sub-agent; etc.
    /// Used by the UI to indent the chain.
    let depth: Int
    let createdAt: Date

    init(agentID: UUID,
         agentName: String,
         agentIconName: String,
         agentTint: String,
         task: String,
         depth: Int) {
        self.id = UUID()
        self.agentID = agentID
        self.agentName = agentName
        self.agentIconName = agentIconName
        self.agentTint = agentTint
        self.task = task
        self.buffer = ""
        self.depth = depth
        self.createdAt = .now
    }
}
