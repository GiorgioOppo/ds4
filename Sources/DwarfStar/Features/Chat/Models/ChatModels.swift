import Foundation
import DS4Engine
import DS4Core

/// A message as shown in the UI: reasoning and visible answer are kept apart so
/// the chain-of-thought can be collapsed. Assistant messages may carry tool
/// calls; tool results are shown as `.tool` messages.
struct UIMessage: Identifiable {
    let id = UUID()
    let role: ChatRole
    var reasoning: String = ""
    var text: String
    var toolStreamText: String = ""   // raw tool markup shown live while it generates
    var toolCalls: [ToolCall] = []
    /// Names of text files imported with this (user) message — shown as badges; the
    /// full content was folded into the turn actually sent to the model.
    var attachments: [String] = []
    /// Set on a `.tool` message that reports an isolated sub-agent run (question,
    /// answer, and a collapsible trace of its internal steps).
    var subAgent: InferenceService.SubAgentRun?
    /// True while that sub-agent is still executing: the card shows a spinner and
    /// the latest internal step live; flipped off when the final run replaces it.
    var subAgentRunning: Bool = false
}

/// A text file staged in the composer: its full content is folded into the next
/// user turn sent to the model; the transcript shows only the filename + size.
struct ChatAttachment: Identifiable, Equatable {
    let id = UUID()
    let name: String
    let content: String
    var bytes: Int { content.utf8.count }
}

