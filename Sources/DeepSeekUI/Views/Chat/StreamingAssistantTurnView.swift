import SwiftUI
import DeepSeekKit
#if canImport(AppKit)
import AppKit
#endif

/// Streaming variant of `AssistantTurnView`. Bound to a single
/// `StreamingRoundController` via `@ObservedObject` so per-token
/// content / reasoning / prefill-trace mutations only invalidate
/// THIS view — the rest of the transcript (other assistant turns,
/// the sidebar, the user bubble that opened this turn) stays
/// Equatable-stable and skips re-render.
///
/// `finalizedRounds` carries the assistant messages already
/// finalised inside the current turn (typically tool-call rounds
/// whose outputs landed before the model emitted the user-facing
/// reply). They render through the same tool / reasoning disclosure
/// chrome the value-driven `AssistantTurnView` uses, so the final
/// bubble looks identical between mid-stream and post-`.done`.
struct StreamingAssistantTurnView: View {
    @ObservedObject var controller: StreamingRoundController
    /// Assistant messages already finalised for this turn (every
    /// pass before the currently-streaming one). Typically tool-
    /// call rounds; usually 0–4 entries.
    let finalizedRounds: [StoredMessage]
    var agentResolver: ((String) -> AgentConfig?)? = nil

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "cpu")
                .foregroundStyle(.white)
                .frame(width: 28, height: 28)
                .background(
                    LinearGradient(
                        colors: [Color.purple, Color.indigo],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing),
                    in: Circle())
                .shadow(color: Color.purple.opacity(0.25),
                         radius: 3, x: 0, y: 1)
            VStack(alignment: .leading, spacing: 8) {
                header
                bubbleContent
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 4)
    }

    // MARK: - chrome

    @ViewBuilder
    private var header: some View {
        HStack(spacing: 8) {
            Text("Assistant")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            // Copy is hidden mid-stream — the content isn't final
            // yet and copying a half-baked reply would be more
            // confusing than helpful. AssistantTurnView puts the
            // button back as soon as `.done` finalises the round.
        }
    }

    @ViewBuilder
    private var bubbleContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let reasoning = combinedReasoning, !reasoning.isEmpty {
                ReasoningDisclosure(reasoning: reasoning)
            }
            ForEach(Array(intermediateContents.enumerated()), id: \.offset) { _, text in
                MarkdownText(raw: text)
            }
            if !allToolPairs.isEmpty {
                toolCallSection
            }
            // Live streaming text with the blinking caret. The
            // controller's `content` mutates per token; SwiftUI
            // re-renders just this Text + caret pair, leaving the
            // turn's other bubble chrome untouched.
            if controller.round.content.isEmpty {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text("Thinking…").foregroundStyle(.secondary)
                }
            } else {
                StreamingCaretText(content: controller.round.content)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(NSColor.controlBackgroundColor).opacity(0.55)))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(Color.secondary.opacity(0.18), lineWidth: 1))
    }

    @ViewBuilder
    private var toolCallSection: some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(allToolPairs.enumerated()), id: \.offset) { _, pair in
                    ToolCallDisclosure(
                        call: pair.call,
                        output: pair.output,
                        agentResolver: agentResolver)
                }
            }
            .padding(.top, 4)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "wrench.and.screwdriver")
                    .font(.caption)
                Text("\(allToolPairs.count) tool call\(allToolPairs.count == 1 ? "" : "s")")
                    .font(.caption)
            }
            .foregroundStyle(.secondary)
        }
        .padding(.top, 6)
    }

    // MARK: - derived state

    /// Reasoning text aggregated across the turn: every finalised
    /// round's `reasoningContent` concatenated, plus the live
    /// reasoning from the controller (remote chats emit it
    /// mid-stream; local chats only at `.done`).
    private var combinedReasoning: String? {
        var parts: [String] = []
        for r in finalizedRounds {
            if let rc = r.reasoningContent, !rc.isEmpty {
                parts.append(rc)
            }
        }
        if let live = controller.round.reasoningContent, !live.isEmpty {
            parts.append(live)
        }
        return parts.isEmpty ? nil : parts.joined(separator: "\n\n")
    }

    /// Non-empty `content` from finalised rounds (rare — usually
    /// the model writes prose only in the final round). Rendered
    /// above the streaming reply so chronological order is
    /// preserved.
    private var intermediateContents: [String] {
        finalizedRounds.compactMap {
            $0.content.isEmpty ? nil : $0.content
        }
    }

    /// `(call, output)` pairs flattened across every finalised
    /// round. The streaming round's own tool calls (if any) land
    /// at `.done` — they're not visible to this view yet.
    private var allToolPairs: [(call: StoredToolCall, output: String?)] {
        finalizedRounds.flatMap { m -> [(call: StoredToolCall, output: String?)] in
            let outputs = m.toolOutputs ?? []
            return m.toolCalls.enumerated().map { idx, call in
                (call, idx < outputs.count ? outputs[idx] : nil)
            }
        }
    }
}
