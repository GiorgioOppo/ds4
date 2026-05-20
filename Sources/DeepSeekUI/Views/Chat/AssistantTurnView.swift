import SwiftUI
import DeepSeekKit
#if canImport(AppKit)
import AppKit
#endif

/// One assistant *turn*, possibly spanning multiple roundtrips.
///
/// The model emits one assistant `Message` per generate pass. When it
/// chains tool calls (`generate → tool calls → tool outputs → generate
/// → …`), each pass appends a fresh `Message` to the conversation. The
/// chat used to render each of these as its own bubble, which made a
/// 5-roundtrip answer look like 5 separate replies even though they
/// were chronologically one response. This view collapses the whole
/// run into a single bubble:
///
///   * `combinedReasoning` — concatenated `reasoningContent` of every
///     round, plus the live `streamingReasoning` buffer when the last
///     message is the active streaming target.
///   * `allToolPairs` — every `(call, output)` from every round,
///     surfaced under one wrench-icon disclosure. Each call inside is
///     a `ToolCallDisclosure` so the user can scan the batch and only
///     open the one they care about.
///   * Intermediate non-empty `content` (rare — usually the model
///     writes its prose only in the final round) is shown in
///     chronological order above the final reply.
///   * Final reply uses the streaming caret while in flight, then
///     promotes to `MarkdownText` once `.done` lands.
struct AssistantTurnView: View, Equatable {
    /// Non-empty list of assistant messages making up this turn, in
    /// chronological order. All elements MUST have `role == .assistant`
    /// — the grouping in ChatView guarantees that.
    let messages: [StoredMessage]
    /// True when the LAST message in `messages` is the currently
    /// streaming target. Drives the caret + "Thinking…" placeholder.
    var isStreamingFinal: Bool = false
    /// Live reasoning buffer for the streaming message, when present.
    /// Overrides the persisted `reasoningContent` of the last message
    /// during streaming.
    var streamingReasoning: String? = nil
    var agentResolver: ((String) -> AgentConfig?)? = nil

    @State private var copied: Bool = false

    /// `Equatable` so `ChatView` can `.equatable()`-wrap this and
    /// skip the body re-evaluation when SwiftUI's parent body
    /// re-runs (per-token `phases` mutate, etc.) but the actual
    /// inputs haven't changed. `agentResolver` is intentionally
    /// excluded from the comparison — it's a closure (not
    /// Equatable) and in practice it's a stable `store.agents`
    /// lookup that doesn't vary turn to turn within one session.
    static func == (lhs: AssistantTurnView,
                     rhs: AssistantTurnView) -> Bool {
        lhs.messages == rhs.messages
            && lhs.isStreamingFinal == rhs.isStreamingFinal
            && lhs.streamingReasoning == rhs.streamingReasoning
    }

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

    /// Card-style container around the model's response. A subtle fill
    /// + hairline border sets the bubble apart from the surrounding
    /// transcript without competing with the user bubble's accent
    /// tint. Reasoning / tool-call disclosures sit inside the card so
    /// the whole turn reads as a single unit.
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
            finalReply
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

    // MARK: - header (Assistant label + Copy action)

    @ViewBuilder
    private var header: some View {
        HStack(spacing: 8) {
            Text("Assistant")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            if !isStreamingFinal, !finalContent.isEmpty {
                Button(action: copyContent) {
                    Label(copied ? "Copied" : "Copy",
                           systemImage: copied ? "checkmark" : "doc.on.doc")
                        .labelStyle(.iconOnly)
                        .font(.caption)
                }
                .buttonStyle(.borderless)
                .help(copied ? "Copied" : "Copy reply")
            }
        }
    }

    // MARK: - tool calls (aggregated across rounds)

    /// Outer wrench-icon disclosure with one ToolCallDisclosure per
    /// call across the entire turn. Collapsed by default; opening it
    /// reveals the per-call rows, each of which is itself collapsible.
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

    // MARK: - final reply (streaming-aware)

    @ViewBuilder
    private var finalReply: some View {
        let last = messages.last!
        if last.content.isEmpty && isStreamingFinal {
            HStack(spacing: 6) {
                ProgressView().controlSize(.mini)
                Text("Thinking…").foregroundStyle(.secondary)
            }
        } else if isStreamingFinal {
            StreamingCaretText(content: last.content)
        } else if !last.content.isEmpty {
            MarkdownText(raw: last.content)
        }
    }

    // MARK: - derived state

    private var finalContent: String {
        messages.last?.content ?? ""
    }

    /// Reasoning blocks from every round, concatenated with a blank-
    /// line gap. The last round prefers `streamingReasoning` over the
    /// stored value so the live buffer is what's visible while
    /// generation is in flight.
    private var combinedReasoning: String? {
        let lastIdx = messages.count - 1
        let parts = messages.enumerated().compactMap { i, m -> String? in
            let isLast = i == lastIdx
            let live = isLast ? streamingReasoning : nil
            let text = (live ?? m.reasoningContent ?? "")
            return text.isEmpty ? nil : text
        }
        if parts.isEmpty { return nil }
        return parts.joined(separator: "\n\n")
    }

    /// Non-empty content from intermediate rounds. The final round's
    /// content is rendered separately by `finalReply` so the streaming
    /// caret + Markdown promotion can apply.
    private var intermediateContents: [String] {
        messages.dropLast().compactMap {
            $0.content.isEmpty ? nil : $0.content
        }
    }

    /// Every `(call, output)` pair from every round, flattened in
    /// chronological order. Calls without an output yet (because the
    /// next roundtrip hasn't finished) get `nil` so the row can show
    /// its 'running…' indicator.
    private var allToolPairs: [(call: StoredToolCall, output: String?)] {
        messages.flatMap { m -> [(call: StoredToolCall, output: String?)] in
            let outputs = m.toolOutputs ?? []
            return m.toolCalls.enumerated().map { idx, call in
                (call, idx < outputs.count ? outputs[idx] : nil)
            }
        }
    }

    // MARK: - copy

    private func copyContent() {
        #if canImport(AppKit)
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(finalContent, forType: .string)
        #endif
        copied = true
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            copied = false
        }
    }
}

/// One collapsible row inside the wrench-icon disclosure. Collapsed
/// shows the tool name + a single-line preview of the call args so the
/// user can scan a batch of calls (`ls {path: "src"}` → `grep
/// {pattern: "foo"}` → …) without opening every one. Expanding reveals
/// the pretty-printed args JSON plus the tool's output. Owns its own
/// `@State` for `isExpanded`, so each row keeps independent open/close
/// state across re-renders even when its sibling rows toggle.
struct ToolCallDisclosure: View {
    let call: StoredToolCall
    let output: String?
    let agentResolver: ((String) -> AgentConfig?)?

    @State private var isExpanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $isExpanded) {
            expandedContent.padding(.top, 4)
        } label: {
            headerLabel
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private var headerLabel: some View {
        HStack(spacing: 6) {
            if let agent = delegation?.agent {
                Image(systemName: agent.iconName)
                    .font(.caption)
                    .foregroundStyle(AgentTint.color(for: agent.tint))
                (Text("Delegated to ")
                    .foregroundStyle(.secondary)
                 + Text(agent.name).bold())
                    .font(.callout)
                    .lineLimit(1)
            } else if delegation != nil {
                Image(systemName: "person.crop.circle.badge.questionmark")
                    .font(.caption)
                    .foregroundStyle(Color.orange)
                Text("Delegation to unknown agent")
                    .font(.callout)
                    .foregroundStyle(.orange)
                    .lineLimit(1)
            } else {
                Image(systemName: "arrow.up.right.square")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                Text(call.name)
                    .font(.callout.monospaced())
                    .lineLimit(1)
            }
            let preview = argsPreview
            if !preview.isEmpty {
                Text(preview)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .layoutPriority(-1)
            }
            Spacer(minLength: 4)
            if output == nil {
                HStack(spacing: 4) {
                    ProgressView().controlSize(.mini)
                    Text("running…")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }

    @ViewBuilder
    private var expandedContent: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let agent = delegation?.agent, !agent.summary.isEmpty {
                Text(agent.summary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let task = delegation?.task, !task.isEmpty {
                Text(task)
                    .font(.callout)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(NSColor.textBackgroundColor),
                                 in: RoundedRectangle(cornerRadius: 4))
            } else {
                let pretty = Self.prettyJSON(call.args)
                if !pretty.isEmpty {
                    Text(pretty)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(NSColor.textBackgroundColor),
                                     in: RoundedRectangle(cornerRadius: 4))
                }
            }
            if let out = output, !out.isEmpty {
                HStack(spacing: 6) {
                    Image(systemName: "arrow.down.left.square")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                    Text("output")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                Text(out)
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(NSColor.textBackgroundColor),
                                 in: RoundedRectangle(cornerRadius: 4))
            }
        }
    }

    private var delegation: (agent: AgentConfig?, task: String)? {
        guard call.name == EncodingDSV4.delegateToolName else { return nil }
        let data = call.args.data(using: .utf8) ?? Data()
        let obj = (try? JSONSerialization.jsonObject(with: data))
            as? [String: Any] ?? [:]
        let agentName = (obj["agent_name"] as? String) ?? ""
        let task = (obj["task"] as? String) ?? ""
        let agent = agentName.isEmpty ? nil : agentResolver?(agentName)
        return (agent, task)
    }

    private var argsPreview: String {
        if let task = delegation?.task, !task.isEmpty {
            return task.replacingOccurrences(of: "\n", with: " ")
        }
        let raw = call.args.trimmingCharacters(in: .whitespacesAndNewlines)
        if raw.isEmpty { return "" }
        guard let data = raw.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data),
              let out = try? JSONSerialization.data(
                withJSONObject: obj,
                options: [.sortedKeys]),
              let str = String(data: out, encoding: .utf8) else {
            return raw.replacingOccurrences(of: "\n", with: " ")
        }
        return str
    }

    static func prettyJSON(_ raw: String) -> String {
        guard !raw.isEmpty,
              let data = raw.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data),
              let out = try? JSONSerialization.data(
                withJSONObject: obj,
                options: [.prettyPrinted, .sortedKeys]),
              let str = String(data: out, encoding: .utf8)
        else { return raw }
        return str
    }
}

/// Streaming assistant text with a blinking caret pinned to the end.
/// Content is split at the LAST paragraph break (`\n\n`): the stable
/// prefix gets the full `MarkdownText` treatment so headings, lists,
/// code blocks and inline formatting render as they stabilize, while
/// the trailing in-progress block stays plain text + caret. Keeping
/// the partial block as plain text avoids the distracting flicker
/// where an unclosed `**` or a half-typed list marker would otherwise
/// briefly render as bold / bullet before the next token corrects it.
/// The caret toggles between full opacity and clear (instead of being
/// removed/added) so the trailing layout position never jumps mid-line,
/// and the surrounding paragraph reflows naturally as tokens arrive.
struct StreamingCaretText: View {
    let content: String

    var body: some View {
        let split = Self.splitStable(content)
        VStack(alignment: .leading, spacing: 10) {
            if !split.stable.isEmpty {
                MarkdownText(raw: split.stable)
            }
            TimelineView(.periodic(from: .now, by: 0.5)) { ctx in
                let on = Int(ctx.date.timeIntervalSince1970 * 2)
                    .isMultiple(of: 2)
                (Text(split.partial)
                 + Text("▌").foregroundColor(on ? .primary : .clear))
                    .font(.callout)
                    .lineSpacing(2)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    /// Find the last block boundary (`\n\n`). Everything up to and
    /// including it is "stable" and can be parsed as markdown without
    /// flicker; everything after is the in-progress block.
    private static func splitStable(
        _ content: String
    ) -> (stable: String, partial: String) {
        guard let range = content.range(of: "\n\n", options: .backwards)
        else { return ("", content) }
        let stable = String(content[..<range.upperBound])
        let partial = String(content[range.upperBound...])
        return (stable, partial)
    }
}
