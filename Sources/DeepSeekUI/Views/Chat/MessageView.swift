import SwiftUI
import DeepSeekKit
#if canImport(AppKit)
import AppKit
#endif

/// One message bubble. User turns get a tinted background pinned to the
/// trailing edge; assistant turns stay full-width with a Copy action.
/// Final-pass assistant content is rendered through `MarkdownText`,
/// which promotes fenced code blocks into distinct artifact cards. The
/// streaming placeholder shows a "Thinking…" indicator until the first
/// token arrives, then a plain text + blinking caret while the buffer
/// grows.
struct MessageView: View {
    let message: StoredMessage
    /// True while this message is the in-progress streaming target so
    /// we can show a blinking caret.
    var isStreaming: Bool = false
    /// Live reasoning buffer fed by the streaming runner (TODO §4
    /// follow-up). When non-nil it overrides
    /// `message.reasoningContent` in the disclosure widget so the
    /// running thinking text is visible mid-turn instead of only at
    /// `.done`. Nil for non-streaming / no-reasoning messages.
    var streamingReasoning: String? = nil
    /// Resolve a `__delegate_to_agent` call's `agent_name` to the
    /// AgentConfig it targets so the tool row can render the
    /// delegate's icon + tint instead of a generic wrench. nil
    /// (the default) makes delegation rows fall back to the
    /// generic rendering, which is fine when MessageView is used
    /// in contexts that don't have an AgentLibrary (previews,
    /// tests, …).
    var agentResolver: ((String) -> AgentConfig?)? = nil

    @State private var copied: Bool = false

    var body: some View {
        switch message.role {
        case .user:        userBubble
        case .assistant:   assistantBubble
        case .system:      systemBubble
        }
    }

    // -------- user --------

    private var userBubble: some View {
        HStack(alignment: .top, spacing: 12) {
            Spacer(minLength: 40)
            VStack(alignment: .trailing, spacing: 4) {
                Text("You")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(message.content)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.accentColor.opacity(0.15),
                                 in: RoundedRectangle(cornerRadius: 10))
            }
            avatar(symbol: "person.fill", tint: .blue)
        }
        .padding(.vertical, 4)
    }

    // -------- assistant --------

    private var assistantBubble: some View {
        HStack(alignment: .top, spacing: 12) {
            avatar(symbol: "cpu", tint: .purple)
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text("Assistant")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    if !isStreaming && !message.content.isEmpty {
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
                // Prefer the live streaming buffer when present so
                // the reasoning shows up as it streams; fall back to
                // the persisted value (which finalizeRemoteIteration
                // writes at .done).
                if let r = streamingReasoning ?? message.reasoningContent,
                   !r.isEmpty
                {
                    ReasoningDisclosure(reasoning: r)
                }
                assistantContent
                if !message.toolCalls.isEmpty {
                    toolCallSection
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 4)
    }

    /// Wrench-icon disclosure containing one ToolCallDisclosure per
    /// (call, output) pair. Rendered below the assistant prose because
    /// the calls chronologically fire *after* the model writes its text
    /// / reasoning. Each inner disclosure is collapsed by default —
    /// expanding it reveals the full args JSON plus the tool's output.
    private var toolCallSection: some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(message.toolCalls.enumerated()), id: \.offset) { idx, call in
                    ToolCallDisclosure(
                        call: call,
                        output: message.toolOutputs?
                            .indices.contains(idx) == true
                            ? message.toolOutputs![idx]
                            : nil,
                        agentResolver: agentResolver)
                }
            }
            .padding(.top, 4)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "wrench.and.screwdriver")
                    .font(.caption)
                Text("\(message.toolCalls.count) tool call\(message.toolCalls.count == 1 ? "" : "s")")
                    .font(.caption)
            }
            .foregroundStyle(.secondary)
        }
        .padding(.top, 6)
    }

    @ViewBuilder
    private var assistantContent: some View {
        if message.content.isEmpty && isStreaming {
            HStack(spacing: 6) {
                ProgressView().controlSize(.mini)
                Text("Thinking…").foregroundStyle(.secondary)
            }
        } else if isStreaming {
            // Partial markdown parses ugly (open `**` / unclosed code
            // fence). Stay on plain Text + blinking caret until done.
            StreamingCaretText(content: message.content)
        } else {
            // Finalized: render through Markdown + artifact splitter.
            MarkdownText(raw: message.content)
        }
    }

    // -------- system --------

    private var systemBubble: some View {
        HStack(alignment: .top, spacing: 12) {
            avatar(symbol: "gear", tint: .gray)
            VStack(alignment: .leading, spacing: 4) {
                Text("System")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(message.content)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .italic()
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 4)
    }

    // -------- helpers --------

    @ViewBuilder
    private func avatar(symbol: String, tint: Color) -> some View {
        Image(systemName: symbol)
            .foregroundStyle(.white)
            .frame(width: 28, height: 28)
            .background(tint, in: Circle())
    }

    private func copyContent() {
        #if canImport(AppKit)
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(message.content, forType: .string)
        #endif
        copied = true
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            copied = false
        }
    }
}

/// Streaming assistant text with a blinking caret pinned to the end.
/// The caret toggles between full opacity and clear (instead of being
/// removed/added) so the trailing layout position never jumps mid-line,
/// and the surrounding paragraph reflows naturally as tokens arrive.
private struct StreamingCaretText: View {
    let content: String

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.5)) { ctx in
            let on = Int(ctx.date.timeIntervalSince1970 * 2)
                .isMultiple(of: 2)
            (Text(content)
             + Text("▌").foregroundColor(on ? .primary : .clear))
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
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
private struct ToolCallDisclosure: View {
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

    // MARK: header

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
            // Args preview: shown next to the tool name so the user can
            // tell `ls /Sources` apart from `ls /Tests` without opening
            // every row. For delegations we surface the sub-task text
            // (what the user cares about) instead of the JSON envelope.
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

    // MARK: expanded body

    @ViewBuilder
    private var expandedContent: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Agent summary line, only when present, mirrors the
            // previous flat layout — useful context that's too long
            // to fit in the collapsed header.
            if let agent = delegation?.agent, !agent.summary.isEmpty {
                Text(agent.summary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            // Args section: for delegations show the sub-task body
            // plainly; for everything else pretty-print the JSON.
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

    // MARK: helpers

    /// Parse a `__delegate_to_agent` call into the resolved target
    /// AgentConfig (nil if the name isn't registered) and the sub-task
    /// text. Returns nil for non-delegation calls.
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

    /// Compact one-line summary of the call's args, used in the
    /// collapsed header. JSON is re-serialized without whitespace so
    /// nested objects fit on one line; truncation is left to SwiftUI.
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
