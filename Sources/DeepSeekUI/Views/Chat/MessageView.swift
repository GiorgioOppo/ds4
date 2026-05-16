import SwiftUI
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
                if let r = message.reasoningContent, !r.isEmpty {
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

    /// Wrench-icon disclosure with one row per (call, output)
    /// pair. Rendered below the assistant prose because the calls
    /// chronologically fire *after* the model writes its text /
    /// reasoning. Output is shown plain because most MCP servers
    /// return raw text or JSON — running it through MarkdownText
    /// would mangle a JSON blob with stray code-block heuristics.
    private var toolCallSection: some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(Array(message.toolCalls.enumerated()), id: \.offset) { idx, call in
                    toolCallRow(call: call,
                                 output: message.toolOutputs?
                                    .indices.contains(idx) == true
                                    ? message.toolOutputs![idx]
                                    : nil)
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

    private func toolCallRow(call: StoredToolCall, output: String?) -> some View {
        // Recognise the synthetic delegation tool so the row can
        // render the *target* agent's identity (icon + tint +
        // name) instead of the generic wrench, and surface the
        // sub-task text plainly instead of the `{agent_name, task}`
        // JSON pair the model emitted.
        let delegation = parseDelegation(call: call)

        return VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                if let agent = delegation?.agent {
                    Image(systemName: agent.iconName)
                        .font(.caption)
                        .foregroundStyle(AgentTint.color(for: agent.tint))
                    Text("Delegated to ")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                    + Text(agent.name)
                        .font(.callout.bold())
                    if !agent.summary.isEmpty {
                        Text("· \(agent.summary)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                } else if delegation != nil {
                    Image(systemName: "person.crop.circle.badge.questionmark")
                        .font(.caption)
                        .foregroundStyle(Color.orange)
                    Text("Delegation to unknown agent")
                        .font(.callout)
                        .foregroundStyle(.orange)
                } else {
                    Image(systemName: "arrow.up.right.square")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                    Text(call.name)
                        .font(.callout.monospaced())
                }
                if output == nil {
                    Spacer()
                    HStack(spacing: 4) {
                        ProgressView().controlSize(.mini)
                        Text("running…")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
            }
            // Args: for a delegation, drop the JSON envelope and
            // show the sub-task as plain text — it's the bit the
            // user actually cares about. For everything else,
            // pretty-print the JSON args so the structure is legible
            // (falls back to raw text on parse failure).
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
                let prettyArgs = MessageView.prettyJSON(call.args)
                if !prettyArgs.isEmpty {
                    Text(prettyArgs)
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
        .padding(.vertical, 2)
    }

    /// Decode a delegation call's payload into the bits the row
    /// needs: the resolved target agent (or nil if the name isn't
    /// registered) and the sub-task text. Returns nil for calls
    /// that aren't `__delegate_to_agent` so the caller can fall
    /// through to the generic rendering.
    private func parseDelegation(call: StoredToolCall)
        -> (agent: AgentConfig?, task: String)?
    {
        guard call.name == EncodingDSV4.delegateToolName else { return nil }
        let data = call.args.data(using: .utf8) ?? Data()
        let obj = (try? JSONSerialization.jsonObject(with: data))
            as? [String: Any] ?? [:]
        let agentName = (obj["agent_name"] as? String) ?? ""
        let task = (obj["task"] as? String) ?? ""
        let agent = agentName.isEmpty ? nil : agentResolver?(agentName)
        return (agent, task)
    }

    private static func prettyJSON(_ raw: String) -> String {
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
