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
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 4)
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
            Text(message.content + "▌")
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
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
