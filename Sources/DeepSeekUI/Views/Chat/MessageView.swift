import SwiftUI

/// One message bubble. Commit 3: raw text + simple role styling.
/// Markdown rendering + collapsible reasoning land in commit 5.
struct MessageView: View {
    let message: StoredMessage
    /// True while this message is the in-progress streaming target so
    /// we can show a blinking caret.
    var isStreaming: Bool = false

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            avatar
            VStack(alignment: .leading, spacing: 4) {
                Text(roleLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if let r = message.reasoningContent, !r.isEmpty {
                    // Commit 5 replaces this with a DisclosureGroup.
                    Text(r)
                        .font(.system(.callout, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .padding(8)
                        .background(Color(NSColor.controlBackgroundColor),
                                     in: RoundedRectangle(cornerRadius: 8))
                }
                streamingContent
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 4)
    }

    private var roleLabel: String {
        switch message.role {
        case .user:      return "You"
        case .assistant: return "Assistant"
        case .system:    return "System"
        }
    }

    @ViewBuilder
    private var avatar: some View {
        let symbol: String = message.role == .user ? "person.fill" : "cpu"
        let tint: Color = message.role == .user ? .blue : .purple
        Image(systemName: symbol)
            .foregroundStyle(.white)
            .frame(width: 28, height: 28)
            .background(tint, in: Circle())
    }

    @ViewBuilder
    private var streamingContent: some View {
        if message.content.isEmpty && isStreaming {
            HStack(spacing: 6) {
                ProgressView().controlSize(.mini)
                Text("Thinking…").foregroundStyle(.secondary)
            }
        } else {
            // Commit 5 swaps Text for an AttributedString(markdown:) view.
            Text(message.content + (isStreaming ? "▌" : ""))
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
