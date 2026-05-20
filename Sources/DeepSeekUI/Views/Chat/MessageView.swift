import SwiftUI
import DeepSeekKit
#if canImport(AppKit)
import AppKit
#endif

/// One message bubble for a *single* StoredMessage. User and system
/// bubbles are full implementations; the `.assistant` case delegates
/// to `AssistantTurnView` so the same rendering serves both this
/// single-message path AND the multi-message turn-grouping path that
/// `ChatView` uses for tool roundtrips.
///
/// This view is preserved as the public single-message API (used by
/// previews, tests, and any caller that hasn't been migrated to the
/// turn-grouping iterator). `ChatView` no longer routes assistant
/// rows through here directly — it builds an `AssistantTurnView` with
/// the full list of messages in the turn instead.
struct MessageView: View {
    let message: StoredMessage
    var isStreaming: Bool = false
    var streamingReasoning: String? = nil
    var agentResolver: ((String) -> AgentConfig?)? = nil

    var body: some View {
        switch message.role {
        case .user:
            userBubble
        case .assistant:
            AssistantTurnView(
                messages: [message],
                isStreamingFinal: isStreaming,
                streamingReasoning: streamingReasoning,
                agentResolver: agentResolver)
        case .system:
            systemBubble
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
                InlineMarkdownText(raw: message.content)
                    .font(.callout)
                    .lineSpacing(2)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(
                        LinearGradient(
                            colors: [
                                Color.accentColor.opacity(0.22),
                                Color.accentColor.opacity(0.12),
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing),
                        in: RoundedRectangle(cornerRadius: 12))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .strokeBorder(
                                Color.accentColor.opacity(0.25),
                                lineWidth: 1))
            }
            avatar(symbol: "person.fill", tint: .blue)
        }
        .padding(.vertical, 4)
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
}
