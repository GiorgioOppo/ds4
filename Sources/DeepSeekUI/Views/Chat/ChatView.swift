import SwiftUI
import DeepSeekKit

/// Single-conversation chat surface: scrollable message list +
/// composer. Reads sampler defaults from `@AppStorage`. Multi-chat
/// (sidebar) and persistence land in commit 4; advanced settings in
/// commit 6.
struct ChatView: View {
    @ObservedObject var store: ChatStore
    @State private var draft: String = ""

    // Sampler defaults — exposed in Settings (commit 6).
    @AppStorage("deepseek.temperature")       private var temperature: Double = 1.0
    @AppStorage("deepseek.topK")              private var topK: Int = 0
    @AppStorage("deepseek.topP")              private var topP: Double = 1.0
    @AppStorage("deepseek.repPenalty")        private var repPenalty: Double = 1.0
    @AppStorage("deepseek.maxTokens")         private var maxTokens: Int = 256
    @AppStorage("deepseek.mode")              private var modeRaw: String = "chat"

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 16) {
                        ForEach(store.conversation.messages) { msg in
                            MessageView(
                                message: msg,
                                isStreaming: isStreamingPlaceholder(msg))
                            .id(msg.id)
                        }
                        if case .streaming(_, let status) = store.phase,
                           !status.isEmpty {
                            Text(status)
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                                .padding(.leading, 40)
                        }
                        if case .error(let msg) = store.phase {
                            Label(msg, systemImage: "exclamationmark.triangle.fill")
                                .foregroundStyle(.red)
                                .font(.callout)
                                .padding(.leading, 40)
                        }
                    }
                    .padding(20)
                }
                .onChange(of: store.conversation.messages.count) { _, _ in
                    scrollToBottom(proxy)
                }
                .onChange(of: scrollBufferLength) { _, _ in
                    scrollToBottom(proxy)
                }
            }
            Divider()
            ComposerView(draft: $draft, phase: store.phase,
                          onSend: sendCurrent, onStop: { store.cancel() })
        }
        .navigationTitle(store.conversation.title)
    }

    private func isStreamingPlaceholder(_ msg: StoredMessage) -> Bool {
        guard case .streaming = store.phase,
              msg.role == .assistant,
              msg.id == store.conversation.messages.last?.id
        else { return false }
        return true
    }

    /// Used to retrigger scroll-to-bottom on every streamed token, not
    /// just on message append.
    private var scrollBufferLength: Int {
        if case .streaming(let buffer, _) = store.phase {
            return buffer.count
        }
        return 0
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        guard let last = store.conversation.messages.last else { return }
        withAnimation(.easeOut(duration: 0.15)) {
            proxy.scrollTo(last.id, anchor: .bottom)
        }
    }

    private func sendCurrent() {
        let text = draft
        draft = ""
        let opts = SamplingOptions(
            temperature: Float(temperature),
            topK: topK,
            topP: Float(topP),
            repetitionPenalty: Float(repPenalty))
        let mode: ThinkingMode = (modeRaw == "max")    ? .max
                                : (modeRaw == "high") ? .high
                                : .chat
        store.send(text: text, mode: mode,
                    options: opts, maxTokens: maxTokens)
    }
}
