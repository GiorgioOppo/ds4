import SwiftUI
import DeepSeekKit

/// Single-conversation chat surface bound to the currently-selected
/// `ChatStore` conversation. Reads sampler defaults from `@AppStorage`.
struct ChatView: View {
    @ObservedObject var store: ChatStore
    @State private var draft: String = ""

    @AppStorage("deepseek.temperature")       private var temperature: Double = 1.0
    @AppStorage("deepseek.topK")              private var topK: Int = 0
    @AppStorage("deepseek.topP")              private var topP: Double = 1.0
    @AppStorage("deepseek.repPenalty")        private var repPenalty: Double = 1.0
    @AppStorage("deepseek.maxTokens")         private var maxTokens: Int = 256
    @AppStorage("deepseek.mode")              private var modeRaw: String = "chat"

    var body: some View {
        if let c = store.selectedConversation {
            content(c)
        } else {
            VStack {
                Spacer()
                Text("No conversation selected.")
                    .foregroundStyle(.secondary)
                Button("New Chat") { store.newChat() }
                Spacer()
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    @ViewBuilder
    private func content(_ c: Conversation) -> some View {
        let phase = store.phase(of: c.id)
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 16) {
                        ForEach(c.messages) { msg in
                            MessageView(
                                message: msg,
                                isStreaming: isStreamingPlaceholder(msg, in: c, phase: phase))
                            .id(msg.id)
                        }
                        if case .streaming(_, let status) = phase, !status.isEmpty {
                            Text(status)
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                                .padding(.leading, 40)
                        }
                        if case .error(let msg) = phase {
                            Label(msg, systemImage: "exclamationmark.triangle.fill")
                                .foregroundStyle(.red)
                                .font(.callout)
                                .padding(.leading, 40)
                        }
                    }
                    .padding(20)
                }
                .onChange(of: c.messages.count) { _, _ in scrollToBottom(proxy, c) }
                .onChange(of: scrollBufferLength(phase)) { _, _ in scrollToBottom(proxy, c) }
            }
            Divider()
            ComposerView(draft: $draft, phase: phase,
                          onSend: sendCurrent, onStop: { store.cancel() })
        }
        .navigationTitle(c.title)
    }

    private func isStreamingPlaceholder(_ msg: StoredMessage,
                                         in c: Conversation,
                                         phase: GenerationPhase) -> Bool {
        guard case .streaming = phase,
              msg.role == .assistant,
              msg.id == c.messages.last?.id else { return false }
        return true
    }

    private func scrollBufferLength(_ phase: GenerationPhase) -> Int {
        if case .streaming(let buffer, _) = phase { return buffer.count }
        return 0
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy, _ c: Conversation) {
        guard let last = c.messages.last else { return }
        withAnimation(.easeOut(duration: 0.15)) {
            proxy.scrollTo(last.id, anchor: .bottom)
        }
    }

    private func sendCurrent() {
        let text = draft
        draft = ""
        let opts = SamplingOptions(
            temperature: Float(temperature),
            topK: topK, topP: Float(topP),
            repetitionPenalty: Float(repPenalty))
        let mode: ThinkingMode = (modeRaw == "max")  ? .max
                                : (modeRaw == "high") ? .high
                                : .chat
        store.send(text: text, mode: mode,
                    options: opts, maxTokens: maxTokens)
    }
}
