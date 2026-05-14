import SwiftUI
import DeepSeekKit

/// Single-conversation chat surface bound to the currently-selected
/// `ChatStore` conversation. Reads sampler defaults from `@AppStorage`.
struct ChatView: View {
    @ObservedObject var store: ChatStore
    @State private var draft: String = ""

    @AppStorage("deepseek.temperature")       private var temperature: Double = 0.7
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
                        if case .prefilling(let promptTokens, let startTime) = phase {
                            PrefillIndicator(promptTokens: promptTokens,
                                              startTime: startTime)
                                .padding(.leading, 40)
                        }
                        if case .streaming(_, let status, let metrics) = phase {
                            ThroughputBar(metrics: metrics, status: status)
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
            if c.messages.allSatisfy({ $0.role != .assistant || $0.content.isEmpty }),
               case .idle = phase {
                Text("First token may take 30 s – 3 min on a small-RAM Mac while weights page in. Subsequent tokens are faster.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .padding(.horizontal, 16)
                    .padding(.bottom, 6)
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
        if case .streaming(let buffer, _, _) = phase { return buffer.count }
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
        // Clamp temperature into the supported [0.5, 1.0] range in case
        // an older @AppStorage value (default used to be 1.0, slider
        // used to span 0…2) is still on disk for a user who never
        // touched the Settings tab.
        let clampedT = min(1.0, max(0.5, temperature))
        let opts = SamplingOptions(
            temperature: Float(clampedT),
            topK: topK, topP: Float(topP),
            repetitionPenalty: Float(repPenalty))
        let mode: ThinkingMode = (modeRaw == "max")  ? .max
                                : (modeRaw == "high") ? .high
                                : .chat
        store.send(text: text, mode: mode,
                    options: opts, maxTokens: maxTokens)
    }
}

/// Animated indicator shown while the prefill forward is running. The
/// elapsed seconds tick live via a TimelineView; the prompt token count
/// stays constant so the user can see how many positions the model is
/// chewing through.
private struct PrefillIndicator: View {
    let promptTokens: Int
    let startTime: Date

    var body: some View {
        TimelineView(.periodic(from: startTime, by: 0.1)) { ctx in
            let elapsed = ctx.date.timeIntervalSince(startTime)
            HStack(spacing: 8) {
                ProgressView()
                    .controlSize(.small)
                Text(String(format: "Prefilling %d tokens · %.1fs",
                            promptTokens, elapsed))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
    }
}

/// Compact two-line throughput readout for the streaming phase. Renders
/// nothing until at least one metric is populated, so very short prompts
/// don't get a blank caption bar before prefillDone arrives.
private struct ThroughputBar: View {
    let metrics: GenerationMetrics
    let status: String

    var body: some View {
        let hasPrefill = metrics.promptTokens > 0
        let hasGen = metrics.generatedTokens > 0
        let hasStatus = !status.isEmpty
        if hasPrefill || hasGen || hasStatus {
            VStack(alignment: .leading, spacing: 2) {
                if hasPrefill {
                    Text(String(format:
                        "Prefill: %d tok in %.2fs · %.0f tok/min",
                        metrics.promptTokens,
                        metrics.prefillElapsed,
                        metrics.prefillTokPerMin))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.tertiary)
                }
                if hasGen {
                    Text(String(format:
                        "Generation: %d tok in %.2fs · %.0f tok/min",
                        metrics.generatedTokens,
                        metrics.generationElapsed,
                        metrics.generationTokPerMin))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.tertiary)
                }
                if hasStatus {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }
}
