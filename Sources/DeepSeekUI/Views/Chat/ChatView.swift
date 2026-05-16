import SwiftUI
import DeepSeekKit

/// Single-conversation chat surface bound to the currently-selected
/// `ChatStore` conversation. Reads sampler defaults from `@AppStorage`.
struct ChatView: View {
    @ObservedObject var store: ChatStore
    @ObservedObject var modelState: ModelState
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
                            // Drop progress indicators where they belong
                            // in the flow: right between the user prompt
                            // and the about-to-be-filled assistant turn.
                            // The placeholder is always the last message
                            // with empty content while we're prefilling
                            // or just starting to stream.
                            if shouldShowInlineProgress(for: msg, in: c, phase: phase) {
                                inlineProgress(phase)
                            }
                            MessageView(
                                message: msg,
                                isStreaming: isStreamingPlaceholder(msg, in: c, phase: phase),
                                agentResolver: { name in
                                    store.agents.agents.first(where: { $0.name == name })
                                })
                            .id(msg.id)
                        }
                        if case .streaming(_, _, let metrics) = phase {
                            // Throughput bar stays in the trailing slot
                            // so the live tok/min readout sits under the
                            // reply as it grows.
                            ThroughputBar(metrics: metrics, status: "")
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
            // Model-state banner: tells the user that the chat is
            // alive but inference isn't (no model loaded / load
            // in progress / load failed). Collapses to EmptyView
            // when a model is ready.
            modelStateBanner
            // Cumulative-cost banner for remote chats. Hidden for
            // local chats (cost is nil) and for fresh remote chats
            // that haven't billed anything yet.
            if let total = c.cumulativeCostUSD, total > 0 {
                HStack(spacing: 6) {
                    Image(systemName: "dollarsign.circle")
                        .foregroundStyle(.secondary)
                    Text("Chat total: \(ThroughputBar.formatUSD(total))")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 4)
            }
            // Live delegation chain: pinned above the composer so
            // the user can watch sub-agents work without losing
            // sight of either the transcript above or the input
            // below. Empty stack collapses to EmptyView, no
            // padding cost.
            let frames = store.activeDelegations[c.id] ?? []
            if !frames.isEmpty {
                DelegationStackView(frames: frames)
                    .padding(.horizontal, 16)
                    .padding(.bottom, 6)
            }
            resumeBanner(c: c, phase: phase)
            thinkingPicker
            ComposerView(draft: $draft,
                          phase: phase,
                          canSend: modelState.isReady,
                          onSend: sendCurrent, onStop: { store.cancel() })
        }
        .navigationTitle(c.title)
    }

    /// Rendered above the composer when a previous generation died
    /// mid-stream and the conversation carries a `pendingTurn`
    /// snapshot. Hidden once `phase` becomes `.streaming`/`.prefilling`
    /// (the resume already kicked off) or `.error`. Tapping Resume
    /// re-runs the same prompt + accumulated ids through the model
    /// so the partial reply on screen keeps extending.
    /// Per-chat thinking-mode picker pinned above the composer.
    /// The selection drives `resolveSampling()` on the next send.
    /// When an agent is attached to this chat its `defaultMode`
    /// wins (same precedence as the sampling sliders), so the
    /// picker reflects the agent's choice as a read-only display
    /// with a small "locked by …" hint — keeps the user from
    /// thinking they're changing something they aren't.
    @ViewBuilder
    private var thinkingPicker: some View {
        let agent = store.selectedConversation?.agentID
            .flatMap { store.agents.agent(id: $0) }
        let effective = agent?.defaultMode ?? modeRaw
        HStack(spacing: 8) {
            Image(systemName: "brain.head.profile")
                .font(.caption)
                .foregroundStyle(.secondary)
            Picker("Thinking", selection: Binding(
                get: { effective },
                set: { newValue in
                    // Agent override: ignore writes — the picker
                    // is disabled in that case anyway, but the
                    // setter still has to be valid.
                    if agent == nil { modeRaw = newValue }
                })) {
                Text("No think").tag("chat")
                Text("High").tag("high")
                Text("Max").tag("max")
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .disabled(agent != nil)
            .frame(maxWidth: 260)
            if let a = agent {
                HStack(spacing: 3) {
                    Image(systemName: "lock.fill")
                    Text("set by \(a.name)")
                }
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .lineLimit(1)
                .truncationMode(.tail)
            }
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.bottom, 4)
    }

    @ViewBuilder
    private var modelStateBanner: some View {
        switch modelState.status {
        case .idle:
            modelBannerRow(
                icon: "tray",
                tint: .secondary,
                title: "No model loaded",
                subtitle: "Pick one from the model menu in the toolbar to start chatting.",
                progress: false)
        case .loading(let ep, let plan):
            modelBannerRow(
                icon: "arrow.down.circle",
                tint: .accentColor,
                title: "Loading \(ep.displayName)…",
                subtitle: plan.map(planSummary) ?? "Probing shards on disk…",
                progress: true)
        case .error(let ep, let msg):
            modelBannerRow(
                icon: "exclamationmark.octagon.fill",
                tint: .orange,
                title: "Could not load \(ep.displayName)",
                subtitle: msg,
                progress: false)
        case .loaded:
            EmptyView()
        }
    }

    private func modelBannerRow(icon: String,
                                 tint: Color,
                                 title: String,
                                 subtitle: String,
                                 progress: Bool) -> some View {
        HStack(alignment: .top, spacing: 8) {
            if progress {
                ProgressView().controlSize(.small)
            } else {
                Image(systemName: icon).foregroundStyle(tint)
            }
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(.callout.bold())
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(tint.opacity(0.08),
                     in: RoundedRectangle(cornerRadius: 6))
        .padding(.horizontal, 16)
        .padding(.bottom, 6)
    }

    private func planSummary(_ plan: LoadPlan) -> String {
        // Light prose extract — the full PreflightSummaryView used
        // to render this in a dedicated screen; here we just need
        // enough to reassure the user something is happening.
        let gb = Double(plan.totalBytes) / 1_073_741_824
        return String(format: "%.1f GB across %d shards · strategy: %@",
                       gb, plan.shards.count, plan.strategy.rawValue)
    }

    @ViewBuilder
    private func resumeBanner(c: Conversation,
                                phase: GenerationPhase) -> some View {
        let isIdle: Bool = {
            switch phase {
            case .idle, .error: return true
            default:           return false
            }
        }()
        if let pt = c.pendingTurn, isIdle {
            HStack(spacing: 10) {
                Image(systemName: "arrow.clockwise.circle.fill")
                    .foregroundStyle(Color.orange)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Generation interrupted")
                        .font(.callout.bold())
                    Text("\(pt.generatedTokens.count) tokens already sampled. Resume continues from where it stopped.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Resume") {
                    resumeCurrent()
                }
                .buttonStyle(.borderedProminent)
                Button {
                    store.discardPendingTurn(of: c.id)
                } label: {
                    Image(systemName: "xmark")
                }
                .buttonStyle(.borderless)
                .help("Discard the partial reply")
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color.orange.opacity(0.08))
        }
    }

    private func isStreamingPlaceholder(_ msg: StoredMessage,
                                         in c: Conversation,
                                         phase: GenerationPhase) -> Bool {
        guard case .streaming = phase,
              msg.role == .assistant,
              msg.id == c.messages.last?.id else { return false }
        return true
    }

    /// True for the empty assistant placeholder at the tail of the
    /// conversation while the model is encoding the prompt or running
    /// the prefill forward — the slot where the inline indicator
    /// should be drawn (directly above the soon-to-be reply).
    private func shouldShowInlineProgress(for msg: StoredMessage,
                                           in c: Conversation,
                                           phase: GenerationPhase) -> Bool {
        guard msg.id == c.messages.last?.id,
              msg.role == .assistant,
              msg.content.isEmpty else { return false }
        switch phase {
        case .prefilling: return true
        case .streaming(_, let status, _): return !status.isEmpty
        default: return false
        }
    }

    @ViewBuilder
    private func inlineProgress(_ phase: GenerationPhase) -> some View {
        switch phase {
        case .prefilling(let promptTokens, let startTime):
            PrefillIndicator(promptTokens: promptTokens,
                              startTime: startTime)
                .padding(.leading, 40)
        case .streaming(_, let status, _) where !status.isEmpty:
            HStack(spacing: 6) {
                ProgressView().controlSize(.mini)
                Text(status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.leading, 40)
        default:
            EmptyView()
        }
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
        // Bail out without clearing local state if a generation is
        // already running (the TextField's onSubmit still fires while
        // the Send button is hidden), or if the draft is empty —
        // losing a half-typed message because Return fired at the
        // wrong moment would be obnoxious.
        if let id = store.selectedID {
            switch store.phase(of: id) {
            case .streaming, .prefilling: return
            default: break
            }
        }
        guard !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return }
        let text = draft
        draft = ""
        let resolved = resolveSampling()
        store.send(text: text,
                    mode: resolved.mode,
                    options: resolved.options,
                    maxTokens: resolved.maxTokens)
    }

    /// Restart a previously-interrupted generation. Reuses the same
    /// sampler / max-tokens that drive `sendCurrent` — including
    /// the attached agent's defaults when one is set — so a
    /// resumed reply continues with the same parameters.
    private func resumeCurrent() {
        guard let id = store.selectedID else { return }
        let resolved = resolveSampling()
        store.resumePendingTurn(of: id,
                                  options: resolved.options,
                                  maxTokens: resolved.maxTokens)
    }

    /// Compose the sampling configuration for this turn. The
    /// attached agent (when set) overrides every Generation-tab
    /// slider — its values were chosen for *this* agent's
    /// behaviour, and silently mixing them with whatever the user
    /// last touched in the global tab would produce confusing
    /// generations. The Generation tab acts as a fallback for
    /// chats with no agent attached and as the global default for
    /// fresh chats.
    private func resolveSampling() -> (mode: ThinkingMode,
                                         options: SamplingOptions,
                                         maxTokens: Int) {
        let agent = store.selectedConversation?.agentID
            .flatMap { store.agents.agent(id: $0) }

        // Sliders write to AppStorage in a slightly wider range
        // than the model accepts; clamp into the supported
        // [0.5, 1.0] window so an out-of-band value from an older
        // build doesn't crash the sampler.
        let temp = agent?.temperature ?? temperature
        let tp   = agent?.topP        ?? topP
        let tk   = agent?.topK        ?? topK
        let rp   = agent?.repetitionPenalty ?? repPenalty
        let mt   = agent?.maxTokens   ?? maxTokens
        let modeStr = agent?.defaultMode ?? modeRaw

        let clampedT = min(1.0, max(0.5, temp))
        let options = SamplingOptions(
            temperature: Float(clampedT),
            topK: tk, topP: Float(tp),
            repetitionPenalty: Float(rp))
        let mode: ThinkingMode = (modeStr == "max")  ? .max
                                : (modeStr == "high") ? .high
                                : .chat
        return (mode, options, mt)
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
        let hasCost = metrics.turnCostUSD != nil
        if hasPrefill || hasGen || hasStatus || hasCost {
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
                if let cost = metrics.turnCostUSD {
                    Text("Turn cost: \(Self.formatUSD(cost))")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                if hasStatus {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }

    /// Compact USD formatter that picks enough significant digits
    /// to make sub-cent costs (typical for tool-call loops on
    /// cheap models) visible. $0.0042 / $0.13 / $1.27.
    static func formatUSD(_ value: Double) -> String {
        if value < 0.01 {
            return String(format: "$%.4f", value)
        } else if value < 1.0 {
            return String(format: "$%.3f", value)
        } else {
            return String(format: "$%.2f", value)
        }
    }
}
