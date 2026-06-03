import SwiftUI

/// Sampler defaults applied to every new generation. Bindings use the
/// same @AppStorage keys ChatView reads, so changes take effect on
/// the next Send (no restart).
struct GenerationSettingsTab: View {
    @AppStorage("deepseek.temperature") private var temperature: Double = 0.1
    @AppStorage("deepseek.topK")        private var topK: Int = 0
    @AppStorage("deepseek.topP")        private var topP: Double = 1.0
    @AppStorage("deepseek.repPenalty")  private var repPenalty: Double = 1.0
    @AppStorage("deepseek.maxTokens")   private var maxTokens: Int = 256
    @AppStorage("deepseek.mode")        private var modeRaw: String = "chat"
    @AppStorage(AppSettingsKey.remoteHistoryTurnsCap)
    private var remoteHistoryTurnsCap: Int = 0

    var body: some View {
        Form {
            Section("Sampling") {
                LabeledContent("Temperature") {
                    HStack {
                        // Range [0, 1]. 0 is greedy argmax — on V4-Flash
                        // MoE very low values can settle into fixed points
                        // (the model loops on filler tokens like 好的好的…);
                        // raise it if you see that. Values around 0.7–0.9
                        // give the most varied-yet-coherent samples.
                        Slider(value: $temperature,
                                in: 0...1,
                                step: 0.05) {
                            EmptyView()
                        } minimumValueLabel: {
                            Text("0").font(.caption2.monospaced())
                                .foregroundStyle(.tertiary)
                        } maximumValueLabel: {
                            Text("1").font(.caption2.monospaced())
                                .foregroundStyle(.tertiary)
                        }
                        Text(String(format: "%.2f", temperature))
                            .frame(width: 48, alignment: .trailing)
                            .font(.system(.body, design: .monospaced))
                    }
                }
                LabeledContent("Top-K (0 = disabled)") {
                    Stepper(value: $topK, in: 0...500, step: 1) {
                        Text("\(topK)")
                            .font(.system(.body, design: .monospaced))
                    }
                }
                LabeledContent("Top-P (1.0 = disabled)") {
                    HStack {
                        Slider(value: $topP, in: 0...1, step: 0.01)
                        Text(String(format: "%.2f", topP))
                            .frame(width: 48, alignment: .trailing)
                            .font(.system(.body, design: .monospaced))
                    }
                }
                LabeledContent("Repetition penalty (1.0 = disabled)") {
                    HStack {
                        Slider(value: $repPenalty, in: 1...2, step: 0.01)
                        Text(String(format: "%.2f", repPenalty))
                            .frame(width: 48, alignment: .trailing)
                            .font(.system(.body, design: .monospaced))
                    }
                }
            }
            Section("Length") {
                LabeledContent("Max tokens per turn") {
                    Stepper(value: $maxTokens, in: 1...8192, step: 16) {
                        Text("\(maxTokens)")
                            .font(.system(.body, design: .monospaced))
                    }
                }
            }
            Section {
                LabeledContent("Sliding window") {
                    Stepper(value: $remoteHistoryTurnsCap,
                             in: 0...100, step: 1) {
                        Text(remoteHistoryTurnsCap == 0
                              ? "Unlimited"
                              : "\(remoteHistoryTurnsCap) turns")
                            .font(.system(.body, design: .monospaced))
                    }
                }
            } header: {
                Text("Remote history")
            } footer: {
                Text("Cap how many user-led turns get re-sent to "
                     + "OpenRouter / Anthropic on the next call. "
                     + "0 = unlimited (full history every turn — "
                     + "what every stateless chat API requires). "
                     + "Set a positive value to bound per-request "
                     + "token cost on long chats; older turns stay "
                     + "visible in the transcript but aren't "
                     + "re-sent. System messages and the agent "
                     + "prompt are never dropped.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Section("Mode") {
                Picker("Thinking mode", selection: $modeRaw) {
                    Text("Chat (no <think>)").tag("chat")
                    Text("High thinking").tag("high")
                    Text("Max thinking").tag("max")
                }
                .pickerStyle(.radioGroup)
            }
        }
        .formStyle(.grouped)
        .padding()
        .onAppear {
            // One-shot migration: clamp a value stored by an older
            // build (ranges 0…2 or 0.5…1.0) into the current [0, 1]
            // range so the slider doesn't render off-track.
            if temperature < 0 { temperature = 0 }
            if temperature > 1 { temperature = 1 }
        }
    }
}
