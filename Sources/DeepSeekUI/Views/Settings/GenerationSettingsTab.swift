import SwiftUI

/// Sampler defaults applied to every new generation. Bindings use the
/// same @AppStorage keys ChatView reads, so changes take effect on
/// the next Send (no restart).
struct GenerationSettingsTab: View {
    @AppStorage("deepseek.temperature") private var temperature: Double = 1.0
    @AppStorage("deepseek.topK")        private var topK: Int = 0
    @AppStorage("deepseek.topP")        private var topP: Double = 1.0
    @AppStorage("deepseek.repPenalty")  private var repPenalty: Double = 1.0
    @AppStorage("deepseek.maxTokens")   private var maxTokens: Int = 256
    @AppStorage("deepseek.mode")        private var modeRaw: String = "chat"

    var body: some View {
        Form {
            Section("Sampling") {
                LabeledContent("Temperature") {
                    HStack {
                        Slider(value: $temperature, in: 0...2, step: 0.05)
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
    }
}
