import SwiftUI

/// Diagnostics panel: tokenize a string with `ds4 --dump-tokens` to inspect how
/// text (including DS4 protocol specials) maps to tokens.
struct DiagnosticsView: View {
    @Bindable var controller: DiagnosticsController

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section("Tokenization (native, backend tokenizer)") {
                    LabeledContent("Model (from Settings)",
                                   value: (controller.modelPath as NSString).lastPathComponent)
                    TextField("Text", text: $controller.text, axis: .vertical)
                        .lineLimit(2...6)
                    Text("Opens the GGUF only for the tokenizer (pure Swift, no subprocess), selected by the model's architecture — DeepSeek or GLM. Choose the model in Settings.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section {
                    if controller.isRunning {
                        Button(role: .destructive) { controller.cancel() } label: {
                            Label("Cancel", systemImage: "stop.fill")
                        }
                    } else {
                        Button { controller.dumpTokens() } label: {
                            Label("Tokenize", systemImage: "text.magnifyingglass")
                        }
                        Button { controller.dumpChatTemplate() } label: {
                            Label("Show chat template + tool format", systemImage: "doc.text.magnifyingglass")
                        }
                    }
                }
            }
            .formStyle(.grouped)
            .frame(maxHeight: 280)

            Divider()

            ScrollView {
                Text(controller.output.isEmpty ? "No output." : controller.output)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(8)
            }
            .frame(maxHeight: 140)
            .background(Color.black.opacity(0.05))

            Divider()
            EngineConsole()
        }
    }
}

/// Live view of the C engine's captured stderr (Metal/kernel diagnostics).
struct EngineConsole: View {
    @State private var log = EngineLog.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Engine Console (stderr)")
                .font(.caption.bold())
                .foregroundStyle(.secondary)
                .padding(.horizontal, 8).padding(.top, 6)
            ScrollView {
                Text(log.text.isEmpty ? "No engine messages." : log.text)
                    .font(.system(.caption2, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(8)
            }
        }
        .background(Color.black.opacity(0.05))
    }
}
