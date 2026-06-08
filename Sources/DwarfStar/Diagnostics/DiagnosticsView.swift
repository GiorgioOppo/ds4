import SwiftUI

/// Diagnostics panel: tokenize a string with `ds4 --dump-tokens` to inspect how
/// text (including DS4 protocol specials) maps to tokens.
struct DiagnosticsView: View {
    @Bindable var controller: DiagnosticsController

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section("Tokenizzazione (nativa, DS4Core.Tokenizer)") {
                    TextField("Modello GGUF", text: $controller.modelPath)
                    Button {
                        if let path = ModelPicker.pickGGUF() { controller.modelPath = path }
                    } label: {
                        Label("Sfoglia…", systemImage: "folder")
                    }
                    TextField("Testo", text: $controller.text, axis: .vertical)
                        .lineLimit(2...6)
                    Text("Apre il GGUF solo per il tokenizer (puro Swift, niente subprocess). Sotto sandbox seleziona il modello con Sfoglia…")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section {
                    if controller.isRunning {
                        Button(role: .destructive) { controller.cancel() } label: {
                            Label("Annulla", systemImage: "stop.fill")
                        }
                    } else {
                        Button { controller.dumpTokens() } label: {
                            Label("Tokenizza", systemImage: "text.magnifyingglass")
                        }
                        Button { controller.dumpChatTemplate() } label: {
                            Label("Mostra chat template + formato tool", systemImage: "doc.text.magnifyingglass")
                        }
                    }
                }
            }
            .formStyle(.grouped)
            .frame(maxHeight: 280)

            Divider()

            ScrollView {
                Text(controller.output.isEmpty ? "Nessun output." : controller.output)
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
            Text("Console motore (stderr)")
                .font(.caption.bold())
                .foregroundStyle(.secondary)
                .padding(.horizontal, 8).padding(.top, 6)
            ScrollView {
                Text(log.text.isEmpty ? "Nessun messaggio dal motore." : log.text)
                    .font(.system(.caption2, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(8)
            }
        }
        .background(Color.black.opacity(0.05))
    }
}
