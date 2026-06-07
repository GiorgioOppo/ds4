import SwiftUI

/// Sheet that downloads a GGUF natively (DS4Engine.ModelDownloader) with progress.
struct DownloadView: View {
    @Bindable var store: ChatStore
    @State private var runner = DownloadRunner()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Scarica un modello")
                .font(.title2).bold()
            Text("Download nativo da Hugging Face in \(store.scriptDir)/gguf. I download parziali riprendono automaticamente.")
                .font(.caption).foregroundStyle(.secondary)

            ForEach(ModelCatalog.downloadTargets) { target in
                HStack {
                    VStack(alignment: .leading) {
                        Text(target.title)
                        Text(target.detail).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Scarica") {
                        runner.run(target: target.id, scriptDir: store.scriptDir)
                    }
                    .disabled(runner.isRunning)
                }
                .padding(.vertical, 2)
            }

            if !runner.log.isEmpty {
                ScrollView {
                    Text(runner.log)
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(height: 160)
                .background(Color.black.opacity(0.05))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            if runner.isRunning {
                ProgressView(value: runner.progress) {
                    Text("Download di \(runner.currentTarget ?? "")… \(runner.progressText)")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }

            HStack {
                if runner.isRunning {
                    Button("Annulla") { runner.cancel() }
                }
                Spacer()
                Button("Chiudi") {
                    store.scanModels()
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)
            }
        }
        .padding()
        .frame(minWidth: 480, minHeight: 420)
    }
}
