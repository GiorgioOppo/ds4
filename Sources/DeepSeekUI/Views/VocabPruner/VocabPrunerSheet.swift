import SwiftUI
import AppKit
import DeepSeekVocabPruner

/// Modal sheet per il vocab pruning italiano-only. Tre pannelli:
///   - top: form (input/output dir + corpus + coverage + dry-run)
///   - middle: live progress (status, progress bar, log scrollabile)
///   - bottom: footer con Start / Cancel / Close
///
/// Mirror strutturale di `ConvertSheet` — locale `@StateObject vm`,
/// nessun environment object necessario.
struct VocabPrunerSheet: View {
    @StateObject private var vm = VocabPrunerViewModel()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                form
                    .padding(20)
            }
            .frame(minHeight: 280)
            Divider()
            progressArea
                .padding(16)
                .frame(minHeight: 220)
            Divider()
            footer
                .padding(12)
        }
        .frame(minWidth: 720, minHeight: 640)
    }

    // ---- Header ----

    @ViewBuilder
    private var header: some View {
        HStack {
            Image(systemName: "scissors")
                .font(.title2)
                .foregroundStyle(.tint)
            VStack(alignment: .leading) {
                Text("Prune vocabulary").font(.headline)
                Text("Riduce il vocabolario a un sottoinsieme italiano/" +
                     "latino-only sulla base di un corpus. Non tocca i " +
                     "pesi del transformer. Vedi docs/VOCAB-PRUNING.md.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                if vm.isRunning { vm.cancel() }
                dismiss()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.borderless)
            .keyboardShortcut(.cancelAction)
        }
        .padding(16)
    }

    // ---- Form ----

    @ViewBuilder
    private var form: some View {
        Form {
            Section("Source") {
                LabeledContent("Input directory") {
                    HStack {
                        Text(vm.inputDir?.path ?? "—")
                            .font(.system(.body, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .foregroundStyle(vm.inputDir == nil ? .secondary : .primary)
                        Spacer()
                        Button("Choose…") {
                            pickDirectory { vm.inputDir = $0 }
                        }
                        .disabled(vm.isRunning)
                    }
                }
                Text("Checkpoint convertito (output di `converter`). Deve " +
                     "contenere tokenizer.json, config.json, model.safetensors.index.json " +
                     "e i shard model-*.safetensors.")
                    .font(.caption).foregroundStyle(.tertiary)
            }

            Section("Destination") {
                LabeledContent("Output directory") {
                    HStack {
                        Text(vm.outputDir?.path ?? "—")
                            .font(.system(.body, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .foregroundStyle(vm.outputDir == nil ? .secondary : .primary)
                        Spacer()
                        Button("Choose…") {
                            pickDirectory { vm.outputDir = $0 }
                        }
                        .disabled(vm.isRunning)
                    }
                }
                Text("Creata se non esiste. Deve essere diversa da Input.")
                    .font(.caption).foregroundStyle(.tertiary)
            }

            Section("Corpus") {
                LabeledContent("Path") {
                    HStack {
                        Text(vm.corpus?.path ?? "—")
                            .font(.system(.body, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .foregroundStyle(vm.corpus == nil ? .secondary : .primary)
                        Spacer()
                        Button("Choose…") {
                            pickCorpus { vm.corpus = $0 }
                        }
                        .disabled(vm.isRunning)
                    }
                }
                Text("File `.txt` / `.jsonl` o directory walkata ricorsivamente. " +
                     "Per JSONL ci si aspetta `{\"text\": \"...\"}` per linea.")
                    .font(.caption).foregroundStyle(.tertiary)
            }

            Section("Coverage") {
                LabeledContent("Soglia cumulativa") {
                    HStack {
                        Slider(value: $vm.coverage, in: 0.99...1.0, step: 0.0005)
                            .disabled(vm.isRunning)
                        Text(String(format: "%.4f", vm.coverage))
                            .font(.system(.body, design: .monospaced))
                            .frame(width: 60, alignment: .trailing)
                    }
                }
                Text("Frazione delle occorrenze del corpus che devono essere " +
                     "coperte dai top-K token. 0.9995 ≈ vocab finale ~32-50k " +
                     "(V4-Flash); 0.9999 ≈ vocab più grande, meno aggressivo.")
                    .font(.caption).foregroundStyle(.tertiary)
            }

            Section("Options") {
                Toggle("Dry-run (solo analisi, nessun output)",
                       isOn: $vm.dryRun)
                    .disabled(vm.isRunning)
            }
        }
        .formStyle(.grouped)
    }

    // ---- Progress / status ----

    @ViewBuilder
    private var progressArea: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Progress").font(.headline)
                Spacer()
                if vm.isRunning {
                    ProgressView().controlSize(.small)
                } else if vm.status.finishedAt != nil {
                    Label("Done", systemImage: "checkmark.seal.fill")
                        .foregroundStyle(.green)
                }
            }

            // Progress bar: significativa solo in Fase 2 (shard).
            if vm.status.shardsTotal > 0 {
                ProgressView(value: vm.progressFraction)
            } else {
                ProgressView()
                    .progressViewStyle(.linear)
                    .opacity(vm.isRunning ? 1 : 0.25)
            }

            // Status one-liner.
            HStack(spacing: 12) {
                statusSummary
                Spacer()
                if vm.status.bytesOut > 0 {
                    Text("\(formatBytes(vm.status.bytesIn)) → \(formatBytes(vm.status.bytesOut))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            // Coverage banner: visibile non appena Fase 1 emette
            // l'evento.
            if vm.status.totalVocab > 0 {
                HStack(spacing: 6) {
                    Image(systemName: "text.word.spacing")
                    Text(String(format:
                        "vocab: %d → %d (coverage %.2f%%)",
                        vm.status.totalVocab,
                        vm.status.keptVocab,
                        vm.status.coveragePct * 100))
                        .font(.callout.monospacedDigit())
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.accentColor.opacity(0.08))
                .cornerRadius(6)
            }

            // Errore.
            if let err = vm.lastError {
                Label(err, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .font(.callout)
                    .padding(.top, 4)
            }

            // Log.
            logView
        }
    }

    @ViewBuilder
    private var statusSummary: some View {
        if vm.status.shardsTotal > 0 {
            Text("shard \(vm.status.shardsWritten) / \(vm.status.shardsTotal)")
                .font(.callout.monospacedDigit())
        } else if vm.status.linesScanned > 0 {
            Text("scanning: \(vm.status.linesScanned) lines, " +
                 "\(vm.status.tokensScanned) tokens")
                .font(.callout.monospacedDigit())
        } else {
            Text(vm.isRunning ? "starting…" : "idle")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var logView: some View {
        ScrollViewReader { proxy in
            ScrollView(.vertical) {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(vm.status.logLines.enumerated()),
                             id: \.offset) { idx, line in
                        Text(line)
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                            .id(idx)
                    }
                }
                .padding(8)
            }
            .background(Color(NSColor.textBackgroundColor))
            .frame(minHeight: 120)
            .onChange(of: vm.status.logLines.count) { _, n in
                if n > 0 {
                    proxy.scrollTo(n - 1, anchor: .bottom)
                }
            }
        }
    }

    // ---- Footer ----

    @ViewBuilder
    private var footer: some View {
        HStack {
            Spacer()
            if vm.isRunning {
                Button("Cancel", role: .destructive) {
                    vm.cancel()
                }
                .keyboardShortcut(.cancelAction)
            } else {
                Button("Close") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button {
                    vm.start()
                } label: {
                    Label(vm.dryRun ? "Analyze (dry-run)" : "Start",
                          systemImage: "play.fill")
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!vm.canStart)
            }
        }
    }

    // ---- File pickers ----

    private func pickDirectory(onPick: @escaping (URL) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        if panel.runModal() == .OK, let u = panel.url {
            onPick(u)
        }
    }

    /// Per il corpus accettiamo file (.txt/.jsonl) E directory.
    private func pickCorpus(onPick: @escaping (URL) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = []   // qualunque tipo, il pruner
                                          // filtra per estensione
        panel.prompt = "Choose"
        if panel.runModal() == .OK, let u = panel.url {
            onPick(u)
        }
    }

    // ---- Helpers ----

    private func formatBytes(_ b: UInt64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useMB, .useGB]
        return formatter.string(fromByteCount: Int64(b))
    }
}
