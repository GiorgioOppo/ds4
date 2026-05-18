import SwiftUI
import AppKit
import UniformTypeIdentifiers
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
    @StateObject private var history = VocabPruneHistory()
    @Environment(\.dismiss) private var dismiss
    @State private var corpusDropTargeted: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            HStack(spacing: 0) {
                VStack(spacing: 0) {
                    ScrollView {
                        form
                            .padding(20)
                    }
                    .frame(minHeight: 280)
                    Divider()
                    progressArea
                        .padding(16)
                        .frame(minHeight: 240)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                Divider()
                historyPane
                    .frame(width: 240)
                    .background(Color(NSColor.controlBackgroundColor))
            }
            Divider()
            footer
                .padding(12)
        }
        .frame(minWidth: 960, minHeight: 700)
        .onAppear {
            vm.history = history
            vm.refreshCheckpointInfo()
        }
        .onChange(of: vm.inputDir) { _, _ in vm.refreshCheckpointInfo() }
        .onChange(of: vm.outputDir) { _, _ in vm.refreshCheckpointInfo() }
        .onChange(of: vm.corpus) { _, _ in vm.refreshCheckpointInfo() }
        .onChange(of: vm.coverage) { _, _ in vm.refreshCheckpointInfo() }
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
                        Text(vm.corpus?.path ?? (corpusDropTargeted ? "Drop here…" : "—"))
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
                    .padding(4)
                    .background(corpusDropTargeted
                                 ? Color.accentColor.opacity(0.18)
                                 : Color.clear)
                    .cornerRadius(4)
                    .onDrop(of: [.fileURL],
                            isTargeted: $corpusDropTargeted) { providers in
                        guard let provider = providers.first, !vm.isRunning
                        else { return false }
                        _ = provider.loadObject(ofClass: URL.self) { url, _ in
                            guard let url = url else { return }
                            DispatchQueue.main.async {
                                vm.corpus = url
                            }
                        }
                        return true
                    }
                }
                Text("File `.txt` / `.jsonl` o directory walkata ricorsivamente. " +
                     "Per JSONL ci si aspetta `{\"text\": \"...\"}` per linea. " +
                     "Anche drag-and-drop da Finder.")
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

                HStack {
                    Text("Concurrency (Fase 1)")
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Stepper(value: $vm.concurrency,
                             in: 1...max(1, ProcessInfo.processInfo.activeProcessorCount * 2),
                             step: 1) {
                        Text("\(vm.concurrency) thread\(vm.concurrency == 1 ? "" : "s")")
                            .font(.system(.body, design: .monospaced))
                            .frame(width: 110, alignment: .trailing)
                    }
                    .disabled(vm.isRunning)
                }
                Text("Tokenizzazione del corpus in parallelo (un thread " +
                     "per file). 1 = sequenziale con save intra-file ogni " +
                     "10k token. Default = \(VocabPruneSpec.defaultConcurrency) " +
                     "(80% dei \(ProcessInfo.processInfo.activeProcessorCount) " +
                     "core attivi).")
                    .font(.caption)
                    .foregroundStyle(.tertiary)

                Toggle("Resume from checkpoint when compatible",
                       isOn: $vm.resumeEnabled)
                    .disabled(vm.isRunning)
                Text("Quando attivo, legge `<output>/.vocab_pruner_checkpoint.json` " +
                     "e riprende invece di ricominciare. Lo spec hash protegge " +
                     "da resume con corpus / coverage diversi.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }

            // Banner che mostra il checkpoint compatibile se trovato.
            if let info = vm.checkpointInfo {
                Section {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "arrow.clockwise.circle.fill")
                            .foregroundStyle(.green)
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Checkpoint compatibile trovato").font(.callout.bold())
                            Text("Phase: \(info.phase). Saved: " +
                                 "\(info.savedAt.formatted(date: .abbreviated, time: .shortened)).")
                                .font(.caption)
                            if info.analyzerFiles > 0 {
                                Text("Analyzer progress: \(info.analyzerFiles) files, " +
                                     "\(info.analyzerLines) lines")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            if info.rewriterShards > 0 {
                                Text("Rewriter progress: \(info.rewriterShards) shard(s)")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        Spacer()
                        Button("Reset") {
                            vm.resetCheckpoint()
                        }
                        .controlSize(.small)
                        .disabled(vm.isRunning)
                    }
                    .padding(.vertical, 4)
                }
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

            // Tabella preview dei token droppati (top-N dal corpus).
            if let decision = vm.lastDecision, !decision.previewDropped.isEmpty {
                droppedTokensTable(decision: decision)
            }

            // Log.
            logView
        }
    }

    @ViewBuilder
    private func droppedTokensTable(decision: KeepDecision) -> some View {
        DisclosureGroup {
            ScrollView(.vertical) {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(decision.previewDropped.prefix(20), id: \.id) { entry in
                        HStack(spacing: 8) {
                            Text("'\(entry.content)'")
                                .font(.caption.monospaced())
                                .lineLimit(1)
                            Spacer()
                            Text("id=\(entry.id)")
                                .font(.caption2.monospaced())
                                .foregroundStyle(.tertiary)
                            Text("×\(entry.count)")
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                                .frame(width: 60, alignment: .trailing)
                        }
                    }
                    if decision.previewDropped.count > 20 {
                        Text("…and \(decision.previewDropped.count - 20) more")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
                .padding(.vertical, 4)
            }
            .frame(maxHeight: 160)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "trash.slash")
                Text("Top \(min(20, decision.previewDropped.count)) " +
                     "dropped tokens by frequency")
                    .font(.callout)
            }
        }
    }

    // ---- History pane ----

    @ViewBuilder
    private var historyPane: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("History").font(.subheadline.bold())
                Spacer()
                if !history.records.isEmpty {
                    Button {
                        history.clear()
                    } label: {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(.borderless)
                    .controlSize(.small)
                    .help("Clear history")
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            Divider()

            if history.records.isEmpty {
                VStack(spacing: 4) {
                    Image(systemName: "clock.badge.questionmark")
                        .font(.system(size: 24))
                        .foregroundStyle(.tertiary)
                    Text("No prior runs")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(spacing: 6) {
                        ForEach(history.records) { rec in
                            historyRow(rec)
                        }
                    }
                    .padding(8)
                }
            }
        }
    }

    @ViewBuilder
    private func historyRow(_ rec: VocabPruneRecord) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(rec.timestamp.formatted(date: .abbreviated,
                                              time: .shortened))
                    .font(.caption.bold())
                Spacer()
                if rec.dryRun {
                    Text("dry")
                        .font(.caption2.bold())
                        .padding(.horizontal, 4)
                        .background(Color.gray.opacity(0.2))
                        .cornerRadius(3)
                }
            }
            Text((rec.outputDir as NSString).lastPathComponent)
                .font(.caption.monospaced())
                .lineLimit(1)
                .truncationMode(.middle)
            HStack(spacing: 4) {
                Text("\(rec.oldVocabSize)").font(.caption2.monospaced())
                Image(systemName: "arrow.right").font(.system(size: 8))
                Text("\(rec.newVocabSize)").font(.caption2.monospaced())
                Spacer()
                if rec.bytesOut > 0 {
                    Text(formatBytes(rec.bytesOut))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(6)
        .background(Color(NSColor.textBackgroundColor))
        .cornerRadius(4)
        .contextMenu {
            Button("Reuse paths") {
                vm.inputDir = URL(fileURLWithPath: rec.inputDir)
                vm.outputDir = URL(fileURLWithPath: rec.outputDir)
                if let c = rec.corpus {
                    vm.corpus = URL(fileURLWithPath: c)
                }
                vm.coverage = rec.coverage
            }
            Button("Remove", role: .destructive) {
                history.remove(rec.id)
            }
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
