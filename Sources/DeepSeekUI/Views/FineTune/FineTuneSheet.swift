import SwiftUI
import AppKit
import DeepSeekTraining

/// Modal sheet for fine-tuning a local DeepSeek checkpoint. Same
/// shape as `ConvertSheet`:
///   - top: header
///   - middle: form (paths + dataset format + hyperparameters)
///   - bottom: live progress (status line, progress bar, scrolling
///     log)
/// The Start button kicks off `FineTuner.run` via the view model;
/// while running the form is locked and a Cancel button replaces
/// Start. The current runner is a stub (validates inputs + emits a
/// plan + throws `FineTuneNotImplemented`), so a "scaffold not wired"
/// banner is rendered in place of the error chrome when the run
/// finishes with that specific error.
struct FineTuneSheet: View {
    @StateObject private var vm = FineTuneViewModel()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                form.padding(20)
            }
            .frame(minHeight: 320)
            Divider()
            progressArea
                .padding(16)
                .frame(minHeight: 240)
            Divider()
            footer.padding(12)
        }
        .frame(minWidth: 760, minHeight: 720)
    }

    // ---- Header ----

    @ViewBuilder
    private var header: some View {
        HStack {
            Image(systemName: "graduationcap")
                .font(.title2)
                .foregroundStyle(.tint)
            VStack(alignment: .leading) {
                Text("Fine-tune model").font(.headline)
                Text("Full fine-tuning of a local DeepSeek checkpoint — all weights are updated. Set the base model, training data, and hyperparameters; the runner validates the spec and emits a step-by-step plan.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
            Button {
                if vm.isRunning { vm.cancel() }
                dismiss()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
        .padding(16)
    }

    // ---- Form ----

    @ViewBuilder
    private var form: some View {
        Form {
            paths
            datasetSection
            hyperparameters
            scheduleAndCheckpoints
        }
        .formStyle(.grouped)
    }

    @ViewBuilder
    private var paths: some View {
        Section("Paths") {
            pathRow(label: "Base model",
                    placeholder: "Pick a converted DeepSeek directory…",
                    url: vm.baseModelPath) { vm.baseModelPath = $0 }
            Text("HF-format or converted checkpoint (`config.json`, `tokenizer.json`, `*.safetensors`).")
                .font(.caption)
                .foregroundStyle(.tertiary)

            pathRow(label: "Training data",
                    placeholder: "Pick a JSONL or text file…",
                    url: vm.datasetPath,
                    pickFiles: true) { vm.datasetPath = $0 }

            pathRow(label: "Eval data (optional)",
                    placeholder: "Hold-out split is used when empty",
                    url: vm.evalDatasetPath,
                    pickFiles: true,
                    clearable: true) { vm.evalDatasetPath = $0 }

            pathRow(label: "Output dir",
                    placeholder: "Where the fine-tuned checkpoint is written…",
                    url: vm.outputPath) { vm.outputPath = $0 }
            Text("Must be different from the base model directory — the runner refuses to overwrite the source.")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder
    private var datasetSection: some View {
        Section("Dataset format") {
            Picker("Layout", selection: $vm.format) {
                ForEach(DatasetFormat.allCases, id: \.self) { f in
                    Text(f.displayName).tag(f)
                }
            }
            .disabled(vm.isRunning)
            .pickerStyle(.menu)
            Text(formatCaption(for: vm.format))
                .font(.caption)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)

            Picker("Precision", selection: $vm.precision) {
                ForEach(TrainingPrecision.allCases, id: \.self) { p in
                    Text(p.displayName).tag(p)
                }
            }
            .disabled(vm.isRunning)
            .pickerStyle(.segmented)
            Text(vm.precision.detail)
                .font(.caption)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private var hyperparameters: some View {
        Section("Optimizer") {
            Picker("Optimizer", selection: $vm.optimizer) {
                ForEach(FineTuneOptimizer.allCases, id: \.self) { o in
                    Text(o.displayName).tag(o)
                }
            }
            .disabled(vm.isRunning)
            .pickerStyle(.segmented)

            LabeledContent("Learning rate") {
                HStack {
                    TextField("5e-5", text: lrTextBinding)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(.body, design: .monospaced))
                        .frame(width: 120)
                        .disabled(vm.isRunning)
                    Stepper("",
                             value: $vm.learningRate,
                             in: 1e-7...1e-2,
                             step: 1e-5)
                        .labelsHidden()
                        .disabled(vm.isRunning)
                }
            }
            LabeledContent("Weight decay") {
                HStack {
                    Slider(value: $vm.weightDecay, in: 0...0.5, step: 0.005)
                        .disabled(vm.isRunning)
                    Text(String(format: "%.3f", vm.weightDecay))
                        .frame(width: 60, alignment: .trailing)
                        .font(.system(.body, design: .monospaced))
                }
            }
        }
        Section("Batch & sequence") {
            LabeledContent("Epochs") {
                Stepper(value: $vm.epochs, in: 1...50) {
                    Text("\(vm.epochs)").font(.system(.body, design: .monospaced))
                }
                .disabled(vm.isRunning)
            }
            LabeledContent("Micro-batch size") {
                Stepper(value: $vm.batchSize, in: 1...64) {
                    Text("\(vm.batchSize)").font(.system(.body, design: .monospaced))
                }
                .disabled(vm.isRunning)
            }
            LabeledContent("Grad-accum steps") {
                Stepper(value: $vm.gradientAccumulationSteps, in: 1...256) {
                    Text("\(vm.gradientAccumulationSteps)")
                        .font(.system(.body, design: .monospaced))
                }
                .disabled(vm.isRunning)
            }
            LabeledContent("Effective batch") {
                Text("\(vm.batchSize * vm.gradientAccumulationSteps)")
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            LabeledContent("Max seq length") {
                HStack {
                    Slider(value: Binding(
                        get: { Double(vm.maxSequenceLength) },
                        set: { vm.maxSequenceLength = Int($0) }
                    ), in: 256...32768, step: 256)
                    .disabled(vm.isRunning)
                    Text("\(vm.maxSequenceLength)")
                        .frame(width: 64, alignment: .trailing)
                        .font(.system(.body, design: .monospaced))
                }
            }
        }
    }

    @ViewBuilder
    private var scheduleAndCheckpoints: some View {
        Section("Schedule & checkpoints") {
            LabeledContent("Warmup steps") {
                Stepper(value: $vm.warmupSteps, in: 0...10000, step: 10) {
                    Text("\(vm.warmupSteps)").font(.system(.body, design: .monospaced))
                }
                .disabled(vm.isRunning)
            }
            LabeledContent("Eval split") {
                HStack {
                    Slider(value: $vm.evalSplit, in: 0...0.3, step: 0.01)
                        .disabled(vm.isRunning || vm.evalDatasetPath != nil)
                    Text(String(format: "%.0f%%", vm.evalSplit * 100))
                        .frame(width: 44, alignment: .trailing)
                        .font(.system(.body, design: .monospaced))
                }
            }
            if vm.evalDatasetPath != nil {
                Text("Eval split is ignored when an explicit eval dataset is provided above.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            LabeledContent("Save every (steps)") {
                Stepper(value: $vm.saveEverySteps, in: 0...100000, step: 50) {
                    Text(vm.saveEverySteps == 0 ? "off"
                          : "\(vm.saveEverySteps)")
                        .font(.system(.body, design: .monospaced))
                }
                .disabled(vm.isRunning)
            }
            LabeledContent("Seed") {
                TextField("42", text: seedTextBinding)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.body, design: .monospaced))
                    .frame(width: 120)
                    .disabled(vm.isRunning)
            }
        }
    }

    @ViewBuilder
    private func pathRow(label: String,
                          placeholder: String,
                          url: URL?,
                          pickFiles: Bool = false,
                          clearable: Bool = false,
                          onPick: @escaping (URL?) -> Void) -> some View {
        LabeledContent(label) {
            HStack {
                Text(url?.path ?? placeholder)
                    .font(.system(.body, design: .monospaced))
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .foregroundStyle(url == nil ? .secondary : .primary)
                Spacer()
                if clearable, url != nil {
                    Button {
                        onPick(nil)
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .disabled(vm.isRunning)
                }
                Button("Choose…") {
                    pick(filesOnly: pickFiles) { onPick($0) }
                }
                .disabled(vm.isRunning)
            }
        }
    }

    // ---- Progress + log ----

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
                        .font(.callout)
                }
            }

            if vm.status.totalSteps > 0 {
                ProgressView(value: vm.status.fraction)
                    .progressViewStyle(.linear)
            } else {
                ProgressView().progressViewStyle(.linear)
                    .opacity(vm.isRunning ? 1 : 0.25)
            }

            HStack(spacing: 16) {
                if vm.status.totalSteps > 0 {
                    metric(label: "Step",
                            value: "\(vm.status.step) / \(vm.status.totalSteps)")
                    metric(label: "Epoch",
                            value: "\(vm.status.currentEpoch)")
                    metric(label: "Loss",
                            value: String(format: "%.4f", vm.status.lastLoss))
                    metric(label: "LR",
                            value: String(format: "%.2e",
                                            vm.status.lastLearningRate))
                } else if vm.status.examples > 0 {
                    metric(label: "Examples",
                            value: "\(vm.status.examples)")
                }
                if let ppl = vm.status.lastPerplexity {
                    metric(label: "PPL",
                            value: String(format: "%.2f", ppl))
                }
                Spacer()
            }

            if let notice = vm.notImplementedNotice {
                notImplementedBanner(notice)
            } else if let err = vm.lastError {
                Label(err, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
            }

            logView
        }
    }

    @ViewBuilder
    private func metric(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.system(.body, design: .monospaced))
        }
    }

    @ViewBuilder
    private func notImplementedBanner(_ detail: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "info.circle.fill")
                .foregroundStyle(.orange)
                .imageScale(.large)
            VStack(alignment: .leading, spacing: 4) {
                Text("Native trainer not wired yet")
                    .font(.subheadline.bold())
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(10)
        .background(Color.orange.opacity(0.12),
                     in: RoundedRectangle(cornerRadius: 8))
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
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(idx)
                    }
                    if vm.status.logLines.isEmpty {
                        Text("(no output yet)")
                            .font(.caption).foregroundStyle(.tertiary)
                    }
                }
                .padding(8)
            }
            .background(Color(NSColor.textBackgroundColor),
                         in: RoundedRectangle(cornerRadius: 8))
            .frame(minHeight: 140)
            .onChange(of: vm.status.logLines.count) { _, n in
                if n > 0 { proxy.scrollTo(n - 1, anchor: .bottom) }
            }
        }
    }

    // ---- Footer ----

    @ViewBuilder
    private var footer: some View {
        HStack {
            Text(canStartHint)
                .font(.caption)
                .foregroundStyle(.tertiary)
            Spacer()
            if vm.isRunning {
                Button("Cancel", role: .destructive) {
                    vm.cancel()
                }
            } else {
                Button("Close") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button {
                    vm.start()
                } label: {
                    Label("Start", systemImage: "play.fill")
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!vm.canStart)
            }
        }
    }

    private var canStartHint: String {
        if vm.canStart { return "Ready — Start kicks off the validation + plan." }
        if vm.baseModelPath == nil { return "Pick a base model directory to continue." }
        if vm.datasetPath == nil  { return "Pick a training dataset to continue." }
        if vm.outputPath == nil   { return "Pick an output directory to continue." }
        return "Check paths and hyperparameters."
    }

    // ---- Helpers ----

    /// String-backed binding for the learning-rate field. Lets the
    /// user type "5e-5" or "0.00005" interchangeably; rejects junk
    /// silently (keeps the previous value).
    private var lrTextBinding: Binding<String> {
        Binding(
            get: { String(format: "%.2e", vm.learningRate) },
            set: { text in
                let trimmed = text.trimmingCharacters(in: .whitespaces)
                if let d = Double(trimmed), d > 0 {
                    vm.learningRate = d
                }
            }
        )
    }

    /// Same idea for the seed (UInt64 doesn't play well with the
    /// built-in .number format style on every platform).
    private var seedTextBinding: Binding<String> {
        Binding(
            get: { String(vm.seed) },
            set: { text in
                let trimmed = text.trimmingCharacters(in: .whitespaces)
                if let v = UInt64(trimmed) {
                    vm.seed = v
                }
            }
        )
    }

    private func formatCaption(for f: DatasetFormat) -> String {
        switch f {
        case .jsonlChat:
            return "One JSON object per line with `messages: [{role, content}, …]`. The OpenAI fine-tuning format — same shape the chat UI emits."
        case .jsonlPromptCompletion:
            return "One JSON object per line with `prompt` + `completion`. Legacy completion format; useful for instruction-tuning corpora."
        case .plainText:
            return "Plain UTF-8 text. Tokenized in max-seq-len chunks; no role / turn structure."
        }
    }

    private func pick(filesOnly: Bool,
                       onPick: @escaping (URL?) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = !filesOnly
        panel.canChooseFiles = filesOnly
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        if panel.runModal() == .OK, let u = panel.url {
            onPick(u)
        }
    }
}
