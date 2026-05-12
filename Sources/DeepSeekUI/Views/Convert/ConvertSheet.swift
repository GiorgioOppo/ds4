import SwiftUI
import AppKit
import DeepSeekConverter

/// Modal sheet for converting a checkpoint directory. Two panes:
///   - top: form (source / dest pickers + target dtype + n-experts
///     + shard size + model-parallel)
///   - bottom: live progress (status line, progress bar, scrolling
///     log)
/// The Start button kicks off `Converter.runQuantize` via the
/// ConvertViewModel; while running the form is locked and a Cancel
/// button replaces Start.
struct ConvertSheet: View {
    @StateObject private var vm = ConvertViewModel()
    @Environment(\.dismiss) private var dismiss
    @AppStorage(AppSettingsKey.converterBinaryPath)
    private var converterBinaryPath: String = ""

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
            Image(systemName: "wand.and.stars")
                .font(.title2)
                .foregroundStyle(.tint)
            VStack(alignment: .leading) {
                Text("Convert checkpoint").font(.headline)
                Text("Quantize HuggingFace weights or transcode an existing INT8 checkpoint to a smaller dtype.")
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
            .buttonStyle(.plain)
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
                        Text(vm.sourcePath?.path ?? "—")
                            .font(.system(.body, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .foregroundStyle(vm.sourcePath == nil ? .secondary : .primary)
                        Spacer()
                        Button("Choose…") {
                            pickDirectory { vm.sourcePath = $0 }
                        }
                        .disabled(vm.isRunning)
                    }
                }
                Text("HF-format checkpoint (`*.safetensors`, `config.json`, `tokenizer.json`) — *or* a previously-converted INT8 directory if you're transcoding to INT4 / INT2.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            Section("Destination") {
                LabeledContent("Output directory") {
                    HStack {
                        Text(vm.destPath?.path ?? "—")
                            .font(.system(.body, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .foregroundStyle(vm.destPath == nil ? .secondary : .primary)
                        Spacer()
                        Button("Choose…") {
                            pickDirectory { vm.destPath = $0 }
                        }
                        .disabled(vm.isRunning)
                    }
                }
                Text("Created if it doesn't exist. Must be different from the source — the converter never writes into its own input dir.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            Section("Target dtype") {
                Picker("Target", selection: $vm.target) {
                    Text("BF16 (fuse FP8/FP4 → 2 bytes)").tag(ConversionTarget.bf16)
                    Text("F16  (fuse FP8/FP4 → 2 bytes)").tag(ConversionTarget.f16)
                    Text("INT8 (W8A16, ~½ × BF16)").tag(ConversionTarget.int8)
                    Text("INT4 (W4A16, ~¼ × BF16)").tag(ConversionTarget.int4)
                    Text("INT2 (W2A16, ~⅛ × BF16, brutal accuracy hit)")
                        .tag(ConversionTarget.int2)
                    Text("Keep (preserve FP8/FP4 native)").tag(ConversionTarget.keep)
                }
                .disabled(vm.isRunning)
                .pickerStyle(.menu)
                Text(targetCaption(for: vm.target))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Section("Sharding & MoE") {
                LabeledContent("Routed experts") {
                    Stepper(value: $vm.nExperts, in: 1...512, step: 1) {
                        Text("\(vm.nExperts)").font(.system(.body, design: .monospaced))
                    }
                    .disabled(vm.isRunning)
                }
                LabeledContent("Shard size (GB)") {
                    HStack {
                        Slider(value: $vm.shardSizeGB, in: 1...20, step: 0.5)
                            .disabled(vm.isRunning)
                        Text(String(format: "%.1f", vm.shardSizeGB))
                            .frame(width: 44, alignment: .trailing)
                            .font(.system(.body, design: .monospaced))
                    }
                }
                LabeledContent("Model-parallel input") {
                    Stepper(value: $vm.modelParallel, in: 1...8) {
                        Text("\(vm.modelParallel)").font(.system(.body, design: .monospaced))
                    }
                    .disabled(vm.isRunning)
                }
                Text("HF V4 ships with `mp=1` — leave at 1 unless you have an explicitly-sharded source.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
        .formStyle(.grouped)
    }

    private func targetCaption(for t: ConversionTarget) -> String {
        switch t {
        case .bf16, .f16:
            return "All FP8/FP4 weights + their E8M0 scales are fused into dense \(t.rawValue.uppercased()). Output is the largest dtype option — roughly 2× input for FP8 source, 4× for FP4."
        case .int8:
            return "Linear weights → INT8 + per-row F16 group scale (128-element blocks). Disk ≈ ½ × BF16. Non-Linear tensors (norms, biases, attn_sink) stay BF16."
        case .int4:
            return "INT4 packed two-per-byte. Disk ≈ ¼ × BF16. RTN simmetrico [-8, 7]; accuracy drop bigger than INT8. Accepts INT8 input for re-quantize."
        case .int2:
            return "INT2 packed four-per-byte. Disk ≈ ⅛ × BF16. RTN over [-2, 1] is brutal — use for memory-bound experiments, not production. Accepts INT8 input."
        case .keep:
            return "No dtype change — FP8/FP4 weights are passed through verbatim (just renamed). Tokenizer + config are still copied."
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

            if vm.status.total > 0 {
                ProgressView(value: vm.status.fraction)
                    .progressViewStyle(.linear)
            } else {
                ProgressView().progressViewStyle(.linear)
                    .opacity(vm.isRunning ? 1 : 0.25)
            }

            HStack(spacing: 12) {
                Text(vm.status.phase.isEmpty ? "—" : vm.status.phase)
                    .font(.caption).foregroundStyle(.secondary)
                if vm.status.total > 0 {
                    Text("\(vm.status.completed) / \(vm.status.total)")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if vm.status.outputBytes > 0 {
                    Text(formatBytes(vm.status.outputBytes) + " written")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
            }

            if let err = vm.lastError {
                Label(err, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
            }

            logView
        }
    }

    @ViewBuilder
    private var logView: some View {
        ScrollViewReader { proxy in
            ScrollView(.vertical) {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(vm.status.logLines.enumerated()), id: \.offset) { idx, line in
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
            .frame(minHeight: 120)
            .onChange(of: vm.status.logLines.count) { _, n in
                if n > 0 { proxy.scrollTo(n - 1, anchor: .bottom) }
            }
        }
    }

    // ---- Footer ----

    @ViewBuilder
    private var footer: some View {
        HStack {
            if converterBinaryPath.isEmpty {
                Text("Converter binary: auto-detect")
                    .font(.caption).foregroundStyle(.tertiary)
            } else {
                Text("Binary: \(converterBinaryPath)")
                    .font(.system(.caption, design: .monospaced))
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .foregroundStyle(.tertiary)
            }
            Spacer()
            if vm.isRunning {
                Button("Cancel", role: .destructive) {
                    vm.cancel()
                }
            } else {
                Button("Close") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button {
                    vm.start(binaryOverridePath:
                                converterBinaryPath.isEmpty ? nil : converterBinaryPath)
                } label: {
                    Label("Start", systemImage: "play.fill")
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!vm.canStart)
            }
        }
    }

    // ---- Helpers ----

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

    private func formatBytes(_ b: UInt64) -> String {
        let gib = 1024.0 * 1024.0 * 1024.0
        if Double(b) >= gib { return String(format: "%.2f GB", Double(b) / gib) }
        let mib = 1024.0 * 1024.0
        return String(format: "%.2f MB", Double(b) / mib)
    }
}
