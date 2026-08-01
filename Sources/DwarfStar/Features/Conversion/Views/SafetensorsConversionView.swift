import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct SafetensorsConversionView: View {
    @Bindable var controller: SafetensorsConversionController

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section("Safetensors → GGUF (DeepSeek V4)") {
                    Picker("Modalità", selection: $controller.mode) {
                        ForEach(ConversionMode.allCases) { Text($0.rawValue).tag($0) }
                    }
                    pathRow("Checkpoint Hugging Face", text: $controller.hfDirectory) {
                        chooseDirectory(title: "Scegli la directory del checkpoint") { controller.hfDirectory = $0 }
                    }
                    if controller.mode == .template {
                        pathRow("GGUF template", text: $controller.templatePath) {
                            chooseFile(title: "Scegli il GGUF template", extensions: ["gguf"]) { controller.templatePath = $0 }
                        }
                    }
                    pathRow("Output GGUF", text: $controller.outputPath) { chooseOutput() }
                    pathRow("Converter", text: $controller.converterPath) {
                        chooseFile(title: "Scegli deepseek4-quantize") { controller.converterPath = $0 }
                    }
                    Text("Il template fornisce metadata, tokenizer, ordine e forme logiche dei tensori; i payload vengono rigenerati dagli shard safetensors.")
                        .font(.caption).foregroundStyle(.secondary)
                }

                if controller.mode == .template {
                    Section("Ricetta di quantizzazione") {
                        HStack {
                            Button("Q2 + imatrix") { controller.applyQ2Recipe() }
                            Button("Q4") { controller.applyQ4Recipe() }
                            Button("Q8_K experts") { controller.applyQ8KRecipe() }
                        }
                        quantPicker("Esperti routed", selection: $controller.experts)
                        quantPicker("Routed W2", selection: $controller.routedW2)
                        quantPicker("Proiezioni attention", selection: $controller.attention)
                        quantPicker("Esperti shared", selection: $controller.shared)
                        quantPicker("Output", selection: $controller.output)
                        pathRow("Imatrix (opzionale)", text: $controller.imatrixPath) {
                            chooseFile(title: "Scegli l'imatrix", extensions: ["dat"]) { controller.imatrixPath = $0 }
                        }
                    }
                }

                Section("Controlli") {
                    Stepper("Thread: \(controller.threads)", value: $controller.threads, in: 1...256)
                    TextField("Confronta un solo tensore (opzionale)", text: $controller.compareTensor)
                    Toggle("Dry run (consigliato prima della conversione completa)", isOn: $controller.dryRun)
                    Toggle("Sovrascrivi un output esistente", isOn: $controller.overwrite)
                    HStack {
                        if controller.isRunning {
                            Button(role: .destructive) { controller.cancel() } label: {
                                Label("Interrompi", systemImage: "stop.fill")
                            }
                            ProgressView().controlSize(.small)
                        } else {
                            Button { controller.start() } label: {
                                Label(controller.dryRun ? "Verifica" : "Converti", systemImage: "arrow.triangle.2.circlepath")
                            }
                            .disabled(!controller.canStart)
                        }
                    }
                }
            }
            .formStyle(.grouped)

            Divider()
            ScrollView {
                Text(controller.log.isEmpty ? "Il comando e l'output del converter appariranno qui." : controller.log)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(10)
            }
            .frame(minHeight: 140, idealHeight: 220)
            .background(Color.black.opacity(0.05))
        }
    }

    @ViewBuilder
    private func pathRow(_ title: String, text: Binding<String>, browse: @escaping () -> Void) -> some View {
        HStack {
            TextField(title, text: text)
            Button("Scegli…", action: browse)
        }
    }

    @ViewBuilder
    private func quantPicker(_ title: String, selection: Binding<ConversionQuant>) -> some View {
        Picker(title, selection: selection) {
            ForEach(ConversionQuant.allCases) { Text($0.rawValue).tag($0) }
        }
    }

    private func chooseDirectory(title: String, assign: (String) -> Void) {
        let panel = NSOpenPanel()
        panel.title = title; panel.canChooseFiles = false; panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        _ = url.startAccessingSecurityScopedResource()
        assign(url.path)
    }

    private func chooseFile(title: String, extensions: [String] = [], assign: (String) -> Void) {
        let panel = NSOpenPanel()
        panel.title = title; panel.canChooseFiles = true; panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if !extensions.isEmpty {
            panel.allowedContentTypes = extensions.compactMap { UTType(filenameExtension: $0) } + [.data]
        }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        _ = url.startAccessingSecurityScopedResource()
        assign(url.path)
    }

    private func chooseOutput() {
        let panel = NSSavePanel()
        panel.title = "Scegli il GGUF di output"
        panel.nameFieldStringValue = "DeepSeek-V4-converted.gguf"
        panel.allowedContentTypes = [UTType(filenameExtension: "gguf") ?? .data]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        _ = url.startAccessingSecurityScopedResource()
        controller.outputPath = url.path
    }
}
