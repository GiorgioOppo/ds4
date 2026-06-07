import SwiftUI

/// Control panel for the OpenAI/Anthropic-compatible `ds4-server` subprocess and
/// a launcher for the interactive `ds4-agent`.
struct ServerView: View {
    @Bindable var controller: ServerController
    /// Whether a model is currently loaded in-process (for the double-load warning).
    let modelLoadedInProcess: Bool

    var body: some View {
        VStack(spacing: 0) {
            Form {
                if modelLoadedInProcess {
                    Section {
                        Label("Un modello è già caricato in-process per la chat. Avviare il server caricherà di nuovo i pesi: su una macchina dove il modello entra una sola volta rischi l'esaurimento di memoria.",
                              systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                            .font(.callout)
                    }
                }

                Section("Server HTTP (ds4-server)") {
                    TextField("Binario", text: $controller.binaryPath)
                    TextField("Modello GGUF", text: $controller.modelPath)
                    HStack {
                        TextField("Host", text: $controller.host)
                        TextField("Porta", value: $controller.port, format: .number.grouping(.never))
                            .frame(width: 80)
                    }
                    Stepper("Contesto: \(controller.contextSize) token",
                            value: $controller.contextSize, in: 1024...1_000_000, step: 1024)
                    Toggle("CORS", isOn: $controller.cors)
                }

                Section("Disk KV cache (opzionale)") {
                    TextField("Cartella KV su disco", text: $controller.kvDiskDir)
                    if !controller.kvDiskDir.isEmpty {
                        Stepper("Spazio: \(controller.kvDiskSpaceMB) MB",
                                value: $controller.kvDiskSpaceMB, in: 256...262144, step: 256)
                    }
                }

                Section("SSD streaming (opzionale)") {
                    Toggle("Abilita streaming expert", isOn: $controller.streamingEnabled)
                    if controller.streamingEnabled {
                        TextField("Budget cache (es. 32GB)", text: $controller.streamingCacheSpec)
                    }
                }

                Section {
                    HStack {
                        if controller.isRunning {
                            Button(role: .destructive) { controller.stop() } label: {
                                Label("Ferma server", systemImage: "stop.fill")
                            }
                            Label("In ascolto su \(controller.endpoint)", systemImage: "dot.radiowaves.left.and.right")
                                .foregroundStyle(.green).font(.callout)
                        } else {
                            Button { controller.start() } label: {
                                Label("Avvia server", systemImage: "play.fill")
                            }
                        }
                    }
                }

                Section("Agent di coding (ds4-agent)") {
                    Text("L'agent è un programma interattivo da terminale. Aprilo in una finestra del Terminale:")
                        .font(.callout).foregroundStyle(.secondary)
                    Button {
                        AgentLauncher.openInTerminal(projectDir: controller.workingDir)
                    } label: {
                        Label("Apri agent nel Terminale", systemImage: "terminal")
                    }
                }
            }
            .formStyle(.grouped)

            if !controller.log.isEmpty {
                Divider()
                ScrollView {
                    Text(controller.log)
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .padding(8)
                }
                .frame(height: 180)
                .background(Color.black.opacity(0.05))
            }
        }
    }
}
