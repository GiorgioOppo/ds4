import SwiftUI

/// Control panel for the native in-process HTTP server (OpenAI-compatible API).
struct ServerView: View {
    @Bindable var controller: ServerController
    /// Whether a model is already loaded in-process for the chat.
    let modelLoadedInProcess: Bool

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section {
                    Label("Server HTTP nativo: espone il modello caricato in-process su un endpoint compatibile con l'API OpenAI. Nessun processo esterno.",
                          systemImage: "server.rack")
                        .font(.callout).foregroundStyle(.secondary)
                    if modelLoadedInProcess {
                        Label("La chat ha già un modello caricato. I pesi sono mmap condivisi (niente doppia copia in RAM), ma KV cache e GPU sono separate: usare chat e server insieme contende le risorse.",
                              systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange).font(.callout)
                    }
                }

                Section("Modello") {
                    HStack {
                        TextField("Modello GGUF", text: $controller.modelPath)
                            .disabled(controller.isRunning)
                        Button("Sfoglia") {
                            if let p = ModelPicker.pickGGUF() { controller.modelPath = p }
                        }
                        .disabled(controller.isRunning)
                    }
                    Stepper("Contesto: \(controller.contextSize) token",
                            value: $controller.contextSize, in: 1024...200_000, step: 1024)
                        .disabled(controller.isRunning)
                    Stepper("Max token per risposta: \(controller.maxTokens)",
                            value: $controller.maxTokens, in: 64...8192, step: 64)
                        .disabled(controller.isRunning)
                }

                Section("Rete") {
                    HStack {
                        TextField("Host", text: $controller.host)
                        TextField("Porta", value: $controller.port, format: .number.grouping(.never))
                            .frame(width: 80)
                    }
                    .disabled(controller.isRunning)
                    Toggle("CORS (Access-Control-Allow-Origin: *)", isOn: $controller.cors)
                        .disabled(controller.isRunning)
                }

                Section {
                    HStack(spacing: 12) {
                        if controller.isRunning {
                            Button(role: .destructive) { controller.stop() } label: {
                                Label("Ferma server", systemImage: "stop.fill")
                            }
                            Label("In ascolto su \(controller.endpoint)", systemImage: "dot.radiowaves.left.and.right")
                                .foregroundStyle(.green).font(.callout)
                        } else if controller.isLoading {
                            ProgressView().controlSize(.small)
                            Text("Caricamento modello…").font(.callout).foregroundStyle(.secondary)
                        } else {
                            Button { controller.start() } label: {
                                Label("Avvia server", systemImage: "play.fill")
                            }
                        }
                    }
                }

                if controller.isRunning {
                    Section("Esempio") {
                        Text(curlExample)
                            .font(.system(.caption2, design: .monospaced))
                            .textSelection(.enabled)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Endpoint supportati") {
                    endpointRow("GET", "/v1/models", "elenco modelli")
                    endpointRow("POST", "/v1/chat/completions", "chat (stream + non-stream)")
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
                .frame(height: 150)
                .background(Color.black.opacity(0.05))
            }
        }
    }

    private func endpointRow(_ method: String, _ path: String, _ desc: String) -> some View {
        HStack(spacing: 8) {
            Text(method)
                .font(.caption2.bold().monospaced())
                .padding(.horizontal, 6).padding(.vertical, 2)
                .background(Color.accentColor.opacity(0.15))
                .clipShape(RoundedRectangle(cornerRadius: 4))
            Text(path).font(.system(.caption, design: .monospaced))
            Spacer()
            Text(desc).font(.caption).foregroundStyle(.secondary)
        }
    }

    private var curlExample: String {
        """
        curl \(controller.endpoint)/chat/completions \\
          -H "Content-Type: application/json" \\
          -d '{"model":"deepseek-v4-flash","stream":true,
               "messages":[{"role":"user","content":"Ciao"}]}'
        """
    }
}
