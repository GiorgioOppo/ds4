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
                    Label("Native HTTP server: exposes the in-process loaded model through an OpenAI-compatible endpoint. No external process.",
                          systemImage: "server.rack")
                        .font(.callout).foregroundStyle(.secondary)
                    if modelLoadedInProcess {
                        Label("Chat already has a model loaded. Weights are shared through mmap (no second RAM copy), but KV cache and GPU work are separate: using chat and server together competes for resources.",
                              systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange).font(.callout)
                    }
                }

                Section("Model (from Settings)") {
                    LabeledContent("GGUF", value: (controller.modelPath as NSString).lastPathComponent)
                    Stepper("Context: \(controller.contextSize) tokens",
                            value: $controller.contextSize, in: 1024...200_000, step: 1024)
                        .disabled(controller.isRunning)
                    Stepper("Max tokens per response: \(controller.maxTokens)",
                            value: $controller.maxTokens, in: 64...8192, step: 64)
                        .disabled(controller.isRunning)
                }

                Section("Network") {
                    HStack {
                        TextField("Host", text: $controller.host)
                        TextField("Port", value: $controller.port, format: .number.grouping(.never))
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
                                Label("Stop Server", systemImage: "stop.fill")
                            }
                            Label("Listening on \(controller.endpoint)", systemImage: "dot.radiowaves.left.and.right")
                                .foregroundStyle(.green).font(.callout)
                        } else if controller.isLoading {
                            ProgressView().controlSize(.small)
                            Text("Loading model...").font(.callout).foregroundStyle(.secondary)
                        } else {
                            Button { controller.start() } label: {
                                Label("Start Server", systemImage: "play.fill")
                            }
                        }
                    }
                }

                if controller.isRunning {
                    Section("Example") {
                        Text(curlExample)
                            .font(.system(.caption2, design: .monospaced))
                            .textSelection(.enabled)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Supported Endpoints") {
                    endpointRow("GET", "/v1/models", "model list")
                    endpointRow("POST", "/v1/chat/completions", "OpenAI chat (stream + non)")
                    endpointRow("POST", "/v1/responses", "OpenAI Responses (stream + non)")
                    endpointRow("POST", "/v1/messages", "Anthropic Messages (stream + non)")
                    endpointRow("POST", "/v1/completions", "OpenAI legacy completion")
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
               "messages":[{"role":"user","content":"Hello"}]}'
        """
    }
}
