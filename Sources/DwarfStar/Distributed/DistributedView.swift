import SwiftUI

/// This Mac as a distributed WORKER: owns a layer slice and listens for the
/// coordinator (the Distribuito sidebar tab). The coordinator lives in the Chat
/// tab → Distribuito mode.
struct WorkerView: View {
    @Bindable var controller: DistributedController

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section {
                    Label("Questo Mac come WORKER: possiede uno slice di layer e resta in ascolto del coordinatore. Avvia i worker, poi connetti il coordinatore (scheda Chat → Distribuito).",
                          systemImage: "rectangle.3.group")
                        .font(.callout).foregroundStyle(.secondary)
                }
                Section("Modello") {
                    HStack {
                        TextField("Modello GGUF", text: $controller.modelPath)
                        Button("Sfoglia") { if let p = ModelPicker.pickGGUF() { controller.modelPath = p } }
                    }
                    Stepper("Contesto: \(controller.contextSize) token",
                            value: $controller.contextSize, in: 1024...200_000, step: 1024)
                }
                .disabled(controller.workerRunning || controller.workerLoading)

                Section("Worker — modello da \(controller.modelLayers) layer (0…\(controller.modelLayers - 1))") {
                    TextField("Porta", value: $controller.port, format: .number.grouping(.never))
                    Stepper("Primo layer: \(controller.layerStart)", value: $controller.layerStart,
                            in: 0...(controller.modelLayers - 1))
                    Stepper("Ultimo layer: \(controller.layerEnd)", value: $controller.layerEnd,
                            in: controller.layerStart...(controller.modelLayers - 1))
                    Toggle("Possiede l'output head (ultimo slice)", isOn: $controller.hasOutput)
                    Text("La route dei worker deve coprire tutti i \(controller.modelLayers) layer in modo contiguo. Su un solo Mac: un worker 0…\(controller.modelLayers - 1) con output head.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .disabled(controller.workerRunning)

                Section {
                    HStack(spacing: 12) {
                        if controller.workerRunning {
                            Button(role: .destructive) { controller.stopWorker() } label: {
                                Label("Ferma worker", systemImage: "stop.fill")
                            }
                            Label(controller.workerSummary, systemImage: "dot.radiowaves.left.and.right")
                                .foregroundStyle(.green).font(.callout)
                        } else if controller.workerLoading {
                            ProgressView().controlSize(.small)
                            Text("Caricamento…").font(.callout).foregroundStyle(.secondary)
                        } else {
                            Button { controller.startWorker() } label: {
                                Label("Avvia worker", systemImage: "play.fill")
                            }
                        }
                    }
                }
            }
            .formStyle(.grouped)

            DistLogView(text: controller.workerLog, height: 140)
        }
    }
}

/// The coordinator: a chat that runs across the worker cluster. Shown inside the
/// Chat tab when "Distribuito" is selected. Before connecting it shows the route
/// config; once connected, the chat transcript + composer.
struct CoordinatorChatView: View {
    @Bindable var controller: DistributedController

    var body: some View {
        VStack(spacing: 0) {
            config
                .formStyle(.grouped)
                .frame(maxHeight: controller.connected ? 230 : .infinity)

            if controller.connected {
                Divider()
                transcript
                Divider()
                composer
            }

            DistLogView(text: controller.coordLog, height: controller.connected ? 80 : 140)
        }
    }

    private var config: some View {
        Form {
            Section {
                Label("Questo Mac come COORDINATORE: connette i worker e chatta sul cluster. Avvia prima i worker (su questo o altri Mac).",
                      systemImage: "rectangle.3.group")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Section("Modello") {
                HStack {
                    TextField("Modello GGUF", text: $controller.modelPath)
                    Button("Sfoglia") { if let p = ModelPicker.pickGGUF() { controller.modelPath = p } }
                }
                Stepper("Contesto: \(controller.contextSize) token",
                        value: $controller.contextSize, in: 1024...200_000, step: 1024)
                Picker("Bit attivazioni (transport)", selection: $controller.activationBits) {
                    Text("32").tag(32); Text("16").tag(16); Text("8").tag(8)
                }
            }
            .disabled(controller.connected || controller.coordLoading)

            if !controller.connected {
                Section("Worker (uno per riga, host:porta, in ordine di layer)") {
                    TextEditor(text: $controller.peersText)
                        .font(.system(.callout, design: .monospaced))
                        .frame(height: 64)
                }
                .disabled(controller.coordLoading)
                Section("Trasporto") {
                    Stepper("Chunk prefill: \(controller.prefillChunk) token",
                            value: $controller.prefillChunk, in: 1...256, step: 8)
                    Toggle("Inoltro worker→worker", isOn: $controller.forwardEnabled)
                    if controller.forwardEnabled {
                        TextField("Host di ritorno (IP LAN di questo Mac)", text: $controller.returnHost)
                        TextField("Porta di ritorno", value: $controller.returnPort, format: .number.grouping(.never))
                            .frame(width: 100)
                    }
                }
                .disabled(controller.coordLoading)
                Section {
                    HStack(spacing: 12) {
                        if controller.coordLoading {
                            ProgressView().controlSize(.small)
                            Text("Connessione…").font(.callout).foregroundStyle(.secondary)
                        } else {
                            Button { controller.connectCoordinator() } label: {
                                Label("Connetti", systemImage: "link")
                            }
                        }
                    }
                }
            } else {
                Section {
                    HStack {
                        Label("Connesso · \(controller.parsePeers().count) worker", systemImage: "link")
                            .foregroundStyle(.green).font(.callout)
                        Spacer()
                        Button(role: .destructive) { controller.disconnectCoordinator() } label: {
                            Label("Disconnetti", systemImage: "xmark.circle")
                        }
                    }
                    Toggle("Thinking", isOn: $controller.think)
                    Stepper("Max token: \(controller.maxTokens)", value: $controller.maxTokens, in: 16...4096, step: 16)
                }
            }
        }
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(controller.messages) { MessageRow(message: $0).id($0.id) }
                }
                .padding()
            }
            .onChange(of: controller.messages.last.map { $0.text.count + $0.reasoning.count }) {
                if let last = controller.messages.last { withAnimation { proxy.scrollTo(last.id, anchor: .bottom) } }
            }
        }
    }

    private var composer: some View {
        HStack(spacing: 8) {
            TextField("Messaggio…", text: $controller.chatInput, axis: .vertical)
                .lineLimit(1...4)
                .textFieldStyle(.roundedBorder)
                .onSubmit { controller.sendChat() }
                .disabled(controller.isGenerating)
            if controller.isGenerating {
                Button(role: .destructive) { controller.stopGeneration() } label: { Image(systemName: "stop.fill") }
            } else {
                Button { controller.sendChat() } label: { Image(systemName: "paperplane.fill") }
                    .disabled(controller.chatInput.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            Button { controller.newChat() } label: { Image(systemName: "square.and.pencil") }
                .help("Nuova chat")
        }
        .padding(8)
    }
}

/// Shared monospaced log strip (hidden when empty).
struct DistLogView: View {
    let text: String
    var height: CGFloat = 140
    var body: some View {
        if !text.isEmpty {
            Divider()
            ScrollView {
                Text(text)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled).padding(8)
            }
            .frame(height: height)
            .background(Color.black.opacity(0.05))
        }
    }
}
