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

/// The coordinator: a chat that runs across the worker cluster, shown inside the
/// Chat tab when "Distribuito" is selected. Before connecting: a full-screen
/// setup form (like the local model-load screen). Once connected: the SAME
/// layout as the local chat — header (model name + route info, Thinking, Nuova
/// chat, Disconnetti), full-screen transcript, composer with live status.
struct CoordinatorChatView: View {
    @Bindable var controller: DistributedController

    var body: some View {
        if controller.connected {
            VStack(spacing: 0) {
                header
                Divider()
                transcript
                Divider()
                composer
            }
        } else {
            setup
        }
    }

    // MARK: Connected — local-chat look

    private var modelName: String { (controller.modelPath as NSString).lastPathComponent }

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(modelName)
                    .font(.headline)
                Text("distribuito · \(controller.modelLayers) layer su \(controller.parsePeers().count) worker · ctx \(controller.contextSize) · \(controller.forwardEnabled ? "inoltro" : "relay")")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Stepper("Max token: \(controller.maxTokens)",
                    value: $controller.maxTokens, in: 16...4096, step: 16)
                .fixedSize()
                .font(.callout)
            Toggle("Thinking", isOn: $controller.think)
                .toggleStyle(.switch)
            Button {
                controller.newChat()
            } label: {
                Label("Nuova chat", systemImage: "square.and.pencil")
            }
            Button(role: .destructive) {
                controller.disconnectCoordinator()
            } label: {
                Label("Disconnetti", systemImage: "xmark.circle")
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
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
        VStack(alignment: .leading, spacing: 4) {
            if controller.isGenerating && !controller.status.isEmpty {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text(controller.status)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            HStack(alignment: .bottom, spacing: 8) {
                TextField("Scrivi un messaggio…", text: $controller.chatInput, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1...6)
                    .onSubmit { controller.sendChat() }
                if controller.isGenerating {
                    Button(role: .destructive) { controller.stopGeneration() } label: {
                        Image(systemName: "stop.fill")
                    }
                } else {
                    Button { controller.sendChat() } label: {
                        Image(systemName: "arrow.up.circle.fill")
                    }
                    .disabled(controller.chatInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
        }
        .padding(10)
    }

    // MARK: Not connected — full-screen setup (mirrors the local load screen)

    private var setup: some View {
        VStack(spacing: 0) {
            Form {
                Section {
                    Label("Questo Mac come COORDINATORE: connette i worker e chatta sul cluster. Avvia prima i worker (scheda Worker, su questo o altri Mac).",
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
                Section("Worker (uno per riga, host:porta, in ordine di layer)") {
                    TextEditor(text: $controller.peersText)
                        .font(.system(.callout, design: .monospaced))
                        .frame(height: 64)
                }
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
            }
            .formStyle(.grouped)
            .disabled(controller.coordLoading)

            DistLogView(text: controller.coordLog, height: 140)
        }
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
