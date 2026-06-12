import SwiftUI

/// Control panel for distributed inference: pick a role (worker or coordinator),
/// configure the slice / route, and — as coordinator — chat across the cluster.
struct DistributedView: View {
    @Bindable var controller: DistributedController

    var body: some View {
        VStack(spacing: 0) {
            config
                .formStyle(.grouped)
                .frame(maxHeight: chatActive ? 230 : .infinity)

            if chatActive {
                Divider()
                transcript
                Divider()
                composer
            }

            if !controller.log.isEmpty {
                Divider()
                ScrollView {
                    Text(controller.log)
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled).padding(8)
                }
                .frame(height: chatActive ? 80 : 140)
                .background(Color.black.opacity(0.05))
            }
        }
    }

    private var chatActive: Bool { controller.role == .coordinator && controller.connected }

    // MARK: Config form

    private var config: some View {
        Form {
            Section {
                Label("Inferenza distribuita: spezza i layer del modello (Flash 43, Pro 61) su piu' Mac (pipeline). Avvia prima i worker, poi connetti il coordinatore.",
                      systemImage: "rectangle.3.group")
                    .font(.callout).foregroundStyle(.secondary)
                Picker("Ruolo", selection: $controller.role) {
                    ForEach(DistributedController.Role.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .disabled(controller.isRunning || controller.isLoading)
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
            .disabled(controller.isRunning || controller.isLoading)

            if controller.role == .worker {
                workerSection
                actionSection
            } else {
                coordinatorSection
            }
        }
    }

    private var workerSection: some View {
        Group {
            Section("Worker — modello da \(controller.modelLayers) layer (0…\(controller.modelLayers - 1))") {
                TextField("Porta", value: $controller.port, format: .number.grouping(.never))
                Stepper("Primo layer: \(controller.layerStart)", value: $controller.layerStart,
                        in: 0...(controller.modelLayers - 1))
                Stepper("Ultimo layer: \(controller.layerEnd)", value: $controller.layerEnd,
                        in: controller.layerStart...(controller.modelLayers - 1))
                Toggle("Possiede l'output head (ultimo slice)", isOn: $controller.hasOutput)
                Text("La route deve coprire tutti i \(controller.modelLayers) layer in modo contiguo. Su un solo Mac: un worker 0…\(controller.modelLayers - 1) con output head.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .disabled(controller.isRunning)
        }
    }

    private var actionSection: some View {
        Section {
            HStack(spacing: 12) {
                if controller.isRunning {
                    Button(role: .destructive) { controller.stopWorker() } label: {
                        Label("Ferma worker", systemImage: "stop.fill")
                    }
                    Label(controller.endpointSummary, systemImage: "dot.radiowaves.left.and.right")
                        .foregroundStyle(.green).font(.callout)
                } else if controller.isLoading {
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

    @ViewBuilder private var coordinatorSection: some View {
        if !controller.connected {
            Section("Worker (uno per riga, host:porta, in ordine di layer)") {
                TextEditor(text: $controller.peersText)
                    .font(.system(.callout, design: .monospaced))
                    .frame(height: 64)
            }
            .disabled(controller.isLoading)
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
            .disabled(controller.isLoading)
            Section {
                HStack(spacing: 12) {
                    if controller.isLoading {
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

    // MARK: Chat (coordinator, connected)

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
                Button(role: .destructive) { controller.stopGeneration() } label: {
                    Image(systemName: "stop.fill")
                }
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
