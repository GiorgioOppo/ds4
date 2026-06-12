import SwiftUI

/// Control panel for distributed inference: pick a role (worker or coordinator),
/// configure the layer slice / peer list, and run.
struct DistributedView: View {
    @Bindable var controller: DistributedController

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section {
                    Label("Inferenza distribuita: spezza i layer del modello (Flash 43, Pro 61) su piu' Mac (pipeline). Avvia prima i worker, poi il coordinatore.",
                          systemImage: "rectangle.3.group")
                        .font(.callout).foregroundStyle(.secondary)
                    Picker("Ruolo", selection: $controller.role) {
                        ForEach(DistributedController.Role.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .disabled(controller.isRunning)
                }

                Section("Modello") {
                    HStack {
                        TextField("Modello GGUF", text: $controller.modelPath)
                        Button("Sfoglia") { if let p = ModelPicker.pickGGUF() { controller.modelPath = p } }
                    }
                    .disabled(controller.isRunning)
                    Stepper("Contesto: \(controller.contextSize) token",
                            value: $controller.contextSize, in: 1024...200_000, step: 1024)
                        .disabled(controller.isRunning)
                    Picker("Bit attivazioni (transport)", selection: $controller.activationBits) {
                        Text("32").tag(32); Text("16").tag(16); Text("8").tag(8)
                    }
                    .disabled(controller.isRunning)
                }

                if controller.role == .worker {
                    Section("Worker — modello da \(controller.modelLayers) layer (0…\(controller.modelLayers - 1))") {
                        TextField("Porta", value: $controller.port, format: .number.grouping(.never))
                            .disabled(controller.isRunning)
                        Stepper("Primo layer: \(controller.layerStart)", value: $controller.layerStart,
                                in: 0...(controller.modelLayers - 1)).disabled(controller.isRunning)
                        Stepper("Ultimo layer: \(controller.layerEnd)", value: $controller.layerEnd,
                                in: controller.layerStart...(controller.modelLayers - 1)).disabled(controller.isRunning)
                        Toggle("Possiede l'output head (ultimo slice)", isOn: $controller.hasOutput)
                            .disabled(controller.isRunning)
                        Text("La route deve coprire tutti i \(controller.modelLayers) layer in modo contiguo. Su un solo Mac: un worker 0…\(controller.modelLayers - 1) con output head.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                } else {
                    Section("Coordinatore — worker (uno per riga, host:porta, in ordine di layer)") {
                        TextEditor(text: $controller.peersText)
                            .font(.system(.callout, design: .monospaced))
                            .frame(height: 70)
                            .disabled(controller.isRunning)
                    }
                    Section("Prompt") {
                        TextField("Prompt", text: $controller.prompt, axis: .vertical)
                            .lineLimit(2...5)
                        Stepper("Max token: \(controller.maxTokens)",
                                value: $controller.maxTokens, in: 16...4096, step: 16)
                        Stepper("Chunk prefill: \(controller.prefillChunk) token",
                                value: $controller.prefillChunk, in: 1...256, step: 8)
                            .disabled(controller.isRunning)
                    }
                    Section("Inoltro worker→worker (opzionale)") {
                        Toggle("Inoltra lo stato HC direttamente tra i worker", isOn: $controller.forwardEnabled)
                            .disabled(controller.isRunning)
                        if controller.forwardEnabled {
                            TextField("Host di ritorno (IP LAN di questo Mac)", text: $controller.returnHost)
                                .disabled(controller.isRunning)
                            TextField("Porta di ritorno", value: $controller.returnPort,
                                      format: .number.grouping(.never))
                                .frame(width: 100)
                                .disabled(controller.isRunning)
                        }
                    }
                }

                Section {
                    HStack(spacing: 12) {
                        if controller.isRunning {
                            Button(role: .destructive) { controller.stop() } label: {
                                Label("Ferma", systemImage: "stop.fill")
                            }
                            Label(controller.endpointSummary, systemImage: "dot.radiowaves.left.and.right")
                                .foregroundStyle(.green).font(.callout)
                        } else if controller.isLoading {
                            ProgressView().controlSize(.small)
                            Text("Caricamento…").font(.callout).foregroundStyle(.secondary)
                        } else {
                            Button { controller.start() } label: {
                                Label(controller.role == .worker ? "Avvia worker" : "Genera", systemImage: "play.fill")
                            }
                        }
                    }
                }

                if controller.role == .coordinator && !controller.output.isEmpty {
                    Section("Risposta") {
                        Text(controller.output).textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
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
                        .textSelection(.enabled).padding(8)
                }
                .frame(height: 140)
                .background(Color.black.opacity(0.05))
            }
        }
    }
}
