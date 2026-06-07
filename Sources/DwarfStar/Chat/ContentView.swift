import SwiftUI
import DS4Engine

struct ContentView: View {
    @Bindable var store: ChatStore

    var body: some View {
        switch store.phase {
        case .needsModel, .failed:
            ModelLoadView(store: store)
        case .loading:
            VStack(spacing: 12) {
                ProgressView()
                Text("Caricamento del modello…")
                    .foregroundStyle(.secondary)
                Text("Mappa il GGUF e compila i kernel Metal. Può richiedere alcuni secondi.")
                    .font(.footnote)
                    .foregroundStyle(.tertiary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .ready:
            ChatView(store: store)
        }
    }
}

/// Pre-load configuration: model selection, SSD streaming, context, system prompt.
struct ModelLoadView: View {
    @Bindable var store: ChatStore
    @State private var showDownload = false

    var body: some View {
        Form {
            Section {
                if store.discoveredModels.isEmpty {
                    Text("Nessun GGUF trovato in \(store.scriptDir) o \(store.scriptDir)/gguf.")
                        .foregroundStyle(.secondary)
                        .font(.callout)
                } else {
                    ForEach(store.discoveredModels) { model in
                        Button {
                            store.modelPath = model.path
                        } label: {
                            HStack {
                                Image(systemName: store.modelPath == model.path
                                      ? "largecircle.fill.circle" : "circle")
                                    .foregroundStyle(.tint)
                                VStack(alignment: .leading) {
                                    Text(model.name).lineLimit(1)
                                    Text(model.displaySize)
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }
            } header: {
                HStack {
                    Text("Modelli disponibili")
                    Spacer()
                    Button { store.scanModels() } label: { Image(systemName: "arrow.clockwise") }
                        .buttonStyle(.borderless)
                    Button { showDownload = true } label: { Label("Scarica…", systemImage: "arrow.down.circle") }
                        .buttonStyle(.borderless)
                }
            }

            Section("Configurazione automatica") {
                Button {
                    store.applyRecommendedPreset()
                } label: {
                    Label("Configura per la tua RAM (\(MemoryInfo.gib(MemoryInfo.physicalBytes)))",
                          systemImage: "wand.and.stars")
                }
                if let note = store.presetNote {
                    Text(note).font(.caption).foregroundStyle(.secondary)
                }
            }

            Section("Percorsi") {
                TextField("Percorso GGUF", text: $store.modelPath)
                // Metal kernels are embedded in the app — no folder to set.
            }

            Section("SSD streaming (modelli più grandi della RAM)") {
                Toggle("Abilita streaming degli expert", isOn: $store.streamingEnabled)
                if store.streamingEnabled {
                    TextField("Budget cache (es. 32GB, vuoto = auto)", text: $store.streamingCacheSpec)
                }
                Toggle("Modalità RAM minima (per-layer paging)", isOn: $store.minimumRAMMode)
                if store.minimumRAMMode {
                    Text("Disabilita residency set Metal e warmup del modello: macOS pagina via i layer freddi. Aiuta sotto i 24 GB.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Toggle("Decode per-layer + eviction (sperimentale)", isOn: $store.perLayerStreaming)
                    .disabled(store.streamingEnabled)
                if store.perLayerStreaming && !store.streamingEnabled {
                    Text("Ogni token decodificato esegue un layer alla volta e chiama MADV_DONTNEED sui pesi del layer appena finito. Working-set ≈ 1 layer in RAM; decode più lento (overhead per-layer × n_layer). Funziona solo con modello completamente residente in RAM.")
                        .font(.caption).foregroundStyle(.secondary)
                } else if store.streamingEnabled {
                    Text("Non disponibile con SSD streaming: il motore fa già per-layer mapping nativamente durante il decode.")
                        .font(.caption).foregroundStyle(.tertiary)
                }
            }

            Section("Contesto e system prompt") {
                Stepper("Contesto: \(store.contextSize) token",
                        value: $store.contextSize, in: 1024...1_000_000, step: 1024)
                TextField("System prompt (opzionale)", text: $store.systemPrompt, axis: .vertical)
                    .lineLimit(2...6)
            }

            if let warning = MemoryInfo.loadWarning(modelPath: store.modelPath,
                                                    streaming: store.streamingEnabled) {
                Section {
                    Label(warning, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                        .font(.callout)
                }
            }

            if case .failed(let message) = store.phase {
                Section {
                    Text(message).foregroundStyle(.red).font(.callout)
                }
            }

            Section {
                Button {
                    store.load()
                } label: {
                    Label("Carica modello", systemImage: "bolt.fill")
                }
                .keyboardShortcut(.return, modifiers: [.command])
                Text("RAM di sistema: \(MemoryInfo.gib(MemoryInfo.physicalBytes))")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
        .onAppear { store.scanModels() }
        .sheet(isPresented: $showDownload) {
            DownloadView(store: store)
        }
    }
}
