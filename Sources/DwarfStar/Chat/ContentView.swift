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
                Button {
                    if let path = ModelPicker.pickGGUF() { store.modelPath = path }
                } label: {
                    Label("Sfoglia…", systemImage: "folder")
                }
                Text("Con l'App Sandbox attiva è necessario selezionare il file qui: il percorso scelto resta accessibile e viene ricordato al prossimo avvio.")
                    .font(.caption).foregroundStyle(.secondary)
                // Metal kernels are embedded in the app — no folder to set.
            }

            Section("Memoria") {
                Text("Lo streaming da SSD è sempre attivo: i pesi non-routed sono mappati no-copy (page cache) e per ogni token vengono letti solo i 6 expert selezionati. Se il modello entra in RAM, la page cache lo tiene residente automaticamente.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Contesto e system prompt") {
                Stepper("Contesto: \(store.contextSize) token",
                        value: $store.contextSize, in: 1024...1_000_000, step: 1024)
                TextField("System prompt (opzionale)", text: $store.systemPrompt, axis: .vertical)
                    .lineLimit(2...6)
            }

            if let warning = MemoryInfo.loadWarning(modelPath: store.modelPath) {
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
        .onAppear { store.restoreModelBookmark(); store.scanModels() }
        .sheet(isPresented: $showDownload) {
            DownloadView(store: store)
        }
    }
}
