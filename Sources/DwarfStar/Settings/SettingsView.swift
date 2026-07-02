import SwiftUI

/// The single place where the model and HOW it runs are configured. Every other
/// screen (chat, server, benchmark, diagnostics, worker) inherits these values.
///
///  • Modello: GGUF + contesto (+ memoria, cache esperti, KV su disco).
///  • Modalità: Locale (motore in-process) o Distribuito (coordina i worker).
///  • Locale: Carica modello (stato della chat locale).
///  • Distribuito: route dei worker + trasporto + Connetti/Disconnetti.
struct SettingsView: View {
    @Bindable var settings: AppSettings
    @Bindable var store: ChatStore
    @Bindable var dist: DistributedController

    var body: some View {
        VStack(spacing: 0) {
            Form {
                modelSection
                Section("Modalità") {
                    Picker("Esecuzione", selection: $settings.mode) {
                        ForEach(AppSettings.EngineMode.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    Text(settings.mode == .local
                         ? "Il modello gira in-process su questo Mac."
                         : "Questo Mac coordina un cluster di worker (i worker si avviano nella scheda Worker, su questo o altri Mac).")
                        .font(.caption).foregroundStyle(.secondary)
                }
                switch settings.mode {
                case .local:       localSection
                case .distributed: distributedSection
                }
            }
            .formStyle(.grouped)

            if settings.mode == .distributed {
                DistLogView(text: dist.coordLog, height: 120)
            }
        }
    }

    // MARK: Modello (shared by both modes)

    private var modelSection: some View {
        Group {
            Section("Modello") {
                HStack {
                    TextField("Modello GGUF", text: $settings.modelPath)
                    Button("Sfoglia") { if let p = ModelPicker.pickGGUF() { settings.modelPath = p } }
                }
                Stepper("Contesto: \(settings.contextSize) token",
                        value: $settings.contextSize, in: 1024...1_000_000, step: 1024)
                if let warning = MemoryInfo.loadWarning(modelPath: settings.modelPath) {
                    Label(warning, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                }
            }
            Section("Memoria") {
                Stepper("Cache esperti: \(store.expertCacheSlots) slot/layer\(store.expertCacheSlots == 0 ? " (off)" : "")",
                        value: $store.expertCacheSlots, in: 0...64, step: 8)
                Toggle("KV su disco (riusa i prefissi tra sessioni)", isOn: $store.diskKVEnabled)
                if store.diskKVEnabled {
                    Stepper("Budget: \(store.diskKVBudgetMB) MB",
                            value: $store.diskKVBudgetMB, in: 512...65536, step: 512)
                }
                Toggle("Raw-KV ring (sperimentale): RAM della KV costante", isOn: $store.rawRingEnabled)
                if store.rawRingEnabled {
                    Label("Tiene in RAM solo la finestra di attenzione (128 righe) invece dell'intero contesto. Sperimentale — verifica gli output dopo un contesto lungo.",
                          systemImage: "flask")
                        .font(.caption).foregroundStyle(.orange)
                }
                Text("Si applicano al prossimo caricamento del modello.")
                    .font(.caption).foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: Locale

    @ViewBuilder private var localSection: some View {
        Section("Motore locale") {
            switch store.phase {
            case .ready:
                HStack {
                    Label("Modello caricato", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    if let info = store.info {
                        Text("\(info.layers) layer · ctx \(info.contextSize)")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Ricarica") { store.load() }
                }
            case .loading:
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Caricamento del modello…").font(.callout).foregroundStyle(.secondary)
                }
            case .needsModel, .failed:
                if case .failed(let message) = store.phase {
                    Label(message, systemImage: "xmark.octagon.fill")
                        .font(.caption).foregroundStyle(.red)
                }
                Button { store.load() } label: {
                    Label("Carica modello", systemImage: "play.fill")
                }
            }
            Text("La chat (scheda Chat) usa questo motore; Tuning e l'agente locale pure.")
                .font(.caption).foregroundStyle(.tertiary)
        }
    }

    // MARK: Distribuito (coordinatore)

    @ViewBuilder private var distributedSection: some View {
        if !dist.connected {
            Section("Worker (uno per riga, host:porta, in ordine di layer)") {
                TextEditor(text: $dist.peersText)
                    .font(.system(.callout, design: .monospaced))
                    .frame(height: 64)
            }
            .disabled(dist.coordLoading)
            Section("Trasporto") {
                Picker("Bit attivazioni", selection: $dist.activationBits) {
                    Text("32").tag(32); Text("16").tag(16); Text("8").tag(8)
                }
                Stepper("Chunk prefill: \(dist.prefillChunk) token",
                        value: $dist.prefillChunk, in: 1...256, step: 8)
                Stepper("Max token per risposta: \(dist.maxTokens)",
                        value: $dist.maxTokens, in: 16...4096, step: 16)
                Toggle("Inoltro worker→worker", isOn: $dist.forwardEnabled)
                if dist.forwardEnabled {
                    TextField("Host di ritorno (IP LAN di questo Mac)", text: $dist.returnHost)
                    TextField("Porta di ritorno", value: $dist.returnPort, format: .number.grouping(.never))
                        .frame(width: 100)
                }
            }
            .disabled(dist.coordLoading)
            Section {
                if dist.coordLoading {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("Connessione…").font(.callout).foregroundStyle(.secondary)
                    }
                } else {
                    Button { dist.connectCoordinator() } label: {
                        Label("Connetti al cluster", systemImage: "link")
                    }
                }
            }
        } else {
            Section("Cluster") {
                HStack {
                    Label("Connesso · \(dist.parsePeers().count) worker · \(dist.modelLayers) layer",
                          systemImage: "link")
                        .foregroundStyle(.green)
                    Spacer()
                    Button(role: .destructive) { dist.disconnectCoordinator() } label: {
                        Label("Disconnetti", systemImage: "xmark.circle")
                    }
                }
                Text("La chat (scheda Chat) ora gira sul cluster.")
                    .font(.caption).foregroundStyle(.tertiary)
            }
        }
    }
}
