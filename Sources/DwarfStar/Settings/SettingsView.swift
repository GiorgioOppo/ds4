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
                Section("Mode") {
                    Picker("Execution", selection: $settings.mode) {
                        ForEach(AppSettings.EngineMode.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    Text(settings.mode == .local
                         ? "The model runs in-process on this Mac."
                         : "This Mac coordinates a worker cluster. Start workers from the Worker tab, on this Mac or other Macs.")
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
            Section("Model") {
                HStack {
                    TextField("GGUF model", text: $settings.modelPath)
                    Button("Browse") { if let p = ModelPicker.pickGGUF() { settings.modelPath = p } }
                }
                Stepper("Context: \(settings.contextSize) tokens",
                        value: $settings.contextSize, in: 1024...1_000_000, step: 1024)
                if let warning = MemoryInfo.loadWarning(modelPath: settings.modelPath) {
                    Label(warning, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                }
            }
            Section("Memory") {
                Stepper("Expert cache: \(store.expertCacheSlots) slots/layer\(store.expertCacheSlots == 0 ? " (off)" : "")",
                        value: $store.expertCacheSlots, in: 0...64, step: 4)
                if store.expertCacheSlots > 8 && store.residentDenseEnabled && MemoryInfo.physicalBytes < 24 * 1_073_741_824 {
                    Label("With resident dense weights on <=16 GB, more than 8 slots can push wired memory into swap (measured drop: 0.05 tok/s).",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                }
                Toggle("Experts via direct pread (F_NOCACHE) - recommended <=16 GB", isOn: $store.expertPreadEnabled)
                Toggle("Resident dense weights (~5 GB, ~1 min load warm-up)", isOn: $store.residentDenseEnabled)
                Toggle("Disk KV (reuse prefixes across sessions)", isOn: $store.diskKVEnabled)
                if store.diskKVEnabled {
                    Stepper("Budget: \(store.diskKVBudgetMB) MB",
                            value: $store.diskKVBudgetMB, in: 512...65536, step: 512)
                }
                Toggle("Raw-KV ring (experimental): constant KV RAM", isOn: $store.rawRingEnabled)
                if store.rawRingEnabled {
                    Label("Keeps only the attention window (128 rows) in RAM instead of the full context. Experimental: verify outputs after a long context.",
                          systemImage: "flask")
                        .font(.caption).foregroundStyle(.orange)
                }
                Text("Applies on the next model load.")
                    .font(.caption).foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: Locale

    @ViewBuilder private var localSection: some View {
        Section("Local Engine") {
            switch store.phase {
            case .ready:
                HStack {
                    Label("Model loaded", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    if let info = store.info {
                        Text("\(info.layers) layer · ctx \(info.contextSize)")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Reload") { store.load() }
                }
            case .loading:
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Loading model...").font(.callout).foregroundStyle(.secondary)
                }
            case .needsModel, .failed:
                if case .failed(let message) = store.phase {
                    Label(message, systemImage: "xmark.octagon.fill")
                        .font(.caption).foregroundStyle(.red)
                }
                Button { store.load() } label: {
                    Label("Load Model", systemImage: "play.fill")
                }
            }
            Text("The Chat tab uses this engine; Tuning and the local agent use it too.")
                .font(.caption).foregroundStyle(.tertiary)
        }
    }

    // MARK: Distribuito (coordinatore)

    @ViewBuilder private var distributedSection: some View {
        if !dist.connected {
            Section("Workers (one per line, host:port, in layer order)") {
                TextEditor(text: $dist.peersText)
                    .font(.system(.callout, design: .monospaced))
                    .frame(height: 64)
            }
            .disabled(dist.coordLoading)
            Section("Transport") {
                Picker("Activation bits", selection: $dist.activationBits) {
                    Text("32").tag(32); Text("16").tag(16); Text("8").tag(8)
                }
                Stepper("Chunk prefill: \(dist.prefillChunk) token",
                        value: $dist.prefillChunk, in: 1...256, step: 8)
                Stepper("Max tokens per response: \(dist.maxTokens)",
                        value: $dist.maxTokens, in: 16...4096, step: 16)
                Toggle("Worker-to-worker forwarding", isOn: $dist.forwardEnabled)
                if dist.forwardEnabled {
                    TextField("Return host (this Mac's LAN IP)", text: $dist.returnHost)
                    TextField("Return port", value: $dist.returnPort, format: .number.grouping(.never))
                        .frame(width: 100)
                }
            }
            .disabled(dist.coordLoading)
            Section {
                if dist.coordLoading {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("Connecting...").font(.callout).foregroundStyle(.secondary)
                    }
                } else {
                    Button { dist.connectCoordinator() } label: {
                        Label("Connect to Cluster", systemImage: "link")
                    }
                }
            }
        } else {
            Section("Cluster") {
                HStack {
                    Label("Connected · \(dist.parsePeers().count) workers · \(dist.modelLayers) layers",
                          systemImage: "link")
                        .foregroundStyle(.green)
                    Spacer()
                    Button(role: .destructive) { dist.disconnectCoordinator() } label: {
                        Label("Disconnect", systemImage: "xmark.circle")
                    }
                }
                Text("The Chat tab now runs on the cluster.")
                    .font(.caption).foregroundStyle(.tertiary)
            }
        }
    }
}
