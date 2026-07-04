import SwiftUI
import AppKit

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
            Section("Benchmark") {
                HStack(spacing: 8) {
                    Button(store.benchRunning ? "Benchmark in corso…" : "Rapido (~3 min)") {
                        store.runSettingsBenchmark(quick: true)
                    }
                    .disabled(store.benchRunning || store.phase != .ready)
                    Button("Completo (~10 min)") {
                        store.runSettingsBenchmark(quick: false)
                    }
                    .disabled(store.benchRunning || store.phase != .ready)
                    if store.benchRunning { ProgressView().controlSize(.small) }
                }
                Text("Misura sul modello caricato i knob del prefill regolabili a caldo e applica/salva la combinazione più veloce. Rapido: solo unione esperti (64/192/256) su 128 token. Completo: anche il chunk (512/1024) su 1024 token — sotto i 512 token un secondo chunk non esiste, quindi il rapido non può misurarlo. I knob che richiedono un reload (percorso matrix-matrix, batch dei route) non sono coperti.")
                    .font(.caption).foregroundStyle(.secondary)
                if let status = store.benchStatus {
                    Label(status, systemImage: store.benchRunning ? "hourglass" : "checkmark.circle")
                        .font(.caption)
                }
                if !store.benchResults.isEmpty {
                    Text(store.benchResults)
                        .font(.system(.caption2, design: .monospaced))
                        .textSelection(.enabled)
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Attivi", value: "union \(store.prefillUnion) · chunk \(store.prefillChunk)")
                    .font(.caption)
            }
            Section("Memory") {
                HStack(spacing: 8) {
                    Button("Align to fast demo config") { store.applyFastDemoDefaults() }
                    Text("Resets every toggle below to the measured-fast set (slots 16, pread + dense stream + mlock + Q4 + bundle ON, ring OFF). Applies on the next model load.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Stepper("Expert cache: \(store.expertCacheSlots) slots/layer\(store.expertCacheSlots == 0 ? " (off)" : "")",
                        value: $store.expertCacheSlots, in: 0...64, step: 4)
                if store.expertCacheSlots > 12 && MemoryInfo.physicalBytes < 24 * 1_073_741_824 {
                    Label("Each slot costs ~0.3 GB of wired memory (6.9 MB × 43 layers): with low RAM too many slots swap and decode collapses.",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                }
                Toggle("Experts via direct pread (F_NOCACHE) - recommended <=16 GB", isOn: $store.expertPreadEnabled)
                Toggle("Expert bundle sidecar (contiguous slabs, ~+25% tok/s)", isOn: $store.expertBundleEnabled)
                if store.expertBundleEnabled {
                    Label("Reuses <model>.expbundle next to the GGUF when readable (e.g. built by the demo); otherwise the first load builds it under Application Support. Same bytes reordered so a cache miss is ONE sequential ~7 MB read (measured: gather 2.7→4.8 GB/s, +27% tok/s). Duplicates the expert region on disk (tens of GB); skipped automatically when space is short. Check the engine log for 'DS4 expbundle:' lines.",
                          systemImage: "externaldrive.badge.plus")
                        .font(.caption).foregroundStyle(.orange)
                    HStack(spacing: 6) {
                        Text("Bundle dir: \(ChatStore.bundleDirectory.path)")
                            .font(.caption2).foregroundStyle(.tertiary)
                            .textSelection(.enabled)
                            .lineLimit(2)
                        Button("Show in Finder") {
                            NSWorkspace.shared.activateFileViewerSelecting([ChatStore.bundleDirectory])
                        }
                        .font(.caption2)
                    }
                    HStack(spacing: 8) {
                        Button("Generate expert bundle now") { store.buildExpertBundleNow() }
                            .font(.caption)
                        if let status = store.bundleBuildStatus {
                            Text(status).font(.caption2).foregroundStyle(.secondary)
                                .lineLimit(2)
                        }
                    }
                }
                Toggle("Dense-weight streaming (reads layer i+1 while computing layer i) - recommended <=16 GB", isOn: $store.denseStreamEnabled)
                Toggle("Pin hot buffers in RAM (mlock ~3.3 GB, keeps the memory compressor away)", isOn: $store.mlockEnabled)
                Toggle("Q4 attention projections (LOSSY, ~+30% speed)", isOn: $store.denseQ4Enabled)
                if store.denseQ4Enabled {
                    Label("Requantizes the three giant attention projections Q8→Q4_K at load (~+30% tok/s measured). Slightly lossy: greedy outputs can occasionally differ while staying coherent. Requires dense-weight streaming.",
                          systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.orange)
                    HStack(spacing: 6) {
                        Text("Q4 cache: \(ChatStore.q4CacheDirectory.path)")
                            .font(.caption2).foregroundStyle(.tertiary)
                            .textSelection(.enabled)
                            .lineLimit(2)
                        Button("Show in Finder") {
                            NSWorkspace.shared.activateFileViewerSelecting([ChatStore.q4CacheDirectory])
                        }
                        .font(.caption2)
                    }
                }
                Toggle("Disk KV (reuse prefixes across sessions)", isOn: $store.diskKVEnabled)
                if store.diskKVEnabled {
                    Stepper("Budget: \(store.diskKVBudgetKTok)k tokens (≈ \(String(format: "%.1f", Double(store.diskKVBudgetKTok) * 0.022)) GB)",
                            value: $store.diskKVBudgetKTok, in: 128...4096, step: 128)
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
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: min(max(store.loadFraction, 0), 1))
                        .progressViewStyle(.linear)
                    Text(store.loadStage.isEmpty ? "Loading model..." : store.loadStage)
                        .font(.caption).foregroundStyle(.secondary)
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
