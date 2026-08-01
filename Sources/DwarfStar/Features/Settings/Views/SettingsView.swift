import SwiftUI
import AppKit
import DS4Engine

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

    // Hugging Face token editor state. The field is write-only: the stored
    // token is never echoed back into it, only the redacted status line.
    @State private var hfTokenField = ""
    @State private var hfRevealToken = false
    @State private var hfStatus: String?
    @State private var showModelDownloads = false

    var body: some View {
        VStack(spacing: 0) {
            Form {
                modelSection
                huggingFaceSection
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
                .disabled(store.benchRunning)
                Group {
                    switch settings.mode {
                    case .local:       localSection
                    case .distributed: distributedSection
                    }
                }
                .disabled(store.benchRunning)
            }
            .formStyle(.grouped)

            if settings.mode == .distributed {
                DistLogView(text: dist.coordLog, height: 120)
            }
        }
        .task(id: "\(settings.mode.rawValue)|\(settings.modelPath)") {
            await store.inspectSelectedModel(path: settings.modelPath)
            // The distributed controller performs its own GGUF metadata pass.
            // Local mode already gets the same descriptor from ChatStore, so a
            // second parse here only adds I/O and temporary allocations while
            // the inference engine may be under memory pressure.
            if settings.mode == .distributed {
                await dist.refreshModelGeometry()
            }
        }
        .sheet(isPresented: $showModelDownloads) {
            DownloadView(store: store)
        }
    }

    // MARK: Hugging Face token

    /// Configuration for the token the model downloader sends as
    /// `Authorization: Bearer`. Stored in the KEYCHAIN (never UserDefaults — a
    /// token is a secret, not a preference) via `HFTokenStore` and passed
    /// explicitly to `ModelDownloader.download`, so it wins over the `HF_TOKEN`
    /// env var and `~/.cache/huggingface/token` fallbacks.
    private var huggingFaceSection: some View {
        Section("Hugging Face") {
            HStack(spacing: 8) {
                Group {
                    if hfRevealToken {
                        TextField("hf_...", text: $hfTokenField)
                    } else {
                        SecureField("hf_...", text: $hfTokenField)
                    }
                }
                .autocorrectionDisabled()
                Button {
                    hfRevealToken.toggle()
                } label: {
                    Image(systemName: hfRevealToken ? "eye.slash" : "eye")
                }
                .buttonStyle(.borderless)
                .help(hfRevealToken ? "Hide the token" : "Show the token")
                Button("Save") {
                    if HFTokenStore.save(hfTokenField) {
                        hfTokenField = ""
                        hfRevealToken = false
                    }
                    refreshHFStatus()
                }
                .disabled(hfTokenField.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Button("Remove", role: .destructive) {
                    HFTokenStore.clear()
                    refreshHFStatus()
                }
                .disabled(HFTokenStore.load() == nil)
            }
            if let status = hfStatus {
                Label("Active token — \(status)", systemImage: "key.fill")
                    .font(.caption)
            } else {
                Label("No token configured. Public models download without one.",
                      systemImage: "key.slash")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text("Used by the model downloader as \u{201C}Authorization: Bearer\u{201D} for gated/private Hugging Face repositories and to avoid anonymous rate limits. Create a read-only token at huggingface.co/settings/tokens. Saved in the macOS Keychain (not UserDefaults) and never shown again in full; saving replaces the previous one. Without a saved token the downloader falls back to the HF_TOKEN environment variable, then ~/.cache/huggingface/token.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .onAppear { refreshHFStatus() }
    }

    private func refreshHFStatus() {
        hfStatus = HFTokenStore.activeSourceDescription()
    }

    // MARK: Modello (shared by both modes)

    /// UI per-backend (classe astratta + implementazioni DeepSeek/GLM):
    /// scelta dal modello caricato o ispezionato, così le sezioni benchmark
    /// e memoria sono sempre quelle del backend selezionato.
    private var backendUI: BackendSettingsUI {
        BackendSettingsUI.make(store: store, dist: dist)
    }

    private var modelSection: some View {
        Group {
            Section("Model") {
                HStack {
                    Text(settings.modelPath.isEmpty ? "Nessun GGUF selezionato" : settings.modelPath)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                    Button("Browse") {
                        if let path = ModelPicker.pickGGUF() { store.selectPickedModel(path: path) }
                    }
                    .disabled(store.phase == .loading)
                    Button {
                        showModelDownloads = true
                    } label: {
                        Label("Scarica…", systemImage: "arrow.down.circle")
                    }
                    .disabled(store.phase == .loading)
                }
                Button("Grant Model Folder Access…") {
                    _ = ModelPicker.pickModelFolder(near: settings.modelPath)
                }
                Text("Sandboxed builds can read ONLY the picked .gguf: sidecar caches next to it (\u{201C}.q4dense\u{201D}, \u{201C}.expbundle\u{201D}, e.g. built by the demo) are invisible, so the app re-creates them inside its container (slow first load, gigabytes duplicated, or a failed write when disk is short). Granting the model's folder reuses them directly — recommended once per model folder.")
                    .font(.caption).foregroundStyle(.secondary)
                Stepper("Context: \(settings.contextSize) tokens",
                        value: $settings.contextSize, in: 1024...1_000_000, step: 1024)
                if let warning = MemoryInfo.loadWarning(modelPath: settings.modelPath) {
                    Label(warning, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                }
            }
            .disabled(store.benchRunning)
            backendUI.benchmarkSection()
            backendUI.memorySection()
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
                        .disabled(store.isGenerating || store.benchRunning)
                        .help(store.isGenerating
                              ? "Stop generation before reloading the model."
                              : "Reload the model with the current settings.")
                }
            case .loading:
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: min(max(store.loadFraction, 0), 1))
                        .progressViewStyle(.linear)
                    // Percentuale a 0,1%: nelle fasi lunghe (riquantizzazione
                    // Q4) il numero prova che il load avanza anche quando la
                    // barra sembra ferma.
                    Text((store.loadStage.isEmpty ? "Loading model..." : store.loadStage)
                         + String(format: " · %.1f%%", min(max(store.loadFraction, 0), 1) * 100))
                        .font(.caption).foregroundStyle(.secondary)
                        .monospacedDigit()
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
                Toggle("Vertical split (expert parallelism)", isOn: $dist.verticalEnabled)
                if dist.verticalEnabled {
                    Label("Workers own EXPERT shards (all layers); the dense backbone (attention/KV/head) runs on THIS Mac. ~41 network round-trips per token: requires a wired link (Thunderbolt bridge or direct Ethernet, RTT < 1 ms) — on Wi-Fi it is slower than the pipeline by design. Chat and the vertical benchmark both run on this route once connected.",
                          systemImage: "arrow.triangle.branch")
                        .font(.caption).foregroundStyle(.orange)
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
                    Label("Connected · \(dist.parsePeers().count) workers · \(dist.modelLayersLabel) layers",
                          systemImage: "link")
                        .foregroundStyle(.green)
                    Spacer()
                    Button(role: .destructive) { dist.disconnectCoordinator() } label: {
                        Label("Disconnect", systemImage: "xmark.circle")
                    }
                }
                if dist.verticalEnabled {
                    Button("Benchmark verticale (96+28 token)") { dist.runVerticalBenchmark() }
                        .disabled(dist.benchmarkActive)
                    Text("Vertical route: expert shards remote, dense backbone local. Result in the coordinator log.")
                        .font(.caption).foregroundStyle(.tertiary)
                } else {
                    Text("The Chat tab now runs on the cluster.")
                        .font(.caption).foregroundStyle(.tertiary)
                }
            }
        }
    }
}
