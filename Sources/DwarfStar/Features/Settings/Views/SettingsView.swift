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
            .disabled(store.benchRunning || store.bundleBuildRunning)
            if store.supportsDeepSeekPerformanceTuning {
                Section("Benchmark DeepSeek V4") {
                HStack(spacing: 8) {
                    Button(store.benchRunning ? "Benchmark in corso…" : "Rapido (~3 min)") {
                        store.runSettingsBenchmark(quick: true)
                    }
                    .disabled(store.benchRunning || store.phase != .ready || store.isGenerating)
                    Button("Completo (~15 min)") {
                        store.runSettingsBenchmark(quick: false)
                    }
                    .disabled(store.benchRunning || store.phase != .ready || store.isGenerating)
                    Button("Auto-tune record-holder (ore)") {
                        store.runAutoTune(
                            distributedRuntimeActive: dist.workerRunning || dist.connected
                                || dist.coordLoading || dist.isGenerating || dist.benchmarkActive
                        )
                    }
                    .disabled(store.benchRunning || store.phase != .ready || store.isGenerating)
                    if store.benchRunning {
                        Button("Stop", role: .destructive) { store.cancelAutoTune() }
                        ProgressView(
                            value: Double(store.benchProgressDone),
                            total: Double(max(store.benchProgressTotal, 1))
                        )
                        .frame(width: 100)
                        .controlSize(.small)
                    }
                }
                Text("Rapido/Completo regolano i knob del prefill sul motore già caricato. Auto-tune record-holder misura una sola volta la baseline calda e ogni configurazione unica; se una configurazione ricompare riusa il risultato in cache senza reload. Per gli expert-cache slot prova un gradino alla volta verso l'alto: dopo una promozione continua soltanto in quella direzione e si ferma al primo peggioramento; prova il vicino inferiore solo se il primo gradino superiore non vince. Il campione completo col decode valido più alto resta il confronto: lo sostituisce qualsiasi decode strettamente superiore che mantenga hash bit-esatto di tutti i logits, prefill entro −8%, stabilità ≥0,75, RAM sopra il floor immutabile e non più di 128 MiB di swap steady. Congela l'usage profile, mantiene Raw-KV ring ON e salva i parametri soltanto dopo warmup dell'agente attivo e probe steady conclusivo. Un journal transazionale ripristina i valori iniziali dopo un'interruzione.")
                    .font(.caption).foregroundStyle(.secondary)
                if let status = store.benchStatus {
                    Label(status, systemImage: store.benchRunning ? "hourglass" :
                          (store.benchSucceeded == false ? "xmark.circle" : "checkmark.circle"))
                        .font(.caption)
                }
                if let reportURL = store.autoTuneReportURL {
                    HStack(spacing: 8) {
                        Text(reportURL.deletingLastPathComponent().path)
                            .font(.caption2).foregroundStyle(.secondary)
                            .lineLimit(1).truncationMode(.middle).textSelection(.enabled)
                        Button("Mostra report") {
                            NSWorkspace.shared.activateFileViewerSelecting([reportURL])
                        }
                        .font(.caption)
                    }
                }
                if !store.benchResults.isEmpty {
                    Text(store.benchResults)
                        .font(.system(.caption2, design: .monospaced))
                        .textSelection(.enabled)
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Attivi (prefill)", value: "union \(store.prefillUnion) · chunk \(store.prefillChunk) · route batch \(store.prefillRouteBatch)")
                    .font(.caption)
                // Il set di CARICAMENTO scelto dall'auto-tune: slot e look-ahead
                // hanno i loro stepper (si aggiornano da soli), ma dense-ahead,
                // async FFN e q8nsg non hanno un controllo — senza questa riga
                // il valore applicato sarebbe visibile solo nel referto.
                LabeledContent("Attivi (load)",
                               value: "slot \(store.expertCacheSlots) · mixed-cache \(store.multiQuantCacheEnabled ? "on" : "off") · " +
                                      "alloc \(store.expertCacheUniform ? "uniforme" : "usage") · pread×\(store.preadSplit) · " +
                                      "ahead \(store.denseAhead) · " +
                                      "asyncFFN \(store.asyncFFNEnabled ? "on" : "off") · " +
                                      "look \(store.expertLookahead) · q8/moe/denseNSG \(store.q8NSG)/\(store.moeNSG)/\(store.denseQ4NSG) · " +
                                      "raw-ring \(store.rawRingEnabled ? "on" : "off") · " +
                                      "MetalIO \(store.metalIOEnabled ? "on" : "off")")
                    .font(.caption)
            }
                Section("Memory · DeepSeek V4") {
                HStack(spacing: 8) {
                    Button("Align to fast demo config") { store.applyFastDemoDefaults() }
                    Text("Resets performance controls to the measured M1 Pro 16 GB preset: 22 slots, mixed-quant cache ON, full Q4, prefill 256/512/32, pread×4 + dense stream + mlock + bundle + MetalIO ON, dense-ahead 2, look-ahead 0, NSG 4, Raw-KV ring ON. MetalIO falls back automatically if slower. Applies on the next model load.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Stepper("Expert cache: \(store.expertCacheSlots) slots/layer\(store.expertCacheSlots == 0 ? " (off)" : "")",
                        value: $store.expertCacheSlots, in: 0...64, step: 4)
                if store.expertCacheSlots > 20 && MemoryInfo.physicalBytes < 24 * 1_073_741_824 {
                    Label("The measured 16 GB Flash preset uses 22 slots. On Flash each extra legacy-size slot costs ~0.25 GB across the IQ2 layers; Pro has 61 layers and larger experts, so do not reuse this RAM estimate. Excessive memory pressure or other heavy apps can trigger swap and collapse decode speed.",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                }
                Stepper("GLM 5.2 · layer residenti: "
                        + (store.glmResidentLayers == 0
                           ? "auto (RAM)" : "\(store.glmResidentLayers)"),
                        value: $store.glmResidentLayers, in: 0...78, step: 1)
                Stepper("GLM 5.2 · esperti attivi: "
                        + (store.glmActiveExperts == 0
                           ? "8 (tutti)" : "\(store.glmActiveExperts)"),
                        value: $store.glmActiveExperts, in: 0...8, step: 1)
                Text("Parametri del backend GLM 5.2, applicati al prossimo caricamento. Layer residenti 0 = adattivo alla RAM fisica; ogni layer residente in più toglie ~230 MiB di SSD da ogni token. Esperti attivi sotto 8 riduce l'I/O per token al costo di qualità.")
                    .font(.caption).foregroundStyle(.secondary)
                Toggle("Mixed-quant expert cache (recommended)", isOn: $store.multiQuantCacheEnabled)
                Text("Caches every routed IQ2/Q4 layer with its real record size under the same total byte budget as the legacy cache. The M1 Pro A/B improved decode by 28.9% and cut expert reads by 31.1%, with all 64 tokens and 2,068,480 logits bit-identical. Turn OFF to restore the legacy off-class bypass. Applies on the next model load.")
                    .font(.caption).foregroundStyle(.secondary)
                Toggle("Uniform expert-cache allocation (A/B option)", isOn: $store.expertCacheUniform)
                Text(store.expertCacheUniform
                     ? "Every routed layer receives the same slot allocation."
                     : "The frozen usage profile distributes the same byte budget toward the layers that benefit most.")
                    .font(.caption).foregroundStyle(.secondary)
                Stepper("Expert look-ahead: \(store.expertLookahead == 0 ? "hash layers only" : "\(store.expertLookahead) speculative/layer")",
                        value: $store.expertLookahead, in: 0...12, step: 2)
                Text("Prefills the NEXT layer's cache slots while the current one computes: exact for the hash layers (always on), top-N from the usage prior when > 0. Speculative I/O runs in the SSD-idle window and yields to the real gather; a wrong guess only wastes idle bandwidth. Applies on the next model load — A/B the tok/s.")
                    .font(.caption).foregroundStyle(.secondary)
                Toggle("Experts via direct pread (F_NOCACHE) - recommended <=16 GB", isOn: $store.expertPreadEnabled)
                if store.expertPreadEnabled {
                    Stepper("Pread split: \(store.preadSplit)", value: $store.preadSplit,
                            in: 1...8, step: 1)
                    Text(store.expertBundleEnabled
                         ? "Il bundle/MetalIO bypassa questo split durante il decode; l'auto-tune lo salta finché il bundle è ON."
                         : "Queue depth NVMe per slab; stessi byte e stessa numerica.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Toggle("Expert bundle sidecar (contiguous slabs, ~+25% tok/s)", isOn: $store.expertBundleEnabled)
                if store.expertBundleEnabled {
                    Toggle("MetalIO direct SSD → GPU buffers (recommended preset)", isOn: $store.metalIOEnabled)
                    Text("Loads expert records directly into MTLBuffer cache slots. Enabled by the measured preset; automatically falls back to pread on an error or low bandwidth. Prefill continues to use parallel pread. Applies on the next model load.")
                        .font(.caption).foregroundStyle(.secondary)
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
                        Button(store.bundleBuildRunning
                               ? "Generating expert bundle…"
                               : "Generate expert bundle now") {
                            store.buildExpertBundleNow()
                        }
                        .disabled(store.bundleBuildRunning || store.phase == .loading)
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
                    Toggle("Q4 also q_a/kv projections (LOSSY, ~+10% measured)", isOn: $store.qkvQ4Enabled)
                    Toggle("Q4 shared-expert FFN (LOSSY, ~+7% measured)", isOn: $store.sharedQ4Enabled)
                    if store.qkvQ4Enabled {
                        Label("Also requantizes the remaining mid-size attention projections (q_a, kv) Q8→Q4_K: ~0.7 GB/token off the SSD stream for ~0.35 GB resident (+10% decode measured on M1 Pro). The existing Q4 cache is extended incrementally (~30 s once). A/B output quality before adopting.",
                              systemImage: "exclamationmark.triangle")
                            .font(.caption).foregroundStyle(.orange)
                    }
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
                .disabled(store.benchRunning)
            } else if let descriptor = store.inspectedModelDescriptor {
                Section("Backend settings") {
                    Label("\(descriptor.displayName) · \(descriptor.architecture.rawValue)",
                          systemImage: "cpu")
                    Text(descriptor.backendAvailability == .recognizedButNotImplemented
                         ? "Modello riconosciuto, ma il backend non è ancora implementato. I controlli DeepSeek-specifici restano nascosti."
                         : "Questa architettura non espone impostazioni runtime specifiche in questa build.")
                        .font(.caption).foregroundStyle(.secondary)
                }
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
