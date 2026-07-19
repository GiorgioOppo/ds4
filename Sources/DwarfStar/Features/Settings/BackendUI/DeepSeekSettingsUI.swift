import SwiftUI
import AppKit
import DS4Engine

/// Implementazione DeepSeek V4 della UI per-backend: sezione memoria completa
/// (cache esperti, streaming denso, Q4 lossy, bundle, Disk KV con budget,
/// Raw-KV ring), benchmark con auto-tune record-holder e pannello Tuning
/// (slot-cache + usage imatrix). È il contenuto che prima viveva hardcoded
/// in `SettingsView`/`TuningView` dietro `supportsDeepSeekPerformanceTuning`.
@MainActor
final class DeepSeekSettingsUI: BackendSettingsUI {
    override var backendName: String? { "DeepSeek V4" }
    override var supportsBenchmark: Bool { true }
    override var quickBenchmarkTitle: String { "Rapido (~3 min)" }
    override var fullBenchmarkTitle: String { "Completo (~15 min)" }
    override var benchmarkCaption: String {
        "Rapido/Completo regolano i knob del prefill sul motore già caricato. Auto-tune record-holder misura una sola volta la baseline calda e ogni configurazione unica; se una configurazione ricompare riusa il risultato in cache senza reload. Per gli expert-cache slot prova un gradino alla volta verso l'alto: dopo una promozione continua soltanto in quella direzione e si ferma al primo peggioramento; prova il vicino inferiore solo se il primo gradino superiore non vince. Il campione completo col decode valido più alto resta il confronto: lo sostituisce qualsiasi decode strettamente superiore che mantenga hash bit-esatto di tutti i logits, prefill entro −8%, stabilità ≥0,75, RAM sopra il floor immutabile e non più di 128 MiB di swap steady. Congela l'usage profile, mantiene Raw-KV ring ON e salva i parametri soltanto dopo warmup dell'agente attivo e probe steady conclusivo. Un journal transazionale ripristina i valori iniziali dopo un'interruzione."
    }

    override func benchmarkExtras() -> AnyView {
        AnyView(DeepSeekBenchmarkExtras(store: store, dist: dist))
    }

    override func memorySection() -> AnyView {
        AnyView(DeepSeekMemorySection(
            store: store,
            diskKVRows: diskKVRows(showBudget: true, note: nil)))
    }

    override func tuningPanel() -> AnyView? {
        AnyView(DeepSeekTuningPanel(store: store))
    }
}

/// Coda della sezione benchmark: auto-tune record-holder, referto e righe
/// "Attivi" (i set applicati da benchmark/auto-tune che non hanno stepper).
private struct DeepSeekBenchmarkExtras: View {
    @Bindable var store: ChatStore
    let dist: DistributedController?

    var body: some View {
        if let dist {
            Button("Auto-tune record-holder (ore)") {
                store.runAutoTune(
                    distributedRuntimeActive: dist.workerRunning || dist.connected
                        || dist.coordLoading || dist.isGenerating || dist.benchmarkActive
                )
            }
            .disabled(store.benchRunning || store.phase != .ready || store.isGenerating)
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
}

/// Sezione "Memory · DeepSeek V4" (identica a prima, senza più i knob GLM
/// che ora vivono nella sezione del loro backend).
private struct DeepSeekMemorySection: View {
    @Bindable var store: ChatStore
    let diskKVRows: AnyView

    var body: some View {
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
                BundleBuildButton(store: store,
                                  idleTitle: "Generate expert bundle now",
                                  busyTitle: "Generating expert bundle…")
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
            diskKVRows
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
    }
}

/// Pannello Tuning DeepSeek: slot-cache esperti + usage imatrix. È il corpo
/// che prima viveva in `TuningView` dietro `supportsExpertTuning`.
struct DeepSeekTuningPanel: View {
    @Bindable var store: ChatStore

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section {
                    Label("Weight fine-tuning is not possible on-device: the engine is inference-only (no backward pass), and quantized 2-bit weights are not trainable. This tab tunes runtime behavior: which experts stay resident in RAM and the usage profile that selects them.",
                          systemImage: "info.circle")
                        .font(.callout).foregroundStyle(.secondary)
                }

                Section("Expert Cache - Persistent + Dynamic") {
                    Stepper("Slots per layer: \(store.expertCacheSlots == 0 ? "off" : "\(store.expertCacheSlots)")",
                            value: $store.expertCacheSlots, in: 0...64, step: 8)
                    Text("Each slot keeps one expert resident on the GPU. On the 43-layer Flash 2-bit model it is about 7 MB/slot/layer (8 slots ≈ 2.4 GB wired); Pro has 61 layers and larger experts, so its cost is substantially higher. Hot experts stay in RAM (hit = zero copies); cold experts rotate out via LRU. Applies on the next model load.")
                        .font(.caption).foregroundStyle(.secondary)
                    Toggle("Cache mixed IQ2/Q4 layers", isOn: $store.multiQuantCacheEnabled)
                    Text("Recommended and exact: uses each layer's real expert size while preserving the legacy total RAM budget. OFF is the single-class fallback. Applies on the next model load.")
                        .font(.caption).foregroundStyle(.secondary)
                    if let info = store.tuningInfo, info.cacheHits + info.cacheMisses > 0 {
                        let rate = Double(info.cacheHits) / Double(info.cacheHits + info.cacheMisses) * 100
                        LabeledContent("Hit rate",
                                       value: String(format: "%.0f%%  (%d hit / %d miss)", rate, info.cacheHits, info.cacheMisses))
                        Text(rate < 15
                             ? "Low hit rate: routing is almost uniform for this workload, so the cache is not paying off. Consider turning it off."
                             : "Useful hit rate: resident experts are saving I/O.")
                            .font(.caption)
                            .foregroundStyle(rate < 15 ? .orange : .green)
                    }
                }

                Section("Expert Usage Profile (Usage Imatrix)") {
                    LabeledContent("Active agent", value: store.selectedAgent.name)
                    if let info = store.tuningInfo {
                        LabeledContent("Recorded routes", value: "\(info.totalRoutes)")
                    }
                    Text("Counts how often the router picks each expert in your real usage. The profile is per-agent (different roles route to different experts): switching agents warms the cache from that profile. Persisted across sessions.")
                        .font(.caption).foregroundStyle(.secondary)
                    HStack {
                        Button { store.refreshTuningInfo() } label: {
                            Label("Refresh", systemImage: "arrow.clockwise")
                        }
                        Button { store.saveExpertUsage() } label: {
                            Label("Save Profile", systemImage: "square.and.arrow.down")
                        }
                        Button(role: .destructive) { store.resetExpertUsage() } label: {
                            Label("Reset", systemImage: "trash")
                        }
                    }
                    .disabled(!store.isReady)
                    if !store.isReady {
                        Text("Load a model in Chat to collect the profile.")
                            .font(.caption).foregroundStyle(.tertiary)
                    }
                }
            }
            .formStyle(.grouped)

            Divider()

            // Per-layer concentration: the honest signal for cache viability.
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Per-layer concentration (share of routing captured by the top-8 experts; ~3% = uniform, high = cache-friendly)")
                        .font(.caption.bold()).foregroundStyle(.secondary)
                        .padding(.bottom, 4)
                    if let info = store.tuningInfo, !info.layerSummaries.isEmpty {
                        ForEach(info.layerSummaries, id: \.self) { line in
                            Text(line)
                                .font(.system(.caption2, design: .monospaced))
                                .textSelection(.enabled)
                        }
                    } else {
                        Text("No data yet - generate a few responses and press Refresh.")
                            .font(.caption).foregroundStyle(.tertiary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(8)
            }
            .background(Color.black.opacity(0.05))
        }
        .onAppear { store.refreshTuningInfo() }
        // This screen binds directly to persisted load-time knobs. Freeze
        // every control for the whole benchmark/auto-tune lease so a tab
        // switch cannot create a hybrid candidate or mutate UserDefaults
        // behind the run journal.
        .disabled(store.benchRunning || EngineActivityGate.shared.activeOwner == .autoTune)
    }
}

/// Bottone di build del sidecar condiviso dai due backend (ExpertBundleTool
/// smista da solo su bundle DeepSeek o sidecar unificato GLM).
struct BundleBuildButton: View {
    @Bindable var store: ChatStore
    let idleTitle: String
    let busyTitle: String

    var body: some View {
        HStack(spacing: 8) {
            Button(store.bundleBuildRunning ? busyTitle : idleTitle) {
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
}
