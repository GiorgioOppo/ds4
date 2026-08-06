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
            if store.expertCacheSlots > 12 && MemoryInfo.physicalBytes < 24 * 1_073_741_824 {
                Label("On 16 GB the engine caps the wired cache automatically (≈12 slots ≈ 3.1 GB): above that, the prefill's transient buffers (~3.4 GB) plus the cache overflow RAM and swap — measured 1275s vs 123s prefill at 22 vs 11 slots. The engine log's 'budget cache esperti' line shows the value actually used; set DS4_EXPERT_CACHE_NO_CLAMP=1 to override. Closing Xcode frees ~3-4 GB more.",
                      systemImage: "exclamationmark.triangle.fill")
                    .font(.caption).foregroundStyle(.orange)
            }
            Toggle("Mixed-quant expert cache (recommended)", isOn: $store.multiQuantCacheEnabled)
            Text("Caches every routed IQ2/Q4 layer with its real record size under the same total byte budget as the legacy cache. The M1 Pro A/B improved decode by 28.9% and cut expert reads by 31.1%, with all 64 tokens and 2,068,480 logits bit-identical. Turn OFF to restore the legacy off-class bypass. Applies on the next model load.")
                .font(.caption).foregroundStyle(.secondary)
            Toggle("DSpark support (runtime sperimentale)", isOn: $store.dsparkEnabled)
            Text("Usa automaticamente il supporto originale/0731 abbinato al modello. I tre transformer Metal, Markov e confidence propongono fino a 5 token, verificati in batch dal modello principale; viene committato solo il prefisso esatto. Attivo con temperatura 0 e repetition penalty 1. Si applica al prossimo caricamento.")
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
            Toggle("Indexed DSA attention (contesti >4k) — prefill lungo ~2.7×", isOn: $store.indexedAttnEnabled)
            Text("Oltre la soglia dell'indexer (~4k token) l'attention legge SOLO le 512 righe compresse selezionate invece dell'intero contesto: prefill 8k misurato 686→256 s (32.3 tok/s, sopra il motore C di riferimento), decode −5-8% nella fascia 4-8k, inerte sotto soglia. Stesso set di righe del percorso a maschera (parità nei test). Si applica al prossimo caricamento del modello.")
                .font(.caption).foregroundStyle(.secondary)
            Toggle("Experts via direct pread (F_NOCACHE) - recommended <=16 GB", isOn: $store.expertPreadEnabled)
            if store.expertPreadEnabled {
                Stepper("Pread split: \(store.preadSplit)", value: $store.preadSplit,
                        in: 1...8, step: 1)
                Text("Queue depth NVMe per slab; default 3, stessi byte e stessa numerica.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Toggle("Dense-weight streaming (reads layer i+1 while computing layer i) - recommended <=16 GB", isOn: $store.denseStreamEnabled)
            Toggle("Pin hot buffers in RAM (mlock ~3.3 GB, keeps the memory compressor away)", isOn: $store.mlockEnabled)
            Toggle("Q4 attention projections (LOSSY, ~+30% speed)", isOn: $store.denseQ4Enabled)
            if store.denseQ4Enabled {
                Label("Requantizes the three giant attention projections Q8→Q4_K at load (~+30% tok/s measured). Slightly lossy: greedy outputs can occasionally differ while staying coherent. Requires dense-weight streaming.",
                      systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange)
                Toggle("Q4 also q_a/kv projections (LOSSY, ~+10% measured)", isOn: $store.qkvQ4Enabled)
                Toggle("Q4 shared-expert FFN (LOSSY, +7% decode but SLOWS PREFILL)",
                       isOn: $store.sharedQ4Enabled)
                if store.sharedQ4Enabled {
                    Label("Not recommended for chat: the +7% short-context decode is paid for by a prefill collapse (routed-expert phase measured 147 → 368 ms/token, bisected 2026-07-22). Leave it off unless you decode long answers from very short prompts.",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                }
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
            Toggle("Raw-KV ring: constant KV RAM", isOn: $store.rawRingEnabled)
            if store.rawRingEnabled {
                Label("Keeps the attention window plus one prefill run (\(128 + store.prefillRouteBatch) rows) in RAM instead of the full context, so raw-KV memory stays constant as the conversation grows. Verified bit-identical to the full cache.",
                      systemImage: "checkmark.seal")
                    .font(.caption).foregroundStyle(.secondary)
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
