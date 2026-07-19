import SwiftUI
import AppKit
import DS4Engine

/// Implementazione GLM 5.2 della UI per-backend. Prima di questa classe i
/// knob GLM stavano DENTRO la sezione "Memory · DeepSeek V4" (visibili solo
/// con un file DeepSeek selezionato) e il benchmark GLM esisteva nel codice
/// ma nessun bottone lo raggiungeva. Ora la sezione appare quando il modello
/// selezionato È GLM, con benchmark, Disk KV (checkpoint singolo, senza
/// budget) e build del sidecar unificato.
@MainActor
final class GLM52SettingsUI: BackendSettingsUI {
    override var backendName: String? { "GLM 5.2" }
    override var supportsBenchmark: Bool { true }
    override var quickBenchmarkTitle: String { "Rapido (64+8 token, ~2-3 min)" }
    override var fullBenchmarkTitle: String { "Completo (192+16 token, ~6-10 min)" }
    override var benchmarkCaption: String {
        "Benchmark di MISURA: prefill sintetico + decode greedy con il profilo per fase in stile DeepSeek (ms/token, quota GPU/host, hit della cache) nel referto. Il tuning DeepSeek dei knob di prefill non si applica al backend GLM; per regolare usa gli stepper qui sotto e ripeti la misura. Non esiste ancora un auto-tune GLM: il preset misurato del bottone qui sopra è il punto di partenza."
    }

    override func benchmarkExtras() -> AnyView {
        AnyView(GLM52AutoTuneExtras(store: store))
    }

    override func memorySection() -> AnyView {
        AnyView(GLM52MemorySection(
            store: store,
            diskKVRows: diskKVRows(
                showBudget: false,
                note: "GLM usa un checkpoint singolo per modello (state.glmkv): riaprire una chat ripristina il prefisso più lungo dell'ultima conversazione. Il budget in token si applica solo allo store DeepSeek.")))
    }

    override func tuningPanel() -> AnyView? {
        AnyView(GLM52TuningPanel(store: store,
                                 memorySection: memorySection()))
    }
}

/// Coda della sezione benchmark: l'auto-tune GLM dei knob di caricamento
/// (parità col bottone record-holder DeepSeek, in versione pragmatica).
private struct GLM52AutoTuneExtras: View {
    @Bindable var store: ChatStore

    var body: some View {
        Button("Auto-tune GLM (~10 min)") { store.runGLMAutoTune() }
            .disabled(store.benchRunning || store.phase != .ready
                      || store.isGenerating)
        Text("Prova le alternative ESATTE dei knob di caricamento (MetalIO, slot streaming, arena esperti, layer residenti, fusione commit) un gradino alla volta, ricaricando il motore per ogni misura (~3 s a load — il vantaggio dello streaming GLM). I logits non cambiano: i knob di qualità restano tuoi. I vincitori vengono applicati, persistiti e il modello ricaricato con la configurazione campione.")
            .font(.caption).foregroundStyle(.secondary)
    }
}

/// Sezione "Memory · GLM 5.2": gli stessi sette controlli di prima, più le
/// righe Disk KV condivise e il bottone del sidecar unificato.
private struct GLM52MemorySection: View {
    @Bindable var store: ChatStore
    let diskKVRows: AnyView

    var body: some View {
        Section("Memory · GLM 5.2") {
            HStack(spacing: 8) {
                Button("Applica preset veloce misurato") {
                    store.applyGLMFastDefaults()
                }
                Text("Preset M1 Pro 16 GB (decode misurato: 6.0 → 4.0 s/token senza sidecar, ~3.1 col sidecar Q4 a 53/75 layer): MetalIO OFF (+18% — contende i commit GPU del decode), 6 esperti attivi (−25% I/O esperti, lieve trade-off qualità), residenza/arena/slot ai default adattivi. Fusione commit, prefill parallelo, kernel vettorizzati, MoE batched e mlock dei residenti sono già default del motore. Si applica al prossimo caricamento.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Stepper("Layer residenti: "
                    + (store.glmResidentLayers == 0
                       ? "auto (RAM)" : "\(store.glmResidentLayers)"),
                    value: $store.glmResidentLayers, in: 0...78, step: 1)
            Stepper("Esperti attivi: "
                    + (store.glmActiveExperts == 0
                       ? "8 (tutti)" : "\(store.glmActiveExperts)"),
                    value: $store.glmActiveExperts, in: 0...8, step: 1)
            Stepper("Arena esperti: "
                    + (store.glmExpertArena == 0
                       ? "24 slot (default)"
                       : "\(store.glmExpertArena) slot"),
                    value: $store.glmExpertArena, in: 0...96, step: 8)
            Stepper("Slot streaming layer: "
                    + (store.glmStreamSlots == 0
                       ? "3 (default)" : "\(store.glmStreamSlots)"),
                    value: $store.glmStreamSlots, in: 0...6, step: 1)
            Toggle("MetalIO SSD → GPU (fallback pread automatico)",
                   isOn: $store.glmMetalIOEnabled)
            Text("Misurato su M1 Pro 16 GB: OFF è +18% in decode — i trasferimenti MetalIO si accodano davanti ai commit sincroni del token. Su macchine con più corsie GPU/IO può vincere ON: misura col benchmark.")
                .font(.caption).foregroundStyle(.secondary)
            Toggle("Staging speculativo esperti (serve banda SSD di riserva)",
                   isOn: $store.glmSpeculativeExperts)
            Text("Misurato NEGATIVO su SSD saturo (il traffico speculativo affama le letture demand e fa stallare il prefetch layer, anche in variante top-N). Accendilo solo con banda SSD davvero libera, e verifica col benchmark.")
                .font(.caption).foregroundStyle(.secondary)
            Toggle("Tensori Q4 dal sidecar (LOSSY, ~2× meno I/O layer)",
                   isOn: $store.glmUseQ4Sidecar)
            DisclosureGroup("Ottimizzazioni misurate (default ON — spegnere solo per diagnosi)") {
                Toggle("Fusione commit (FFN N + trunk N+1, ~metà attese)",
                       isOn: $store.glmFuseEnabled)
                Toggle("MoE batched (esperti instradati in 2 dispatch)",
                       isOn: $store.glmMoEBatchEnabled)
                Toggle("Router fuso su GPU (−18% prefill misurato)",
                       isOn: $store.glmGpuRouterEnabled)
                Toggle("mlock dei pesi residenti (−394 ms/token sul head)",
                       isOn: $store.glmMlockEnabled)
                Stepper("Pread split prefill: "
                        + (store.glmReadSplit == 0
                           ? "4 (default)" : "\(store.glmReadSplit)"),
                        value: $store.glmReadSplit, in: 0...8, step: 1)
                Stepper("NSG kernel matvec: "
                        + (store.glmNSG == 0
                           ? "4 (default)" : "\(store.glmNSG)"),
                        value: $store.glmNSG, in: 0...8, step: 1)
                Text("Ognuna è il default del motore col verdetto misurato in sessione; i toggle esistono per A/B e diagnosi (i percorsi di riferimento restano nel codice). Si applicano al prossimo caricamento.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text("Parametri del backend GLM 5.2, applicati al prossimo caricamento. Layer residenti 0 = adattivo: su RAM stretta resta al minimo dei 3 layer dense — misurato: pochi residenti extra vengono paginati dal sistema e costano ~750 ms/token di residency ai commit, più di quanto risparmino di SSD. Arena esperti: ~10 MiB per slot, più riuso nel prefill. Slot streaming: ~250 MiB l'uno (con la fusione commit il motore ne usa almeno 4). Tensori Q4 OFF = layer Q8 dal GGUF (gli esperti unificati del sidecar restano attivi: sono lossless).")
                .font(.caption).foregroundStyle(.secondary)
            diskKVRows
            BundleBuildButton(store: store,
                              idleTitle: "Genera sidecar GLM ora",
                              busyTitle: "Generazione sidecar GLM…")
            Text("Sidecar unificato (esperti contigui + tensori layer Q4), stessa politica del bundle DeepSeek: riusa quello accanto al GGUF quando già esiste (es. costruito dalla demo); altrimenti l'app lo possiede in Application Support senza toccare la cartella del modello. Ripremere riprende dal primo layer mancante; QUALSIASI sottoinsieme di layer è utile.")
                .font(.caption).foregroundStyle(.secondary)
            if !store.modelPath.isEmpty {
                HStack(spacing: 6) {
                    Text("Sidecar dir: \(ChatStore.glmResolvedLayerQ4Directory(modelPath: store.modelPath))")
                        .font(.caption2).foregroundStyle(.tertiary)
                        .textSelection(.enabled)
                        .lineLimit(2)
                    Button("Show in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting(
                            [URL(fileURLWithPath: ChatStore
                                .glmResolvedLayerQ4Directory(
                                    modelPath: store.modelPath))])
                    }
                    .font(.caption2)
                }
            }
            Text("Applies on the next model load.")
                .font(.caption).foregroundStyle(.tertiary)
        }
        .disabled(store.benchRunning)
    }
}

/// Pannello Tuning GLM: il tuning runtime coincide con gli stepper del
/// backend (la residenza è adattiva, non c'è usage imatrix); il pannello li
/// riespone accanto a una nota che rimanda al benchmark per la misura.
private struct GLM52TuningPanel: View {
    @Bindable var store: ChatStore
    let memorySection: AnyView

    var body: some View {
        Form {
            Section {
                Label("GLM 5.2 persiste una usage imatrix per modello (.glm-usage.json) ma non ha ancora uno slot-cache per-layer che la consumi come DeepSeek: gli esperti passano dall'arena keyed condivisa. Il routing GLM misurato è quasi uniforme (top esperti globali ≈ 15% delle route a 2 GB), quindi un hit-rate basso qui sotto è atteso, non un guasto. Regola gli stepper e misura l'effetto con il Benchmark in Settings.",
                      systemImage: "info.circle")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Section("Arena esperti — hit/miss") {
                if let counters = store.glmArenaCounters,
                   counters.hits + counters.misses > 0 {
                    let rate = Double(counters.hits)
                        / Double(counters.hits + counters.misses) * 100
                    LabeledContent("Hit rate", value: String(
                        format: "%.0f%%  (%d hit / %d miss)",
                        rate, counters.hits, counters.misses))
                    Text(rate < 15
                         ? "Hit rate basso: per questo carico l'arena non sta pagando (atteso col routing quasi uniforme di GLM; il riuso arriva soprattutto dal prefill union e dalla speculazione)."
                         : "Hit rate utile: i record riusati stanno risparmiando I/O.")
                        .font(.caption)
                        .foregroundStyle(rate < 15 ? .orange : .green)
                } else {
                    Text("Nessun dato: genera qualche risposta e premi Aggiorna.")
                        .font(.caption).foregroundStyle(.tertiary)
                }
                Button {
                    store.refreshGLMArenaCounters()
                } label: {
                    Label("Aggiorna", systemImage: "arrow.clockwise")
                }
                .disabled(!store.isReady)
            }
            memorySection
        }
        .formStyle(.grouped)
        .onAppear { store.refreshGLMArenaCounters() }
        .disabled(store.benchRunning || EngineActivityGate.shared.activeOwner == .autoTune)
    }
}
