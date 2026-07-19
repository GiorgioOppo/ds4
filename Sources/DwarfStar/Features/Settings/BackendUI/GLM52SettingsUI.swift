import SwiftUI
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
        "Benchmark di MISURA: prefill sintetico + decode greedy con il profilo streaming per fase nel referto. Il tuning DeepSeek dei knob di prefill non si applica al backend GLM; per regolare usa gli stepper qui sotto e ripeti la misura. L'auto-tune non è necessario: la residenza layer è già adattiva alla RAM."
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

/// Sezione "Memory · GLM 5.2": gli stessi sette controlli di prima, più le
/// righe Disk KV condivise e il bottone del sidecar unificato.
private struct GLM52MemorySection: View {
    @Bindable var store: ChatStore
    let diskKVRows: AnyView

    var body: some View {
        Section("Memory · GLM 5.2") {
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
            Toggle("Staging speculativo esperti (serve banda SSD di riserva)",
                   isOn: $store.glmSpeculativeExperts)
            Toggle("Tensori Q4 dal sidecar (LOSSY, ~2× meno I/O layer)",
                   isOn: $store.glmUseQ4Sidecar)
            Text("Parametri del backend GLM 5.2, applicati al prossimo caricamento. Layer residenti 0 = adattivo alla RAM fisica (~230 MiB di SSD in meno per token ciascuno). Arena esperti: ~10 MiB per slot, più riuso nel prefill. Slot streaming: ~250 MiB l'uno, più fill SSD in volo. Tensori Q4 OFF = layer Q8 dal GGUF (gli esperti unificati del sidecar restano attivi: sono lossless).")
                .font(.caption).foregroundStyle(.secondary)
            diskKVRows
            BundleBuildButton(store: store,
                              idleTitle: "Genera sidecar GLM ora",
                              busyTitle: "Generazione sidecar GLM…")
            Text("Sidecar unificato (esperti contigui + tensori layer Q4): riusa quello accanto al GGUF quando esiste; se la cartella del modello non è scrivibile (sandbox) la build va in Application Support. Ripremere riprende dal primo layer mancante.")
                .font(.caption).foregroundStyle(.secondary)
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
                Label("GLM 5.2 non ha una usage imatrix né uno slot-cache DeepSeek da profilare: la residenza dei layer è adattiva alla RAM e gli esperti passano dall'arena keyed. Regola gli stepper qui sotto e misura l'effetto con il Benchmark in Settings.",
                      systemImage: "info.circle")
                    .font(.callout).foregroundStyle(.secondary)
            }
            memorySection
        }
        .formStyle(.grouped)
        .disabled(store.benchRunning || EngineActivityGate.shared.activeOwner == .autoTune)
    }
}
