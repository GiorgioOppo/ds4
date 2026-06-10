import SwiftUI
import DS4Engine

/// Runtime-tuning panel: expert slot-cache configuration ("persistent + changing
/// experts") and the routing-usage profile (the "usage imatrix") that pre-warms
/// it. Weight-level fine-tuning is NOT possible on-device (see the note below).
struct TuningView: View {
    @Bindable var store: ChatStore

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section {
                    Label("Il fine-tuning dei pesi non è possibile on-device: il motore è solo-inferenza (nessun backward pass) e i pesi 2-bit quantizzati non sono addestrabili. Questa scheda ottimizza il runtime: quali esperti tenere residenti in RAM e il profilo d'uso che li seleziona.",
                          systemImage: "info.circle")
                        .font(.callout).foregroundStyle(.secondary)
                }

                Section("Cache esperti — persistenti + dinamici") {
                    Stepper("Slot per layer: \(store.expertCacheSlots == 0 ? "off" : "\(store.expertCacheSlots)")",
                            value: $store.expertCacheSlots, in: 0...64, step: 8)
                    Text("Ogni slot tiene un esperto residente in GPU (≈7 MB/slot × 43 layer sul modello 2-bit: 8 slot ≈ 2,4 GB wired). Gli esperti caldi restano in RAM (hit = zero copie); i freddi ruotano via LRU. Si applica al prossimo caricamento del modello.")
                        .font(.caption).foregroundStyle(.secondary)
                    if let info = store.tuningInfo, info.cacheHits + info.cacheMisses > 0 {
                        let rate = Double(info.cacheHits) / Double(info.cacheHits + info.cacheMisses) * 100
                        LabeledContent("Hit rate",
                                       value: String(format: "%.0f%%  (%d hit / %d miss)", rate, info.cacheHits, info.cacheMisses))
                        Text(rate < 15
                             ? "Hit rate basso: il routing è quasi uniforme su questo carico — la cache non sta ripagando, valuta di spegnerla."
                             : "Hit rate utile: gli esperti persistenti stanno risparmiando I/O.")
                            .font(.caption)
                            .foregroundStyle(rate < 15 ? .orange : .green)
                    }
                }

                Section("Profilo d'uso esperti (\"imatrix d'uso\")") {
                    LabeledContent("Agente attivo", value: store.selectedAgent.name)
                    if let info = store.tuningInfo {
                        LabeledContent("Routing registrati", value: "\(info.totalRoutes)")
                    }
                    Text("Conta quanto spesso il router sceglie ogni esperto nel TUO uso reale. Il profilo è PER-AGENTE (ruoli diversi instradano verso esperti diversi): cambiando agente la cache si ri-scalda con il suo profilo. Persistito tra le sessioni.")
                        .font(.caption).foregroundStyle(.secondary)
                    HStack {
                        Button { store.refreshTuningInfo() } label: {
                            Label("Aggiorna", systemImage: "arrow.clockwise")
                        }
                        Button { store.saveExpertUsage() } label: {
                            Label("Salva profilo", systemImage: "square.and.arrow.down")
                        }
                        Button(role: .destructive) { store.resetExpertUsage() } label: {
                            Label("Azzera", systemImage: "trash")
                        }
                    }
                    .disabled(!store.isReady)
                    if !store.isReady {
                        Text("Carica un modello nella scheda Chat per raccogliere il profilo.")
                            .font(.caption).foregroundStyle(.tertiary)
                    }
                }
            }
            .formStyle(.grouped)

            Divider()

            // Per-layer concentration: the honest signal for cache viability.
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Concentrazione per layer (quota dei routing catturata dai top-8 esperti; ~3% = uniforme, alto = cache conveniente)")
                        .font(.caption.bold()).foregroundStyle(.secondary)
                        .padding(.bottom, 4)
                    if let info = store.tuningInfo, !info.layerSummaries.isEmpty {
                        ForEach(info.layerSummaries, id: \.self) { line in
                            Text(line)
                                .font(.system(.caption2, design: .monospaced))
                                .textSelection(.enabled)
                        }
                    } else {
                        Text("Nessun dato — genera qualche risposta e premi Aggiorna.")
                            .font(.caption).foregroundStyle(.tertiary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(8)
            }
            .background(Color.black.opacity(0.05))
        }
        .onAppear { store.refreshTuningInfo() }
    }
}
