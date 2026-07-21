import SwiftUI
import DS4Engine

/// Classe base "astratta" della UI per-backend: implementa le PARTI COMUNI
/// (scaffold del benchmark, righe Disk KV, sezione informativa di fallback)
/// e definisce i punti di override che le sottoclassi concrete riempiono.
/// `DeepSeekSettingsUI` e `GLM52SettingsUI` sono le due implementazioni; la
/// factory `make` sceglie in base all'architettura del modello caricato o
/// ispezionato, così Settings e Tuning mostrano SEMPRE i controlli del
/// backend selezionato — non più un'unica sezione gated su una capability
/// DeepSeek che nascondeva i knob GLM proprio quando servivano.
///
/// Swift non ha classi astratte: la base è istanziabile di proposito e fa da
/// fallback neutro (Qwen/sconosciuto/nessun modello) mostrando solo la
/// sezione informativa.
@MainActor
class BackendSettingsUI {
    let store: ChatStore
    /// Serve solo al bottone auto-tune DeepSeek (stato del runtime
    /// distribuito). Nil quando il chiamante non ha un controller (Tuning).
    let dist: DistributedController?

    required init(store: ChatStore, dist: DistributedController?) {
        self.store = store
        self.dist = dist
    }

    /// Sceglie l'implementazione dal modello CARICATO (info) o, prima del
    /// load, da quello ispezionato — stessa precedenza di
    /// `ChatStore.modelCapabilities`.
    static func make(store: ChatStore,
                     dist: DistributedController?) -> BackendSettingsUI {
        guard let architecture = store.info?.architecture
            ?? store.inspectedModelDescriptor?.architecture else {
            return BackendSettingsUI(store: store, dist: dist)
        }
        if architecture == .deepSeekV4 {
            return DeepSeekSettingsUI(store: store, dist: dist)
        }
        if architecture == GLM52BackendDefinition.supportedArchitecture,
           GLM52BackendDefinition.runtimeEnabled {
            return GLM52SettingsUI(store: store, dist: dist)
        }
        return BackendSettingsUI(store: store, dist: dist)
    }

    // MARK: - Punti di override (la superficie per-backend)

    /// Nome del backend nei titoli di sezione; nil = nessuna sezione
    /// benchmark/memoria (solo fallback informativo).
    var backendName: String? { nil }
    var supportsBenchmark: Bool { false }
    var quickBenchmarkTitle: String { "Rapido" }
    var fullBenchmarkTitle: String { "Completo" }
    var benchmarkCaption: String { "" }

    /// Sezione memoria/knob del backend. La base mostra la sezione
    /// informativa di fallback per le architetture senza controlli.
    func memorySection() -> AnyView {
        guard let descriptor = store.inspectedModelDescriptor else {
            return AnyView(EmptyView())
        }
        return AnyView(BackendFallbackSection(descriptor: descriptor))
    }

    /// Contenuto extra in coda alla sezione benchmark (auto-tune, righe
    /// "Attivi" DeepSeek). La base non aggiunge nulla.
    func benchmarkExtras() -> AnyView { AnyView(EmptyView()) }

    /// Pannello della scheda Tuning; nil = ContentUnavailableView.
    func tuningPanel() -> AnyView? { nil }

    // MARK: - UI comune implementata nella base

    /// Scaffold del benchmark, identico per ogni backend: Rapido/Completo,
    /// Stop, progresso, stato e referto. `runSettingsBenchmark` smista già
    /// da solo sul motore vivo (DeepSeek o GLM).
    func benchmarkSection() -> AnyView {
        guard supportsBenchmark, let name = backendName else {
            return AnyView(EmptyView())
        }
        return AnyView(BackendBenchmarkSection(
            store: store,
            title: name,
            quickTitle: quickBenchmarkTitle,
            fullTitle: fullBenchmarkTitle,
            caption: benchmarkCaption,
            extras: benchmarkExtras()))
    }

    /// Righe Disk KV condivise: toggle e budget in token sono comuni a
    /// entrambi gli store (riuso prefissi tra sessioni con eviction);
    /// cambia solo il costo per token, da cui la stima in GB.
    func diskKVRows(showBudget: Bool, gbPerKTok: Double = 0.022,
                    note: String?) -> AnyView {
        AnyView(DiskKVRows(store: store, showBudget: showBudget,
                           gbPerKTok: gbPerKTok, note: note))
    }
}

// MARK: - Viste comuni della base

/// Sezione benchmark condivisa: stessa struttura per entrambi i backend.
private struct BackendBenchmarkSection: View {
    @Bindable var store: ChatStore
    let title: String
    let quickTitle: String
    let fullTitle: String
    let caption: String
    let extras: AnyView

    var body: some View {
        Section("Benchmark \(title)") {
            HStack(spacing: 8) {
                Button(store.benchRunning ? "Benchmark in corso…" : quickTitle) {
                    store.runSettingsBenchmark(quick: true)
                }
                .disabled(store.benchRunning || store.phase != .ready || store.isGenerating)
                Button(fullTitle) {
                    store.runSettingsBenchmark(quick: false)
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
            if !caption.isEmpty {
                Text(caption)
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let status = store.benchStatus {
                Label(status, systemImage: store.benchRunning ? "hourglass" :
                      (store.benchSucceeded == false ? "xmark.circle" : "checkmark.circle"))
                    .font(.caption)
            }
            if !store.benchResults.isEmpty {
                Text(store.benchResults)
                    .font(.system(.caption2, design: .monospaced))
                    .textSelection(.enabled)
                    .foregroundStyle(.secondary)
            }
            extras
        }
    }
}

/// Righe Disk KV condivise dai due backend.
private struct DiskKVRows: View {
    @Bindable var store: ChatStore
    let showBudget: Bool
    let gbPerKTok: Double
    let note: String?

    var body: some View {
        Toggle("Disk KV (reuse prefixes across sessions)", isOn: $store.diskKVEnabled)
        if store.diskKVEnabled {
            if showBudget {
                Stepper("Budget: \(store.diskKVBudgetKTok)k tokens (≈ \(String(format: "%.1f", Double(store.diskKVBudgetKTok) * gbPerKTok)) GB)",
                        value: $store.diskKVBudgetKTok, in: 128...4096, step: 128)
            }
            if let note {
                Text(note)
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

/// Fallback per architetture senza controlli runtime (Qwen, sconosciute).
private struct BackendFallbackSection: View {
    let descriptor: RuntimeModelDescriptor

    var body: some View {
        Section("Backend settings") {
            Label("\(descriptor.displayName) · \(descriptor.architecture.rawValue)",
                  systemImage: "cpu")
            Text(descriptor.backendAvailability == .recognizedButNotImplemented
                 ? "Modello riconosciuto, ma il backend non è ancora implementato. I controlli specifici del backend restano nascosti."
                 : "Questa architettura non espone impostazioni runtime specifiche in questa build.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}
