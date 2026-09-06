import SwiftUI

/// The Chat tab. The engine (local or distributed) is chosen ONCE in the
/// Impostazioni tab; this view just renders the right chat for it:
///   • Locale       — the in-process engine (ChatView once loaded).
///   • Distribuito  — the coordinator chat across the worker cluster.
/// When the engine isn't ready, a placeholder points to Impostazioni.
struct ChatTabView: View {
    @Bindable var store: ChatStore
    @Bindable var dist: DistributedController
    let settings: AppSettings
    var openSettings: () -> Void = {}

    var body: some View {
        switch settings.mode {
        case .local:
            switch store.phase {
            case .ready:
                ChatView(store: store)
            case .loading:
                VStack(spacing: 18) {
                    Image(systemName: "shippingbox")
                        .font(.system(size: 36, weight: .light))
                        .foregroundStyle(.tint)
                    Text("Preparazione del modello")
                        .font(.title2.weight(.semibold))
                    Text(store.inspectedModelDescriptor?.displayName ?? "Modello GGUF locale")
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                    ProgressView(value: min(max(store.loadFraction, 0), 1))
                        .progressViewStyle(.linear)
                        .frame(maxWidth: 360)
                        .accessibilityLabel("Caricamento del modello")
                    // Percentuale a 0,1%: le fasi lunghe (riquantizzazione Q4)
                    // muovono la barra di frazioni di punto — il numero prova
                    // che il caricamento sta AVANZANDO anche quando la barra
                    // sembra ferma.
                    Text((store.loadStage.isEmpty ? "Caricamento in corso…" : store.loadStage)
                         + String(format: " · %.1f%%", min(max(store.loadFraction, 0), 1) * 100))
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                    Text("Il primo avvio può richiedere più tempo per preparare i kernel Metal. Puoi continuare a usare la barra laterale.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: 390)
                }
                .multilineTextAlignment(.center)
                .padding(32)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            case .failed(let message):
                ContentUnavailableView {
                    Label("Impossibile caricare il modello", systemImage: "exclamationmark.triangle")
                } description: {
                    Text(message)
                        .textSelection(.enabled)
                } actions: {
                    Button("Controlla le impostazioni", action: openSettings)
                        .buttonStyle(.borderedProminent)
                    Button("Riprova") { store.load() }
                        .disabled(store.modelPath.isEmpty)
                }
            case .needsModel:
                welcome
            }
        case .distributed:
            CoordinatorChatView(controller: dist, openSettings: openSettings)
        }
    }

    private var welcome: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                Image(systemName: "bubble.left.and.bubble.right")
                    .font(.system(size: 42, weight: .light))
                    .foregroundStyle(.tint)
                VStack(alignment: .leading, spacing: 8) {
                    Text("La tua prossima idea, in locale.")
                        .font(.largeTitle.weight(.semibold))
                    Text("Prepara un modello sul Mac per scrivere, esplorare documenti e lavorare sul codice.")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }
                VStack(alignment: .leading, spacing: 18) {
                    setupStep("1", title: "Scegli un modello", detail: "Scaricalo dal catalogo oppure seleziona un file GGUF già presente sul Mac.")
                    setupStep("2", title: "Carica il modello", detail: "Le impostazioni ti aiutano a scegliere una configurazione adatta alla memoria disponibile.")
                    setupStep("3", title: "Inizia una conversazione", detail: "Torna qui: le conversazioni vengono salvate automaticamente.")
                }
                Button(action: openSettings) {
                    Label("Scegli e carica un modello", systemImage: "arrow.right")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }
            .frame(maxWidth: 540, alignment: .leading)
            .padding(40)
            .frame(maxWidth: .infinity, alignment: .center)
        }
        .background(Color.secondary.opacity(0.025))
    }

    private func setupStep(_ number: String, title: String, detail: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text(number)
                .font(.callout.weight(.semibold))
                .foregroundStyle(.tint)
                .frame(width: 28, height: 28)
                .background(Color.accentColor.opacity(0.1), in: Circle())
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.headline)
                Text(detail).font(.callout).foregroundStyle(.secondary)
            }
        }
    }
}
