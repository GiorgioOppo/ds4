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
                VStack(spacing: 12) {
                    ProgressView(value: min(max(store.loadFraction, 0), 1))
                        .progressViewStyle(.linear)
                        .frame(maxWidth: 360)
                    Text(store.loadStage.isEmpty ? "Loading model..." : store.loadStage)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            case .needsModel, .failed:
                ContentUnavailableView {
                    Label("No Model Loaded", systemImage: "shippingbox")
                } description: {
                    Text(placeholderText)
                } actions: {
                    Button("Open Settings") { openSettings() }
                }
            }
        case .distributed:
            CoordinatorChatView(controller: dist, openSettings: openSettings)
        }
    }

    private var placeholderText: String {
        if case .failed(let message) = store.phase {
            return "Load failed: \(message)\nConfigure the model in Settings."
        }
        return "Choose a GGUF model and load it in Settings."
    }
}
