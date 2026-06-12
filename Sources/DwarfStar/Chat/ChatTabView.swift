import SwiftUI

/// The Chat tab. A segmented switch chooses how the chat runs:
///   • Locale       — the in-process engine on this Mac (ContentView).
///   • Distribuito  — this Mac coordinates the worker cluster (CoordinatorChatView).
/// Each mode shows its own config then chat; the WORKER role lives in the
/// separate "Worker" sidebar tab.
struct ChatTabView: View {
    @Bindable var store: ChatStore
    @Bindable var dist: DistributedController
    @State private var mode: Mode = .local

    enum Mode: String, CaseIterable, Identifiable {
        case local = "Locale"
        case distributed = "Distribuito"
        var id: String { rawValue }
    }

    var body: some View {
        VStack(spacing: 0) {
            Picker("Modo", selection: $mode) {
                ForEach(Mode.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .fixedSize()
            .padding(.horizontal).padding(.vertical, 8)
            Divider()
            switch mode {
            case .local:       ContentView(store: store)
            case .distributed: CoordinatorChatView(controller: dist)
            }
        }
    }
}
