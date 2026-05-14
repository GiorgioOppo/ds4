import SwiftUI
import DeepSeekKit

enum AppPhase {
    case picking
    case loading(URL)
    case ready(URL, ModelConfig)
}

/// Top-level view. Receives the shared `InferenceService` from the
/// App scene and routes between picker → loading → ready.
struct ContentView: View {
    let service: InferenceService
    @State private var phase: AppPhase = .picking

    var body: some View {
        Group {
            switch phase {
            case .picking:
                ModelPickerView { url in
                    phase = .loading(url)
                }
                .onAppear {
                    // Auto-resume the previously loaded model if reachable.
                    if let last = AppSettings.lastModelDir,
                       FileManager.default.fileExists(atPath: last) {
                        phase = .loading(URL(fileURLWithPath: last))
                    }
                }

            case .loading(let url):
                LoadingView(modelDir: url,
                            service: service,
                            onLoaded: { cfg in phase = .ready(url, cfg) },
                            onCancel:  { phase = .picking })

            case .ready(let url, _):
                ChatContainer(
                    store: ChatStore(modelDirPath: url.path,
                                      service: service),
                    onUnload: { phase = .picking })
            }
        }
    }
}

/// Hosts the `ChatStore` for the lifetime of the .ready phase and
/// lays out the NavigationSplitView (sidebar + detail). An "Unload"
/// toolbar action lets the user drop back to the picker without
/// quitting the app, and a "Convert…" action opens the
/// ConvertSheet for offline checkpoint quantization.
private struct ChatContainer: View {
    @StateObject var store: ChatStore
    var onUnload: () -> Void

    @State private var showConvert: Bool = false

    var body: some View {
        NavigationSplitView {
            ConversationListView(store: store)
                .frame(minWidth: 200)
        } detail: {
            ChatView(store: store)
                .toolbar {
                    ToolbarItem {
                        Button {
                            showConvert = true
                        } label: {
                            Label("Convert model…", systemImage: "wand.and.stars")
                        }
                    }
                    ToolbarItem(placement: .destructiveAction) {
                        Button {
                            onUnload()
                        } label: {
                            Label("Unload model", systemImage: "eject")
                        }
                    }
                }
        }
        .sheet(isPresented: $showConvert) {
            ConvertSheet()
        }
    }
}
