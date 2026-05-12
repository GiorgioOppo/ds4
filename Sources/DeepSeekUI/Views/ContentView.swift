import SwiftUI
import DeepSeekKit

enum AppPhase {
    case picking
    case loading(URL)
    case ready(URL, ModelConfig)
}

/// Top-level view. Owns the `InferenceService` (one per app launch)
/// and routes between picker → loading → ready (chat happy-path lands
/// in commit 3, for now ready is a placeholder).
struct ContentView: View {
    @State private var phase: AppPhase = .picking
    private let service = InferenceService()

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
                ChatContainer(store: ChatStore(
                    modelDirPath: url.path,
                    service: service))
            }
        }
    }
}

/// Hosts the `ChatStore` for the lifetime of the .ready phase and
/// lays out the NavigationSplitView (sidebar + detail).
private struct ChatContainer: View {
    @StateObject var store: ChatStore

    var body: some View {
        NavigationSplitView {
            ConversationListView(store: store)
                .frame(minWidth: 200)
        } detail: {
            ChatView(store: store)
        }
    }
}
