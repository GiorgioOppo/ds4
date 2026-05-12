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

            case .ready(let url, let cfg):
                readyPlaceholder(url: url, cfg: cfg)
            }
        }
    }

    /// Commit 3 replaces this with the chat surface (NavigationSplitView).
    @ViewBuilder
    private func readyPlaceholder(url: URL, cfg: ModelConfig) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Model ready", systemImage: "checkmark.seal.fill")
                .foregroundStyle(.green)
                .font(.title3)
            Text(url.path)
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(.secondary)
            Divider()
            Text("layers: \(cfg.nLayers) · heads: \(cfg.nHeads) · dim: \(cfg.dim)")
                .font(.callout)
            Text("Chat UI lands in the next commit.")
                .foregroundStyle(.tertiary)
                .padding(.top, 12)
            HStack {
                Button("Unload & pick another") {
                    phase = .picking
                }
                Spacer()
            }
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
