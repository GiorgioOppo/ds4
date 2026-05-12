import SwiftUI
import AppKit

/// First-launch surface: invites the user to pick the model directory
/// (must contain `tokenizer.json`, `config.json`, and `*.safetensors`
/// shards). Opens an NSOpenPanel restricted to directories. Also
/// surfaces a "Recent" list pulled from `AppSettings.recentDirs()`.
struct ModelPickerView: View {
    /// Called with the URL the user selects. Caller decides what to do
    /// next (typically: kick off `InferenceService.loadModel`).
    var onSelect: (URL) -> Void
    @State private var recents: [String] = AppSettings.recentDirs()

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "cpu")
                .font(.system(size: 56, weight: .light))
                .foregroundStyle(.tint)
            VStack(spacing: 6) {
                Text("DeepSeek-V4")
                    .font(.largeTitle)
                Text("Choose a converted model directory to begin.")
                    .foregroundStyle(.secondary)
            }
            Button("Choose Model Folder…", action: showPanel)
                .keyboardShortcut(.defaultAction)
                .controlSize(.large)
            Text("Expected contents: tokenizer.json, config.json, model-*.safetensors")
                .font(.caption)
                .foregroundStyle(.tertiary)

            if !recents.isEmpty {
                Divider().padding(.horizontal, 80)
                VStack(alignment: .leading, spacing: 6) {
                    Text("Recent").font(.caption).foregroundStyle(.secondary)
                    ForEach(recents, id: \.self) { path in
                        HStack {
                            Button(action: { select(path) }) {
                                Text(path)
                                    .font(.system(.callout, design: .monospaced))
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                            }
                            .buttonStyle(.link)
                            Spacer()
                            Button {
                                AppSettings.forgetRecentDir(path)
                                recents = AppSettings.recentDirs()
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.tertiary)
                            }
                            .buttonStyle(.borderless)
                        }
                    }
                }
                .frame(maxWidth: 500, alignment: .leading)
            }
        }
        .padding(48)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func select(_ path: String) {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: path) else {
            AppSettings.forgetRecentDir(path)
            recents = AppSettings.recentDirs()
            return
        }
        onSelect(url)
    }

    private func showPanel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        panel.message = "Select the converted model directory."
        if panel.runModal() == .OK, let url = panel.url {
            onSelect(url)
        }
    }
}
