import SwiftUI
import AppKit

/// First-launch surface: invites the user to pick the model directory
/// (must contain `tokenizer.json`, `config.json`, and `*.safetensors`
/// shards). Opens an NSOpenPanel restricted to directories.
struct ModelPickerView: View {
    /// Called with the URL the user selects. Caller decides what to do
    /// next (typically: kick off `InferenceService.loadModel`).
    var onSelect: (URL) -> Void

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
        }
        .padding(48)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
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

#Preview {
    ModelPickerView { url in print("picked \(url)") }
}
