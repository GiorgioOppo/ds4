import SwiftUI

/// Top-level view. Routes between the model-picker (no model selected)
/// and the chat/load surfaces (model selected). Commit 1 only wires
/// the picker case; the loading + chat panes land in commits 2-3.
struct ContentView: View {
    @State private var selectedModelDir: URL?

    var body: some View {
        Group {
            if let dir = selectedModelDir {
                Text("Model dir: \(dir.path)")
                    .font(.system(.body, design: .monospaced))
                    .padding()
                // commit 2: replace with LoadingView(modelDir: dir)
            } else {
                ModelPickerView { url in
                    selectedModelDir = url
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
