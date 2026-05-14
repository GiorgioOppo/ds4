import SwiftUI
import DeepSeekKit

/// Bottom input bar: multiline text field + send button. Send becomes
/// Stop while the chat is in `.streaming` phase (commit 7 wires the
/// stop action through to `ChatStore.cancel`).
struct ComposerView: View {
    @Binding var draft: String
    let phase: GenerationPhase
    var onSend: () -> Void
    var onStop: () -> Void

    var body: some View {
        HStack(alignment: .bottom, spacing: 12) {
            TextField("Message the model…", text: $draft, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...8)
                .onSubmit(onSend)

            switch phase {
            case .streaming, .prefilling:
                Button(action: onStop) {
                    Label("Stop", systemImage: "stop.fill")
                }
                .tint(.red)
                .controlSize(.large)
            default:
                Button(action: onSend) {
                    Label("Send", systemImage: "paperplane.fill")
                }
                .disabled(draft.trimmingCharacters(in: .whitespaces).isEmpty)
                .keyboardShortcut(.return, modifiers: .command)
                .controlSize(.large)
            }
        }
        .padding(12)
        .background(.bar)
    }
}
