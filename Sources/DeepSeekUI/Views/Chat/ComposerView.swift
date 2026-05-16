import SwiftUI
import DeepSeekKit

/// Bottom input bar: text field + send button. Documents are now imported
/// through the dedicated `DocumentsView` (Settings → Documents) and
/// attached to the active conversation through its own picker — there is
/// no inline file attach here any more.
///
/// Send becomes Stop while the chat is in `.streaming` or `.prefilling`.
struct ComposerView: View {
    @Binding var draft: String
    let phase: GenerationPhase
    /// Gated by the chat's model-load state. When false, Send is
    /// disabled and the TextField shows a "load a model" prompt
    /// — the user can still type, the draft is preserved, but
    /// the send path won't fire on an empty service.
    var canSend: Bool = true
    var onSend: () -> Void
    var onStop: () -> Void

    // Explicit focus binding for the TextField. NSOpenPanel and other
    // modal flows can leave the key window's first-responder elsewhere;
    // we re-acquire focus at the points listed below so typing always
    // lands here.
    @FocusState private var composerFocused: Bool

    private var placeholderText: String {
        canSend
            ? "Message the model…"
            : "Load a model from the toolbar to start chatting"
    }

    var body: some View {
        HStack(alignment: .bottom, spacing: 12) {
            TextField(placeholderText, text: $draft, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...8)
                .focused($composerFocused)
                .onSubmit { if canSend { onSend() } }

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
                .disabled(!canSend
                          || draft.trimmingCharacters(in: .whitespaces).isEmpty)
                .keyboardShortcut(.return, modifiers: .command)
                .controlSize(.large)
            }
        }
        .padding(12)
        .background(.bar)
        .onAppear { composerFocused = true }
        .onChange(of: phase) { _, newPhase in
            if case .idle = newPhase { composerFocused = true }
        }
    }
}
