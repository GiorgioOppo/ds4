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
    var onSend: () -> Void
    var onStop: () -> Void

    // Explicit focus binding for the TextField. NSOpenPanel and other
    // modal flows can leave the key window's first-responder elsewhere;
    // we re-acquire focus at the points listed below so typing always
    // lands here.
    @FocusState private var composerFocused: Bool

    var body: some View {
        HStack(alignment: .bottom, spacing: 12) {
            TextField("Message the model…", text: $draft, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...8)
                .focused($composerFocused)
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
        .onAppear { composerFocused = true }
        .onChange(of: phase) { _, newPhase in
            if case .idle = newPhase { composerFocused = true }
        }
    }
}
