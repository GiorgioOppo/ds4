import SwiftUI
import DeepSeekKit

/// Bottom input bar: text field + send button + paperclip to attach
/// text documents. Attachments show up as removable chips above the
/// field; the actual contents are prepended to the message by
/// `ChatView.sendCurrent` when the user hits Send. Send becomes Stop
/// while the chat is in `.streaming` or `.prefilling`.
struct ComposerView: View {
    @Binding var draft: String
    @Binding var attachments: [DocumentAttachment]
    let phase: GenerationPhase
    var onSend: () -> Void
    var onStop: () -> Void

    // Explicit focus binding for the TextField. Without it the
    // paperclip button (the first focusable element in the HStack)
    // can end up holding the key window's first-responder slot, and
    // NSOpenPanel dismissal doesn't always hand focus back to the
    // field — the symptom is "the chat won't let me type any more".
    @FocusState private var composerFocused: Bool

    var body: some View {
        VStack(spacing: 8) {
            if !attachments.isEmpty {
                AttachmentStrip(attachments: $attachments)
            }
            HStack(alignment: .bottom, spacing: 12) {
                Button(action: pickAttachments) {
                    Label("Attach", systemImage: "paperclip")
                        .labelStyle(.iconOnly)
                }
                .controlSize(.large)
                .help("Attach text documents")
                .disabled(isGenerating)

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
                    .disabled(isSendDisabled)
                    .keyboardShortcut(.return, modifiers: .command)
                    .controlSize(.large)
                }
            }
        }
        .padding(12)
        .background(.bar)
        .onAppear {
            // Take focus on first mount and after the composer is
            // re-mounted via a sidebar selection change.
            composerFocused = true
        }
        .onChange(of: phase) { _, newPhase in
            // When the model finishes (idle), pull focus back into the
            // field so the next prompt can be typed without a click.
            if case .idle = newPhase { composerFocused = true }
        }
    }

    private var isGenerating: Bool {
        switch phase {
        case .streaming, .prefilling: return true
        default:                       return false
        }
    }

    private var isSendDisabled: Bool {
        // Allow sending with attachments + empty draft (user might want
        // the model to just react to the file). Only disable when both
        // are empty.
        let trimmed = draft.trimmingCharacters(in: .whitespaces)
        return trimmed.isEmpty && attachments.isEmpty
    }

    private func pickAttachments() {
        let picked = AttachmentPicker.present()
        // Always restore focus to the composer after dismissing
        // NSOpenPanel — even when the user cancelled — otherwise the
        // text field stays unfocused and key strokes go nowhere.
        composerFocused = true
        guard !picked.isEmpty else { return }
        attachments.append(contentsOf: picked)
    }
}

/// Horizontal strip of attachment chips above the composer. Each chip
/// shows the filename, byte size, and an "x" to remove. The strip
/// scrolls horizontally so a dozen attachments don't push the
/// composer off-screen.
private struct AttachmentStrip: View {
    @Binding var attachments: [DocumentAttachment]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(attachments) { a in
                    chip(a)
                }
            }
            .padding(.horizontal, 2)
        }
        .frame(maxHeight: 32)
    }

    @ViewBuilder
    private func chip(_ a: DocumentAttachment) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "doc.text")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(a.name)
                .font(.caption)
                .lineLimit(1)
            Text(byteString(a.byteCount))
                .font(.caption2.monospaced())
                .foregroundStyle(.tertiary)
            Button {
                attachments.removeAll { $0.id == a.id }
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.borderless)
            .help("Remove attachment")
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.secondary.opacity(0.12),
                     in: Capsule())
    }

    private func byteString(_ n: Int) -> String {
        if n < 1024 { return "\(n) B" }
        let kb = Double(n) / 1024.0
        if kb < 1024 { return String(format: "%.1f KB", kb) }
        let mb = kb / 1024.0
        return String(format: "%.1f MB", mb)
    }
}
