import SwiftUI
import DeepSeekTools

/// Modal sheet sollevato da `NativeToolHost.pendingRequest` quando
/// una tool nativa richiede il consenso dell'utente. Senza questo
/// presenter, le call gated (shell, write, edit, apply_patch,
/// repo_clone, …) bloccano la `withCheckedContinuation` in
/// `NativeToolHost.present(request:)` per sempre — il loop agentico
/// rimane fermo in `Thinking…` perché la tool non torna mai.
///
/// Tre azioni:
///   - `Deny`: il registry torna `.deny`, il dispatch ritorna
///     un `ToolOutput.error` che il modello vede come `[error: …]`.
///   - `Allow once`: la singola call procede; le successive sullo
///     stesso `(tool, category)` chiedono di nuovo.
///   - `Always allow`: la decisione viene cached per la sessione
///     (vedi `ToolRegistry.sessionAllowCache`).
struct PermissionSheet: View {
    let request: PermissionRequest
    let onDecision: (PermissionDecision) -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: iconName)
                    .font(.system(size: 28))
                    .foregroundStyle(tint)
                    .frame(width: 36)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Tool permission request")
                        .font(.title3.bold())
                    HStack(spacing: 8) {
                        Text(request.tool)
                            .font(.callout.monospaced())
                        Text("·")
                            .foregroundStyle(.tertiary)
                        Text(request.category.rawValue)
                            .font(.caption.monospaced())
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(tint.opacity(0.18),
                                         in: RoundedRectangle(cornerRadius: 4))
                            .foregroundStyle(tint)
                        Text("·")
                            .foregroundStyle(.tertiary)
                        Text(request.mode.displayName)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }
                Spacer()
            }

            Divider()

            Text(request.summary)
                .font(.callout)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)

            if let detail = request.detail, !detail.isEmpty {
                ScrollView(.vertical, showsIndicators: true) {
                    Text(detail)
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(8)
                }
                .frame(maxHeight: 120)
                .background(Color(NSColor.controlBackgroundColor),
                             in: RoundedRectangle(cornerRadius: 6))
            }

            Spacer(minLength: 8)

            HStack {
                Button(role: .destructive) {
                    onDecision(.deny)
                    dismiss()
                } label: {
                    Text("Deny")
                }
                .keyboardShortcut(.escape)
                Spacer()
                Button("Allow once") {
                    onDecision(.allowOnce)
                    dismiss()
                }
                Button("Always allow this session") {
                    onDecision(.alwaysAllow)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(20)
        .frame(width: 480, minHeight: 240)
    }

    private var iconName: String {
        switch request.category {
        case .readOnly:  return "doc.text.magnifyingglass"
        case .planning:  return "list.bullet"
        case .mutating:  return "pencil.line"
        case .network:   return "globe"
        case .dangerous: return "exclamationmark.triangle.fill"
        }
    }

    private var tint: Color {
        switch request.category {
        case .readOnly, .planning: return .secondary
        case .mutating:            return .orange
        case .network:             return .blue
        case .dangerous:           return .red
        }
    }
}
