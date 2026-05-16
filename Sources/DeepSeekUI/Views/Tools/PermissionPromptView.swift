import SwiftUI
import DeepSeekTools

/// Modal sheet shown when a tool dispatch hits a gated category and
/// the user hasn't pre-decided. Three buttons mirror the
/// `PermissionDecision` cases (Deny / Allow once / Always allow).
/// Cmd-period maps to Deny so the user can dismiss with the same
/// shortcut as a regular cancel.
struct PermissionPromptView: View {
    let request: PermissionRequest
    let onDecide: (PermissionDecision) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 10) {
                Image(systemName: icon(for: request.category))
                    .foregroundStyle(tint(for: request.category))
                    .font(.title2)
                VStack(alignment: .leading) {
                    Text("\(request.tool) wants to run")
                        .font(.headline)
                    Text(modeLabel)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Text(request.summary)
                .font(.body.monospaced())
                .padding(.vertical, 4)
                .padding(.horizontal, 8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary.opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            if let detail = request.detail {
                Text(detail)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            HStack {
                Button("Deny") { onDecide(.deny) }
                    .keyboardShortcut(".", modifiers: .command)
                Spacer()
                Button("Allow once") { onDecide(.allowOnce) }
                Button("Always allow") { onDecide(.alwaysAllow) }
                    .keyboardShortcut(.return)
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(20)
        .frame(width: 440)
    }

    private var modeLabel: String {
        "\(request.category.rawValue) · \(request.mode.displayName) mode"
    }

    private func icon(for category: ToolCategory) -> String {
        switch category {
        case .readOnly:  return "eye"
        case .mutating:  return "pencil"
        case .dangerous: return "exclamationmark.triangle.fill"
        case .network:   return "network"
        case .planning:  return "list.bullet.rectangle"
        }
    }

    private func tint(for category: ToolCategory) -> Color {
        switch category {
        case .readOnly:  return .blue
        case .mutating:  return .orange
        case .dangerous: return .red
        case .network:   return .purple
        case .planning:  return .green
        }
    }
}
