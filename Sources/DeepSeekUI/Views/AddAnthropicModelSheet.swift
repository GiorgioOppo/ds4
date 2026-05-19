import SwiftUI

/// Toolbar-launched sheet to load an Anthropic-native endpoint
/// (TODO §10.4 / T4). Lighter than `AddOpenRouterModelSheet` — we
/// don't fetch a live catalog here, because Anthropic's catalog
/// changes rarely and a hardcoded shortlist of common slugs covers
/// the typical workflow. Power users can paste any other valid
/// `model` id into the text field and load it.
///
/// "Load" triggers `ModelState.load(.anthropic(modelID:))`, which
/// validates the Keychain-stored Anthropic API key against
/// `/v1/models` and then marks the chat ready.
struct AddAnthropicModelSheet: View {
    @ObservedObject var modelState: ModelState

    @Environment(\.dismiss) private var dismiss
    @State private var customID: String = ""
    @State private var hasAPIKey: Bool = false

    /// A static shortlist of common Anthropic models. Not exhaustive
    /// — the text field below accepts any id Anthropic publishes.
    private let suggested: [(id: String, label: String, note: String)] = [
        ("claude-opus-4-5",
         "Claude Opus 4.5",
         "Frontier reasoning. Highest cost / quality."),
        ("claude-sonnet-4-5",
         "Claude Sonnet 4.5",
         "Balanced. Best $/quality for most tasks."),
        ("claude-haiku-4-5",
         "Claude Haiku 4.5",
         "Fastest + cheapest. Good for tool loops."),
        ("claude-3-5-sonnet-20241022",
         "Claude 3.5 Sonnet (legacy)",
         "Stable pin if you want reproducibility."),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            if !hasAPIKey {
                noKeyBanner
            }
            list
            customRow
            footer
        }
        .padding(20)
        .frame(width: 600, height: 480)
        .onAppear {
            hasAPIKey = KeychainStore.exists(
                account: KeychainAccount.anthropicAPIKey)
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Add Anthropic model").font(.title3.bold())
                Text("Native `api.anthropic.com/v1/messages` route — "
                     + "enables prompt caching unavailable via OpenRouter.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var noKeyBanner: some View {
        HStack(spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(Color.orange)
            Text("No Anthropic API key configured. Add it under Settings → API Keys before loading a model.")
                .font(.caption)
            Spacer()
        }
        .padding(8)
        .background(Color.orange.opacity(0.12),
                     in: RoundedRectangle(cornerRadius: 6))
    }

    private var list: some View {
        List(suggested, id: \.id) { entry in
            Button { select(modelID: entry.id) } label: {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(alignment: .firstTextBaseline) {
                        Text(entry.label).font(.callout.bold())
                        Spacer()
                        Text(entry.id)
                            .font(.caption.monospaced())
                            .foregroundStyle(.tertiary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    Text(entry.note)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
        .listStyle(.inset)
    }

    private var customRow: some View {
        HStack(spacing: 8) {
            Image(systemName: "pencil")
                .foregroundStyle(.secondary)
            TextField("…or paste a custom model id (e.g. claude-3-5-haiku-20241022)",
                       text: $customID)
                .textFieldStyle(.roundedBorder)
            Button("Load") {
                let trimmed = customID.trimmingCharacters(in: .whitespaces)
                guard !trimmed.isEmpty else { return }
                select(modelID: trimmed)
            }
            .disabled(customID.trimmingCharacters(in: .whitespaces).isEmpty)
        }
    }

    private var footer: some View {
        HStack {
            Spacer()
            Button("Close") { dismiss() }
                .keyboardShortcut(.escape)
        }
    }

    private func select(modelID: String) {
        Task {
            await modelState.load(.anthropic(modelID: modelID))
            await MainActor.run { dismiss() }
        }
    }
}
