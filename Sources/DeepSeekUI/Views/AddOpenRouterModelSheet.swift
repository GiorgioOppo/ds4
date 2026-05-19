import SwiftUI

/// Sheet launched from the toolbar's model picker that lets the
/// user pick an OpenRouter-hosted model to load. The catalog
/// auto-refreshes if it's older than 24 h; the picker has a
/// manual reload button for when the user just updated.
///
/// "Load" triggers `ModelState.load(.openRouter(modelID:))`, which
/// validates against `/auth/key` (so a missing key surfaces here
/// instead of failing on the first turn) and then marks the chat
/// ready immediately — no weights to map.
struct AddOpenRouterModelSheet: View {
    @ObservedObject var catalog: OpenRouterCatalog
    @ObservedObject var modelState: ModelState
    /// Opzionale: quando presente, il sheet lega l'endpoint scelto
    /// alla chat selezionata invece di limitarsi a chiamare
    /// `modelState.load`. Permette `chat A locale + chat B remota`
    /// — la chat A continua sul local model, la B passa alla remota.
    /// Nil → fallback al vecchio comportamento globale.
    var store: ChatStore? = nil

    @Environment(\.dismiss) private var dismiss
    @State private var search: String = ""
    @State private var hasAPIKey: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            if !hasAPIKey {
                noKeyBanner
            }
            searchField
            if let err = catalog.lastError {
                Text(err)
                    .font(.caption)
                    .foregroundStyle(Color.red)
                    .lineLimit(2)
            }
            listOrEmpty
            footer
        }
        .padding(20)
        .frame(width: 600, height: 540)
        .onAppear {
            hasAPIKey = KeychainStore.exists(
                account: KeychainAccount.openRouterAPIKey)
            let key = KeychainStore.get(
                account: KeychainAccount.openRouterAPIKey)
            catalog.touch(apiKey: key)
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Add OpenRouter model").font(.title3.bold())
                Text("Pick from \(catalog.models.count) models. Cached for 24 h.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if catalog.isLoading {
                ProgressView().controlSize(.small)
            }
            Button {
                let key = KeychainStore.get(
                    account: KeychainAccount.openRouterAPIKey)
                Task { await catalog.refresh(apiKey: key, force: true) }
            } label: {
                Label("Reload catalog", systemImage: "arrow.clockwise")
            }
            .help("Re-fetch the model list from OpenRouter")
        }
    }

    private var noKeyBanner: some View {
        HStack(spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(Color.orange)
            Text("No OpenRouter API key configured. Add it under Settings → API Keys before loading a model.")
                .font(.caption)
            Spacer()
        }
        .padding(8)
        .background(Color.orange.opacity(0.12),
                     in: RoundedRectangle(cornerRadius: 6))
    }

    private var searchField: some View {
        HStack {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField("Search by provider, name, or slug…", text: $search)
                .textFieldStyle(.plain)
        }
        .padding(8)
        .background(Color(NSColor.textBackgroundColor),
                     in: RoundedRectangle(cornerRadius: 6))
    }

    @ViewBuilder
    private var listOrEmpty: some View {
        if catalog.models.isEmpty {
            VStack(spacing: 8) {
                Image(systemName: "tray")
                    .font(.system(size: 32))
                    .foregroundStyle(.tertiary)
                Text(catalog.isLoading
                      ? "Loading catalog…"
                      : "No models cached yet. Reload to fetch the OpenRouter catalog.")
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            List(filtered) { model in
                Button { select(model) } label: { row(model) }
                    .buttonStyle(.plain)
                    .contentShape(Rectangle())
            }
            .listStyle(.inset)
        }
    }

    private var footer: some View {
        HStack {
            if let last = catalog.lastRefreshedAt {
                Text("Last refresh: \(last.formatted(date: .abbreviated, time: .shortened))")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            Spacer()
            Button("Close") { dismiss() }
                .keyboardShortcut(.escape)
        }
    }

    private func row(_ model: OpenRouterModel) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline) {
                Text(model.displayName).font(.callout.bold())
                Spacer()
                Text(model.id)
                    .font(.caption.monospaced())
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            HStack(spacing: 8) {
                if let ctx = model.contextLength {
                    Label(formatContext(ctx), systemImage: "text.alignleft")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                if let pricing = pricingLabel(model.pricing) {
                    Label(pricing, systemImage: "dollarsign.circle")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            if let desc = model.description, !desc.isEmpty {
                Text(desc)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 2)
    }

    private var filtered: [OpenRouterModel] {
        let needle = search.trimmingCharacters(in: .whitespaces).lowercased()
        guard !needle.isEmpty else { return catalog.models }
        return catalog.models.filter { m in
            if m.id.lowercased().contains(needle) { return true }
            if let n = m.name?.lowercased(), n.contains(needle) { return true }
            if let d = m.description?.lowercased(), d.contains(needle) { return true }
            return false
        }
    }

    private func select(_ model: OpenRouterModel) {
        let endpoint = ModelEndpoint.openRouter(modelID: model.id)
        // Bind alla chat selezionata se abbiamo lo store; questo
        // permette di tenere il local model caricato e usare la
        // remote SOLO per questa chat.
        if let store = store, let id = store.selectedID {
            store.setEndpoint(endpoint, for: id)
        }
        Task {
            await modelState.load(endpoint)
            await MainActor.run { dismiss() }
        }
    }

    private func formatContext(_ tokens: Int) -> String {
        if tokens >= 1_000_000 {
            return String(format: "%.1fM ctx", Double(tokens) / 1_000_000)
        } else if tokens >= 1000 {
            return "\(tokens / 1000)k ctx"
        }
        return "\(tokens) ctx"
    }

    private func pricingLabel(_ p: OpenRouterModel.Pricing?) -> String? {
        guard let p,
              let prompt = p.promptPerToken,
              let completion = p.completionPerToken else { return nil }
        let pm = prompt * 1_000_000
        let cm = completion * 1_000_000
        if pm == 0 && cm == 0 { return "free" }
        return String(format: "$%.2f/M in · $%.2f/M out", pm, cm)
    }
}
