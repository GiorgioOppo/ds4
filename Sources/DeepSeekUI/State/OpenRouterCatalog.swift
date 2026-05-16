import Foundation
import SwiftUI

/// Live cache of OpenRouter's model catalog. The "Add OpenRouter
/// model" picker pulls from here so the user gets autocompletion
/// across the ~300 currently-supported models instead of typing
/// the slug from memory.
///
/// Refresh policy:
///   - First touch loads from disk if a recent cache exists, then
///     fires a background refresh through the network.
///   - `staleAfter` = 24 h. Beyond that, `models` is empty until
///     the network call returns.
///   - Manual refresh via `refresh(force: true)` for the picker's
///     "Reload" button.
///
/// The catalog is independent of whether the user has *any*
/// OpenRouter endpoint configured — it's useful in the "browse"
/// stage to decide which model to add.
@MainActor
final class OpenRouterCatalog: ObservableObject {
    @Published private(set) var models: [OpenRouterModel] = []
    @Published private(set) var isLoading: Bool = false
    @Published private(set) var lastError: String?
    @Published private(set) var lastRefreshedAt: Date?

    private let staleAfter: TimeInterval = 24 * 60 * 60
    private let client = OpenRouterClient()

    init() {
        loadFromDisk()
    }

    /// Public touch: load disk cache, kick a network refresh if
    /// the cache is stale or absent. Idempotent — multiple calls
    /// in flight coalesce on `isLoading`.
    func touch(apiKey: String?) {
        if shouldRefresh {
            Task { await refresh(apiKey: apiKey, force: false) }
        }
    }

    /// Force-refresh from network. Picker's "Reload" calls this.
    func refresh(apiKey: String?, force: Bool) async {
        if isLoading { return }
        if !force, !shouldRefresh { return }
        isLoading = true
        lastError = nil
        do {
            let fetched = try await client.fetchModels(apiKey: apiKey)
            self.models = fetched.sorted { $0.id < $1.id }
            self.lastRefreshedAt = .now
            saveToDisk()
        } catch {
            self.lastError = (error as? LocalizedError)?.errorDescription
                ?? error.localizedDescription
        }
        isLoading = false
    }

    /// Return the model entry matching an endpoint's `modelID`,
    /// if the catalog has seen it. Used by the picker's row
    /// rendering + cost banner to look up `pricing`.
    func model(for modelID: String) -> OpenRouterModel? {
        models.first(where: { $0.id == modelID })
    }

    // MARK: - private

    private var shouldRefresh: Bool {
        if models.isEmpty { return true }
        guard let last = lastRefreshedAt else { return true }
        return Date.now.timeIntervalSince(last) > staleAfter
    }

    private func loadFromDisk() {
        guard let url = try? PersistencePaths.openRouterCatalogURL(),
              let data = try? Data(contentsOf: url) else { return }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        if let cached = try? decoder.decode(CatalogCache.self, from: data) {
            models = cached.models
            lastRefreshedAt = cached.refreshedAt
        }
    }

    private func saveToDisk() {
        guard let url = try? PersistencePaths.openRouterCatalogURL() else { return }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let cache = CatalogCache(
            models: models,
            refreshedAt: lastRefreshedAt ?? .now)
        if let data = try? encoder.encode(cache) {
            try? data.write(to: url, options: .atomic)
        }
    }

    private struct CatalogCache: Codable {
        let models: [OpenRouterModel]
        let refreshedAt: Date
    }
}
