import Foundation
import SwiftUI

/// One persisted entry in the toolbar's model picker: a named
/// endpoint plus the last-used timestamp the picker uses to order
/// the recents list.
///
/// Identity is the wrapped endpoint's id (path for local
/// directories) so opening the same folder twice via Browse
/// dedupes naturally instead of stacking duplicate entries.
struct ConfiguredModelEntry: Codable, Identifiable, Hashable {
    let endpoint: ModelEndpoint
    var lastUsedAt: Date
    /// Optional user-supplied label; falls back to the endpoint's
    /// natural `displayName` when nil/empty. Lets the user keep
    /// two checkpoints that share a folder name distinguishable
    /// ("v4-flash latest" vs "v4-flash 2025-04-08").
    var displayLabel: String?

    var id: String { endpoint.id }
    var name: String {
        if let label = displayLabel?.trimmingCharacters(in: .whitespaces),
           !label.isEmpty { return label }
        return endpoint.displayName
    }
}

/// Persisted registry of model endpoints the user has ever loaded
/// (or pre-configured). The toolbar picker reads `entries` for the
/// recents list; `touch(_:)` is called whenever a load completes
/// so the most-recent model floats to the top.
///
/// On-disk format mirrors the other library JSONs (mcp.json,
/// agents.json, …) so a future "export my setup" feature can scoop
/// them all up uniformly.
@MainActor
final class ModelLibrary: ObservableObject {
    @Published private(set) var entries: [ConfiguredModelEntry] = []

    init() {
        load()
    }

    /// Insert or update an entry for this endpoint and bump its
    /// `lastUsedAt` to now. Called after a successful load so the
    /// picker treats it as the most recent. `displayLabel`, when
    /// provided, overrides the existing label.
    func touch(_ endpoint: ModelEndpoint, label: String? = nil) {
        if let idx = entries.firstIndex(where: { $0.endpoint == endpoint }) {
            entries[idx].lastUsedAt = .now
            if let label, !label.isEmpty {
                entries[idx].displayLabel = label
            }
        } else {
            entries.append(ConfiguredModelEntry(
                endpoint: endpoint,
                lastUsedAt: .now,
                displayLabel: label))
        }
        save()
    }

    /// Remove an endpoint from the recents list without unloading
    /// the model (the runtime decision is on `ModelState`). Used
    /// by the picker's "Forget" affordance.
    func forget(_ endpoint: ModelEndpoint) {
        entries.removeAll { $0.endpoint == endpoint }
        save()
    }

    /// Entries sorted by `lastUsedAt` descending — newest first,
    /// matching what users expect from a "Recent" submenu. Stale
    /// entries pointing at no-longer-existing paths are still
    /// returned (the picker labels them); pruning is the user's
    /// call via `forget`.
    func recents() -> [ConfiguredModelEntry] {
        entries.sorted { $0.lastUsedAt > $1.lastUsedAt }
    }

    // MARK: - persistence

    private func load() {
        guard let url = try? PersistencePaths.modelsConfigURL(),
              let data = try? Data(contentsOf: url) else { return }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        if let decoded = try? decoder.decode([ConfiguredModelEntry].self,
                                               from: data) {
            entries = decoded
        }
    }

    private func save() {
        guard let url = try? PersistencePaths.modelsConfigURL() else { return }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(entries) {
            try? data.write(to: url, options: .atomic)
        }
    }
}
