import Foundation
import SwiftUI
import DeepSeekTools

/// Durable per-tool permission defaults. Distinct from
/// `ToolRegistry.sessionAllowCache` (in-memory, per session) — this
/// store persists "always allow" / "always ask" / "always deny"
/// decisions across launches.
///
/// Storage: a single JSON blob inside Application Support, keyed by
/// `<tool>:<category>`. Kept hand-rolled (not `@AppStorage`) so we
/// can present per-rule editing in the Settings tab without trying
/// to encode `Set<…>` into a property list.
@MainActor
final class PermissionStore: ObservableObject {
    enum DefaultDecision: String, Codable {
        case ask, alwaysAllow, alwaysDeny

        var displayName: String {
            switch self {
            case .ask:         return "Ask"
            case .alwaysAllow: return "Always allow"
            case .alwaysDeny:  return "Always deny"
            }
        }
    }

    @Published private(set) var defaults: [String: DefaultDecision] = [:]

    init() { load() }

    func decision(for tool: String, category: ToolCategory) -> DefaultDecision {
        defaults["\(tool):\(category.rawValue)"] ?? .ask
    }

    func setDecision(_ decision: DefaultDecision,
                     for tool: String,
                     category: ToolCategory) {
        defaults["\(tool):\(category.rawValue)"] = decision
        save()
    }

    func reset() {
        defaults.removeAll()
        save()
    }

    // MARK: - persistence

    private func load() {
        guard let url = try? PersistencePaths.permissionsConfigURL(),
              let data = try? Data(contentsOf: url) else { return }
        if let decoded = try? JSONDecoder().decode(
                [String: DefaultDecision].self, from: data) {
            defaults = decoded
        }
    }

    private func save() {
        guard let url = try? PersistencePaths.permissionsConfigURL() else { return }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(defaults) {
            try? data.write(to: url, options: .atomic)
        }
    }
}
