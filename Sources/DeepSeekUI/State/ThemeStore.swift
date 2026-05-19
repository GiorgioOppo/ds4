import Foundation
import SwiftUI
import DeepSeekTools

/// Active theme selection + helpers that turn `Theme` hex strings
/// into SwiftUI `Color`. Stored separately from `AppSettings` because
/// the theme catalog evolves (user-defined themes will land later),
/// so we want JSON not UserDefaults.
@MainActor
final class ThemeStore: ObservableObject {
    @Published var activeThemeID: String = BuiltInThemes.system.name
    @Published private(set) var catalog: [Theme] = BuiltInThemes.all

    init() { load() }

    var active: Theme {
        catalog.first { $0.id == activeThemeID } ?? BuiltInThemes.system
    }

    func setActive(_ themeID: String) {
        activeThemeID = themeID
        save()
    }

    /// Add (or replace) a user-defined theme and persist (TODO §12).
    /// Themes whose name collides with a built-in are rejected —
    /// otherwise the catalog merge in `load()` would silently
    /// shadow them on next launch.
    func addCustomTheme(_ theme: Theme) {
        guard !BuiltInThemes.all.contains(where: { $0.id == theme.id })
        else { return }
        var c = catalog
        if let idx = c.firstIndex(where: { $0.id == theme.id }) {
            c[idx] = theme
        } else {
            c.append(theme)
        }
        catalog = c.sorted { $0.name < $1.name }
        save()
    }

    /// Drop a user-defined theme. Built-in themes are protected.
    /// If the deleted theme was active, the active selection
    /// falls back to `BuiltInThemes.system`.
    func removeCustomTheme(_ themeID: String) {
        guard !BuiltInThemes.all.contains(where: { $0.id == themeID })
        else { return }
        catalog.removeAll { $0.id == themeID }
        if activeThemeID == themeID {
            activeThemeID = BuiltInThemes.system.id
        }
        save()
    }

    /// Effective ColorScheme override for the SwiftUI root.
    /// `nil` → follow the system.
    var preferredColorScheme: ColorScheme? {
        switch active.appearance {
        case .system: return nil
        case .light:  return .light
        case .dark:   return .dark
        }
    }

    private func load() {
        guard let url = try? PersistencePaths.themeConfigURL(),
              let data = try? Data(contentsOf: url),
              let payload = try? JSONDecoder().decode(Payload.self, from: data) else {
            return
        }
        activeThemeID = payload.activeThemeID
        var merged: [String: Theme] = [:]
        for t in BuiltInThemes.all { merged[t.id] = t }
        for t in payload.customThemes { merged[t.id] = t }
        catalog = merged.values.sorted { $0.name < $1.name }
    }

    private func save() {
        let builtinNames = Set(BuiltInThemes.all.map(\.id))
        let custom = catalog.filter { !builtinNames.contains($0.id) }
        let payload = Payload(activeThemeID: activeThemeID, customThemes: custom)
        guard let url = try? PersistencePaths.themeConfigURL() else { return }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(payload) {
            try? data.write(to: url, options: .atomic)
        }
    }

    private struct Payload: Codable {
        var activeThemeID: String
        var customThemes: [Theme]
    }
}

/// Convert a `#RRGGBB` string to SwiftUI `Color`. Returns `nil` for
/// empty / malformed strings so callers can fall through to the
/// system default.
func swiftUIColor(hex: String) -> Color? {
    var raw = hex.trimmingCharacters(in: .whitespaces)
    if raw.hasPrefix("#") { raw.removeFirst() }
    guard raw.count == 6, let n = UInt32(raw, radix: 16) else { return nil }
    let r = Double((n >> 16) & 0xFF) / 255.0
    let g = Double((n >> 8) & 0xFF) / 255.0
    let b = Double(n & 0xFF) / 255.0
    return Color(red: r, green: g, blue: b)
}
