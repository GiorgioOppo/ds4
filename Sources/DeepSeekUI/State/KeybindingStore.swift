import Foundation
import SwiftUI
import DeepSeekTools

/// Loads and persists the user's keyboard-shortcut overrides. The
/// values are pure data (`Keybinding`); the UI layer translates each
/// entry to a `KeyboardShortcut` at render time via
/// `keyboardShortcut(for:)`.
@MainActor
final class KeybindingStore: ObservableObject {
    @Published private(set) var bindings: [String: Keybinding] = [:]

    init() { load() }

    func binding(for action: String) -> Keybinding? {
        bindings[action]
    }

    func setBinding(_ binding: Keybinding) {
        bindings[binding.action] = binding
        save()
    }

    func reset() {
        bindings = Dictionary(uniqueKeysWithValues:
            BuiltInKeybindings.all.map { ($0.action, $0) })
        save()
    }

    private func load() {
        var merged: [String: Keybinding] = Dictionary(
            uniqueKeysWithValues: BuiltInKeybindings.all.map { ($0.action, $0) })
        if let url = try? PersistencePaths.keybindingsConfigURL(),
           let data = try? Data(contentsOf: url),
           let overrides = try? JSONDecoder().decode([Keybinding].self, from: data) {
            for kb in overrides { merged[kb.action] = kb }
        }
        bindings = merged
    }

    private func save() {
        // Only persist entries that differ from the built-in defaults.
        let defaults = Dictionary(
            uniqueKeysWithValues: BuiltInKeybindings.all.map { ($0.action, $0) })
        let overrides = bindings.values.filter { kb in
            defaults[kb.action] != kb
        }
        guard let url = try? PersistencePaths.keybindingsConfigURL() else { return }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(Array(overrides)) {
            try? data.write(to: url, options: .atomic)
        }
    }
}

/// Translate a `Keybinding` to a SwiftUI `KeyboardShortcut`. Returns
/// `nil` when the binding has no representable key — the UI then
/// falls back to whatever default the view declared inline.
func swiftUIShortcut(for binding: Keybinding) -> KeyboardShortcut? {
    let key: KeyEquivalent
    switch binding.key.lowercased() {
    case "return", "enter": key = .return
    case "escape":          key = .escape
    case "tab":             key = .tab
    case "space":           key = .space
    case "up":              key = .upArrow
    case "down":            key = .downArrow
    case "left":            key = .leftArrow
    case "right":           key = .rightArrow
    case "delete":          key = .delete
    default:
        guard let scalar = binding.key.unicodeScalars.first else { return nil }
        key = KeyEquivalent(Character(scalar))
    }
    var mods: EventModifiers = []
    for m in binding.modifiers {
        switch m {
        case .command:  mods.insert(.command)
        case .option:   mods.insert(.option)
        case .control:  mods.insert(.control)
        case .shift:    mods.insert(.shift)
        }
    }
    return KeyboardShortcut(key, modifiers: mods)
}
