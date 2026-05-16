import Foundation

/// Configurable keyboard shortcuts. SwiftUI's `KeyboardShortcut`
/// can't be modelled across processes, so we represent bindings as
/// `(action, key, modifiers)` triples and let the UI translate them.
///
/// Actions are stable identifiers — the same set across versions —
/// so a user's overrides survive renames at the UI layer. Bindings
/// with conflicting `(key, modifiers)` are resolved in registration
/// order; the host should detect and report duplicates.
public struct Keybinding: Codable, Sendable, Identifiable, Hashable {
    public var id: String { action }
    public var action: String
    public var key: String
    public var modifiers: Set<Modifier>

    public enum Modifier: String, Codable, Sendable, CaseIterable {
        case command, option, control, shift
    }

    public init(action: String,
                key: String,
                modifiers: Set<Modifier> = []) {
        self.action = action
        self.key = key
        self.modifiers = modifiers
    }
}

public enum BuiltInKeybindings {
    public static let all: [Keybinding] = [
        .init(action: "chat.send",                key: "return", modifiers: [.command]),
        .init(action: "chat.stop",                key: ".",      modifiers: [.command]),
        .init(action: "conversation.new",         key: "n",      modifiers: [.command]),
        .init(action: "conversation.clear",       key: "k",      modifiers: [.command]),
        .init(action: "agent.toggleMode",         key: "tab",    modifiers: [.command]),
        .init(action: "palette.slashCommands",    key: "p",      modifiers: [.command, .shift]),
        .init(action: "palette.tools",            key: "t",      modifiers: [.command, .shift]),
        .init(action: "settings.open",            key: ",",      modifiers: [.command]),
        .init(action: "model.switch",             key: "m",      modifiers: [.command, .shift]),
        .init(action: "permissions.review",       key: "g",      modifiers: [.command, .shift]),
    ]
}
