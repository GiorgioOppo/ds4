import SwiftUI
import DeepSeekTools

/// Read-only keybinding inspector. Each action shows its current
/// shortcut; the reset button restores every action to the
/// `BuiltInKeybindings` defaults. Inline editing is intentionally
/// out of scope for this iteration — it needs a key-grab widget,
/// conflict detection, and a confirmation pass before overwriting
/// system shortcuts. Tracked in TODO.md.
struct KeybindingsSettingsTab: View {
    @ObservedObject var store: KeybindingStore

    private var actions: [Keybinding] {
        store.bindings.values.sorted { $0.action < $1.action }
    }

    var body: some View {
        Form {
            Section("Shortcuts") {
                ForEach(actions) { binding in
                    HStack {
                        Text(binding.action).font(.body.monospaced())
                        Spacer()
                        Text(format(binding))
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            Section {
                Button("Reset to defaults") { store.reset() }
            } footer: {
                Text("Inline rebind is not implemented yet — see TODO.md.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }

    private func format(_ binding: Keybinding) -> String {
        var parts: [String] = []
        if binding.modifiers.contains(.control) { parts.append("⌃") }
        if binding.modifiers.contains(.option)  { parts.append("⌥") }
        if binding.modifiers.contains(.shift)   { parts.append("⇧") }
        if binding.modifiers.contains(.command) { parts.append("⌘") }
        parts.append(binding.key.uppercased())
        return parts.joined()
    }
}
