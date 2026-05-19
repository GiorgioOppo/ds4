import SwiftUI
import DeepSeekTools

/// Keybinding editor (TODO §12). Each action row shows its current
/// shortcut + a "Rebind" button that opens a sheet listening for
/// the next key chord. Conflict detection runs at save time: if
/// the requested chord is already assigned to another action, the
/// sheet surfaces a warning and the user has to either pick a
/// different chord or confirm the overwrite (which clears the
/// other action's binding).
struct KeybindingsSettingsTab: View {
    @ObservedObject var store: KeybindingStore

    @State private var rebinding: Keybinding?

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
                        Button("Rebind") {
                            rebinding = binding
                        }
                        .buttonStyle(.borderless)
                        .controlSize(.small)
                    }
                }
            }
            Section {
                Button("Reset to defaults") { store.reset() }
            } footer: {
                Text("Rebind opens a capture sheet — press the new chord "
                     + "(modifiers + key). Conflicts with another action are "
                     + "highlighted before save.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .sheet(item: $rebinding) { binding in
            KeybindingRebindSheet(
                store: store,
                original: binding,
                onCancel: { rebinding = nil },
                onSave: { _ in rebinding = nil })
        }
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

// MARK: - Rebind sheet

/// Modal that captures the next keystroke and proposes it as the
/// new shortcut for `original.action`. Listens via SwiftUI's
/// `.onKeyPress` (macOS 14+); the user can also pick a key
/// manually from a TextField if the capture missed the chord.
private struct KeybindingRebindSheet: View {
    @ObservedObject var store: KeybindingStore
    let original: Keybinding
    let onCancel: () -> Void
    let onSave: (Keybinding) -> Void

    @State private var capturedKey: String = ""
    @State private var capturedModifiers: Set<Keybinding.Modifier> = []
    @State private var conflictAction: String?
    @FocusState private var captureFieldFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Rebind shortcut").font(.title3.bold())
            Text("Action: \(original.action)")
                .font(.callout)
                .foregroundStyle(.secondary)

            HStack {
                Text("Press the new shortcut, or type below:")
                    .font(.callout)
                Spacer()
            }
            captureBox

            HStack {
                Text("Modifiers:")
                Toggle("⌘", isOn: bindingFor(.command))
                Toggle("⌥", isOn: bindingFor(.option))
                Toggle("⌃", isOn: bindingFor(.control))
                Toggle("⇧", isOn: bindingFor(.shift))
            }
            .toggleStyle(.button)

            TextField("Key (e.g. \"k\", \"return\", \"escape\")",
                       text: $capturedKey)
                .textFieldStyle(.roundedBorder)
                .font(.body.monospaced())
                .focused($captureFieldFocused)
                .onKeyPress(phases: .down) { press in
                    // Capture printable characters + a small set of
                    // named keys. Modifiers come from the event's
                    // `.modifiers` SwiftUI flags.
                    capturedModifiers = mapModifiers(press.modifiers)
                    if let named = namedKey(from: press) {
                        capturedKey = named
                    } else if let ch = press.characters.first {
                        capturedKey = String(ch)
                    }
                    refreshConflict()
                    return .handled
                }

            if let other = conflictAction {
                Label("Conflicts with: \(other). Saving will overwrite that "
                      + "action's binding.",
                      systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(Color.orange)
            }

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { onCancel() }
                    .keyboardShortcut(.escape)
                Button("Save") { save() }
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(capturedKey
                               .trimmingCharacters(in: .whitespaces)
                               .isEmpty)
            }
        }
        .padding(20)
        .frame(minWidth: 460, minHeight: 320)
        .onAppear {
            capturedKey = original.key
            capturedModifiers = original.modifiers
            captureFieldFocused = true
        }
    }

    @ViewBuilder
    private var captureBox: some View {
        HStack(spacing: 6) {
            ForEach(orderedMods(capturedModifiers), id: \.self) { m in
                Text(symbol(for: m))
            }
            Text(capturedKey.isEmpty ? "—" : capturedKey.uppercased())
                .bold()
        }
        .font(.title2.monospaced())
        .frame(maxWidth: .infinity, alignment: .center)
        .padding(.vertical, 10)
        .background(Color.gray.opacity(0.12),
                     in: RoundedRectangle(cornerRadius: 8))
    }

    private func bindingFor(_ m: Keybinding.Modifier) -> Binding<Bool> {
        Binding(
            get: { capturedModifiers.contains(m) },
            set: { on in
                if on { capturedModifiers.insert(m) }
                else  { capturedModifiers.remove(m) }
                refreshConflict()
            })
    }

    private func mapModifiers(_ m: EventModifiers) -> Set<Keybinding.Modifier> {
        var out: Set<Keybinding.Modifier> = []
        if m.contains(.command)  { out.insert(.command) }
        if m.contains(.option)   { out.insert(.option) }
        if m.contains(.control)  { out.insert(.control) }
        if m.contains(.shift)    { out.insert(.shift) }
        return out
    }

    /// Map a key press to one of our canonical key names. Anything
    /// outside this set falls back to the raw character.
    private func namedKey(from press: KeyPress) -> String? {
        switch press.key {
        case .return:       return "return"
        case .escape:       return "escape"
        case .tab:          return "tab"
        case .space:        return "space"
        case .delete:       return "delete"
        case .leftArrow:    return "leftArrow"
        case .rightArrow:   return "rightArrow"
        case .upArrow:      return "upArrow"
        case .downArrow:    return "downArrow"
        default:            return nil
        }
    }

    private func symbol(for m: Keybinding.Modifier) -> String {
        switch m {
        case .control: return "⌃"
        case .option:  return "⌥"
        case .shift:   return "⇧"
        case .command: return "⌘"
        }
    }

    private func orderedMods(_ s: Set<Keybinding.Modifier>) -> [Keybinding.Modifier] {
        let order: [Keybinding.Modifier] = [.control, .option, .shift, .command]
        return order.filter { s.contains($0) }
    }

    private func refreshConflict() {
        let key = capturedKey.trimmingCharacters(in: .whitespaces)
        guard !key.isEmpty else {
            conflictAction = nil
            return
        }
        let other = store.bindings.values.first(where: {
            $0.action != original.action
            && $0.key.caseInsensitiveCompare(key) == .orderedSame
            && $0.modifiers == capturedModifiers
        })
        conflictAction = other?.action
    }

    private func save() {
        // Clear the conflicting action's binding (set it to empty
        // key, which the format helper renders as "?" — UX hint
        // that something needs attention).
        if let conflict = conflictAction,
           let target = store.bindings.values.first(where: {
               $0.action == conflict })
        {
            store.setBinding(Keybinding(action: target.action,
                                          key: "",
                                          modifiers: []))
        }
        let updated = Keybinding(action: original.action,
                                   key: capturedKey,
                                   modifiers: capturedModifiers)
        store.setBinding(updated)
        onSave(updated)
    }
}
