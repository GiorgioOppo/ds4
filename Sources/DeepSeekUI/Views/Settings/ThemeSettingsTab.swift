import SwiftUI
import DeepSeekTools

/// Theme picker with live swatches. Selecting a row sets the active
/// theme on `ThemeStore`; the root scene observes that and applies
/// `preferredColorScheme(_:)` plus accent overrides. The "Create
/// custom theme…" button at the bottom opens a sheet that writes a
/// fresh `Theme` into the store — `addCustomTheme` persists it
/// alongside the built-ins.
struct ThemeSettingsTab: View {
    @ObservedObject var store: ThemeStore

    @State private var showCreateSheet = false
    @State private var draft: Theme = Self.emptyDraft()

    var body: some View {
        Form {
            Section("Active theme") {
                Text("Picked theme: \(store.active.name)")
                    .font(.headline)
                Text(store.active.summary)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Section("Catalog") {
                ForEach(store.catalog) { theme in
                    HStack {
                        Button {
                            store.setActive(theme.id)
                        } label: {
                            themeRow(theme,
                                     isActive: theme.id == store.activeThemeID)
                        }
                        .buttonStyle(.plain)
                        if !isBuiltIn(theme) {
                            Button(role: .destructive) {
                                store.removeCustomTheme(theme.id)
                            } label: {
                                Image(systemName: "trash")
                            }
                            .buttonStyle(.borderless)
                            .help("Delete custom theme")
                        }
                    }
                }
            }
            Section {
                Button {
                    draft = Self.emptyDraft()
                    showCreateSheet = true
                } label: {
                    Label("Create custom theme…",
                          systemImage: "paintbrush.pointed")
                }
            } footer: {
                Text("Custom themes are stored alongside the built-ins. "
                     + "Existing chats pick the new theme as soon as it's "
                     + "set active.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .sheet(isPresented: $showCreateSheet) {
            ThemeEditorSheet(draft: $draft) {
                store.addCustomTheme(draft)
                showCreateSheet = false
            } onCancel: {
                showCreateSheet = false
            }
        }
    }

    @ViewBuilder
    private func themeRow(_ theme: Theme, isActive: Bool) -> some View {
        HStack(spacing: 12) {
            swatch(hex: theme.accent)
            VStack(alignment: .leading, spacing: 2) {
                Text(theme.name).font(.body)
                Text(theme.summary).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if isActive {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.tint)
            }
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private func swatch(hex: String) -> some View {
        let color = swiftUIColor(hex: hex) ?? .accentColor
        RoundedRectangle(cornerRadius: 6)
            .fill(color)
            .frame(width: 22, height: 22)
            .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(.quaternary))
    }

    private func isBuiltIn(_ theme: Theme) -> Bool {
        BuiltInThemes.all.contains { $0.id == theme.id }
    }

    private static func emptyDraft() -> Theme {
        Theme(name: "My Theme",
              summary: "",
              accent: "#7F7FFF",
              background: "#101010",
              foreground: "#F0F0F0",
              assistantBubble: "#1B1B22",
              userBubble: "#27272E",
              appearance: .dark)
    }
}

// MARK: - Sheet

/// Create / edit a custom theme. Six color slots driven by
/// `ColorPicker` so the user doesn't have to know hex. The picker
/// produces a SwiftUI `Color`; we round-trip it through
/// `Color → RGB doubles → #RRGGBB` so the saved Theme stays in the
/// same string format as the built-ins.
private struct ThemeEditorSheet: View {
    @Binding var draft: Theme
    let onSave: () -> Void
    let onCancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Create custom theme")
                .font(.title3.bold())
            Form {
                Section("Metadata") {
                    TextField("Name", text: $draft.name)
                    TextField("Summary", text: $draft.summary)
                    Picker("Appearance", selection: $draft.appearance) {
                        Text("System").tag(Theme.Appearance.system)
                        Text("Light").tag(Theme.Appearance.light)
                        Text("Dark").tag(Theme.Appearance.dark)
                    }
                    .pickerStyle(.segmented)
                }
                Section("Colors") {
                    colorRow(label: "Accent",
                             hex: $draft.accent)
                    colorRow(label: "Background",
                             hex: $draft.background)
                    colorRow(label: "Foreground",
                             hex: $draft.foreground)
                    colorRow(label: "Assistant bubble",
                             hex: $draft.assistantBubble)
                    colorRow(label: "User bubble",
                             hex: $draft.userBubble)
                }
            }
            .formStyle(.grouped)
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { onCancel() }
                    .keyboardShortcut(.escape)
                Button("Save") { onSave() }
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(draft.name
                               .trimmingCharacters(in: .whitespaces)
                               .isEmpty)
            }
        }
        .padding(20)
        .frame(minWidth: 480, minHeight: 460)
    }

    @ViewBuilder
    private func colorRow(label: String, hex: Binding<String>) -> some View {
        HStack {
            ColorPicker(label,
                        selection: Binding(
                            get: { swiftUIColor(hex: hex.wrappedValue) ?? .gray },
                            set: { hex.wrappedValue = Self.hexString(from: $0) }
                        ),
                        supportsOpacity: false)
            TextField("#RRGGBB", text: hex)
                .textFieldStyle(.roundedBorder)
                .frame(width: 100)
                .font(.body.monospaced())
        }
    }

    /// SwiftUI Color → `#RRGGBB`. Uses NSColor on macOS for a
    /// lossless conversion through the device-RGB color space.
    private static func hexString(from color: Color) -> String {
        #if canImport(AppKit)
        let nsColor = NSColor(color).usingColorSpace(.deviceRGB)
            ?? NSColor.black
        let r = Int(round(nsColor.redComponent * 255))
        let g = Int(round(nsColor.greenComponent * 255))
        let b = Int(round(nsColor.blueComponent * 255))
        return String(format: "#%02X%02X%02X", r, g, b)
        #else
        return "#000000"
        #endif
    }
}
