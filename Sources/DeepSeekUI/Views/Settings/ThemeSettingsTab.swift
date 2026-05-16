import SwiftUI
import DeepSeekTools

/// Theme picker with live swatches. Selecting a row sets the active
/// theme on `ThemeStore`; the root scene observes that and applies
/// `preferredColorScheme(_:)` plus accent overrides.
struct ThemeSettingsTab: View {
    @ObservedObject var store: ThemeStore

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
                    Button {
                        store.setActive(theme.id)
                    } label: {
                        themeRow(theme,
                                 isActive: theme.id == store.activeThemeID)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .formStyle(.grouped)
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
}
