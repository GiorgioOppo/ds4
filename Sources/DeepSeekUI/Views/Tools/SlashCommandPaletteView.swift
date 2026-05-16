import SwiftUI
import DeepSeekTools

/// Autocomplete strip rendered above the composer while the user is
/// typing a slash command. Filters the registered command list by
/// prefix and lets the user pick with arrow keys + Return, or
/// dismiss with Escape.
///
/// The composer asks for this view (instead of owning the list
/// itself) so the lookup logic stays in `SlashCommandLibrary` and
/// the view is a thin renderer.
struct SlashCommandPaletteView: View {
    let library: SlashCommandLibrary
    let prefix: String
    @Binding var highlightedIndex: Int
    let onPick: (SlashCommand) -> Void
    let onDismiss: () -> Void

    private var matches: [SlashCommand] {
        let trimmed = prefix.lowercased().trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { return library.commands }
        return library.commands.filter { $0.name.hasPrefix(trimmed) }
    }

    var body: some View {
        if matches.isEmpty {
            EmptyView()
        } else {
            VStack(spacing: 0) {
                ForEach(Array(matches.enumerated()), id: \.element.id) { i, cmd in
                    Button {
                        onPick(cmd)
                    } label: {
                        row(cmd, isHighlighted: i == highlightedIndex)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.vertical, 4)
            .background(.bar)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .onExitCommand(perform: onDismiss)
        }
    }

    @ViewBuilder
    private func row(_ cmd: SlashCommand, isHighlighted: Bool) -> some View {
        HStack(spacing: 10) {
            Text("/\(cmd.name)")
                .font(.body.monospaced())
                .foregroundStyle(.primary)
            Text(cmd.summary)
                .font(.callout)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(isHighlighted ? Color.accentColor.opacity(0.18) : .clear)
    }
}
