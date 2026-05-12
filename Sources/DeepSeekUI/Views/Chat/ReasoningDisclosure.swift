import SwiftUI

/// Collapsible `<think>...</think>` rendering. Reasoning content is
/// preserved as monospaced plain text (not markdown — the model
/// often emits half-finished code and notation that markdown
/// parsing distorts). Default collapsed; persistence of the
/// collapse state across sessions isn't worth the complexity.
struct ReasoningDisclosure: View {
    let reasoning: String
    @State private var expanded: Bool = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            ScrollView(.horizontal, showsIndicators: false) {
                Text(reasoning)
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .padding(8)
            }
            .frame(maxHeight: 240)
            .background(Color(NSColor.controlBackgroundColor),
                         in: RoundedRectangle(cornerRadius: 8))
        } label: {
            Label("Reasoning", systemImage: "brain")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}
