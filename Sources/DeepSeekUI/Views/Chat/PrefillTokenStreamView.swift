import SwiftUI

/// Dedicated indicator shown while the model prefills the prompt.
/// Streams the prompt's real BPE tokens into a wrapping grid of
/// chips — one chip per token id — so the prefill can be watched
/// token by token. Bound to the round's `StreamingRoundController`
/// via `@ObservedObject`, so only this view re-renders as tokens
/// arrive.
///
/// Transient: visible during the `.prefilling` phase, gone once the
/// first reply token streams. The persistent "what the model saw"
/// record stays the job of `PrefillTraceDisclosure`.
struct PrefillTokenStreamView: View {
    @ObservedObject var controller: StreamingRoundController

    /// Tokens decoded so far this prefill.
    private var done: Int { controller.prefillTokenSteps.count }
    /// Total tokens in the prefill delta. `max` guards the brief
    /// window where steps outpace the `.prefillStart` total.
    private var total: Int { max(controller.prefillTokenTotal, done) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            header
            if done > 0 {
                tokenGrid
            }
        }
    }

    @ViewBuilder
    private var header: some View {
        HStack(spacing: 6) {
            Image(systemName: "square.grid.2x2")
                .font(.caption2)
            Text("Prefill · \(done) / \(total) token")
                .font(.caption)
                .monospacedDigit()
            if done < total {
                ProgressView().controlSize(.mini)
            }
            Spacer()
        }
        .foregroundStyle(.secondary)
    }

    @ViewBuilder
    private var tokenGrid: some View {
        ScrollViewReader { proxy in
            ScrollView(.vertical, showsIndicators: true) {
                PrefillFlowLayout(spacing: 4) {
                    ForEach(Array(controller.prefillTokenSteps.enumerated()),
                            id: \.offset) { idx, token in
                        PrefillTokenChip(text: token,
                                          isLatest: idx == done - 1)
                            .id(idx)
                    }
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 200)
            .background(Color(NSColor.controlBackgroundColor),
                         in: RoundedRectangle(cornerRadius: 8))
            .onChange(of: done) { _, newValue in
                if newValue > 0 {
                    proxy.scrollTo(newValue - 1, anchor: .bottom)
                }
            }
        }
    }
}

/// One token rendered as a chip. Whitespace and control characters
/// are swapped for visible glyphs so a space- or newline-only token
/// is not an invisible box; an empty token (a partial multi-byte
/// char whose completion lands on the next id) shows a faint dot.
private struct PrefillTokenChip: View {
    let text: String
    let isLatest: Bool

    var body: some View {
        Text(display)
            .font(.system(.caption2, design: .monospaced))
            .foregroundStyle(isLatest ? Color.accentColor : .primary)
            .padding(.horizontal, 4)
            .padding(.vertical, 2)
            .background(
                RoundedRectangle(cornerRadius: 3)
                    .fill(isLatest
                          ? Color.accentColor.opacity(0.20)
                          : Color.secondary.opacity(0.10)))
    }

    private var display: String {
        if text.isEmpty { return "·" }
        var out = ""
        for ch in text {
            switch ch {
            case " ":          out += "␣"
            case "\n", "\r":   out += "⏎"
            case "\t":         out += "⇥"
            default:           out.append(ch)
            }
        }
        return out
    }
}

/// Minimal flow layout: lays subviews left-to-right, wrapping to a
/// new row when the next subview would overflow the proposed width.
/// Uses macOS 14's `Layout` protocol; no cache — the chip set is
/// small and rebuilt only during the brief prefill window.
private struct PrefillFlowLayout: Layout {
    var spacing: CGFloat = 4

    func sizeThatFits(proposal: ProposedViewSize,
                      subviews: Subviews,
                      cache: inout Void) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0
        for sv in subviews {
            let size = sv.sizeThatFits(.unspecified)
            if x > 0 && x + size.width > maxWidth {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
        let width = maxWidth.isFinite ? maxWidth : x
        return CGSize(width: width, height: y + rowHeight)
    }

    func placeSubviews(in bounds: CGRect,
                       proposal: ProposedViewSize,
                       subviews: Subviews,
                       cache: inout Void) {
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0
        for sv in subviews {
            let size = sv.sizeThatFits(.unspecified)
            if x > 0 && x + size.width > bounds.width {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            sv.place(at: CGPoint(x: bounds.minX + x, y: bounds.minY + y),
                     proposal: ProposedViewSize(size))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}
