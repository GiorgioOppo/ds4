import SwiftUI
#if canImport(AppKit)
import AppKit
#endif

/// Live view of every sub-agent currently running on behalf of the
/// chat the user is looking at. Renders as a stacked card pinned
/// above the composer so a delegation chain stays visible without
/// stealing focus from the main transcript.
///
/// Each frame shows:
///   - the target agent's SF Symbol + tinted dot, so the chain
///     reads visually at a glance ("blue circle, green circle, …");
///   - "Delegated to <name>" + the agent's summary;
///   - the task the host asked for, in a faint quote block;
///   - the streaming reply as it arrives (clamped to the last
///     few lines so a chatty agent doesn't push the composer
///     off-screen — the full reply lands in the tool-call
///     disclosure when the delegation finishes).
///
/// `frames` is ordered from outermost (depth 1, host's direct
/// callee) to innermost (the deepest sub currently running). The
/// view indents each level by `depth - 1` so the chain visually
/// recedes into the right margin.
struct DelegationStackView: View {
    let frames: [DelegationFrame]

    var body: some View {
        if frames.isEmpty {
            EmptyView()
        } else {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Image(systemName: "arrow.triangle.branch")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("Delegation in progress")
                        .font(.caption.bold())
                        .foregroundStyle(.secondary)
                }
                ForEach(frames) { frame in
                    frameRow(frame)
                        .padding(.leading,
                                  CGFloat(max(0, frame.depth - 1)) * 14)
                }
            }
            .padding(10)
            .background(Color(NSColor.controlBackgroundColor)
                            .opacity(0.85),
                         in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8)
                .strokeBorder(Color.secondary.opacity(0.25), lineWidth: 0.5))
        }
    }

    private func frameRow(_ frame: DelegationFrame) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: frame.agentIconName)
                    .font(.caption)
                    .foregroundStyle(AgentTint.color(for: frame.agentTint))
                Text(frame.agentName)
                    .font(.callout.bold())
                Spacer()
                ProgressView()
                    .controlSize(.mini)
            }
            if !frame.task.isEmpty {
                Text(frame.task)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .italic()
                    .lineLimit(2)
                    .truncationMode(.tail)
                    .padding(.leading, 18)
            }
            if !frame.buffer.isEmpty {
                Text(tailLines(frame.buffer, max: 6))
                    .font(.caption.monospaced())
                    .foregroundStyle(.primary)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.leading, 18)
                    .padding(.top, 2)
            }
        }
    }

    /// Keep only the last N lines of the buffer to bound vertical
    /// space; trims long single-line outputs to a sensible width
    /// too. Tail-windowing matches how the user reads streaming
    /// text — they want the most recent tokens, not the prefix.
    private func tailLines(_ s: String, max n: Int) -> String {
        let lines = s.split(separator: "\n", omittingEmptySubsequences: false)
        let kept = lines.suffix(n).map(String.init)
        return kept.joined(separator: "\n")
    }
}
