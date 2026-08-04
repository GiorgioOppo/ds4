import SwiftUI
import DS4Engine
import DS4Core

/// The raw tool-call markup as it streams, shown live during generation. Once the
/// block closes it is replaced by the formatted ToolCallView card.
struct ToolStreamView: View {
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Label("Generating tool call...", systemImage: "wrench.and.screwdriver")
                .font(.caption.bold())
                .foregroundStyle(.orange.opacity(0.7))
            Text(text)
                .font(.system(.caption2, design: .monospaced))
                .textSelection(.enabled)
                .foregroundStyle(.secondary)
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.05))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Color.orange.opacity(0.25),
                                                                 style: StrokeStyle(lineWidth: 1, dash: [4])))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

/// A tool the model decided to call.
struct ToolCallView: View {
    let call: ToolCall

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Label("Tool call: \(call.name)", systemImage: "wrench.and.screwdriver.fill")
                .font(.caption.bold())
                .foregroundStyle(.orange)
            Text(call.argumentsJSON)
                .font(.system(.caption2, design: .monospaced))
                .textSelection(.enabled)
                .foregroundStyle(.secondary)
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

/// An isolated sub-agent run: target + answer, with a collapsible trace of the
/// internal steps (which never entered the main conversation's context). While
/// `running`, the card shows a spinner and the LATEST step live instead of the
/// (not yet available) answer — the execution detail as it happens.
struct SubAgentView: View {
    let run: InferenceService.SubAgentRun
    var running: Bool = false
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Label("Sub-agent · \(run.target)", systemImage: "person.3.sequence")
                    .font(.caption.bold()).foregroundStyle(.purple)
                if running { ProgressView().controlSize(.small) }
            }
            if !run.question.isEmpty {
                Text(run.question).font(.caption2).foregroundStyle(.secondary).lineLimit(2)
            }
            if running {
                Text(run.steps.last ?? "starting…")
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            } else {
                Text(run.answer).font(.callout).textSelection(.enabled)
            }
            if !run.steps.isEmpty {
                DisclosureGroup(isExpanded: $expanded) {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(run.steps.enumerated()), id: \.offset) { _, step in
                            Text(step)
                                .font(.system(.caption2, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                } label: {
                    Label("Internal steps (\(run.steps.count))", systemImage: "list.bullet.indent")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.purple.opacity(0.07))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Color.purple.opacity(0.25)))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

/// The result of a tool, fed back to the model.
struct ToolResultRow: View {
    let text: String
    private var isMalformedCall: Bool {
        text.hasPrefix("Chiamata tool incompleta o non valida")
    }

    var body: some View {
        HStack(alignment: .top, spacing: 7) {
            Image(systemName: isMalformedCall ? "exclamationmark.triangle.fill" : "arrow.uturn.left")
                .padding(.top, 1)
            Text(text)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .font(.system(.caption, design: .monospaced))
        .foregroundStyle(.secondary)
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background((isMalformedCall ? Color.orange : Color.green).opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

/// Collapsible chain-of-thought block.
struct ReasoningView: View {
    let text: String
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            Text(text)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("Reasoning", systemImage: "brain")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(8)
        .background(Color.secondary.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}
