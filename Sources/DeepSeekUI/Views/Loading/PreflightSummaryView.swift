import SwiftUI
import DeepSeekKit

/// Renders the seven discrete fields of a `LoadPlan` as a vertical
/// label/value list, matching the columns of the CLI's
/// `LoadPlan.summary()` but with each field individually styled.
struct PreflightSummaryView: View {
    let plan: LoadPlan

    private static let gib: Double = 1024 * 1024 * 1024
    private static func g(_ bytes: UInt64) -> String {
        String(format: "%.2f GB", Double(bytes) / gib)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Section(header: header("System")) {
                row("Physical RAM",   PreflightSummaryView.g(plan.physicalRAM))
                row("Available RAM",  PreflightSummaryView.g(plan.availableRAM))
                row("CPU cores",      "\(plan.cores)")
                row("GPU rec. working-set", PreflightSummaryView.g(plan.mtlWorkingSet))
            }
            Section(header: header("Checkpoint")) {
                row("Shards",         "\(plan.shards.count)")
                row("Total bytes",    PreflightSummaryView.g(plan.totalBytes))
                row("Largest shard",  PreflightSummaryView.g(plan.maxShardBytes))
            }
            Section(header: header("Strategy")) {
                row(plan.strategy.rawValue.uppercased(), plan.reason,
                    valueColor: .secondary)
            }
        }
        .padding(20)
        .background(Color(NSColor.controlBackgroundColor),
                     in: RoundedRectangle(cornerRadius: 12))
        .frame(maxWidth: 480)
    }

    private func header(_ text: String) -> some View {
        Text(text)
            .font(.headline)
            .foregroundStyle(.secondary)
            .padding(.bottom, 2)
    }

    private func row(_ label: String, _ value: String,
                      valueColor: Color = .primary) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .foregroundStyle(.secondary)
                .frame(width: 160, alignment: .leading)
            Text(value)
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(valueColor)
            Spacer(minLength: 0)
        }
    }
}
