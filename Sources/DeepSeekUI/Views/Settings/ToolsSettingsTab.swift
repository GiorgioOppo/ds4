import SwiftUI
import DeepSeekTools

/// Read-only inventory of native tools currently registered with the
/// `NativeToolHost`, grouped by category. Lets the user verify
/// what's available without parsing the chat's system block. Wiring
/// (enable/disable per agent) lives in the Agent editor; this tab
/// is purely informational.
struct ToolsSettingsTab: View {
    @ObservedObject var host: NativeToolHost

    private var grouped: [(ToolCategory, [ToolSchema])] {
        let byCategory = Dictionary(grouping: host.schemas, by: { $0.category })
        let order: [ToolCategory] = [.readOnly, .mutating, .dangerous, .network, .planning]
        return order.compactMap { cat in
            guard let items = byCategory[cat], !items.isEmpty else { return nil }
            return (cat, items.sorted { $0.name < $1.name })
        }
    }

    var body: some View {
        Form {
            ForEach(grouped, id: \.0) { (category, tools) in
                Section(header: Text(label(for: category))) {
                    ForEach(tools, id: \.name) { schema in
                        VStack(alignment: .leading, spacing: 2) {
                            HStack {
                                Image(systemName: icon(for: category))
                                    .foregroundStyle(.secondary)
                                Text(schema.name)
                                    .font(.body.monospaced())
                                Spacer()
                                Text(category.rawValue)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            Text(schema.description)
                                .font(.callout)
                                .foregroundStyle(.secondary)
                                .lineLimit(3)
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
        }
        .formStyle(.grouped)
    }

    private func label(for cat: ToolCategory) -> String {
        switch cat {
        case .readOnly:  return "Read-only"
        case .mutating:  return "Mutating"
        case .dangerous: return "Dangerous"
        case .network:   return "Network"
        case .planning:  return "Planning"
        }
    }

    private func icon(for cat: ToolCategory) -> String {
        switch cat {
        case .readOnly:  return "eye"
        case .mutating:  return "pencil"
        case .dangerous: return "exclamationmark.triangle"
        case .network:   return "network"
        case .planning:  return "list.bullet.rectangle"
        }
    }
}
