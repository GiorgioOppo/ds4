import SwiftUI
import DeepSeekTools
import DeepSeekIntegrations

/// Inventory of native tools registered with the `NativeToolHost`,
/// grouped by category. Adds a Sandbox section (TODO §9) that
/// surfaces the `useShellSandbox` toggle + an "Initialize default
/// profile" button — `ShellTool(useSandbox:)` reads the toggle at
/// NativeToolHost init time, so flipping the switch requires
/// restarting the app for the change to take effect.
struct ToolsSettingsTab: View {
    @ObservedObject var host: NativeToolHost
    @AppStorage(AppSettingsKey.useShellSandbox)
    private var useShellSandbox: Bool = false
    @State private var profileStatus: String? = nil

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
            sandboxSection
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

    private var sandboxSection: some View {
        Section {
            Toggle("Wrap ShellTool calls in `sandbox-exec`",
                    isOn: $useShellSandbox)
            HStack {
                Button("Initialize default profile") {
                    let root = URL(fileURLWithPath: NSHomeDirectory())
                        .appendingPathComponent(".deepseek")
                    do {
                        try Sandbox.ensureDefaultProfile(at: root)
                        profileStatus = "Wrote profile to \(root.path)/sandbox/default.sb"
                    } catch {
                        profileStatus = "Failed: \(error.localizedDescription)"
                    }
                }
                .disabled(!useShellSandbox)
                Spacer()
            }
            if let status = profileStatus {
                Text(status).font(.caption).foregroundStyle(.secondary)
            }
        } header: {
            Text("Sandbox")
        } footer: {
            Text("Sandbox uses macOS `sandbox-exec` with the profile at "
                 + "`~/.deepseek/sandbox/default.sb`. The default profile is "
                 + "strict (read-only outside the root, no network); tune it "
                 + "before relying on the toggle. Changes require restart.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}
