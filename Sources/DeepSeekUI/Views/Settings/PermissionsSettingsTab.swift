import SwiftUI
import DeepSeekTools

/// Per-tool default permission editor. Lets the user pre-decide
/// (ask / always allow / always deny) for each registered tool so the
/// confirmation sheet doesn't fire mid-stream. Also surfaces the
/// per-session "always allow" grants from `NativeToolHost` with a
/// reset button.
struct PermissionsSettingsTab: View {
    @ObservedObject var host: NativeToolHost
    @ObservedObject var store: PermissionStore

    var body: some View {
        Form {
            Section("Session grants") {
                if host.sessionGrants.isEmpty {
                    Text("No tools have been granted alwaysAllow this session.")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(Array(host.sessionGrants).sorted(), id: \.self) { grant in
                        Text(grant).font(.body.monospaced())
                    }
                    Button("Clear session grants") {
                        host.resetPermissions()
                    }
                }
            }

            Section("Per-tool defaults") {
                ForEach(host.schemas, id: \.name) { schema in
                    Picker(schema.name, selection: Binding(
                        get: { store.decision(for: schema.name, category: schema.category) },
                        set: { store.setDecision($0,
                                                 for: schema.name,
                                                 category: schema.category) }
                    )) {
                        ForEach([PermissionStore.DefaultDecision.ask,
                                 .alwaysAllow,
                                 .alwaysDeny], id: \.self) { d in
                            Text(d.displayName).tag(d)
                        }
                    }
                    .pickerStyle(.menu)
                }
                Button("Reset all to Ask") { store.reset() }
            }
        }
        .formStyle(.grouped)
    }
}
