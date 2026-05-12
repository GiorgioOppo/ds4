import SwiftUI

/// Pre-flight knobs: strategy override + force-load + recent dirs.
/// Changes affect the next model load, not the current session.
struct LoadingSettingsTab: View {
    @AppStorage(AppSettingsKey.loadStrategy) private var loadStrategy: String = "auto"
    @AppStorage(AppSettingsKey.forceLoad)    private var forceLoad: Bool = false
    @AppStorage(AppSettingsKey.lastModelDir) private var lastModelDir: String = ""

    var body: some View {
        Form {
            Section("Strategy") {
                Picker("Load strategy", selection: $loadStrategy) {
                    Text("Auto").tag("auto")
                    Text("Preload").tag("preload")
                    Text("Mmap").tag("mmap")
                }
                .pickerStyle(.segmented)
                Toggle("Bypass RAM-safety refusals (--force-load)",
                        isOn: $forceLoad)
                Text("Force-load skips the 70 % shard cap and the 25× total-RAM oversubscription guard. Use when you accept the risk of paging-driven freezes.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Section("Last loaded model") {
                if lastModelDir.isEmpty {
                    Text("None yet.")
                        .foregroundStyle(.secondary)
                } else {
                    HStack {
                        Text(lastModelDir)
                            .font(.system(.body, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Button("Forget") { lastModelDir = "" }
                    }
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}
