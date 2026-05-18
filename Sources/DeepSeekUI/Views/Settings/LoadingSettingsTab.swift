import SwiftUI
import AppKit

/// Pre-flight knobs: strategy override + force-load + recent dirs +
/// converter binary path. Changes affect the next model load /
/// conversion, not the current session.
struct LoadingSettingsTab: View {
    @AppStorage(AppSettingsKey.loadStrategy) private var loadStrategy: String = "auto"
    @AppStorage(AppSettingsKey.forceLoad)    private var forceLoad: Bool = false
    @AppStorage(AppSettingsKey.lastModelDir) private var lastModelDir: String = ""
    @AppStorage(AppSettingsKey.converterBinaryPath)
    private var converterPath: String = ""
    @AppStorage(AppSettingsKey.warmupOnLoad) private var warmupOnLoad: Bool = false
    @AppStorage(AppSettingsKey.commonPrefixRewind)
    private var commonPrefixRewind: Bool = false
    @AppStorage(AppSettingsKey.useMapSharedWeights)
    private var useMapSharedWeights: Bool = false

    var body: some View {
        Form {
            Section("Strategy") {
                Picker("Load strategy", selection: $loadStrategy) {
                    Text("Auto").tag("auto")
                    Text("Preload").tag("preload")
                    Text("Mmap").tag("mmap")
                    Text("Streaming").tag("streaming")
                }
                .pickerStyle(.segmented)
                Toggle("Bypass RAM-safety refusals (--force-load)",
                        isOn: $forceLoad)
                Text("Auto picks streaming when the checkpoint is more than 10× the effective unified-memory budget — between-layer madvise hints keep the working set bounded so the OS doesn't freeze. Force-load skips the 50% shard cap entirely. Preload copies everything to a fresh MTLBuffer (rare on small Macs).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Performance tweaks (ds4-inspired)") {
                Toggle("Warm pages on load",
                        isOn: $warmupOnLoad)
                Text("Pre-fault tutte le pagine dei weight shards al " +
                     "model load. Riduce il time-to-first-token. " +
                     "Skip automatico se model size > RAM × 1.5 " +
                     "(safe su Mac con memoria limitata).")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Toggle("Common-prefix rewind (experimental)",
                        isOn: $commonPrefixRewind)
                Text("Riusa la KV cache anche quando l'utente edita " +
                     "il proprio ultimo messaggio (common-prefix " +
                     "match invece di strict-prefix). Reset esplicito " +
                     "di scoreState al window boundary (round-down a " +
                     "LCM dei compressRatio del modello). " +
                     "Default OFF; abilita solo dopo testing.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Toggle("MAP_SHARED for Metal weights (experimental)",
                        isOn: $useMapSharedWeights)
                Text("Usa `MAP_SHARED` invece di `MAP_PRIVATE` per " +
                     "l'mmap dei weight shards. Su Apple Silicon + " +
                     "APFS dovrebbe permettere zero-copy MTLBuffer " +
                     "wrap (come ds4). Fallback automatico a " +
                     "MAP_PRIVATE se mmap fallisce. Default OFF " +
                     "per safety vs Darwin VM panic.")
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
            Section("Converter binary") {
                LabeledContent("Path") {
                    HStack {
                        TextField("Auto-detect", text: $converterPath)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(.body, design: .monospaced))
                            .frame(minWidth: 240)
                        Button("Choose…") {
                            let panel = NSOpenPanel()
                            panel.canChooseFiles = true
                            panel.canChooseDirectories = false
                            panel.allowsMultipleSelection = false
                            panel.prompt = "Select"
                            if panel.runModal() == .OK, let u = panel.url {
                                converterPath = u.path
                            }
                        }
                    }
                }
                Text("Used by Convert model… When empty, the runner searches Bundle Resources, the running executable's sibling dir, `.build/{debug,release}/converter`, and `/usr/local/bin/converter` in that order.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Section("Recent model folders") {
                if recents.isEmpty {
                    Text("No recents.")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(recents, id: \.self) { path in
                        HStack {
                            Text(path)
                                .font(.system(.body, design: .monospaced))
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer()
                            Button("Forget") {
                                AppSettings.forgetRecentDir(path)
                                recents = AppSettings.recentDirs()
                            }
                            .controlSize(.small)
                        }
                    }
                }
            }
        }
        .formStyle(.grouped)
        .padding()
        .onAppear { recents = AppSettings.recentDirs() }
    }

    @State private var recents: [String] = AppSettings.recentDirs()
}
