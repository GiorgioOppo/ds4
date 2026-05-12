import SwiftUI
import AppKit

/// Inspect / clear the on-disk conversation history. Doesn't touch
/// the live `ChatStore` — the user would have to restart the app to
/// see the empty state.
struct StorageSettingsTab: View {
    @State private var dirURL: URL? = nil
    @State private var fileCount: Int = 0
    @State private var totalBytes: UInt64 = 0
    @State private var showingClearConfirm = false

    var body: some View {
        Form {
            Section("History location") {
                if let url = dirURL {
                    LabeledContent("Path") {
                        Text(url.path)
                            .font(.system(.body, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                    }
                    LabeledContent("Saved") {
                        Text("\(fileCount) conversation(s), \(formatBytes(totalBytes))")
                            .font(.system(.body, design: .monospaced))
                    }
                    HStack {
                        Button("Reveal in Finder") {
                            NSWorkspace.shared.activateFileViewerSelecting([url])
                        }
                        Spacer()
                        Button("Clear all…", role: .destructive) {
                            showingClearConfirm = true
                        }
                    }
                } else {
                    Text("Directory unavailable.")
                        .foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
        .padding()
        .task { refresh() }
        .alert("Delete all saved conversations?",
                isPresented: $showingClearConfirm) {
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                clearAll()
            }
        } message: {
            Text("This removes every JSON file in the history folder. Currently-open chats in the app are not affected until you quit and relaunch.")
        }
    }

    private func refresh() {
        guard let dir = try? PersistencePaths.conversationsDir() else { return }
        dirURL = dir
        let files = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        let jsons = files.filter { $0.pathExtension == "json" }
        fileCount = jsons.count
        totalBytes = jsons.reduce(0 as UInt64) { acc, url in
            let s = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            return acc + UInt64(s)
        }
    }

    private func clearAll() {
        guard let dir = dirURL else { return }
        let files = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        for url in files where url.pathExtension == "json" {
            try? FileManager.default.removeItem(at: url)
        }
        refresh()
    }

    private func formatBytes(_ b: UInt64) -> String {
        let mib = 1024.0 * 1024.0
        if Double(b) < mib { return "\(b / 1024) KB" }
        return String(format: "%.2f MB", Double(b) / mib)
    }
}
