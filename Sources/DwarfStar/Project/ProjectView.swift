import SwiftUI
import AppKit
import DS4Engine

/// Sandbox-friendly project folder selection (mirror of ModelPicker): the picked
/// directory gets session-long security-scoped access + a persisted bookmark.
@MainActor
enum ProjectPicker {
    private static let bookmarkKey = "ds4.projectBookmark"

    static func pickFolder() -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.title = "Importa una cartella di progetto"
        panel.prompt = "Importa"
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        _ = url.startAccessingSecurityScopedResource()
        saveBookmark(url)
        return url
    }

    static func restoreBookmark() -> URL? {
        guard let data = UserDefaults.standard.data(forKey: bookmarkKey) else { return nil }
        var stale = false
        guard let url = try? URL(resolvingBookmarkData: data, options: .withSecurityScope,
                                 relativeTo: nil, bookmarkDataIsStale: &stale) else { return nil }
        guard url.startAccessingSecurityScopedResource() else { return nil }
        if stale { saveBookmark(url) }
        return url
    }

    static func clearBookmark() {
        UserDefaults.standard.removeObject(forKey: bookmarkKey)
    }

    private static func saveBookmark(_ url: URL) {
        if let data = try? url.bookmarkData(options: .withSecurityScope,
                                            includingResourceValuesForKeys: nil, relativeTo: nil) {
            UserDefaults.standard.set(data, forKey: bookmarkKey)
        }
    }
}

/// The project memory container: import a folder into a cache SEPARATE from the
/// chat memory. The model explores it via the project_* tools, so only the parts
/// it actually reads enter the conversation.
struct ProjectView: View {
    @Bindable var store: ChatStore
    @State private var info: ProjectCache.Info?
    @State private var preview: [String] = []
    @State private var restored = false

    var body: some View {
        Form {
            Section {
                Text("Il progetto importato vive in una cache separata dalla memoria della chat: l'import non consuma contesto. L'agente (es. Coding) lo esplora con i tool project_list / project_read / project_search, e solo le parti lette entrano in conversazione.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Progetto importato") {
                if let p = info {
                    LabeledContent("Nome", value: p.name)
                    LabeledContent("File indicizzati", value: "\(p.fileCount)")
                    LabeledContent("Dimensione testo", value: ByteCountFormatter.string(
                        fromByteCount: Int64(p.totalBytes), countStyle: .file))
                    Button(role: .destructive) {
                        ProjectCache.shared.clear()
                        ProjectPicker.clearBookmark()
                        info = nil
                        preview = []
                    } label: {
                        Label("Rimuovi progetto", systemImage: "trash")
                    }
                } else {
                    Text("Nessun progetto importato.").foregroundStyle(.secondary)
                }
                Button {
                    if let url = ProjectPicker.pickFolder() { importProject(url) }
                } label: {
                    Label(info == nil ? "Importa cartella…" : "Cambia progetto…",
                          systemImage: "folder.badge.plus")
                }
                Text("Solo file di testo (≤1 MB ciascuno, max 3000); cartelle come .git e node_modules sono escluse. Sotto sandbox la cartella resta accessibile e viene ricordata.")
                    .font(.caption).foregroundStyle(.tertiary)
            }

            Section("Uso con gli agenti") {
                Text("L'agente Coding predefinito ha già i tool di progetto abilitati. Per altri agenti, attivali nella scheda Agenti (project_list, project_read, project_search). Ricorda: i tool si applicano all'avvio di una nuova chat.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            if !preview.isEmpty {
                Section("Anteprima file (\(preview.count) di \(info?.fileCount ?? 0))") {
                    ForEach(preview, id: \.self) { f in
                        Text(f).font(.system(.caption, design: .monospaced))
                    }
                }
            }
        }
        .formStyle(.grouped)
        .onAppear {
            guard !restored else { return }
            restored = true
            if let i = ProjectCache.shared.info() {
                info = i
                preview = ProjectCache.shared.sampleFiles(50)
            } else if let url = ProjectPicker.restoreBookmark() {
                importProject(url)
            }
        }
    }

    private func importProject(_ url: URL) {
        let i = ProjectCache.shared.load(root: url)
        info = i
        preview = ProjectCache.shared.sampleFiles(50)
    }
}
