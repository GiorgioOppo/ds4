import SwiftUI
import AppKit
import DS4Engine

/// Library of saved projects: each entry keeps the folder's security-scoped
/// bookmark (sandbox-safe across launches). ONE project at a time is active in
/// the ProjectCache (the project_* tools read from it); switching just reloads
/// the cache — the chat memory is never touched.
@MainActor
enum ProjectLibrary {
    struct SavedProject: Codable, Identifiable, Equatable {
        let id: String
        var name: String
        let bookmark: Data
    }

    private static let listKey = "ds4.projectLibrary"
    private static let activeKey = "ds4.projectActive"
    private static let legacyKey = "ds4.projectBookmark"

    static func all() -> [SavedProject] {
        migrateLegacyIfNeeded()
        guard let data = UserDefaults.standard.data(forKey: listKey),
              let list = try? JSONDecoder().decode([SavedProject].self, from: data) else { return [] }
        return list
    }

    static func save(_ list: [SavedProject]) {
        if let data = try? JSONEncoder().encode(list) {
            UserDefaults.standard.set(data, forKey: listKey)
        }
    }

    static var activeId: String? {
        get { UserDefaults.standard.string(forKey: activeKey) }
        set { UserDefaults.standard.set(newValue, forKey: activeKey) }
    }

    /// Pick a folder and add it to the library (deduplicated by resolved path).
    static func pickAndAdd() -> SavedProject? {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.title = "Importa una cartella di progetto"
        panel.prompt = "Importa"
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        _ = url.startAccessingSecurityScopedResource()
        guard let bm = try? url.bookmarkData(options: .withSecurityScope,
                                             includingResourceValuesForKeys: nil, relativeTo: nil) else { return nil }
        var list = all()
        let project = SavedProject(id: UUID().uuidString, name: url.lastPathComponent, bookmark: bm)
        list.removeAll { resolveURL($0)?.path == url.path }   // re-import replaces
        list.append(project)
        save(list)
        return project
    }

    static func remove(id: String) {
        var list = all()
        list.removeAll { $0.id == id }
        save(list)
        if activeId == id { activeId = nil }
    }

    /// Resolve a saved project's folder and start security-scoped access.
    static func resolveURL(_ p: SavedProject) -> URL? {
        var stale = false
        guard let url = try? URL(resolvingBookmarkData: p.bookmark, options: .withSecurityScope,
                                 relativeTo: nil, bookmarkDataIsStale: &stale) else { return nil }
        guard url.startAccessingSecurityScopedResource() else { return nil }
        return url
    }

    /// One-time migration from the old single-project bookmark key.
    private static func migrateLegacyIfNeeded() {
        let ud = UserDefaults.standard
        guard let bm = ud.data(forKey: legacyKey) else { return }
        ud.removeObject(forKey: legacyKey)
        var stale = false
        guard let url = try? URL(resolvingBookmarkData: bm, options: .withSecurityScope,
                                 relativeTo: nil, bookmarkDataIsStale: &stale) else { return }
        var list: [SavedProject] = []
        if let data = ud.data(forKey: listKey),
           let existing = try? JSONDecoder().decode([SavedProject].self, from: data) {
            list = existing
        }
        let p = SavedProject(id: UUID().uuidString, name: url.lastPathComponent, bookmark: bm)
        list.append(p)
        if let data = try? JSONEncoder().encode(list) { ud.set(data, forKey: listKey) }
        if ud.string(forKey: activeKey) == nil { ud.set(p.id, forKey: activeKey) }
    }
}

/// The project memory container: a LIBRARY of saved projects (multiple folders),
/// one active at a time in a cache SEPARATE from the chat memory. The model
/// explores the active one via the project_* tools, so only the parts it
/// actually reads enter the conversation.
struct ProjectView: View {
    @Bindable var store: ChatStore
    @State private var projects: [ProjectLibrary.SavedProject] = []
    @State private var info: ProjectCache.Info?
    @State private var preview: [String] = []
    @State private var message = ""
    @State private var restored = false

    var body: some View {
        Form {
            Section {
                Text("I progetti importati vivono in una cache separata dalla memoria della chat: l'import non consuma contesto. L'agente (es. Coding) esplora il progetto ATTIVO con i tool project_list / project_read / project_search; solo le parti lette entrano in conversazione.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Progetti salvati") {
                if projects.isEmpty {
                    Text("Nessun progetto salvato.").foregroundStyle(.secondary)
                }
                ForEach(projects) { project in
                    HStack {
                        Label(project.name, systemImage: "folder")
                        if project.id == ProjectLibrary.activeId {
                            Text("attivo")
                                .font(.caption)
                                .padding(.horizontal, 6).padding(.vertical, 2)
                                .background(Color.accentColor.opacity(0.2))
                                .clipShape(Capsule())
                        }
                        Spacer()
                        Button("Attiva") { activate(project) }
                            .disabled(project.id == ProjectLibrary.activeId)
                        Button(role: .destructive) { remove(project) } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.borderless)
                        .help("Rimuovi dalla libreria (la cartella su disco non viene toccata)")
                    }
                }
                Button {
                    if let p = ProjectLibrary.pickAndAdd() {
                        projects = ProjectLibrary.all()
                        activate(p)
                    }
                } label: {
                    Label("Importa cartella…", systemImage: "folder.badge.plus")
                }
                Text("Solo file di testo (≤1 MB ciascuno, max 3000); cartelle come .git e node_modules escluse. Le cartelle restano accessibili tra i riavvii (bookmark sandbox).")
                    .font(.caption).foregroundStyle(.tertiary)
            }

            Section("Progetto attivo") {
                if let p = info {
                    LabeledContent("Nome", value: p.name)
                    LabeledContent("File indicizzati", value: "\(p.fileCount)")
                    LabeledContent("Dimensione testo", value: ByteCountFormatter.string(
                        fromByteCount: Int64(p.totalBytes), countStyle: .file))
                } else {
                    Text("Nessun progetto attivo.").foregroundStyle(.secondary)
                }
                if !message.isEmpty {
                    Text(message).font(.caption).foregroundStyle(.orange)
                }
            }

            Section("Uso con gli agenti") {
                Text("L'agente Coding predefinito ha già i tool di progetto. Cambiando progetto attivo, i tool leggono il nuovo; i risultati già in conversazione restano (apri una Nuova chat per ripartire puliti).")
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
            projects = ProjectLibrary.all()
            guard !restored else { return }
            restored = true
            if let i = ProjectCache.shared.info() {
                info = i
                preview = ProjectCache.shared.sampleFiles(50)
            } else if let activeId = ProjectLibrary.activeId,
                      let p = projects.first(where: { $0.id == activeId }) {
                activate(p)
            }
        }
    }

    private func activate(_ project: ProjectLibrary.SavedProject) {
        message = ""
        guard let url = ProjectLibrary.resolveURL(project) else {
            message = "Cartella non più accessibile (spostata o eliminata?). Rimuovila e re-importala."
            return
        }
        let i = ProjectCache.shared.load(root: url)
        ProjectLibrary.activeId = project.id
        info = i
        preview = ProjectCache.shared.sampleFiles(50)
        projects = ProjectLibrary.all()
    }

    private func remove(_ project: ProjectLibrary.SavedProject) {
        let wasActive = project.id == ProjectLibrary.activeId
        ProjectLibrary.remove(id: project.id)
        projects = ProjectLibrary.all()
        if wasActive {
            ProjectCache.shared.clear()
            info = nil
            preview = []
        }
    }
}
