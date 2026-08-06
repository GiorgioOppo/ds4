import SwiftUI
import AppKit
import DS4Engine

/// Library of saved projects: folders imported by the user keep the folder's
/// security-scoped bookmark (sandbox-safe across launches); repositories cloned
/// by the github_clone tool live inside the app container and are tracked by
/// plain path (no bookmark needed there). ONE project at a time is active in
/// the ProjectCache (the project_* tools read from it); switching just reloads
/// the cache — the chat memory is never touched.
@MainActor
enum ProjectLibrary {
    struct SavedProject: Codable, Identifiable, Equatable {
        let id: String
        var name: String
        let bookmark: Data?      // user-imported folder (security-scoped)
        let path: String?        // github_clone folder inside the app container

        init(id: String, name: String, bookmark: Data?, path: String? = nil) {
            self.id = id; self.name = name; self.bookmark = bookmark; self.path = path
        }
    }

    /// True for entries that are clones made by the github_clone tool: they are
    /// disposable copies inside the app container, so "remove" DELETES the
    /// folder (otherwise the next sync would just re-list it).
    static func isCloned(_ p: SavedProject) -> Bool {
        p.path?.hasPrefix(GitHubTool.projectsDirectory().path) == true
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
        panel.title = "Add a Local Project"
        panel.prompt = "Add Project"
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
        if let p = list.first(where: { $0.id == id }), isCloned(p), let path = p.path {
            try? FileManager.default.removeItem(atPath: path)
        }
        list.removeAll { $0.id == id }
        save(list)
        if activeId == id { activeId = nil }
    }

    /// Keep the library in sync with the github_clone destination folder:
    /// every cloned repo appears as a path-based entry, entries whose folder
    /// was deleted are pruned, and when the ProjectCache's current root IS one
    /// of those folders (the model just ran github_clone in chat) that entry is
    /// marked active so the GUI reflects what the tools already see. Call it
    /// before reading `all()` in any view that lists projects.
    static func syncClonedRepos() {
        let fm = FileManager.default
        let dir = GitHubTool.projectsDirectory()
        let folders = ((try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.isDirectoryKey],
                                                    options: [.skipsHiddenFiles])) ?? [])
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true }
        var list = all()
        let before = list
        list.removeAll { p in
            guard isCloned(p), let path = p.path else { return false }
            return !fm.fileExists(atPath: path)
        }
        for f in folders where !list.contains(where: { $0.path == f.path }) {
            // Stable id per folder: a re-clone of the same repo keeps its
            // identity (and its active state) instead of duplicating the entry.
            list.append(SavedProject(id: "github:" + f.lastPathComponent,
                                     name: f.lastPathComponent + " (GitHub)",
                                     bookmark: nil, path: f.path))
        }
        if list != before { save(list) }
        if let root = ProjectCache.shared.rootURL(),
           let match = list.first(where: { $0.path == root.path }),
           activeId != match.id {
            activeId = match.id
        }
    }

    /// Resolve, load into the ProjectCache and mark active. Returns the project
    /// info, or nil when the folder is no longer resolvable.
    @discardableResult
    static func activate(_ project: SavedProject) -> ProjectCache.Info? {
        guard let url = resolveURL(project) else { return nil }
        let info = ProjectCache.shared.load(root: url)
        activeId = project.id
        return info
    }

    /// Resolve a saved project's folder. Path-based entries (github_clone) live
    /// inside the app container — plain access, no security scope; bookmark
    /// entries resolve the scoped bookmark and start access.
    static func resolveURL(_ p: SavedProject) -> URL? {
        if let path = p.path {
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir),
                  isDir.boolValue else { return nil }
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        guard let bm = p.bookmark else { return nil }
        var stale = false
        guard let url = try? URL(resolvingBookmarkData: bm, options: .withSecurityScope,
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
                Text("Imported projects live in a cache separate from chat memory: importing does not consume context. The agent (for example Coding) explores the active project with project_list / project_read / project_search; only the parts it reads enter the conversation.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Saved Projects") {
                if projects.isEmpty {
                    Text("No saved projects.").foregroundStyle(.secondary)
                }
                ForEach(projects) { project in
                    let cloned = ProjectLibrary.isCloned(project)
                    HStack {
                        Label(project.name, systemImage: cloned ? "arrow.down.circle" : "folder")
                        if project.id == ProjectLibrary.activeId {
                            Text("active")
                                .font(.caption)
                                .padding(.horizontal, 6).padding(.vertical, 2)
                                .background(Color.accentColor.opacity(0.2))
                                .clipShape(Capsule())
                        }
                        Spacer()
                        Button("Activate") { activate(project) }
                            .disabled(project.id == ProjectLibrary.activeId)
                        Button(role: .destructive) { remove(project) } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.borderless)
                        .help(cloned
                              ? "Delete the cloned copy from disk (re-clone it anytime with github_clone)"
                              : "Remove from the library (the folder on disk is not touched)")
                    }
                }
                Button {
                    if let p = ProjectLibrary.pickAndAdd() {
                        projects = ProjectLibrary.all()
                        activate(p)
                    }
                } label: {
                    Label("Add New Project…", systemImage: "folder.badge.plus")
                }
                Text("Text files only (<=1 MB each, max 3000); folders like .git and node_modules are excluded. Folders remain accessible across launches through sandbox bookmarks. Repositories cloned in chat with github_clone appear here automatically with a \u{2913} icon (stored in Application Support; removing one deletes the cloned copy).")
                    .font(.caption).foregroundStyle(.tertiary)
            }

            Section("Active Project") {
                if let p = info {
                    LabeledContent("Name", value: p.name)
                    LabeledContent("Indexed files", value: "\(p.fileCount)")
                    LabeledContent("Text size", value: ByteCountFormatter.string(
                        fromByteCount: Int64(p.totalBytes), countStyle: .file))
                } else {
                    Text("No active project.").foregroundStyle(.secondary)
                }
                if !message.isEmpty {
                    Text(message).font(.caption).foregroundStyle(.orange)
                }
            }

            Section("Use with Agents") {
                Text("The default Coding agent already has project tools. When you change the active project, tools read the new one; results already in the conversation remain. Start a New Chat for a clean slate.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            if !preview.isEmpty {
                Section("File Preview (\(preview.count) of \(info?.fileCount ?? 0))") {
                    ForEach(preview, id: \.self) { f in
                        Text(f).font(.system(.caption, design: .monospaced))
                    }
                }
            }
        }
        .formStyle(.grouped)
        .onAppear {
            // Sync first so a repo just cloned in chat shows up (and is marked
            // active) the moment this tab opens; then ALWAYS refresh the active
            // info — the cache may have been swapped by github_clone since the
            // last appearance.
            ProjectLibrary.syncClonedRepos()
            projects = ProjectLibrary.all()
            if let i = ProjectCache.shared.info() {
                info = i
                preview = ProjectCache.shared.sampleFiles(50)
                restored = true
                return
            }
            info = nil
            preview = []
            guard !restored else { return }
            restored = true
            if let activeId = ProjectLibrary.activeId,
               let p = projects.first(where: { $0.id == activeId }) {
                activate(p)
            }
        }
    }

    private func activate(_ project: ProjectLibrary.SavedProject) {
        message = ""
        guard let i = ProjectLibrary.activate(project) else {
            message = "Folder is no longer accessible (moved or deleted?). Remove it and import it again."
            return
        }
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
