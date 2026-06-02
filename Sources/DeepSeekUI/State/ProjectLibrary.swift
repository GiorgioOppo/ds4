import Foundation
import SwiftUI

/// A user-defined project: a name plus a list of source paths (files
/// or directories). Indexing walks those paths and produces one
/// `VectorizedDocument` per discovered text file, tagged with the
/// project's id so the Projects tab can group them.
///
/// `lastIndexedAt` is nil until the user runs an index; re-indexing
/// nukes every document tagged with this project and re-runs the
/// scan. `modelFingerprint` is recorded at index time so the chat
/// loader (later step) can refuse to use a project whose tokens were
/// produced against a different model.
struct Project: Codable, Identifiable, Hashable {
    let id: UUID
    var name: String
    /// Absolute filesystem paths. May point at single files or whole
    /// directories — `ProjectIndexer` walks directories recursively.
    var sourcePaths: [String]
    /// App-scoped security bookmarks, one per entry in `sourcePaths`
    /// (positional match). Required so the macOS sandbox keeps granting
    /// reads after the original `NSOpenPanel` session expires —
    /// otherwise tools fail with "couldn't be opened because you don't
    /// have permission to view it" when they follow a symlink in the
    /// per-project farm down to the user's real disk.
    ///
    /// Optional + nil-default for backward compatibility: projects
    /// persisted before the bookmark wiring landed decode without this
    /// field, and `addPath` re-grants them lazily as the user re-picks
    /// folders. When non-nil, the array length matches `sourcePaths`;
    /// individual entries may still be empty `Data()` if bookmark
    /// generation failed for that specific URL.
    var sourceBookmarks: [Data]?
    var createdAt: Date
    var lastIndexedAt: Date?
    var modelFingerprint: String?

    /// Override per-progetto del modo di presentazione al modello.
    /// `nil` → usa `AppSettings.projectContextMode` (default
    /// `.pathsOnly`). Campo opzionale per retro-compatibilità coi
    /// progetti già persistiti.
    var contextMode: ProjectContextMode?

    /// Override per-progetto del cap sull'inventario in modalità
    /// `pathsOnly`. `nil` → usa
    /// `AppSettings.projectInventoryMaxFiles`. 0 = senza limite
    /// (sconsigliato per progetti grandi).
    var maxInventoryFiles: Int?

    /// Parent directories of symlink targets that landed outside
    /// every entry in `sourcePaths` during the last farm rebuild.
    /// The ProjectDetailView surfaces them with a "Grant access"
    /// affordance so the user can extend the security-scoped
    /// bookmark set with one NSOpenPanel pick instead of grinding
    /// through each file's permission failure at read time.
    ///
    /// Recomputed on every `ProjectRootBuilder.rebuild` — the list
    /// shrinks as the user grants access (the granted dir becomes
    /// a `sourcePaths` entry, so its descendants no longer count
    /// as external) and grows when new external symlinks appear in
    /// the source. Optional + nil-default for backward compat with
    /// projects persisted before this field existed.
    var pendingSymlinkRoots: [String]?

    init(id: UUID = UUID(),
         name: String,
         sourcePaths: [String] = [],
         sourceBookmarks: [Data]? = nil,
         createdAt: Date = .now,
         lastIndexedAt: Date? = nil,
         modelFingerprint: String? = nil,
         contextMode: ProjectContextMode? = nil,
         maxInventoryFiles: Int? = nil,
         pendingSymlinkRoots: [String]? = nil) {
        self.id = id
        self.name = name
        self.sourcePaths = sourcePaths
        self.sourceBookmarks = sourceBookmarks
        self.createdAt = createdAt
        self.lastIndexedAt = lastIndexedAt
        self.modelFingerprint = modelFingerprint
        self.contextMode = contextMode
        self.maxInventoryFiles = maxInventoryFiles
        self.pendingSymlinkRoots = pendingSymlinkRoots
    }

    /// Modalità effettiva (override per-progetto o default
    /// globale).
    var effectiveContextMode: ProjectContextMode {
        contextMode ?? AppSettings.projectContextMode
    }

    /// Cap effettivo per l'inventario (override per-progetto o
    /// default globale).
    var effectiveMaxInventoryFiles: Int {
        if let n = maxInventoryFiles { return n }
        return AppSettings.projectInventoryMaxFiles
    }
}

/// Global list of projects + persistence. Documents themselves live in
/// `DocumentLibrary` (tagged with `projectID`); `ProjectLibrary` only
/// owns the project metadata and orchestrates indexing.
@MainActor
final class ProjectLibrary: ObservableObject {
    @Published private(set) var projects: [Project] = []

    /// Active security-scoped resource accessors, one vault per project
    /// id. Vault `acquire(bookmarks:)` is called whenever a project's
    /// bookmark blob changes (initial load, edit, re-add) so the macOS
    /// sandbox keeps granting reads to the user's source folders across
    /// app launches. Removing a project (or replacing its sources) drops
    /// the vault, which `stopAccessingSecurityScopedResource()`s every
    /// URL it was holding.
    private var sourceAccess: [UUID: ProjectSourceAccess] = [:]

    init() {
        load()
    }

    func create(name: String) -> Project {
        var p = Project(name: name)
        // Empty sourcePaths → rebuild is a no-op; no discoveries
        // possible. Still call it so the empty farm root exists on
        // disk for downstream code that checks for it.
        if let result = try? ProjectRootBuilder.rebuild(p) {
            p.pendingSymlinkRoots = result.externalSymlinkRoots.isEmpty
                ? nil
                : result.externalSymlinkRoots
        }
        projects.insert(p, at: 0)
        saveIndex()
        return p
    }

    func update(_ project: Project) {
        guard let idx = projects.firstIndex(where: { $0.id == project.id }) else { return }
        let prev = projects[idx]
        var next = project
        // Rebuild BEFORE the save so the discovery result lands in
        // `next.pendingSymlinkRoots` before we persist — otherwise
        // we'd write twice (once with stale discovery, once with
        // fresh).
        if prev.sourcePaths != project.sourcePaths,
           let result = try? ProjectRootBuilder.rebuild(project)
        {
            next.pendingSymlinkRoots = result.externalSymlinkRoots.isEmpty
                ? nil
                : result.externalSymlinkRoots
        }
        projects[idx] = next
        saveIndex()
        // Always refresh the vault when bookmarks change — even if the
        // path list is identical, the user may have re-picked a folder
        // to recover sandbox access after a stale grant.
        if prev.sourceBookmarks != project.sourceBookmarks {
            refreshAccess(for: project)
        }
    }

    /// Promote a previously-discovered external symlink root into a
    /// real `sourcePaths` entry now that the user has granted access
    /// via NSOpenPanel. `path` is the URL.path the picker returned
    /// (may be exactly the discovered path or a chosen ancestor of
    /// it — the rebuild's allowed-roots check handles either form).
    /// `bookmark` is the security-scoped blob, persisted alongside
    /// the path so the grant survives the next launch.
    ///
    /// Re-uses `update(_:)`'s pipeline: the rebuild fires, discovery
    /// re-runs, and the granted path's descendants drop out of
    /// `pendingSymlinkRoots` on their own.
    func grantSymlinkRoot(path: String,
                           bookmark: Data,
                           for projectID: UUID) {
        guard let idx = projects.firstIndex(where: { $0.id == projectID })
        else { return }
        var updated = projects[idx]
        var bookmarks = updated.sourceBookmarks ?? []
        while bookmarks.count < updated.sourcePaths.count {
            bookmarks.append(Data())
        }
        if let existing = updated.sourcePaths.firstIndex(of: path) {
            bookmarks[existing] = bookmark
        } else {
            updated.sourcePaths.append(path)
            bookmarks.append(bookmark)
        }
        updated.sourceBookmarks = bookmarks
        update(updated)
    }

    /// Add `path` to a project's `pendingSymlinkRoots` from outside
    /// the rebuild pipeline. Called by `ChatStore` when a file-
    /// reading tool catches an EPERM mid-chat on a path that
    /// resolved through a symlink: the discovery rebuild only runs
    /// at create/edit time, so a link created (or first followed)
    /// after the last rebuild wouldn't otherwise show up in the
    /// Grant Access list until the user manually triggered a
    /// refresh.
    ///
    /// No-op when the path is already in `sourcePaths` (race: the
    /// user granted access just as a tool was running) or already
    /// in `pendingSymlinkRoots`. Cheap to call repeatedly — the
    /// dedupe keeps the disk write rare.
    func notePendingSymlinkRoot(path: String, for projectID: UUID) {
        guard let idx = projects.firstIndex(where: { $0.id == projectID })
        else { return }
        let project = projects[idx]
        // Already covered by a granted source → nothing to do; the
        // EPERM was a stale bookmark or a race, not a missing grant.
        if project.sourcePaths.contains(path) { return }
        var pending = project.pendingSymlinkRoots ?? []
        if pending.contains(path) { return }
        pending.append(path)
        pending.sort()  // mirror ProjectRootBuilder's stable ordering
        projects[idx].pendingSymlinkRoots = pending
        saveIndex()
    }

    /// User dismissed an external symlink root from the "Grant
    /// access" list — they know the link is intentional dead weight
    /// (build artefact, generated tag, etc.) and don't want the
    /// banner to keep nagging. The dismissal is one-shot: the next
    /// rebuild rediscovers the same path if the source-side link
    /// is still there. For a sticky ignore, the user should remove
    /// the link upstream or skip the source folder altogether.
    func dismissSymlinkRoot(path: String, for projectID: UUID) {
        guard let idx = projects.firstIndex(where: { $0.id == projectID })
        else { return }
        guard var pending = projects[idx].pendingSymlinkRoots else { return }
        pending.removeAll { $0 == path }
        projects[idx].pendingSymlinkRoots = pending.isEmpty ? nil : pending
        saveIndex()
    }

    func delete(_ id: UUID, documents: DocumentLibrary) {
        documents.purge(projectID: id)
        projects.removeAll { $0.id == id }
        saveIndex()
        ProjectRootBuilder.remove(id: id)
        sourceAccess.removeValue(forKey: id) // releases via deinit
    }

    func project(id: UUID) -> Project? {
        projects.first(where: { $0.id == id })
    }

    // MARK: - persistence

    private func load() {
        guard let url = try? PersistencePaths.projectsIndexURL(),
              let data = try? Data(contentsOf: url) else { return }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        if let entries = try? decoder.decode([Project].self, from: data) {
            projects = entries.sorted { $0.createdAt > $1.createdAt }
            // Re-establish sandbox access to every project's source
            // folders before any tool dispatch can touch them. Without
            // this, reads through the per-project symlink farm fail
            // with permission errors after a restart.
            for project in projects {
                refreshAccess(for: project)
            }
        }
    }

    /// (Re)create the per-project access vault from the project's
    /// current `sourceBookmarks`. If any bookmark resolves stale, the
    /// vault re-mints it and we persist the refreshed blob so the next
    /// launch starts from a clean state. Bookmark-less projects (old
    /// JSON, or freshly created without paths) get an empty vault —
    /// safe, but reads to those paths will fail until the user adds
    /// them through the picker.
    private func refreshAccess(for project: Project) {
        let bookmarks = project.sourceBookmarks ?? []
        let vault = ProjectSourceAccess()
        let resolved = vault.acquire(bookmarks: bookmarks)
        sourceAccess[project.id] = vault

        // Re-mint stale bookmarks so we don't keep paying the resolve
        // cost (and keep working if the original bookmark eventually
        // expires for real). Only persist if something actually changed
        // to avoid touching the JSON on every launch.
        let hasStale = resolved.contains(where: { $0?.isStale == true })
        guard hasStale else { return }
        var newBookmarks = bookmarks
        for (i, entry) in resolved.enumerated()
        where i < newBookmarks.count {
            guard let r = entry, r.isStale else { continue }
            if let fresh = ProjectSourceAccess.makeBookmark(for: r.url) {
                newBookmarks[i] = fresh
            }
        }
        if newBookmarks != bookmarks,
           let idx = projects.firstIndex(where: { $0.id == project.id }) {
            var updated = projects[idx]
            updated.sourceBookmarks = newBookmarks
            projects[idx] = updated
            saveIndex()
        }
    }

    func saveIndex() {
        guard let url = try? PersistencePaths.projectsIndexURL() else { return }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(projects) {
            try? data.write(to: url, options: .atomic)
        }
    }
}
