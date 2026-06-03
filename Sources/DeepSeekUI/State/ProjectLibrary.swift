import Foundation
import SwiftUI

/// How the project's `sourcePaths` get materialised into the farm
/// root. Picked at create time, persisted on the `Project`, applied
/// by `ProjectRootBuilder.rebuild`. Legacy projects (decoded from
/// JSON written before this field existed) default to
/// `.symlinkFarm` so their behaviour doesn't change at upgrade.
enum ProjectImportStrategy: Codable, Hashable {
    /// Original behaviour: real directories with one symlink per
    /// file pointing back at the user's on-disk source. Picks up
    /// upstream edits live; requires an active security-scoped
    /// bookmark for every read; surfaces out-of-source links
    /// through the discover+grant flow.
    case symlinkFarm

    /// Resolve every symlink and copy bytes from `sourcePaths` into
    /// the farm. Self-contained after the import: tools read/write
    /// the in-container copy, never the user's real folders. The
    /// farm is initialised as a Git repo (`git init` + baseline
    /// commit) so the user can `git diff` to see what the agent
    /// changed and `git checkout` to roll back. Updates are
    /// explicit via "Re-import" in `ProjectDetailView`.
    case copy

    /// Shallow `git clone` of a remote repository into the farm.
    /// Self-contained; "Pull" updates from upstream. Requires the
    /// app to be allowed to spawn `git` (non-App-Store builds, or
    /// stub with future entitlements).
    case gitClone(repoURL: String, branch: String?)

    /// Stable id used for the SwiftUI Picker selection. Each
    /// variant collapses to one canonical id even when the
    /// associated values differ.
    var pickerID: String {
        switch self {
        case .symlinkFarm: return "symlink"
        case .copy:        return "copy"
        case .gitClone:    return "git"
        }
    }

    /// User-facing label.
    var displayName: String {
        switch self {
        case .symlinkFarm: return "Symlink farm"
        case .copy:        return "Copy"
        case .gitClone:    return "Clone from Git"
        }
    }
}

/// State of a per-project rebuild — used by the UI to render a
/// spinner or banner while `.copy` is grinding through a large
/// source tree or `.gitClone` is shelling out to `git`. Symlink
/// rebuilds are fast enough that we don't bother surfacing them.
enum ProjectRebuildStatus: Equatable {
    case idle
    case running
    case failed(message: String)
}

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
    /// projects persisted before this field existed. Only populated
    /// for `importStrategy == .symlinkFarm` — copy / git modes are
    /// self-contained and don't need a grant flow.
    var pendingSymlinkRoots: [String]?

    /// How `ProjectRootBuilder` materialises `sourcePaths` into the
    /// farm. Optional + nil-default for projects persisted before
    /// the field existed — they migrate on the next rebuild via
    /// `ProjectLibrary`. New projects default to `.copy` so the
    /// sandbox isn't paying the bookmark-per-read tax for everyone.
    var importStrategy: ProjectImportStrategy?

    /// Effective strategy used by the rebuild pipeline. Legacy
    /// projects with a nil field map to `.symlinkFarm` so their
    /// behaviour stays bit-identical until the user (or
    /// `ProjectLibrary`'s migration path) flips them.
    var effectiveImportStrategy: ProjectImportStrategy {
        importStrategy ?? .symlinkFarm
    }

    init(id: UUID = UUID(),
         name: String,
         sourcePaths: [String] = [],
         sourceBookmarks: [Data]? = nil,
         createdAt: Date = .now,
         lastIndexedAt: Date? = nil,
         modelFingerprint: String? = nil,
         contextMode: ProjectContextMode? = nil,
         maxInventoryFiles: Int? = nil,
         pendingSymlinkRoots: [String]? = nil,
         importStrategy: ProjectImportStrategy? = nil) {
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
        self.importStrategy = importStrategy
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

    /// Per-project rebuild state. The UI binds to this so the
    /// detail view can render a spinner while a `.copy` walks
    /// through a large source tree or `.gitClone` is talking to
    /// the network. Symlink rebuilds finish synchronously and never
    /// land here — keeping the dict small.
    @Published private(set) var rebuildStates: [UUID: ProjectRebuildStatus] = [:]

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

    /// Default strategy used by `create(name:)`. New projects land
    /// on `.copy` so the agent's tools mutate the in-container copy
    /// rather than the user's real files, and the sandbox doesn't
    /// pay the bookmark-per-read tax. Legacy projects (decoded
    /// with `importStrategy == nil`) are upgraded by the
    /// `load()` path.
    static let defaultImportStrategy: ProjectImportStrategy = .copy

    func create(name: String,
                strategy: ProjectImportStrategy = ProjectLibrary
                    .defaultImportStrategy) -> Project {
        let p = Project(name: name, importStrategy: strategy)
        projects.insert(p, at: 0)
        saveIndex()
        startRebuild(p.id)
        return p
    }

    /// Re-run the farm rebuild for a project, refreshing the import.
    /// Used by the "Re-import" / "Pull" buttons in
    /// `ProjectDetailView`. For `.symlinkFarm` it's a no-op-ish
    /// recompute of the discovered list; for `.copy` it re-reads
    /// the source folders and overwrites the farm; for
    /// `.gitClone` it `git pull`s on top of the existing clone
    /// instead of doing a fresh shallow clone.
    func refresh(_ projectID: UUID) {
        startRebuild(projectID)
    }

    /// Kick off whichever rebuild the project's strategy needs.
    /// Symlink mode runs synchronously — it's just creating
    /// directory entries, no point spinning up a Task. Copy / git
    /// modes detach to a background priority Task because they can
    /// take seconds (large repos, network), and we don't want the
    /// main thread blocked while the user is staring at a frozen
    /// sheet. The `rebuildStates` published dict drives the
    /// spinner in `ProjectDetailView`.
    private func startRebuild(_ projectID: UUID) {
        guard let project = projects.first(where: { $0.id == projectID })
        else { return }
        switch project.effectiveImportStrategy {
        case .symlinkFarm:
            applySyncSymlinkRebuild(project)
        case .copy, .gitClone:
            // Coalesce reruns: a second click on "Re-import" while
            // the first copy is still grinding would race the
            // detached Task and produce a half-overwritten farm.
            // Drop the request; the user can retry once the
            // current Task finishes.
            if rebuildStates[projectID] == .running { return }
            rebuildStates[projectID] = .running
            Task.detached(priority: .userInitiated) { [weak self] in
                let outcome: Result<ProjectRootBuilder.RebuildResult, Error>
                do {
                    if case .gitClone = project.effectiveImportStrategy,
                       await self?.farmAlreadyCloned(for: projectID) == true
                    {
                        // Existing clone → "Pull" semantics, not a
                        // fresh shallow clone over an already-populated
                        // farm root.
                        ProjectRootBuilder.pullClone(project)
                        outcome = .success(.init(
                            root: try PersistencePaths.projectRootDir(
                                id: projectID),
                            externalSymlinkRoots: []))
                    } else {
                        let result = try ProjectRootBuilder.rebuild(project)
                        outcome = .success(result)
                    }
                } catch {
                    outcome = .failure(error)
                }
                await self?.finishRebuild(projectID, outcome: outcome)
            }
        }
    }

    /// Fast-path sync rebuild for `.symlinkFarm`. Stays on the main
    /// actor because creating symlinks is cheap and the call site
    /// (`update`) already expects synchronous state mutation.
    private func applySyncSymlinkRebuild(_ project: Project) {
        guard let idx = projects.firstIndex(where: { $0.id == project.id })
        else { return }
        rebuildStates.removeValue(forKey: project.id)
        if let result = try? ProjectRootBuilder.rebuild(project) {
            projects[idx].pendingSymlinkRoots =
                result.externalSymlinkRoots.isEmpty
                    ? nil
                    : result.externalSymlinkRoots
        }
        saveIndex()
    }

    /// Apply the outcome of a copy / git rebuild back onto the
    /// published state. Called from the detached Task via a hop
    /// to the main actor.
    private func finishRebuild(
        _ projectID: UUID,
        outcome: Result<ProjectRootBuilder.RebuildResult, Error>
    ) {
        guard let idx = projects.firstIndex(where: { $0.id == projectID })
        else {
            rebuildStates.removeValue(forKey: projectID)
            return
        }
        switch outcome {
        case .success(let result):
            projects[idx].pendingSymlinkRoots =
                result.externalSymlinkRoots.isEmpty
                    ? nil
                    : result.externalSymlinkRoots
            rebuildStates.removeValue(forKey: projectID)
            saveIndex()
        case .failure(let error):
            rebuildStates[projectID] = .failed(
                message: error.localizedDescription)
        }
    }

    /// True when the project's farm already contains a `.git`
    /// directory — used by `startRebuild` to pick between a fresh
    /// shallow clone and a `git pull` on the existing tree.
    /// `nonisolated` so the detached rebuild Task can call it
    /// without re-hopping to the main actor for every check; the
    /// filesystem read it performs doesn't touch any actor-isolated
    /// state.
    private func farmAlreadyCloned(for projectID: UUID) -> Bool {
        guard let root = try? PersistencePaths.projectRootDir(id: projectID)
        else { return false }
        let gitDir = root.appendingPathComponent(".git",
                                                  isDirectory: true)
        return FileManager.default.fileExists(atPath: gitDir.path)
    }

    func update(_ project: Project) {
        guard let idx = projects.firstIndex(where: { $0.id == project.id }) else { return }
        let prev = projects[idx]
        let sourcesChanged = prev.sourcePaths != project.sourcePaths
        let strategyChanged = prev.effectiveImportStrategy
            != project.effectiveImportStrategy
        projects[idx] = project
        saveIndex()
        // Always refresh the vault when bookmarks change — even if the
        // path list is identical, the user may have re-picked a folder
        // to recover sandbox access after a stale grant. Done before
        // the rebuild so a `.copy` reading source files sees the
        // freshly-activated security-scoped resources.
        if prev.sourceBookmarks != project.sourceBookmarks {
            refreshAccess(for: project)
        }
        // Symlink rebuilds finish synchronously and update
        // `pendingSymlinkRoots` in place; copy / git detach to a
        // background Task and the result lands later via
        // `finishRebuild`. The save above carries the new
        // sourcePaths regardless, so even if the user closes the
        // app mid-copy, the project metadata is persisted.
        if sourcesChanged || strategyChanged {
            startRebuild(project.id)
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
            migrateLegacyProjectsToCopy()
        }
    }

    /// Walk every loaded project; for any with `importStrategy == nil`
    /// (i.e. persisted before the field existed), flip the strategy
    /// to `.copy` and kick off a background rebuild so the legacy
    /// symlink farm gets replaced with an in-container copy.
    /// Called from `load()` after `refreshAccess` activates the
    /// security-scoped bookmarks — the rebuild reads source folders
    /// directly, which only works while the bookmarks are live.
    ///
    /// Async by design: a large project with a multi-GB source tree
    /// would freeze app startup if we did the copy synchronously.
    /// The metadata flip + save lands immediately; the actual file
    /// I/O runs in a detached Task and the UI surfaces "Importing"
    /// in the meantime.
    ///
    /// Failure mode: a failed rebuild leaves the strategy at `.copy`
    /// but the farm empty, with `rebuildStates[id] = .failed`. The
    /// detail view shows the error; the user can re-pick sources or
    /// fall back to `.symlinkFarm` from a future affordance.
    private func migrateLegacyProjectsToCopy() {
        var migratedIDs: [UUID] = []
        for i in projects.indices {
            guard projects[i].importStrategy == nil,
                  !projects[i].sourcePaths.isEmpty
            else { continue }
            projects[i].importStrategy = .copy
            migratedIDs.append(projects[i].id)
        }
        guard !migratedIDs.isEmpty else { return }
        saveIndex()
        for id in migratedIDs {
            startRebuild(id)
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
