import Foundation

/// Materializes a `Project` on disk as a tree of real directories with
/// symlinks pointing at each source file. The resulting root is what
/// `ToolContext.rootDirectory` is set to, so tools (shell, swift build,
/// xcodebuild, git, …) execute against a stable cwd that reflects the
/// project layout without us copying any bytes from the user's source
/// folders.
///
/// Layout — for each entry in `Project.sourcePaths`:
/// - directory → mirrored as real directories under the project root,
///   with every contained file replaced by a symlink to the original
/// - file → symlinked directly under the project root by basename
///
/// Symlinks inside the source tree are handled with a "trust boundary"
/// computed from the project's *full* `sourcePaths` list (not just the
/// folder being mirrored):
/// - A symlinked **file** whose resolved target stays inside ANY
///   `sourcePaths` entry is mirrored as a farm symlink pointing at the
///   real target. This covers internal convenience links
///   (`current -> v2/build.swift`) and cross-source links once the
///   user has granted access to both folders.
/// - A symlinked file whose target falls **outside** every granted
///   source is skipped, and its target's parent directory is collected
///   into `RebuildResult.externalSymlinkRoots` so the UI can offer the
///   user a one-click "Grant access" affordance instead of a confusing
///   "permission denied" at first read.
/// - A symlinked **directory** is skipped — descending risks scan
///   cycles, and the real dir is reachable via another `sourcePaths`
///   entry when needed.
///
/// Name collisions between source entries (or files that share a
/// basename at the same level) get `-2`, `-3`, … suffixes.
enum ProjectRootBuilder {

    /// Outcome of one rebuild pass. The URL is the project root the
    /// caller wires into `ToolContext.rootDirectory`; the symlink
    /// roots are unique parent directories of every file-symlink
    /// target that fell outside the project's currently-granted
    /// sources. The caller persists the latter onto
    /// `Project.pendingSymlinkRoots` so `ProjectDetailView` can
    /// surface them with a "Grant access" button.
    struct RebuildResult {
        let root: URL
        let externalSymlinkRoots: [String]
    }

    /// Wipes the project's root and rebuilds the symlink farm from
    /// scratch. Cheap for small projects; large source trees rebuild
    /// in the low hundreds of ms. Call after `Project.sourcePaths`
    /// changes.
    @discardableResult
    static func rebuild(_ project: Project) throws -> RebuildResult {
        let root = try PersistencePaths.projectRootDir(id: project.id)
        try clearContents(of: root)
        var used: Set<String> = []
        let allowedRoots = canonicalRoots(of: project.sourcePaths)
        var discovered: Set<String> = []
        for sourcePath in project.sourcePaths {
            let src = URL(fileURLWithPath: sourcePath)
            let name = uniqueName(src.lastPathComponent, in: &used)
            let dst = root.appendingPathComponent(name)
            try linkSource(src,
                            to: dst,
                            allowedRoots: allowedRoots,
                            discovered: &discovered)
        }
        // Sort for stable UI ordering across rebuilds — set-iteration
        // order would shuffle the "Grant access" list on every
        // refresh, which is jarring even when the contents are stable.
        let sortedDiscovered = discovered.sorted()
        return RebuildResult(root: root,
                              externalSymlinkRoots: sortedDiscovered)
    }

    /// Returns the project root URL, rebuilding the symlink farm if
    /// it's missing or empty but the project has sources. Used by
    /// `ChatStore` on every tool dispatch so projects loaded from
    /// disk after an app restart self-heal lazily.
    static func ensureBuilt(_ project: Project) -> URL? {
        do {
            let root = try PersistencePaths.projectRootDir(id: project.id)
            let isEmpty = (try? FileManager.default
                .contentsOfDirectory(atPath: root.path).isEmpty) ?? true
            if isEmpty && !project.sourcePaths.isEmpty {
                return try rebuild(project).root
            }
            return root
        } catch {
            return nil
        }
    }

    /// Deletes the project's symlink-farm root. Idempotent — missing
    /// directories are ignored.
    static func remove(id: UUID) {
        let fm = FileManager.default
        guard let root = try? PersistencePaths.projectRootDir(id: id) else {
            return
        }
        try? fm.removeItem(at: root)
    }

    // MARK: - internals

    /// Canonical (post `resolvingSymlinksInPath`) form of every
    /// source path, used for the symlink trust check. Pre-canonicalised
    /// once so the per-link comparison stays O(n) over a small array
    /// rather than re-resolving on each hit.
    private static func canonicalRoots(of paths: [String]) -> [String] {
        var out: [String] = []
        out.reserveCapacity(paths.count)
        for p in paths {
            let real = (p as NSString).resolvingSymlinksInPath
            if !out.contains(real) { out.append(real) }
        }
        return out
    }

    private static func clearContents(of dir: URL) throws {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil) else { return }
        for item in items {
            try fm.removeItem(at: item)
        }
    }

    private static func uniqueName(_ proposed: String,
                                    in used: inout Set<String>) -> String {
        if !used.contains(proposed) {
            used.insert(proposed)
            return proposed
        }
        var counter = 2
        while used.contains("\(proposed)-\(counter)") { counter += 1 }
        let name = "\(proposed)-\(counter)"
        used.insert(name)
        return name
    }

    private static func linkSource(_ src: URL,
                                    to dst: URL,
                                    allowedRoots: [String],
                                    discovered: inout Set<String>) throws {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: src.path, isDirectory: &isDir) else {
            return
        }
        if isDir.boolValue {
            try fm.createDirectory(at: dst,
                                    withIntermediateDirectories: true)
            try mirrorDirectory(src,
                                 into: dst,
                                 allowedRoots: allowedRoots,
                                 discovered: &discovered)
        } else {
            try fm.createSymbolicLink(at: dst, withDestinationURL: src)
        }
    }

    private static func mirrorDirectory(_ src: URL,
                                         into dst: URL,
                                         allowedRoots: [String],
                                         discovered: inout Set<String>) throws {
        let fm = FileManager.default
        let keys: [URLResourceKey] = [.isDirectoryKey, .isSymbolicLinkKey]
        // Include hidden entries (.git, .swiftpm, dotfiles): tools
        // like `git status` and `swift build` need them. Mirroring
        // here is cheap because we symlink, not copy.
        guard let it = fm.enumerator(
            at: src,
            includingPropertiesForKeys: keys,
            options: [])
        else { return }

        while let item = it.nextObject() as? URL {
            let vals = try? item.resourceValues(forKeys: Set(keys))
            let isLink = vals?.isSymbolicLink == true
            let isDir = vals?.isDirectory == true
            // Directory symlink: skip descent to avoid scan cycles. We
            // still don't emit a symlink for the directory itself —
            // the user can add the real dir to `sourcePaths` if they
            // need it materialised in the farm.
            if isLink && isDir {
                it.skipDescendants()
                continue
            }
            let prefix = src.path
            guard item.path.hasPrefix(prefix + "/") else { continue }
            let rel = String(item.path.dropFirst(prefix.count + 1))
            let target = dst.appendingPathComponent(rel)
            if isDir {
                try fm.createDirectory(at: target,
                                        withIntermediateDirectories: true)
            } else if isLink {
                // File symlink. Mirror it if the resolved target falls
                // inside ANY granted source (the current one or
                // another `sourcePaths` entry, which would have its
                // own bookmark). Otherwise record the target's parent
                // so the UI can offer a "Grant access" affordance.
                let resolved = (item.path as NSString).resolvingSymlinksInPath
                let withinGranted = allowedRoots.contains { allowed in
                    resolved == allowed
                        || resolved.hasPrefix(allowed + "/")
                }
                if withinGranted {
                    let realTarget = URL(fileURLWithPath: resolved)
                    try fm.createSymbolicLink(at: target,
                                              withDestinationURL: realTarget)
                } else {
                    // Stash the target's parent: granting access to
                    // the *containing dir* (rather than the file
                    // itself) lets one NSOpenPanel pick unlock every
                    // sibling link landing in the same folder.
                    let parent = (resolved as NSString).deletingLastPathComponent
                    if !parent.isEmpty && parent != "/" {
                        discovered.insert(parent)
                    }
                }
            } else {
                try fm.createSymbolicLink(at: target,
                                          withDestinationURL: item)
            }
        }
    }
}
