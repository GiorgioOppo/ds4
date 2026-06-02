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
/// Symlinks inside the source tree are handled as follows:
/// - A symlinked **file whose target stays inside this `sourcePaths`
///   entry** is mirrored as a farm symlink pointing at the resolved
///   real target. Internal convenience links (e.g. `current ->
///   v2/build.swift`) survive into the farm and are reachable by
///   every read/write tool.
/// - A symlinked file whose target falls **outside** the source
///   folder is skipped. The macOS sandbox bookmark is granted per
///   `sourcePaths` entry, so the farm can't actually read through
///   those links — surfacing them would leave the model staring at
///   a `read` that fails with a confusing "permission denied"
///   instead of just not seeing the file.
/// - A symlinked **directory** is skipped — descending risks scan
///   cycles, and the real dir is reachable via another `sourcePaths`
///   entry when needed.
///
/// Name collisions between source entries (or files that share a
/// basename at the same level) get `-2`, `-3`, … suffixes.
enum ProjectRootBuilder {

    /// Wipes the project's root and rebuilds the symlink farm from
    /// scratch. Cheap for small projects; large source trees rebuild
    /// in the low hundreds of ms. Call after `Project.sourcePaths`
    /// changes.
    @discardableResult
    static func rebuild(_ project: Project) throws -> URL {
        let root = try PersistencePaths.projectRootDir(id: project.id)
        try clearContents(of: root)
        var used: Set<String> = []
        for sourcePath in project.sourcePaths {
            let src = URL(fileURLWithPath: sourcePath)
            let name = uniqueName(src.lastPathComponent, in: &used)
            let dst = root.appendingPathComponent(name)
            try linkSource(src, to: dst)
        }
        return root
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
                return try rebuild(project)
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

    private static func linkSource(_ src: URL, to dst: URL) throws {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: src.path, isDirectory: &isDir) else {
            return
        }
        if isDir.boolValue {
            try fm.createDirectory(at: dst,
                                    withIntermediateDirectories: true)
            try mirrorDirectory(src, into: dst)
        } else {
            try fm.createSymbolicLink(at: dst, withDestinationURL: src)
        }
    }

    private static func mirrorDirectory(_ src: URL, into dst: URL) throws {
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
                // File symlink. Only mirror it when its resolved
                // target stays inside this source folder — a link
                // out to `/Users/...` or another sourcePath would
                // either fail at read time (the macOS sandbox grants
                // a security-scoped bookmark per `sourcePaths` entry
                // only) or duplicate an entry that another mirror
                // pass already surfaces. Surfacing a known-broken
                // link in the farm is worse than hiding it: the
                // model sees a `*.md` file, tries `read`, and gets a
                // "couldn't be opened because you don't have
                // permission" error.
                let resolved = (item.path as NSString).resolvingSymlinksInPath
                let srcReal = (src.path as NSString).resolvingSymlinksInPath
                let withinSrc = resolved == srcReal
                    || resolved.hasPrefix(srcReal + "/")
                guard withinSrc else { continue }
                let realTarget = URL(fileURLWithPath: resolved)
                try fm.createSymbolicLink(at: target,
                                          withDestinationURL: realTarget)
            } else {
                try fm.createSymbolicLink(at: target,
                                          withDestinationURL: item)
            }
        }
    }
}
