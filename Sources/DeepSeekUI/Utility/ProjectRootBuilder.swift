import Foundation

/// Materializes a `Project` on disk as a tree of real directories that
/// `ToolContext.rootDirectory` is set to, so tools (shell, swift build,
/// xcodebuild, git, …) execute against a stable cwd that reflects the
/// project layout.
///
/// The exact way `sourcePaths` get into the farm depends on
/// `Project.effectiveImportStrategy`:
///
/// - **`.symlinkFarm`**: real directories with one symlink per file
///   pointing at the user's on-disk source. Live updates; needs a
///   security-scoped bookmark for every read; surfaces out-of-source
///   symlinks via `RebuildResult.externalSymlinkRoots`.
/// - **`.copy`**: walk each source, **resolve symlinks**, and copy the
///   actual bytes into the farm. After import the farm is fully self-
///   contained — no sandbox bookmark needed for reads, and tool
///   mutations land on the in-container copy rather than the user's
///   real files. The farm is initialised as a Git repo so the user
///   can `git diff` to see what the agent changed.
/// - **`.gitClone`**: `git clone --depth 1` the upstream into the farm
///   root. Self-contained; updates are explicit via "Pull" in the
///   detail view.
///
/// Name collisions between source entries (or files that share a
/// basename at the same level) get `-2`, `-3`, … suffixes.
enum ProjectRootBuilder {

    /// Outcome of one rebuild pass. The URL is the project root the
    /// caller wires into `ToolContext.rootDirectory`. The symlink
    /// roots field is only populated under `.symlinkFarm` — copy
    /// and git clones resolve every link at import time so the list
    /// stays empty.
    struct RebuildResult {
        let root: URL
        let externalSymlinkRoots: [String]
    }

    /// Wipes the project's root and rebuilds the farm from scratch.
    /// Cheap for small projects under `.symlinkFarm`; the cost of
    /// `.copy` scales with the source tree's byte count.
    @discardableResult
    static func rebuild(_ project: Project) throws -> RebuildResult {
        let root = try PersistencePaths.projectRootDir(id: project.id)
        try clearContents(of: root)
        switch project.effectiveImportStrategy {
        case .symlinkFarm:
            return try rebuildSymlinkFarm(project, root: root)
        case .copy:
            try rebuildCopy(project, root: root)
            initGitRepoIfNeeded(at: root)
            return RebuildResult(root: root, externalSymlinkRoots: [])
        case .gitClone(let repoURL, let branch):
            try rebuildGitClone(repoURL: repoURL, branch: branch, root: root)
            return RebuildResult(root: root, externalSymlinkRoots: [])
        }
    }

    /// Returns the project root URL, rebuilding the farm if it's
    /// missing or empty but the project has sources. Used by
    /// `ChatStore` on every tool dispatch so projects loaded from
    /// disk after an app restart self-heal lazily.
    static func ensureBuilt(_ project: Project) -> URL? {
        do {
            let root = try PersistencePaths.projectRootDir(id: project.id)
            let isEmpty = (try? FileManager.default
                .contentsOfDirectory(atPath: root.path).isEmpty) ?? true
            if isEmpty && hasContent(project) {
                return try rebuild(project).root
            }
            return root
        } catch {
            return nil
        }
    }

    /// `git pull` an existing `.gitClone` farm so the user picks up
    /// upstream changes without losing local edits the agent may
    /// have committed. No-op for projects using a different
    /// strategy. Best-effort: a failed pull leaves the farm in its
    /// pre-pull state.
    static func pullClone(_ project: Project) {
        guard case .gitClone = project.effectiveImportStrategy,
              let root = try? PersistencePaths.projectRootDir(id: project.id)
        else { return }
        _ = run(git: ["pull", "--ff-only"], cwd: root)
    }

    /// Deletes the project's farm root. Idempotent — missing
    /// directories are ignored.
    static func remove(id: UUID) {
        let fm = FileManager.default
        guard let root = try? PersistencePaths.projectRootDir(id: id) else {
            return
        }
        try? fm.removeItem(at: root)
    }

    /// True when the project has something to materialise — used by
    /// `ensureBuilt` to skip the "rebuild on empty" optimisation for
    /// projects that genuinely should be empty.
    private static func hasContent(_ project: Project) -> Bool {
        switch project.effectiveImportStrategy {
        case .symlinkFarm, .copy:
            return !project.sourcePaths.isEmpty
        case .gitClone:
            return true
        }
    }

    // MARK: - symlink farm (original behaviour)

    private static func rebuildSymlinkFarm(_ project: Project,
                                            root: URL) throws -> RebuildResult {
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
        let sortedDiscovered = discovered.sorted()
        return RebuildResult(root: root,
                              externalSymlinkRoots: sortedDiscovered)
    }

    // MARK: - copy mode

    /// Walk every source folder once and copy its contents into the
    /// farm. Symlinks are dereferenced — we want the actual bytes
    /// in-container, not a link back to the user's machine.
    private static func rebuildCopy(_ project: Project,
                                     root: URL) throws {
        var used: Set<String> = []
        let fm = FileManager.default
        for sourcePath in project.sourcePaths {
            let src = URL(fileURLWithPath: sourcePath)
            let name = uniqueName(src.lastPathComponent, in: &used)
            let dst = root.appendingPathComponent(name)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: src.path, isDirectory: &isDir) else {
                continue
            }
            if isDir.boolValue {
                try fm.createDirectory(at: dst,
                                        withIntermediateDirectories: true)
                try copyDirectoryResolved(src, into: dst)
            } else {
                try copyFileResolved(src, to: dst)
            }
        }
    }

    /// Recursively copy a source folder, materialising every file
    /// (including those reached through a symlink) as a real file
    /// in the destination. We deliberately don't preserve symlinks
    /// inside the farm — the whole point of copy mode is that the
    /// destination doesn't depend on the user's on-disk state.
    private static func copyDirectoryResolved(_ src: URL,
                                                into dst: URL) throws {
        let fm = FileManager.default
        let keys: [URLResourceKey] = [.isDirectoryKey, .isSymbolicLinkKey]
        guard let it = fm.enumerator(
            at: src,
            includingPropertiesForKeys: keys,
            options: [])
        else { return }
        while let item = it.nextObject() as? URL {
            let vals = try? item.resourceValues(forKeys: Set(keys))
            let isLink = vals?.isSymbolicLink == true
            let isDir = vals?.isDirectory == true
            // Symlinked directory: skip descent to avoid scan loops.
            // The target — if it's inside `sourcePaths` — will be
            // walked separately, and external targets are out of
            // scope for copy mode (`.copy` is about isolation).
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
            } else {
                try copyFileResolved(item, to: target)
            }
        }
    }

    /// Copy a single file, following any symlink so the destination
    /// holds the actual bytes. Skips entries whose read fails (a
    /// dangling link, a sandbox EPERM) instead of aborting the
    /// whole import — the user can re-import after fixing
    /// permissions and the file appears.
    private static func copyFileResolved(_ src: URL, to dst: URL) throws {
        let fm = FileManager.default
        // Ensure parent exists; `copyDirectoryResolved` already
        // created the chain for nested files, but a top-level
        // source file lands directly under root and needs no extra
        // work.
        let parent = dst.deletingLastPathComponent()
        if !fm.fileExists(atPath: parent.path) {
            try fm.createDirectory(at: parent,
                                    withIntermediateDirectories: true)
        }
        // Resolve symlinks once so the read goes straight at the
        // real target — avoids the symlink-chain-permission failure
        // mode where `Data(contentsOf:)` fails because the link's
        // target is outside the sandbox bookmark.
        let resolvedPath = (src.path as NSString).resolvingSymlinksInPath
        let resolvedURL = URL(fileURLWithPath: resolvedPath)
        guard let data = try? Data(contentsOf: resolvedURL) else {
            return  // skip unreadable files; user can re-import later
        }
        try data.write(to: dst, options: .atomic)
        // Preserve the source's executable bit so the farm copy of
        // `Tools/generate-xcodeproj.sh` stays runnable.
        if let attrs = try? fm.attributesOfItem(atPath: resolvedPath),
           let perms = (attrs[.posixPermissions] as? NSNumber)?.intValue
        {
            try? fm.setAttributes(
                [.posixPermissions: NSNumber(value: perms)],
                ofItemAtPath: dst.path)
        }
    }

    /// `git init` the farm + stage everything + baseline commit so
    /// the user can `git diff` to see what the agent changed. Best-
    /// effort: failures (sandbox blocks `Process()`, no `git` on
    /// PATH) leave the farm without a repo but don't fail the
    /// rebuild. The agent's tools still work; the user just doesn't
    /// get the diff affordance.
    private static func initGitRepoIfNeeded(at root: URL) {
        let fm = FileManager.default
        let gitDir = root.appendingPathComponent(".git",
                                                  isDirectory: true)
        if fm.fileExists(atPath: gitDir.path) { return }
        let init_ = run(git: ["init", "-q"], cwd: root)
        guard init_ else { return }
        _ = run(git: ["add", "-A"], cwd: root)
        // Identity is required for `commit` to succeed even on a
        // brand-new repo. We set it as a one-shot via env so the
        // user's global gitconfig stays untouched.
        _ = run(
            git: ["-c", "user.email=deepseek@local",
                  "-c", "user.name=DeepSeek Import",
                  "commit", "-q", "-m", "Initial import"],
            cwd: root)
    }

    // MARK: - git clone mode

    /// Shallow-clone the upstream repo into the farm root. When the
    /// user picked a branch, that's what we clone; otherwise we let
    /// git resolve the default. Failures throw so the caller knows
    /// the rebuild didn't produce a usable farm.
    private static func rebuildGitClone(repoURL: String,
                                          branch: String?,
                                          root: URL) throws {
        // Clone wants the destination to NOT exist (otherwise it
        // errors out). `clearContents` left us with an empty root,
        // so we clone into a sibling and then move it on top.
        let parent = root.deletingLastPathComponent()
        let stagingURL = parent.appendingPathComponent(
            ".clone-staging-\(UUID().uuidString)",
            isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: stagingURL)
        }
        var args = ["clone", "--depth", "1"]
        if let branch, !branch.isEmpty {
            args.append("--branch")
            args.append(branch)
        }
        args.append(repoURL)
        args.append(stagingURL.path)
        if !run(git: args, cwd: parent) {
            throw NSError(
                domain: "ProjectRootBuilder", code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                            "git clone failed for \(repoURL)"])
        }
        // Move the cloned tree into the farm root. We can't just
        // `mv` over a non-empty dir, so we transfer item by item.
        let fm = FileManager.default
        let children = (try? fm.contentsOfDirectory(
            at: stagingURL, includingPropertiesForKeys: nil)) ?? []
        for child in children {
            let dst = root.appendingPathComponent(child.lastPathComponent)
            try fm.moveItem(at: child, to: dst)
        }
    }

    // MARK: - internals

    /// Run `git` with the given arguments synchronously, returning
    /// true on exit 0. Used by the import paths; the app needs to
    /// be allowed to spawn `Process()` for this to do anything
    /// useful (non-App-Store builds).
    @discardableResult
    private static func run(git args: [String], cwd: URL) -> Bool {
        let process = Process()
        process.launchPath = "/usr/bin/env"
        process.arguments = ["git"] + args
        process.currentDirectoryURL = cwd
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        do {
            try process.run()
        } catch {
            return false
        }
        process.waitUntilExit()
        return process.terminationStatus == 0
    }

    /// Canonical (post `resolvingSymlinksInPath`) form of every
    /// source path, used by `.symlinkFarm`'s trust check.
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
        guard let it = fm.enumerator(
            at: src,
            includingPropertiesForKeys: keys,
            options: [])
        else { return }

        while let item = it.nextObject() as? URL {
            let vals = try? item.resourceValues(forKeys: Set(keys))
            let isLink = vals?.isSymbolicLink == true
            let isDir = vals?.isDirectory == true
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
