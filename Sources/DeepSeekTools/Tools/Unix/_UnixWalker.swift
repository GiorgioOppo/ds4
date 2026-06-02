import Foundation

/// Filesystem walker for the Unix toolbox. Wraps `FileManager.enumerator`
/// with three policies that differ from the existing `GrepTool`/`GlobTool`:
///   1. **Symlinks are gated by a trust boundary.** The existing tools
///      rely implicitly on `.isRegularFileKey` to skip symlinks-to-
///      directories, but symlinks-to-regular-files DO get traversed —
///      which means a malicious link inside the agent root could
///      smuggle a read from outside it. We filter via
///      `.isSymbolicLinkKey` and only let an entry through when either
///      `followSymlinks` is explicitly true OR the link's resolved
///      target falls inside `trustedRoots`. The farm strategy
///      (`ProjectRootBuilder`) populates `trustedRoots` with
///      `rootDirectory + additionalReadRoots`, so farm symlinks pointing
///      at the user's real sources are followed transparently while
///      a sneaky link to `/etc/passwd` is still rejected.
///   2. **Cycle detection.** `FileManager.enumerator` happily loops if
///      there's a symlink cycle (`a/b/c -> a`). When we follow a
///      symlink, we resolve each URL with `resolvingSymlinksInPath` and
///      skip ones we've already visited.
///   3. **Trusted roots are NOT a "follow everything" hatch.** Only
///      symlinks whose target resolves *inside* a trusted root are
///      followed. A symlink whose target is `/Users/me/other-project/`
///      is rejected unless that path was added to `trustedRoots`.
///
/// Pure-Swift, no subprocess. Used by `ls`, `du`, `find`, `rm` (when
/// recursive), and any future Unix tool that needs a tree walk.
public enum UnixWalker {

    public struct Options: Sendable {
        public var followSymlinks: Bool
        /// Additional roots whose contents we consider safe to follow
        /// even when `followSymlinks` is false. A symlink is followed
        /// when its resolved target is `root` itself, lives under
        /// `root`, or under any entry in `trustedRoots`. Typically the
        /// caller passes `context.rootDirectory + context.additionalReadRoots`
        /// so the project's symlink farm walks through to the user's
        /// real source folders without flipping `followSymlinks` open
        /// for arbitrary targets.
        public var trustedRoots: [URL]
        public var skipHidden: Bool
        public var skipPackageDescendants: Bool
        public var maxDepth: Int?

        public init(followSymlinks: Bool = false,
                    trustedRoots: [URL] = [],
                    skipHidden: Bool = true,
                    skipPackageDescendants: Bool = true,
                    maxDepth: Int? = nil) {
            self.followSymlinks = followSymlinks
            self.trustedRoots = trustedRoots
            self.skipHidden = skipHidden
            self.skipPackageDescendants = skipPackageDescendants
            self.maxDepth = maxDepth
        }
    }

    public struct Entry: Sendable {
        public let url: URL
        public let relativePath: String
        public let isDirectory: Bool
        public let isSymlink: Bool
        public let depth: Int
    }

    /// Synchronously enumerate `root` according to `options`, calling
    /// `visit` on each entry until it returns false. Returns the number
    /// of entries visited. Synchronous because every caller wants to
    /// build a bounded list and the cancellation hook is checked
    /// inside the loop.
    @discardableResult
    public static func walk(root: URL,
                            options: Options = Options(),
                            isCancelled: @Sendable () -> Bool = { false },
                            visit: (Entry) -> Bool) -> Int
    {
        let rootStd = root.standardizedFileURL
        let rootPath = rootStd.path
        var enumOpts: FileManager.DirectoryEnumerationOptions = []
        if options.skipHidden { enumOpts.insert(.skipsHiddenFiles) }
        if options.skipPackageDescendants { enumOpts.insert(.skipsPackageDescendants) }

        let resourceKeys: Set<URLResourceKey> = [
            .isDirectoryKey,
            .isSymbolicLinkKey,
            .isRegularFileKey,
        ]
        guard let walker = FileManager.default.enumerator(
            at: rootStd,
            includingPropertiesForKeys: Array(resourceKeys),
            options: enumOpts
        ) else { return 0 }

        // Pre-canonicalise the trust boundary once so the per-entry
        // hot path stays cheap. Includes `root` itself: a symlink whose
        // target is under the walk's own root is always safe.
        var trustedPaths: [String] = [rootPath]
        for url in options.trustedRoots {
            let p = (url.standardizedFileURL.path as NSString).resolvingSymlinksInPath
            if !trustedPaths.contains(p) { trustedPaths.append(p) }
        }

        var visited = Set<String>()
        var count = 0
        while let next = walker.nextObject() as? URL {
            if isCancelled() { break }
            let resolved = next.standardizedFileURL
            let values = (try? resolved.resourceValues(forKeys: resourceKeys))
            let isDir = values?.isDirectory ?? false
            let isLink = values?.isSymbolicLink ?? false

            // Symlink policy. A link passes when `followSymlinks` is
            // explicitly on OR when its resolved target lands inside a
            // trusted root. We compute the real path once and reuse it
            // below for cycle detection.
            var realPath: String? = nil
            if isLink {
                let resolvedTarget = (resolved.path as NSString)
                    .resolvingSymlinksInPath
                realPath = resolvedTarget
                let withinTrust = trustedPaths.contains { trustedRoot in
                    pathIsWithinPrefix(resolvedTarget, root: trustedRoot)
                }
                if !options.followSymlinks && !withinTrust {
                    if isDir { walker.skipDescendants() }
                    continue
                }
            }

            // Cycle detection. Any followed symlink (either by explicit
            // `followSymlinks` or via the trust boundary) gets canonical
            // path checked; already-visited targets are skipped and the
            // enumerator is told not to descend.
            if isLink {
                let key = realPath ?? (resolved.path as NSString).resolvingSymlinksInPath
                if visited.contains(key) {
                    if isDir { walker.skipDescendants() }
                    continue
                }
                visited.insert(key)
            }

            let depth = (walker.level - 1)
            if let maxDepth = options.maxDepth, depth > maxDepth {
                walker.skipDescendants()
                continue
            }

            // Compute relative path. `standardizedFileURL.path` strips
            // the trailing slash, so adding 1 covers the separator after
            // the root path.
            let rel: String
            if resolved.path == rootPath {
                rel = ""
            } else if resolved.path.hasPrefix(rootPath + "/") {
                rel = String(resolved.path.dropFirst(rootPath.count + 1))
            } else {
                rel = resolved.lastPathComponent
            }

            let entry = Entry(url: resolved,
                              relativePath: rel,
                              isDirectory: isDir,
                              isSymlink: isLink,
                              depth: depth)
            count += 1
            if !visit(entry) { break }
        }
        return count
    }

    /// True when `path` is `root` itself or sits under it. Guards
    /// against the classic prefix-match bug where `/foo/bar` would
    /// erroneously qualify as a child of `/foo/ba`. Both inputs must
    /// already be canonical (post `resolvingSymlinksInPath`).
    private static func pathIsWithinPrefix(_ path: String, root: String) -> Bool {
        if path == root { return true }
        let rootWithSep = root.hasSuffix("/") ? root : root + "/"
        return path.hasPrefix(rootWithSep)
    }
}
