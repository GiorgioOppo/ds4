import Foundation

/// Filesystem walker for the Unix toolbox. Wraps `FileManager.enumerator`
/// with two policies that differ from the existing `GrepTool`/`GlobTool`:
///   1. **Symlinks are not followed by default.** The existing tools
///      rely implicitly on `.isRegularFileKey` to skip symlinks-to-
///      directories, but symlinks-to-regular-files DO get traversed —
///      which means a malicious link inside the agent root could
///      smuggle a read from outside it. Default `followSymlinks: false`
///      filters via `.isSymbolicLinkKey` regardless of the link target.
///   2. **Cycle detection.** `FileManager.enumerator` happily loops if
///      there's a symlink cycle (`a/b/c -> a`). When `followSymlinks`
///      is true we resolve each URL with `realpathSync` and skip ones
///      we've already visited.
///
/// Pure-Swift, no subprocess. Used by `ls`, `du`, `find`, `rm` (when
/// recursive), and any future Unix tool that needs a tree walk.
public enum UnixWalker {

    public struct Options: Sendable {
        public var followSymlinks: Bool
        public var skipHidden: Bool
        public var skipPackageDescendants: Bool
        public var maxDepth: Int?

        public init(followSymlinks: Bool = false,
                    skipHidden: Bool = true,
                    skipPackageDescendants: Bool = true,
                    maxDepth: Int? = nil) {
            self.followSymlinks = followSymlinks
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

        var visited = Set<String>()
        var count = 0
        while let next = walker.nextObject() as? URL {
            if isCancelled() { break }
            let resolved = next.standardizedFileURL
            let values = (try? resolved.resourceValues(forKeys: resourceKeys))
            let isDir = values?.isDirectory ?? false
            let isLink = values?.isSymbolicLink ?? false

            // Symlink policy. Skip the entry outright when not following;
            // the walker won't descend into linked directories either,
            // so this is the simplest correct behaviour.
            if isLink && !options.followSymlinks {
                continue
            }

            // Cycle detection when following symlinks. We canonicalize
            // by realpath; if already seen, skip and tell the enumerator
            // not to descend.
            if options.followSymlinks {
                let realPath = (resolved.path as NSString).resolvingSymlinksInPath
                if visited.contains(realPath) {
                    walker.skipDescendants()
                    continue
                }
                visited.insert(realPath)
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
}
