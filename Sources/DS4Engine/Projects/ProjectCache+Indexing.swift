import Foundation

extension ProjectCache {
    // MARK: - Import / clear (GUI side)

    /// Walk `rootURL` and build the file index. Returns the project info.
    /// (The caller holds the security-scoped access for the session.)
    @discardableResult
    public func load(root rootURL: URL) -> Info {
        var found: [String] = []
        var total = 0
        let fm = FileManager.default
        // Strip the root prefix robustly: the enumerator can return URLs whose
        // prefix is the symlink-RESOLVED root (e.g. macOS /var → /private/var)
        // while `rootURL.path` is the unresolved form — a plain string replace
        // then leaves the "relative" path absolute and every project_* lookup
        // misses. Match the plain prefix first (the common, no-symlink case: no
        // extra syscall); fall back to the resolved prefix. The stored root stays
        // `rootURL`, so reads go through the original (security-scoped) path.
        let plainPrefix = rootURL.path.hasSuffix("/") ? rootURL.path : rootURL.path + "/"
        let resolvedRoot = rootURL.resolvingSymlinksInPath().path
        let resolvedPrefix = resolvedRoot.hasSuffix("/") ? resolvedRoot : resolvedRoot + "/"
        func relativize(_ url: URL) -> String {
            if url.path.hasPrefix(plainPrefix) { return String(url.path.dropFirst(plainPrefix.count)) }
            let resolved = url.resolvingSymlinksInPath().path
            if resolved.hasPrefix(resolvedPrefix) { return String(resolved.dropFirst(resolvedPrefix.count)) }
            if resolved.hasPrefix(plainPrefix) { return String(resolved.dropFirst(plainPrefix.count)) }
            return url.lastPathComponent
        }
        if let en = fm.enumerator(at: rootURL,
                                  includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey, .isSymbolicLinkKey],
                                  options: [.skipsHiddenFiles]) {
            for case let url as URL in en {
                let vals = try? url.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey, .isSymbolicLinkKey])
                // Never index a symlink: reads follow it, and a link carried by
                // an imported/cloned repo may point anywhere outside the root.
                if vals?.isSymbolicLink == true { continue }
                if vals?.isDirectory == true {
                    if Self.skipDirs.contains(url.lastPathComponent) { en.skipDescendants() }
                    continue
                }
                let size = vals?.fileSize ?? 0
                guard size > 0, size <= Self.maxFileBytes else { continue }
                guard Self.looksTextual(url) else { continue }
                found.append(relativize(url))
                total += size
                if found.count >= Self.maxFiles { break }
            }
        }
        found.sort()
        let info = Info(name: rootURL.lastPathComponent, fileCount: found.count, totalBytes: total)
        lock.lock()
        root = rootURL
        files = found
        contents = [:]
        cachedBytes = 0
        infoValue = info
        lock.unlock()
        return info
    }

    /// Re-walk the current root and rebuild the index (content cache cleared):
    /// for changes made OUTSIDE the tools — `git stash` via the git tool,
    /// scripts, the user's editor. Nil when no project is active.
    @discardableResult
    public func reload() -> Info? {
        guard let r = rootURL() else { return nil }
        return load(root: r)
    }

    public func clear() {
        lock.lock()
        root = nil; files = []; contents = [:]; cachedBytes = 0; infoValue = nil
        lock.unlock()
    }

    public func info() -> Info? { lock.lock(); defer { lock.unlock() }; return infoValue }

    /// The active project root (the git tool runs there). nil = no project.
    public func rootURL() -> URL? { lock.lock(); defer { lock.unlock() }; return root }

    /// First `n` indexed paths (GUI preview).
    public func sampleFiles(_ n: Int) -> [String] {
        lock.lock(); defer { lock.unlock() }
        return Array(files.prefix(n))
    }

    /// All indexed relative paths (sub-agent target search / project map).
    public func fileList() -> [String] { lock.lock(); defer { lock.unlock() }; return files }

    /// The entire indexed text of `relPath` (to seed a sub-agent's KV context);
    /// nil if not in the index. Capped by the same per-file limit as the index.
    public func fullText(of relPath: String) -> String? {
        guard !relPath.contains("..") else { return nil }
        return fileContents(relPath)
    }

    static func looksTextual(_ url: URL) -> Bool {
        if textExtensions.contains(url.pathExtension.lowercased()) { return true }
        guard let fh = try? FileHandle(forReadingFrom: url),
              let head = try? fh.read(upToCount: 2048) else { return false }
        defer { try? fh.close() }
        return !head.isEmpty && !head.contains(0) && String(data: head, encoding: .utf8) != nil
    }

}
