import Foundation

/// A memory container for an imported project, SEPARATE from the chat memory:
/// importing a folder builds an index (and a lazy content cache) that does NOT
/// touch the conversation KV. The model explores it through the project_* tools,
/// so only the parts it actually reads enter the chat (as tool results).
///
/// Thread-safe singleton (tools run from the inference actor, the GUI from the
/// main actor). All tool outputs are hard-capped: on a slow local machine every
/// token of tool result is prefill cost.
public final class ProjectCache: @unchecked Sendable {
    public static let shared = ProjectCache()
    private let lock = NSLock()

    private var root: URL?
    private var files: [String] = []              // sorted relative paths
    private var contents: [String: String] = [:]  // lazy per-file cache
    private var cachedBytes = 0

    public struct Info: Sendable {
        public let name: String
        public let fileCount: Int
        public let totalBytes: Int
    }
    private var infoValue: Info?

    // Limits: keep the index and any single tool answer small.
    static let maxFiles = 3000
    static let maxFileBytes = 1 << 20            // index files up to 1 MB
    static let maxCacheBytes = 24 << 20          // content cache budget
    static let maxListEntries = 200
    static let readChunkLines = 120              // default lines per project_read call
    static let maxReadLines = 400                // hard cap when the model asks for more
    static let maxSearchHits = 30

    static let skipDirs: Set<String> = [".git", ".build", ".swiftpm", "node_modules",
                                        "DerivedData", "Pods", ".venv", "__pycache__",
                                        "dist", "build", ".idea", ".vscode"]
    static let textExtensions: Set<String> = [
        "swift", "m", "mm", "h", "c", "cc", "cpp", "hpp", "metal",
        "py", "js", "ts", "tsx", "jsx", "json", "md", "txt", "rst",
        "yml", "yaml", "toml", "ini", "cfg", "sh", "bash", "zsh",
        "html", "css", "xml", "plist", "entitlements", "modulemap",
        "java", "kt", "rs", "go", "rb", "sql", "gradle", "cmake", "make",
        "tex", "bib", "cls", "sty", "latex",
    ]

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

    // MARK: - Tool surface (text in / text out, hard-capped)

    /// List the entries one level under `relPath` ("" / "." = project root).
    public func listTool(path relPath: String) -> String {
        lock.lock(); defer { lock.unlock() }
        guard root != nil else { return "No project imported." }
        // Normalize root sentinels: "" / "." / "./" all mean the project root, and a
        // leading "./" is stripped (indexed paths have no "./" prefix).
        var p = relPath
        if p == "." || p == "./" { p = "" }
        else if p.hasPrefix("./") { p = String(p.dropFirst(2)) }
        guard p.isEmpty || !p.contains("..") else { return "Invalid path." }
        let prefix = p.isEmpty ? "" : (p.hasSuffix("/") ? p : p + "/")
        var dirs = Set<String>()
        var leaf: [String] = []
        for f in files where f.hasPrefix(prefix) {
            let rest = String(f.dropFirst(prefix.count))
            if let slash = rest.firstIndex(of: "/") {
                dirs.insert(String(rest[..<slash]) + "/")
            } else {
                leaf.append(rest)
            }
        }
        if dirs.isEmpty && leaf.isEmpty { return "No files under '\(relPath)'." }
        var out = (dirs.sorted() + leaf.sorted()).prefix(Self.maxListEntries).joined(separator: "\n")
        let n = dirs.count + leaf.count
        if n > Self.maxListEntries { out += "\n... (+\(n - Self.maxListEntries) more)" }
        return out
    }

    /// Read `relPath` from `fromLine` (1-based) for up to `maxLines` lines
    /// (default readChunkLines, hard-capped at maxReadLines), with line numbers
    /// so the model can paginate. Long files should be read in FEW large chunks:
    /// every round-trip costs a full prefill+decode on a local model.
    public func readTool(path relPath: String, fromLine: Int, maxLines: Int? = nil) -> String {
        guard !relPath.contains("..") else { return "Invalid path." }
        guard let text = fileContents(relPath) else {
            return info() == nil
                ? "No project imported."
                : "File not found in the index: '\(relPath)'. Use project_list to explore."
        }
        let lines = text.components(separatedBy: "\n")
        let start = max(1, fromLine)
        guard start <= lines.count else { return "'\(relPath)' has only \(lines.count) lines." }
        let chunk = min(max(1, maxLines ?? Self.readChunkLines), Self.maxReadLines)
        let end = min(lines.count, start + chunk - 1)
        var out = "\(relPath) - lines \(start)-\(end) of \(lines.count):\n"
        for i in (start - 1)..<end {
            out += "\(i + 1)\t\(lines[i])\n"
        }
        if end < lines.count {
            out += "... (continues: call project_read with from_line=\(end + 1))"
        }
        return out
    }

    /// Compact overview of the whole project in ONE call: the directories up to
    /// `maxDepth` levels (each with the number of indexed files beneath it) plus
    /// the root files. Complements project_list, which shows one level per call.
    public func treeTool(maxDepth: Int = 3) -> String {
        lock.lock(); let snapshot = files; let inf = infoValue; lock.unlock()
        guard let inf else { return "No project imported." }
        let depth = max(1, min(maxDepth, 6))
        var counts: [String: Int] = [:]          // dir path (≤ depth components) → files beneath
        var rootFiles: [String] = []
        for f in snapshot {
            let comps = f.split(separator: "/").map(String.init)
            guard comps.count > 1 else { rootFiles.append(f); continue }
            for d in 1...min(comps.count - 1, depth) {
                counts[comps.prefix(d).joined(separator: "/"), default: 0] += 1
            }
        }
        var lines: [String] = []
        for dir in counts.keys.sorted() {
            let comps = dir.components(separatedBy: "/")
            lines.append(String(repeating: "  ", count: comps.count - 1)
                         + (comps.last ?? dir) + "/ (\(counts[dir] ?? 0))")
        }
        lines.append(contentsOf: rootFiles)
        var out = "Project \"\(inf.name)\": \(inf.fileCount) files. Directories to depth \(depth) (file count in parentheses), then root files:"
        out += "\n" + lines.prefix(Self.maxListEntries).joined(separator: "\n")
        if lines.count > Self.maxListEntries {
            out += "\n... (+\(lines.count - Self.maxListEntries) more entries; use project_list on a subfolder)"
        }
        return out
    }

    /// Find indexed files whose PATH matches `pattern`: case-insensitive
    /// substring, with '*' as a wildcard for any run of characters.
    /// Complements searchTool, which searches file CONTENTS.
    public func findTool(pattern: String) -> String {
        lock.lock(); let snapshot = files; let hasRoot = root != nil; lock.unlock()
        guard hasRoot else { return "No project imported." }
        let p = pattern.trimmingCharacters(in: .whitespaces)
        guard p.count >= 2 else { return "Pattern too short (min 2 characters)." }
        let hits: [String]
        if p.contains("*") {
            let escaped = NSRegularExpression.escapedPattern(for: p)
                .replacingOccurrences(of: "\\*", with: ".*")
            guard let re = try? NSRegularExpression(pattern: escaped, options: [.caseInsensitive]) else {
                return "Invalid pattern."
            }
            hits = snapshot.filter {
                re.firstMatch(in: $0, range: NSRange($0.startIndex..., in: $0)) != nil
            }
        } else {
            let q = p.lowercased()
            hits = snapshot.filter { $0.lowercased().contains(q) }
        }
        guard !hits.isEmpty else {
            return "No files match '\(pattern)'. Use project_search to search file contents instead."
        }
        var out = hits.prefix(Self.maxSearchHits).joined(separator: "\n")
        if hits.count > Self.maxSearchHits { out += "\n... (limit of \(Self.maxSearchHits) results reached)" }
        return out
    }

    /// Case-insensitive substring search across the indexed files. An optional
    /// `pathPrefix` restricts the search to one subfolder (or a single file) —
    /// fewer hits burned on the wrong part of the tree, faster on big projects.
    public func searchTool(query: String, pathPrefix: String? = nil) -> String {
        let q = query.lowercased()
        guard q.count >= 2 else { return "Query too short." }
        lock.lock(); var snapshot = files; lock.unlock()
        if var p = pathPrefix?.trimmingCharacters(in: .whitespaces), !p.isEmpty, p != "." {
            guard !p.contains("..") else { return "Invalid path." }
            if p.hasPrefix("./") { p = String(p.dropFirst(2)) }
            let dirPrefix = p.hasSuffix("/") ? p : p + "/"
            snapshot = snapshot.filter { $0 == p || $0.hasPrefix(dirPrefix) }
            if snapshot.isEmpty { return "No indexed files under '\(p)'." }
        }
        var hits: [String] = []
        for f in snapshot {
            guard let text = searchContents(f) else { continue }
            for (i, line) in text.components(separatedBy: "\n").enumerated()
            where line.lowercased().contains(q) {
                hits.append("\(f):\(i + 1): \(String(line.trimmingCharacters(in: .whitespaces).prefix(160)))")
                if hits.count >= Self.maxSearchHits { break }
            }
            if hits.count >= Self.maxSearchHits { break }
        }
        if hits.isEmpty { return "No results for '\(query)'." }
        var out = hits.joined(separator: "\n")
        if hits.count >= Self.maxSearchHits { out += "\n... (limit of \(Self.maxSearchHits) results reached)" }
        return out
    }

    /// True when `url` (which may not exist yet) stays inside `root` AFTER
    /// resolving symlinks. The lexical prefix checks catch "..", but a symlink
    /// INSIDE the project (e.g. carried by a cloned repo) can point anywhere;
    /// resolving only the existing components keeps not-yet-created write
    /// targets valid, and resolving both sides also normalizes macOS
    /// /var → /private/var.
    private static func withinRoot(_ url: URL, root: URL) -> Bool {
        let resolvedRoot = root.resolvingSymlinksInPath().standardizedFileURL.path
        let prefix = resolvedRoot.hasSuffix("/") ? resolvedRoot : resolvedRoot + "/"
        return url.resolvingSymlinksInPath().standardizedFileURL.path.hasPrefix(prefix)
    }

    /// Load (and cache) a file's contents; nil if not in the index.
    private func fileContents(_ relPath: String) -> String? {
        lock.lock()
        if let c = contents[relPath] { lock.unlock(); return c }
        guard let root, files.contains(relPath) else { lock.unlock(); return nil }
        let url = root.appendingPathComponent(relPath)
        lock.unlock()
        guard Self.withinRoot(url, root: root),
              let data = try? Data(contentsOf: url),
              let text = String(data: data, encoding: .utf8) else { return nil }
        lock.lock()
        if cachedBytes + data.count > Self.maxCacheBytes { contents = [:]; cachedBytes = 0 }
        contents[relPath] = text
        cachedBytes += data.count
        lock.unlock()
        return text
    }

    // MARK: - Write surface (the agentic "code mode" tools)

    /// Validate a relative path for writing: inside the root, no traversal,
    /// textual extension only (this is a code assistant, not a binary editor).
    private func writableURL(_ relPath: String) -> (URL, String)? {
        guard let root else { return nil }
        guard !relPath.isEmpty, !relPath.hasPrefix("/"), !relPath.contains("..") else { return nil }
        let url = root.appendingPathComponent(relPath).standardizedFileURL
        guard url.path.hasPrefix(root.standardizedFileURL.path + "/") else { return nil }
        guard Self.withinRoot(url, root: root) else { return nil }
        guard Self.textExtensions.contains(url.pathExtension.lowercased()) else { return nil }
        return (url, relPath)
    }

    /// Create or overwrite `relPath` with `content` (creates intermediate dirs).
    /// The index and cache are updated so the next read/search sees the change.
    public func writeTool(path relPath: String, content: String) -> String {
        guard info() != nil else { return "No project imported." }
        guard let (url, rel) = writableURL(relPath) else {
            return "Invalid path or non-text extension: '\(relPath)'."
        }
        guard content.utf8.count <= Self.maxFileBytes else {
            return "Content too large (max \(Self.maxFileBytes / 1024) KB)."
        }
        let existed = FileManager.default.fileExists(atPath: url.path)
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try content.write(to: url, atomically: true, encoding: .utf8)
        } catch {
            return "Write failed: \(error.localizedDescription)"
        }
        upsertIndex(rel, content: content)
        let lines = content.components(separatedBy: "\n").count
        return "\(existed ? "Overwrote" : "Created") '\(rel)' (\(lines) lines)."
    }

    /// Replace ONE exact occurrence of `find` with `replace` in `relPath`.
    /// Refuses 0 matches (wrong/old text) and >1 matches (ambiguous — the model
    /// must include more surrounding context), like an agentic editor should.
    public func editTool(path relPath: String, find: String, replace: String) -> String {
        guard info() != nil else { return "No project imported." }
        guard let (url, rel) = writableURL(relPath) else {
            return "Invalid path or non-text extension: '\(relPath)'."
        }
        guard !find.isEmpty else { return "'find' is empty." }
        // The edit base is reread from DISK, not from the content cache: the
        // file may have changed since it was cached (the git tool, the user's
        // editor) and applying the replacement to a stale base would silently
        // revert those changes on write. upsertIndex refreshes the cache below.
        guard let text = freshContents(rel) else {
            return "File not found in the index: '\(rel)'. Use project_list / project_read before editing."
        }
        let occurrences = text.components(separatedBy: find).count - 1
        guard occurrences != 0 else {
            return "Text to replace was not found in '\(rel)'. Reread the file (project_read) and use the exact text, including indentation."
        }
        guard occurrences == 1 else {
            return "Ambiguous text: \(occurrences) occurrences in '\(rel)'. Include more context (neighboring lines) to make it unique."
        }
        guard let range = text.range(of: find) else { return "Text not found." }
        let updated = text.replacingCharacters(in: range, with: replace)
        do {
            try updated.write(to: url, atomically: true, encoding: .utf8)
        } catch {
            return "Write failed: \(error.localizedDescription)"
        }
        upsertIndex(rel, content: updated)
        let line = text[text.startIndex..<range.lowerBound].components(separatedBy: "\n").count
        return "Edited '\(rel)' around line \(line) (1 replacement)."
    }

    /// Contents for SEARCH: the cache when hot, otherwise straight from disk
    /// WITHOUT inserting. Searching a project larger than the cache budget
    /// used to flush the whole cache repeatedly (thrash), evicting exactly the
    /// files the model is actively reading; a search visit is not a signal of
    /// future reads, so it must not cost cache residency.
    private func searchContents(_ relPath: String) -> String? {
        lock.lock()
        if let c = contents[relPath] { lock.unlock(); return c }
        lock.unlock()
        return freshContents(relPath)
    }

    /// Like fileContents but ALWAYS reread from disk (index membership still
    /// required). For edit bases: see the comment in editTool.
    private func freshContents(_ relPath: String) -> String? {
        lock.lock()
        guard let r = root, files.contains(relPath) else { lock.unlock(); return nil }
        lock.unlock()
        let url = r.appendingPathComponent(relPath)
        guard Self.withinRoot(url, root: r),
              let data = try? Data(contentsOf: url) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Insert/update a file in the index and content cache after a write.
    private func upsertIndex(_ rel: String, content: String) {
        lock.lock()
        if let i = files.firstIndex(where: { $0 >= rel }) {
            if files[i] != rel { files.insert(rel, at: i) }
        } else {
            files.append(rel)
        }
        // Replace, don't double-count: subtract the previous cached size first
        // (repeated edits of one file used to inflate cachedBytes until the
        // whole cache got flushed for nothing), then apply the same budget
        // flush as fileContents.
        if let old = contents[rel] { cachedBytes = max(0, cachedBytes - old.utf8.count) }
        if cachedBytes + content.utf8.count > Self.maxCacheBytes { contents = [:]; cachedBytes = 0 }
        contents[rel] = content
        cachedBytes += content.utf8.count
        if let inf = infoValue {
            infoValue = Info(name: inf.name, fileCount: files.count, totalBytes: inf.totalBytes)
        }
        lock.unlock()
    }

    // MARK: - Raw file read/write (any file inside the project root)

    /// Validate a relative path against the project root: inside it, no traversal.
    /// Unlike `writableURL`, no text-extension restriction (raw files allowed).
    private func rootRelativeURL(_ relPath: String) -> URL? {
        guard let root = rootURL() else { return nil }
        guard !relPath.isEmpty, !relPath.hasPrefix("/"), !relPath.contains("..") else { return nil }
        let url = root.appendingPathComponent(relPath).standardizedFileURL
        guard url.path.hasPrefix(root.standardizedFileURL.path + "/") else { return nil }
        guard Self.withinRoot(url, root: root) else { return nil }
        return url
    }

    /// Read ANY file inside the project root (not just the index), decoded as text.
    /// Without a range: the whole file (capped). With `fromLine`/`toLine` (1-based,
    /// inclusive): only those lines, numbered. Errors on binary/unreadable content.
    public func readFileTool(path relPath: String, fromLine: Int? = nil, toLine: Int? = nil) -> String {
        guard let url = rootRelativeURL(relPath) else {
            return "No project imported or invalid path: '\(relPath)'."
        }
        guard let data = try? Data(contentsOf: url) else {
            return "File not found or unreadable: '\(relPath)'."
        }
        // Line range: return only [from, to] with line numbers (like project_read).
        if fromLine != nil || toLine != nil {
            guard let text = String(data: data, encoding: .utf8) else {
                return "Non-text file ('\(relPath)', \(data.count) bytes): cannot read as text."
            }
            let lines = text.components(separatedBy: "\n")
            let n = lines.count
            let from = max(1, fromLine ?? 1)
            guard from <= n else { return "'\(relPath)' has only \(n) lines." }
            let to = min(n, toLine ?? n)
            guard to >= from else { return "Invalid range: to_line (\(to)) < from_line (\(from))." }
            var out = "\(relPath) - lines \(from)-\(to) of \(n):\n"
            for i in (from - 1)..<to { out += "\(i + 1)\t\(lines[i])\n" }
            return out
        }
        // Whole file, capped. 24 KB ≈ 6k tokens: on the default low-RAM
        // context (4096) even that is a big bite, so anything larger must be
        // read by range — the old 96 KB cap could swallow several contexts.
        let cap = 24 * 1024
        var end = min(cap, data.count)
        var text = String(data: data.prefix(end), encoding: .utf8)
        // A cap landing mid-character makes a valid text file undecodable:
        // back off up to 3 bytes (the longest UTF-8 continuation run).
        while text == nil, data.count > cap, end > cap - 3 {
            end -= 1
            text = String(data: data.prefix(end), encoding: .utf8)
        }
        guard let text else {
            return "Non-text file ('\(relPath)', \(data.count) bytes): cannot read as text."
        }
        var out = "\(relPath) (\(data.count) bytes):\n\(text)"
        if data.count > end {
            out += "\n... (truncated to \(cap / 1024) KB of \(data.count / 1024) KB; use from_line/to_line to read the rest in ranges)"
        }
        return out
    }

    /// Count the lines (and bytes) of a file in the project root. Uses the SAME
    /// split as file_read/file_modify/file_add so the line numbers are consistent.
    public func lineCountTool(path relPath: String) -> String {
        guard let url = rootRelativeURL(relPath) else {
            return "No project imported or invalid path: '\(relPath)'."
        }
        guard let data = try? Data(contentsOf: url) else {
            return "File not found or unreadable: '\(relPath)'."
        }
        guard let text = String(data: data, encoding: .utf8) else {
            return "Non-text file ('\(relPath)', \(data.count) bytes)."
        }
        let lines = text.components(separatedBy: "\n").count
        return "\(relPath): \(lines) lines, \(data.count) bytes."
    }

    /// Create or overwrite the WHOLE file inside the project root (any extension;
    /// creates intermediate dirs). For line-level changes use addLinesTool /
    /// modifyLinesTool. Updates the index when the file is textual.
    public func writeFileTool(path relPath: String, content: String) -> String {
        guard let url = rootRelativeURL(relPath) else {
            return "No project imported or invalid path: '\(relPath)'."
        }
        guard content.utf8.count <= Self.maxFileBytes else {
            return "Content too large (max \(Self.maxFileBytes / 1024) KB)."
        }
        let existed = FileManager.default.fileExists(atPath: url.path)
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try content.write(to: url, atomically: true, encoding: .utf8)
        } catch {
            return "Write failed: \(error.localizedDescription)"
        }
        if Self.textExtensions.contains(url.pathExtension.lowercased()) { upsertIndex(relPath, content: content) }
        let lines = content.components(separatedBy: "\n").count
        return "\(existed ? "Overwrote" : "Created") '\(relPath)' (\(lines) lines)."
    }

    /// ADD lines WITHOUT overwriting: insert `content` before line `atLine` (1-based);
    /// `atLine` omitted or beyond the end appends. Creates the file if absent.
    /// Updates the index when the file is textual.
    public func addLinesTool(path relPath: String, content: String, atLine: Int? = nil) -> String {
        guard let url = rootRelativeURL(relPath) else {
            return "No project imported or invalid path: '\(relPath)'."
        }
        guard !content.isEmpty else { return "Nothing to add ('content' is empty)." }
        guard content.utf8.count <= Self.maxFileBytes else {
            return "Content too large (max \(Self.maxFileBytes / 1024) KB)."
        }
        // New file: just create it.
        guard FileManager.default.fileExists(atPath: url.path) else {
            return writeFileTool(path: relPath, content: content)
        }
        guard let data = try? Data(contentsOf: url), let existing = String(data: data, encoding: .utf8) else {
            return "Non-text file: '\(relPath)'."
        }
        var lines = existing.components(separatedBy: "\n")
        let n = lines.count
        let newLines = content.components(separatedBy: "\n")
        let requested = min(max(1, atLine ?? (n + 1)), n + 1)  // insert BEFORE this line; n+1 = append
        // Appending to a newline-terminated file: the trailing "" element is
        // the phantom line AFTER the final newline — insert before it, or the
        // join manufactures a blank line between old content and new (and the
        // file keeps its trailing newline this way).
        let at = (requested == n + 1 && lines.last == "") ? n : requested
        lines.insert(contentsOf: newLines, at: at - 1)
        let updated = lines.joined(separator: "\n")
        guard updated.utf8.count <= Self.maxFileBytes else {
            return "Resulting file too large (max \(Self.maxFileBytes / 1024) KB)."
        }
        do { try updated.write(to: url, atomically: true, encoding: .utf8) }
        catch { return "Write failed: \(error.localizedDescription)" }
        if Self.textExtensions.contains(url.pathExtension.lowercased()) { upsertIndex(relPath, content: updated) }
        return "Added \(newLines.count) lines to '\(relPath)' \(requested > n ? "at the end" : "before line \(at)")."
    }

    /// MODIFY by REPLACING lines [fromLine, toLine] (1-based, inclusive) of an
    /// existing text file with `content` (`toLine` omitted = a single line; empty
    /// `content` = delete those lines). Updates the index when the file is textual.
    public func modifyLinesTool(path relPath: String, content: String, fromLine: Int, toLine: Int? = nil) -> String {
        guard let url = rootRelativeURL(relPath) else {
            return "No project imported or invalid path: '\(relPath)'."
        }
        guard content.utf8.count <= Self.maxFileBytes else {
            return "Content too large (max \(Self.maxFileBytes / 1024) KB)."
        }
        guard let data = try? Data(contentsOf: url), let existing = String(data: data, encoding: .utf8) else {
            return "File not found or non-text: '\(relPath)'."
        }
        var lines = existing.components(separatedBy: "\n")
        let n = lines.count
        guard fromLine >= 1, fromLine <= n else {
            return "from_line \(fromLine) is out of range (1...\(n)). Use file_add to add lines."
        }
        let from = fromLine
        let to = min(max(from, toLine ?? from), n)
        let newLines = content.isEmpty ? [] : content.components(separatedBy: "\n")
        let removed = to - from + 1
        lines.replaceSubrange((from - 1)..<to, with: newLines)
        let updated = lines.joined(separator: "\n")
        guard updated.utf8.count <= Self.maxFileBytes else {
            return "Resulting file too large (max \(Self.maxFileBytes / 1024) KB)."
        }
        do { try updated.write(to: url, atomically: true, encoding: .utf8) }
        catch { return "Write failed: \(error.localizedDescription)" }
        if Self.textExtensions.contains(url.pathExtension.lowercased()) { upsertIndex(relPath, content: updated) }
        return "Modified lines \(from)-\(to) of '\(relPath)': \(removed) removed -> \(newLines.count) inserted."
    }

    /// DELETE one file inside the project root (never directories). Destructive
    /// but confined to the imported project; with a git repo the file is
    /// recoverable. The index and content cache are updated.
    public func deleteFileTool(path relPath: String) -> String {
        guard let url = rootRelativeURL(relPath) else {
            return "No project imported or invalid path: '\(relPath)'."
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else {
            return "File not found: '\(relPath)'."
        }
        guard !isDir.boolValue else {
            return "'\(relPath)' is a directory: only single files can be deleted."
        }
        do { try FileManager.default.removeItem(at: url) }
        catch { return "Delete failed: \(error.localizedDescription)" }
        lock.lock()
        files.removeAll { $0 == relPath }
        if let old = contents.removeValue(forKey: relPath) {
            cachedBytes = max(0, cachedBytes - old.utf8.count)
        }
        if let inf = infoValue {
            infoValue = Info(name: inf.name, fileCount: files.count, totalBytes: inf.totalBytes)
        }
        lock.unlock()
        return "Deleted '\(relPath)'."
    }
}
