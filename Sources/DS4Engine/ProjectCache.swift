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
    static let readChunkLines = 120              // lines per project_read call
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
    ]

    // MARK: - Import / clear (GUI side)

    /// Walk `rootURL` and build the file index. Returns the project info.
    /// (The caller holds the security-scoped access for the session.)
    @discardableResult
    public func load(root rootURL: URL) -> Info {
        var found: [String] = []
        var total = 0
        let fm = FileManager.default
        if let en = fm.enumerator(at: rootURL, includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey],
                                  options: [.skipsHiddenFiles]) {
            for case let url as URL in en {
                let vals = try? url.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey])
                if vals?.isDirectory == true {
                    if Self.skipDirs.contains(url.lastPathComponent) { en.skipDescendants() }
                    continue
                }
                let size = vals?.fileSize ?? 0
                guard size > 0, size <= Self.maxFileBytes else { continue }
                guard Self.looksTextual(url) else { continue }
                let rel = url.path.replacingOccurrences(of: rootURL.path + "/", with: "")
                found.append(rel)
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

    public func clear() {
        lock.lock()
        root = nil; files = []; contents = [:]; cachedBytes = 0; infoValue = nil
        lock.unlock()
    }

    public func info() -> Info? { lock.lock(); defer { lock.unlock() }; return infoValue }

    /// First `n` indexed paths (GUI preview).
    public func sampleFiles(_ n: Int) -> [String] {
        lock.lock(); defer { lock.unlock() }
        return Array(files.prefix(n))
    }

    static func looksTextual(_ url: URL) -> Bool {
        if textExtensions.contains(url.pathExtension.lowercased()) { return true }
        guard let fh = try? FileHandle(forReadingFrom: url),
              let head = try? fh.read(upToCount: 2048) else { return false }
        defer { try? fh.close() }
        return !head.isEmpty && !head.contains(0) && String(data: head, encoding: .utf8) != nil
    }

    // MARK: - Tool surface (text in / text out, hard-capped)

    /// List the entries one level under `relPath` ("" = project root).
    public func listTool(path relPath: String) -> String {
        lock.lock(); defer { lock.unlock() }
        guard root != nil else { return "Nessun progetto importato." }
        let prefix = relPath.isEmpty ? "" : relPath.hasSuffix("/") ? relPath : relPath + "/"
        guard relPath.isEmpty || !relPath.contains("..") else { return "Percorso non valido." }
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
        if dirs.isEmpty && leaf.isEmpty { return "Nessun file sotto '\(relPath)'." }
        var out = (dirs.sorted() + leaf.sorted()).prefix(Self.maxListEntries).joined(separator: "\n")
        let n = dirs.count + leaf.count
        if n > Self.maxListEntries { out += "\n… (+\(n - Self.maxListEntries) altri)" }
        return out
    }

    /// Read `relPath` from `fromLine` (1-based) for up to readChunkLines lines,
    /// with line numbers so the model can paginate.
    public func readTool(path relPath: String, fromLine: Int) -> String {
        guard !relPath.contains("..") else { return "Percorso non valido." }
        guard let text = fileContents(relPath) else {
            return "File non trovato nell'indice: '\(relPath)'. Usa project_list per esplorare."
        }
        let lines = text.components(separatedBy: "\n")
        let start = max(1, fromLine)
        guard start <= lines.count else { return "'\(relPath)' ha solo \(lines.count) righe." }
        let end = min(lines.count, start + Self.readChunkLines - 1)
        var out = "\(relPath) — righe \(start)-\(end) di \(lines.count):\n"
        for i in (start - 1)..<end {
            out += "\(i + 1)\t\(lines[i])\n"
        }
        if end < lines.count {
            out += "… (continua: richiama project_read con from_line=\(end + 1))"
        }
        return out
    }

    /// Case-insensitive substring search across the indexed files.
    public func searchTool(query: String) -> String {
        let q = query.lowercased()
        guard q.count >= 2 else { return "Query troppo corta." }
        lock.lock(); let snapshot = files; lock.unlock()
        var hits: [String] = []
        for f in snapshot {
            guard let text = fileContents(f) else { continue }
            for (i, line) in text.components(separatedBy: "\n").enumerated()
            where line.lowercased().contains(q) {
                hits.append("\(f):\(i + 1): \(String(line.trimmingCharacters(in: .whitespaces).prefix(160)))")
                if hits.count >= Self.maxSearchHits { break }
            }
            if hits.count >= Self.maxSearchHits { break }
        }
        if hits.isEmpty { return "Nessun risultato per '\(query)'." }
        var out = hits.joined(separator: "\n")
        if hits.count >= Self.maxSearchHits { out += "\n… (limite di \(Self.maxSearchHits) risultati raggiunto)" }
        return out
    }

    /// Load (and cache) a file's contents; nil if not in the index.
    private func fileContents(_ relPath: String) -> String? {
        lock.lock()
        if let c = contents[relPath] { lock.unlock(); return c }
        guard let root, files.contains(relPath) else { lock.unlock(); return nil }
        let url = root.appendingPathComponent(relPath)
        lock.unlock()
        guard let data = try? Data(contentsOf: url),
              let text = String(data: data, encoding: .utf8) else { return nil }
        lock.lock()
        if cachedBytes + data.count > Self.maxCacheBytes { contents = [:]; cachedBytes = 0 }
        contents[relPath] = text
        cachedBytes += data.count
        lock.unlock()
        return text
    }
}
