import Foundation

extension ProjectCache {
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

    /// Load (and cache) a file's contents; nil if not in the index.
    func fileContents(_ relPath: String) -> String? {
        guard let url = confinedProjectURL(for: relPath) else { return nil }
        lock.lock()
        if let c = contents[relPath] { lock.unlock(); return c }
        guard root != nil, files.contains(relPath) else { lock.unlock(); return nil }
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
