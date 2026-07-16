import Foundation

extension ProjectCache {
    // MARK: - Write surface (the agentic "code mode" tools)

    /// Validate a relative path for writing: inside the root, no traversal,
    /// textual extension only (this is a code assistant, not a binary editor).
    private func writableURL(_ relPath: String) -> (URL, String)? {
        guard let url = confinedProjectURL(for: relPath) else { return nil }
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
            guard let (checkedURL, _) = writableURL(relPath) else {
                return "Invalid path or non-text extension: '\(relPath)'."
            }
            try content.write(to: checkedURL, atomically: true, encoding: .utf8)
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
        guard let (_, rel) = writableURL(relPath) else {
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
        guard let (checkedURL, _) = writableURL(relPath) else {
            return "Invalid path or non-text extension: '\(relPath)'."
        }
        do {
            try updated.write(to: checkedURL, atomically: true, encoding: .utf8)
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
    func searchContents(_ relPath: String) -> String? {
        guard confinedProjectURL(for: relPath) != nil else { return nil }
        lock.lock()
        if let c = contents[relPath] { lock.unlock(); return c }
        lock.unlock()
        return freshContents(relPath)
    }

    /// Like fileContents but ALWAYS reread from disk (index membership still
    /// required). For edit bases: see the comment in editTool.
    private func freshContents(_ relPath: String) -> String? {
        lock.lock()
        guard root != nil, files.contains(relPath) else { lock.unlock(); return nil }
        lock.unlock()
        guard let url = confinedProjectURL(for: relPath),
              let data = try? Data(contentsOf: url) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Insert/update a file in the index and content cache after a write.
    func upsertIndex(_ rel: String, content: String) {
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

}
