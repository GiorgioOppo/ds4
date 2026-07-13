import Foundation

extension ProjectCache {
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
