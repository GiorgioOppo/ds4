import Foundation

/// The `github_clone` tool: downloads a PUBLIC GitHub repository as an HTTPS
/// tarball (codeload.github.com — no git binary, no credentials), extracts it
/// under Application Support, and imports it into ProjectCache as the ACTIVE
/// project so the project_* tools can explore it.
///
/// The point is CONTEXT FRUGALITY on a local model: the clone costs the chat
/// only the returned orientation summary (tree + documentation files), and the
/// model then explores with project_find / project_search and reads only the
/// relevant ranges with project_read — instead of paying prefill for the whole
/// repository.
///
/// Safety: this tool runs on model-emitted arguments, so the request is pinned
/// to ONE fixed host and owner/name/ref are validated against strict ASCII
/// character sets — the model cannot steer the fetch anywhere else. The archive
/// is size-capped and extracted into a private per-repo directory (never inside
/// an existing project).
public enum GitHubTool {

    struct Repo: Equatable {
        let owner: String
        let name: String
        let ref: String?          // branch, tag, or commit; nil = default branch
    }

    static let maxArchiveBytes = 128 << 20            // compressed tarball cap
    static let downloadTimeout: TimeInterval = 120
    static let extractTimeout: TimeInterval = 60
    static let maxDocFiles = 15

    // MARK: - Argument parsing (pure; unit-tested)

    /// Accepts "owner/name", an http(s)://[www.]github.com/owner/name URL
    /// (optionally with ".git", a trailing slash, or "/tree/<ref>"), or the SSH
    /// form "git@github.com:owner/name.git". `explicitRef` (the tool's 'ref'
    /// argument) wins over a ref embedded in the URL. Returns nil for any other
    /// host or any component outside the allowed character sets.
    static func parseRepo(_ raw: String, ref explicitRef: String? = nil) -> Repo? {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !s.isEmpty else { return nil }
        if let cut = s.firstIndex(where: { $0 == "?" || $0 == "#" }) { s = String(s[..<cut]) }
        if s.hasPrefix("git@github.com:") {
            s = String(s.dropFirst("git@github.com:".count))
        } else if let r = s.range(of: "github.com/") {
            // Only a real github.com URL may precede the match — reject
            // "evil.com/github.com/…" and any other embedding.
            let head = String(s[..<r.lowerBound]).lowercased()
            guard head.isEmpty || head == "www." || head.hasSuffix("://") || head.hasSuffix("://www.") else {
                return nil
            }
            s = String(s[r.upperBound...])
        } else if s.contains("://") || s.contains("@") {
            return nil                                 // a URL, but not github.com
        }
        while s.hasSuffix("/") { s.removeLast() }
        let comps = s.split(separator: "/").map(String.init)
        guard comps.count >= 2 else { return nil }
        let owner = comps[0]
        var name = comps[1]
        if name.lowercased().hasSuffix(".git") { name = String(name.dropLast(4)) }
        guard validOwner(owner), validName(name) else { return nil }
        var ref = explicitRef
        if ref == nil, comps.count >= 4, comps[2] == "tree" || comps[2] == "commit" {
            // "/tree/feature/x" is the branch "feature/x". (A /tree/<branch>/<subdir>
            // URL is indistinguishable from a slashed branch name; when a plain
            // branch download 404s, the model should pass 'ref' explicitly.)
            ref = comps[3...].joined(separator: "/")
        }
        if let r = ref, !validRef(r) { return nil }
        return Repo(owner: owner, name: name, ref: ref)
    }

    private static func ascii(_ s: String, plus extra: Set<Character>) -> Bool {
        s.allSatisfy { ($0.isASCII && ($0.isLetter || $0.isNumber)) || extra.contains($0) }
    }

    static func validOwner(_ s: String) -> Bool {
        !s.isEmpty && s.count <= 39 && !s.hasPrefix("-") && ascii(s, plus: ["-"])
    }

    static func validName(_ s: String) -> Bool {
        !s.isEmpty && s.count <= 100 && s != "." && s != ".." && ascii(s, plus: ["-", "_", "."])
    }

    static func validRef(_ s: String) -> Bool {
        !s.isEmpty && s.count <= 200 && !s.contains("..")
            && !s.hasPrefix("-") && !s.hasPrefix("/") && !s.hasSuffix("/")
            && ascii(s, plus: ["-", "_", ".", "/"])
    }

    /// The pinned download endpoint. The strict charsets above guarantee the
    /// string is a valid URL, so the force-unwrap cannot fire.
    static func archiveURL(for repo: Repo) -> URL {
        URL(string: "https://codeload.github.com/\(repo.owner)/\(repo.name)/tar.gz/\(repo.ref ?? "HEAD")")!
    }

    // MARK: - Orientation summary (pure; unit-tested)

    /// The documentation shortlist for the clone summary, in reading order:
    /// root README first, other root-level .md files, then docs/-style folders,
    /// then nested READMEs. Deduplicated, uncapped (the caller caps for display).
    static func docCandidates(in files: [String]) -> [String] {
        var picked: [String] = []
        var seen = Set<String>()
        func add(_ f: String) { if seen.insert(f).inserted { picked.append(f) } }
        let lower = files.map { $0.lowercased() }
        for (i, f) in files.enumerated() where !f.contains("/") && lower[i].hasPrefix("readme") { add(f) }
        for (i, f) in files.enumerated() where !f.contains("/") && lower[i].hasSuffix(".md") { add(f) }
        for (i, f) in files.enumerated() {
            let l = lower[i]
            guard l.hasPrefix("docs/") || l.hasPrefix("doc/") || l.contains("/docs/") else { continue }
            if l.hasSuffix(".md") || l.hasSuffix(".rst") || l.hasSuffix(".txt") { add(f) }
        }
        for (i, f) in files.enumerated() where lower[i].hasSuffix("/readme.md") { add(f) }
        return picked
    }

    // MARK: - Tool entry point

    static func run(repoArg: String, ref refArg: String?) -> String {
        let trimmedRef = refArg?.trimmingCharacters(in: .whitespacesAndNewlines)
        let explicitRef = (trimmedRef?.isEmpty == false) ? trimmedRef : nil
        if let r = explicitRef, !validRef(r) {
            return "Invalid 'ref': use a branch, tag, or commit SHA (letters, digits, . _ - /)."
        }
        guard let repo = parseRepo(repoArg, ref: explicitRef) else {
            return "Invalid 'repo': pass 'owner/name' (e.g. \"apple/swift\") or a github.com URL."
        }
        let (file, dlError) = downloadArchive(archiveURL(for: repo))
        guard let tarFile = file else {
            return "Download failed for github.com/\(repo.owner)/\(repo.name): \(dlError ?? "unknown error")"
        }
        defer { try? FileManager.default.removeItem(at: tarFile) }
        let dest = projectsDirectory().appendingPathComponent("\(repo.owner)-\(repo.name)", isDirectory: true)
        if let err = extractArchive(tarFile, into: dest) {
            return "Extraction failed: \(err)"
        }
        let info = ProjectCache.shared.load(root: dest)
        guard info.fileCount > 0 else {
            return "Repository downloaded to \(dest.path), but no indexable text files were found (empty or binary-only repo?)."
        }
        let at = repo.ref.map { "@\($0)" } ?? ""
        var out = "Imported github.com/\(repo.owner)/\(repo.name)\(at) as the ACTIVE project"
        out += " (\(info.fileCount) text files, \(info.totalBytes / 1024) KB indexed; any previous project was replaced."
        out += " The user can switch projects anytime from the app's Project tab, where this clone is now listed).\n\n"
        out += ProjectCache.shared.treeTool(maxDepth: 2)
        let docs = docCandidates(in: ProjectCache.shared.fileList())
        if !docs.isEmpty {
            var list = docs.prefix(maxDocFiles).joined(separator: "\n")
            if docs.count > maxDocFiles { list += "\n... (+\(docs.count - maxDocFiles) more documentation files)" }
            out += "\n\nDocumentation to orient from (project_read, first chunk first):\n" + list
        }
        out += "\n\nNext: locate code with project_find (file names) and project_search (contents, scoped with 'path'); read only the relevant line ranges with project_read. Do not read the repository wholesale — every token of tool output is prefill cost."
        return out
    }

    /// Where imported repositories live: one folder per owner-repo, replaced on
    /// re-clone. Inside the app sandbox this is always writable. Public so the
    /// GUI's ProjectLibrary can list cloned repos as selectable projects.
    public static func projectsDirectory() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DwarfStar/github-projects", isDirectory: true)
    }

    // MARK: - Download / extract (side effects)

    /// Synchronous size-capped download to a private staging file. Returns the
    /// staged file or an error message. Synchronous by design: tools run off
    /// the main thread inside the InferenceService actor, and BuiltinTool.run
    /// is a sync closure (same pattern as WebClient).
    private static func downloadArchive(_ url: URL) -> (file: URL?, error: String?) {
        var req = URLRequest(url: url, timeoutInterval: downloadTimeout)
        req.setValue("DwarfStar/1.0 (+local DeepSeek-V4 agent)", forHTTPHeaderField: "User-Agent")
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForResource = downloadTimeout
        let session = URLSession(configuration: config)
        defer { session.finishTasksAndInvalidate() }
        let sem = DispatchSemaphore(value: 0)
        // nonisolated(unsafe): written ONLY by the completion handler, read by
        // the caller AFTER sem.wait() — the semaphore orders the two accesses.
        nonisolated(unsafe) var outcome: (URL?, String?) = (nil, "no response")
        let task = session.downloadTask(with: req) { tmp, resp, err in
            defer { sem.signal() }
            if let err {
                outcome = (nil, "network error: \(err.localizedDescription)"); return
            }
            guard let http = resp as? HTTPURLResponse, let tmp else {
                outcome = (nil, "empty response"); return
            }
            guard http.statusCode == 200 else {
                outcome = (nil, http.statusCode == 404
                    ? "repository or ref not found (HTTP 404). Only PUBLIC repositories are supported; check owner/name and 'ref'."
                    : "HTTP \(http.statusCode)")
                return
            }
            // The system deletes `tmp` when this handler returns: stage it first.
            let staging = FileManager.default.temporaryDirectory
                .appendingPathComponent("dwarfstar-ghclone-\(UUID().uuidString).tar.gz")
            do { try FileManager.default.moveItem(at: tmp, to: staging) } catch {
                outcome = (nil, "could not stage the download: \(error.localizedDescription)"); return
            }
            let attrs = try? FileManager.default.attributesOfItem(atPath: staging.path)
            let size = (attrs?[.size] as? NSNumber)?.intValue ?? 0
            guard size <= maxArchiveBytes else {
                try? FileManager.default.removeItem(at: staging)
                outcome = (nil, "archive too large (\(size >> 20) MB > \(maxArchiveBytes >> 20) MB cap)")
                return
            }
            outcome = (staging, nil)
        }
        task.resume()
        if sem.wait(timeout: .now() + downloadTimeout + 10) == .timedOut {
            task.cancel()
            return (nil, "timeout after \(Int(downloadTimeout))s")
        }
        return outcome
    }

    /// Extract the tarball into a FRESH `dir` with /usr/bin/tar (part of the
    /// macOS base system; the child inherits the app sandbox and only touches
    /// our own Application Support). Returns an error message or nil.
    private static func extractArchive(_ tar: URL, into dir: URL) -> String? {
        let fm = FileManager.default
        do {
            if fm.fileExists(atPath: dir.path) { try fm.removeItem(at: dir) }
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        } catch {
            return "could not prepare '\(dir.path)': \(error.localizedDescription)"
        }
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        // GitHub tarballs wrap everything in one "<repo>-<sha>/" directory;
        // --strip-components 1 unwraps it so `dir` IS the repo root.
        proc.arguments = ["-x", "-z", "-f", tar.path, "-C", dir.path, "--strip-components", "1"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        do { try proc.run() } catch {
            return "tar not runnable: \(error.localizedDescription)"
        }
        // Drain concurrently (a full pipe would deadlock the timeout loop).
        nonisolated(unsafe) var out = Data()
        let reader = Thread {
            while let chunk = try? pipe.fileHandleForReading.read(upToCount: 64 * 1024), !chunk.isEmpty {
                out.append(chunk)
            }
        }
        reader.start()
        let deadline = Date().addingTimeInterval(extractTimeout)
        while proc.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if proc.isRunning {
            proc.terminate()
            return "tar stopped: timeout after \(Int(extractTimeout))s"
        }
        Thread.sleep(forTimeInterval: 0.05)
        guard proc.terminationStatus == 0 else {
            let msg = String(data: out, encoding: .utf8) ?? ""
            return "tar failed (exit \(proc.terminationStatus)): \(String(msg.prefix(400)))"
        }
        // A hostile repo's tarball can carry symlinks pointing anywhere on
        // disk. The project tools refuse to follow them (ProjectCache resolves
        // and re-checks the root), but drop them at the source too — a
        // code-analysis import has no use for symlinks.
        if let en = fm.enumerator(at: dir, includingPropertiesForKeys: [.isSymbolicLinkKey]) {
            for case let item as URL in en
            where (try? item.resourceValues(forKeys: [.isSymbolicLinkKey]))?.isSymbolicLink == true {
                try? fm.removeItem(at: item)
            }
        }
        return nil
    }
}
