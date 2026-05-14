import Foundation

/// Walks a Project's `sourcePaths`, filters to text-like files, and
/// produces `IndexCandidate` entries that the caller turns into
/// `VectorizedDocument`s via the tokenizer. The walker itself is
/// pure (no model dependency); the actual tokenize + persist loop
/// lives in the import flow so it can stream progress to the UI.
enum ProjectIndexer {
    /// Per-file cap. Files larger than this are skipped — pulling a
    /// multi-megabyte log into the index would tokenize to tens of
    /// thousands of ids per file and saturate any usable context.
    static let maxFileBytes = 1 * 1024 * 1024

    /// Whitelist of extensions we consider text-like and worth
    /// tokenising. Keeps `.png`, `.zip`, `.so`, weight blobs, etc.
    /// out of the index even when they sit next to source files.
    static let textExtensions: Set<String> = [
        "swift", "py", "js", "jsx", "ts", "tsx", "html", "htm",
        "css", "scss", "less",
        "json", "yaml", "yml", "toml", "xml", "csv", "tsv",
        "md", "markdown", "txt", "rst", "log",
        "c", "cc", "cpp", "cxx", "h", "hpp", "hxx", "m", "mm",
        "metal", "glsl", "wgsl",
        "rs", "go", "java", "kt", "kts", "scala",
        "rb", "php", "lua", "pl", "sh", "bash", "zsh", "fish",
        "sql", "proto", "thrift", "graphql",
        "ini", "cfg", "env", "conf",
        "dockerfile", "makefile", "gitignore", "gitattributes",
        "lock"
    ]

    /// Directory names that get skipped during recursion. Excluding
    /// them at directory level (not just per-file) avoids walking
    /// huge node_modules / .git trees.
    static let excludedDirectories: Set<String> = [
        ".git", ".svn", ".hg",
        "node_modules", "vendor", "Pods", "Carthage",
        ".build", "build", "DerivedData", "dist", "out",
        "__pycache__", ".venv", "venv", ".tox", ".pytest_cache",
        "target",                    // Rust / Java
        ".next", ".nuxt", ".turbo",  // JS bundlers
        ".DS_Store"
    ]

    struct IndexCandidate {
        let url: URL              // absolute path on disk
        let displayPath: String   // path relative to the project root that contained it
        let byteCount: Int        // size at scan time
    }

    /// Scan every `sourcePath` of `project` and return the candidate
    /// files in the order they were discovered. Symlinks are not
    /// followed (avoids cycles); unreadable directories are skipped
    /// silently.
    static func scan(_ project: Project) -> [IndexCandidate] {
        var out: [IndexCandidate] = []
        let fm = FileManager.default
        for path in project.sourcePaths {
            let root = URL(fileURLWithPath: path)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: root.path, isDirectory: &isDir) else { continue }
            let rootName = root.lastPathComponent
            if isDir.boolValue {
                walk(root: root, rootDisplay: rootName, into: &out)
            } else if let cand = candidate(for: root, displayPath: rootName) {
                out.append(cand)
            }
        }
        return out
    }

    // MARK: - internals

    private static func walk(root: URL, rootDisplay: String,
                              into out: inout [IndexCandidate]) {
        let fm = FileManager.default
        let keys: [URLResourceKey] = [.isDirectoryKey, .isSymbolicLinkKey, .fileSizeKey]
        guard let it = fm.enumerator(
            at: root,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles, .skipsSymbolicLinks, .skipsPackageDescendants])
        else { return }

        while let item = it.nextObject() as? URL {
            let comp = item.lastPathComponent
            // Directory-level skips: prune subtree.
            if let vals = try? item.resourceValues(forKeys: Set(keys)),
               vals.isDirectory == true {
                if excludedDirectories.contains(comp) {
                    it.skipDescendants()
                }
                continue
            }
            guard let cand = candidate(for: item,
                                        displayPath: relativePath(of: item,
                                                                    underRoot: root,
                                                                    rootDisplay: rootDisplay))
            else { continue }
            out.append(cand)
        }
    }

    private static func candidate(for url: URL,
                                   displayPath: String) -> IndexCandidate? {
        // Extension/name whitelist. The lowercased basename catches
        // Makefile / Dockerfile-style filenames without extension.
        let ext = url.pathExtension.lowercased()
        let basename = url.lastPathComponent.lowercased()
        let isTextExt = textExtensions.contains(ext)
            || textExtensions.contains(basename)
        guard isTextExt else { return nil }

        // Size + binary checks.
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? Int else {
            return nil
        }
        guard size > 0, size <= maxFileBytes else { return nil }
        if looksBinary(at: url) { return nil }

        return IndexCandidate(url: url,
                               displayPath: displayPath,
                               byteCount: size)
    }

    private static func looksBinary(at url: URL) -> Bool {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return true }
        defer { try? handle.close() }
        let head = (try? handle.read(upToCount: 4096)) ?? Data()
        return head.contains(0)
    }

    private static func relativePath(of url: URL,
                                      underRoot root: URL,
                                      rootDisplay: String) -> String {
        let rootComps = root.standardizedFileURL.pathComponents
        let urlComps  = url.standardizedFileURL.pathComponents
        guard urlComps.count > rootComps.count,
              Array(urlComps.prefix(rootComps.count)) == rootComps else {
            return url.lastPathComponent
        }
        let tail = urlComps.dropFirst(rootComps.count).joined(separator: "/")
        return "\(rootDisplay)/\(tail)"
    }
}
