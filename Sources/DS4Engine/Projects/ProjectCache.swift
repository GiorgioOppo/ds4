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
    let lock = NSLock()

    var root: URL?
    var files: [String] = []              // sorted relative paths
    var contents: [String: String] = [:]  // lazy per-file cache
    var cachedBytes = 0

    public struct Info: Sendable {
        public let name: String
        public let fileCount: Int
        public let totalBytes: Int
    }
    var infoValue: Info?

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

    /// Resolve a tool-supplied relative path below the active project without
    /// ever accepting a symbolic link below the imported root.  Checking only
    /// `resolvingSymlinksInPath()` on the final URL is insufficient when the
    /// leaf does not exist: Foundation then leaves an earlier directory link
    /// unresolved (for example `link/new-file.txt`).  Walking every existing
    /// component catches both file and directory links while still allowing a
    /// write to create genuinely missing nested directories.
    ///
    /// The imported root itself is the trusted boundary and may use an OS-level
    /// alias such as `/tmp` -> `/private/tmp`; only components *below* it are
    /// rejected.  Errors other than "not found" fail closed.
    func confinedProjectURL(for relativePath: String) -> URL? {
        guard let root = rootURL(),
              !relativePath.isEmpty,
              !relativePath.hasPrefix("/") else { return nil }

        let rawComponents = relativePath.split(separator: "/", omittingEmptySubsequences: false)
        guard !rawComponents.isEmpty else { return nil }
        let components = rawComponents.map(String.init)
        guard components.allSatisfy({ component in
            !component.isEmpty
                && component != "."
                && component != ".."
                && !component.utf8.contains(0)
        }) else { return nil }

        var candidate = root.standardizedFileURL
        var missingAncestor = false
        for (index, component) in components.enumerated() {
            candidate.appendPathComponent(component)
            guard !missingAncestor else { continue }

            do {
                let values = try candidate.resourceValues(
                    forKeys: [.isSymbolicLinkKey, .isDirectoryKey]
                )
                guard values.isSymbolicLink != true else { return nil }
                if index < components.count - 1, values.isDirectory != true {
                    return nil
                }
            } catch {
                let nsError = error as NSError
                guard nsError.domain == NSCocoaErrorDomain,
                      nsError.code == CocoaError.Code.fileReadNoSuchFile.rawValue else {
                    return nil
                }
                missingAncestor = true
            }
        }
        return candidate
    }

}
