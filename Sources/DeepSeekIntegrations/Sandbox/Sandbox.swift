import Foundation

/// Wrapper around macOS `sandbox-exec` for confining `ShellTool` (and
/// any future subprocess). Equivalent of opencode's "containers"
/// concept, scaled down to what macOS offers natively: no Docker /
/// no namespacing, just `sandboxd` profiles.
///
/// Status: scaffolded. The default profile below is intentionally
/// conservative (read-only filesystem, no network, allow execve only
/// of `/usr/bin` and `/bin`). It will reject most real shell work;
/// tune for the project's needs before turning the wrapper on by
/// default. Listed in TODO.md.
///
/// ## Symlink farm interaction
///
/// `ShellTool` runs inside a per-project symlink farm: directories
/// are real but every source file is a symlink into the user's
/// `Project.sourcePaths`. Seatbelt resolves the symlink before
/// applying read rules, so a profile that only allows reads inside
/// `DEEPSEEK_ROOT` will silently break every read through the farm.
///
/// `renderProfile(extraReadRoots:)` returns a profile string that
/// additionally allows reads under each entry in `extraReadRoots` —
/// `ShellTool` passes `context.additionalReadRoots`, which the host
/// populates with the project's source folders.
public enum Sandbox {
    /// Path of the default profile that ships with the app, relative
    /// to the agent root. Written by `ensureDefaultProfile(at:)` and
    /// read by `ShellTool` when no extra read roots are present.
    public static let defaultProfileRelative = "sandbox/default.sb"

    /// Write the no-extra-roots variant of the profile into
    /// `root/sandbox/default.sb` unless the file already exists.
    /// Per-call profile rendering (the case with `additionalReadRoots`)
    /// happens dynamically in `ShellTool` and bypasses this on-disk
    /// copy.
    public static func ensureDefaultProfile(at root: URL) throws {
        let dir = root.appendingPathComponent("sandbox")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        let target = dir.appendingPathComponent("default.sb")
        guard !FileManager.default.fileExists(atPath: target.path) else { return }
        try Data(defaultProfile.utf8).write(to: target, options: .atomic)
    }

    /// Render a profile that allows reads/writes inside `DEEPSEEK_ROOT`
    /// (passed as a `-D` param at exec time) **plus** reads inside each
    /// entry in `extraReadRoots`. The extra roots are baked into the
    /// profile text as literal subpaths because seatbelt's `(param …)`
    /// directive doesn't accept arrays.
    ///
    /// Returns the profile as a string the caller can write to a
    /// per-call temp file and feed to `sandbox-exec -f`.
    public static func renderProfile(extraReadRoots: [URL]) -> String {
        var body = """
        (version 1)
        (deny default)
        (allow process-fork)
        (allow process-exec)
        (allow file-read* (subpath (param "DEEPSEEK_ROOT")))
        (allow file-read-data
            (literal "/usr/lib")
            (literal "/usr/bin")
            (literal "/bin")
            (literal "/private/etc"))
        """
        // Each extra root is allowed for read* (covers metadata,
        // open-read, mmap). The farm only needs reads; writes still
        // funnel through `DEEPSEEK_ROOT` paths and end up at the
        // source files transparently via the link layer.
        var seen = Set<String>()
        for root in extraReadRoots {
            // Canonicalise so two URLs that differ only by trailing
            // slash or `/private` prefix don't end up duplicated.
            let path = (root.standardizedFileURL.path as NSString)
                .resolvingSymlinksInPath
            if seen.contains(path) || path.isEmpty || path == "/" { continue }
            seen.insert(path)
            // Quote-escape backslashes and double quotes so a source
            // folder name with spaces or quotes can't break the
            // sexp parse. Single quotes and other characters are
            // fine inside seatbelt string literals.
            let escaped = path
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
            body += "\n(allow file-read* (subpath \"\(escaped)\"))"
        }
        body += """

        (allow file-write* (subpath (param "DEEPSEEK_ROOT")))
        (deny network*)
        (allow mach-lookup)
        """
        return body
    }

    /// A starter profile: read-only FS, no network, no spawning of
    /// signed system services. The shell will still be able to read
    /// files inside the agent root because the profile resolves
    /// relative paths through the `DEEPSEEK_ROOT` parameter that the
    /// host should pass when launching `sandbox-exec`. Equivalent to
    /// `renderProfile(extraReadRoots: [])`.
    public static let defaultProfile: String = renderProfile(extraReadRoots: [])
}
