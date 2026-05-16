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
public enum Sandbox {
    /// Path of the default profile that ships with the app, relative
    /// to the agent root.
    public static let defaultProfileRelative = "sandbox/default.sb"

    /// Write `defaultProfile` into `root/sandbox/default.sb` unless
    /// the file already exists.
    public static func ensureDefaultProfile(at root: URL) throws {
        let dir = root.appendingPathComponent("sandbox")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        let target = dir.appendingPathComponent("default.sb")
        guard !FileManager.default.fileExists(atPath: target.path) else { return }
        try Data(defaultProfile.utf8).write(to: target, options: .atomic)
    }

    /// A starter profile: read-only FS, no network, no spawning of
    /// signed system services. The shell will still be able to read
    /// files inside the agent root because the profile resolves
    /// relative paths through the `DEEPSEEK_ROOT` parameter that the
    /// host should pass when launching `sandbox-exec`.
    public static let defaultProfile: String = """
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
    (allow file-write* (subpath (param "DEEPSEEK_ROOT")))
    (deny network*)
    (allow mach-lookup)
    """
}
