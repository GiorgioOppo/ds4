import Foundation

/// Per-call context handed to every `Tool.run`. Carries the working
/// directory the agent should treat as root, the active agent mode,
/// and the permission delegate that the tool calls into when it
/// needs explicit user consent. Kept as a struct of value types so
/// the same context can be safely captured across actors.
public struct ToolContext: Sendable {
    /// Filesystem root for resolving relative paths. Tools that
    /// accept paths should sandbox themselves to this root unless
    /// the user has explicitly enabled `allowEscapingRoot`.
    public let rootDirectory: URL

    /// If false, any path resolution that resolves above
    /// `rootDirectory` is rejected with `permissionDenied`. Default
    /// `false`. Surface to the user as a per-agent toggle if needed.
    public let allowEscapingRoot: Bool

    /// Additional absolute roots that are considered safe alongside
    /// `rootDirectory`. The path-resolution check accepts a
    /// resolved path that lives under EITHER `rootDirectory` OR any
    /// of these extra roots — without flipping the broader
    /// `allowEscapingRoot` escape hatch.
    ///
    /// Use case: a chat attached to a `Project` runs tools inside a
    /// symlink-farm at `~/Library/Application Support/.../projects/
    /// {id}/`, but the model may want to reference the user's real
    /// repo path (`/Users/me/code/foo/...`) directly because that's
    /// what `git`, `swift build`, or other tools surface in their
    /// output. The host populates `additionalReadRoots` with the
    /// project's `sourcePaths` so both addressing styles work.
    public let additionalReadRoots: [URL]

    /// The agent mode under which this call is executing. Tools can
    /// inspect this to e.g. dry-run instead of write.
    public let mode: AgentMode

    /// Permission gate. The registry calls
    /// `decide(category: ToolCategory, tool: String, summary: String)`
    /// before dispatch; the delegate returns `.allow` / `.deny` /
    /// `.allowOnce` / `.alwaysAllow`. The registry caches
    /// `.alwaysAllow` per session.
    public let permission: PermissionDelegate

    /// Caller-supplied environment for tools that spawn subprocesses.
    /// `nil` → inherit the host's environment.
    public let environment: [String: String]?

    /// Best-effort cancellation hook. Tools that perform long work
    /// (`shell`, `webfetch`) should check this periodically. When the
    /// chat stop button is pressed, the host marks this `true`.
    public let isCancelled: @Sendable () -> Bool

    /// Optional callback fired when a file-reading tool fails an
    /// `open()` with macOS sandbox EPERM on a path that resolves
    /// through a symlink. The argument is the **parent directory of
    /// the resolved target** — exactly the directory the host can
    /// pass to `NSOpenPanel` for a one-click grant. Wired by the GUI
    /// host to push the parent onto the active project's
    /// `pendingSymlinkRoots` so the user sees it in Settings → Project
    /// without having to trigger a fresh rebuild first.
    ///
    /// Nil for hosts that don't have a project context (CLI / tests).
    public let reportSymlinkTargetNeeded: (@Sendable (URL) -> Void)?

    public init(rootDirectory: URL,
                allowEscapingRoot: Bool = false,
                additionalReadRoots: [URL] = [],
                mode: AgentMode = .build,
                permission: PermissionDelegate,
                environment: [String: String]? = nil,
                isCancelled: @escaping @Sendable () -> Bool = { false },
                reportSymlinkTargetNeeded: (@Sendable (URL) -> Void)? = nil) {
        self.rootDirectory = rootDirectory
        self.allowEscapingRoot = allowEscapingRoot
        self.additionalReadRoots = additionalReadRoots
        self.mode = mode
        self.permission = permission
        self.environment = environment
        self.isCancelled = isCancelled
        self.reportSymlinkTargetNeeded = reportSymlinkTargetNeeded
    }
}

/// Resolution of a path argument against the context root. Rejects
/// any path that escapes the root unless explicitly allowed. Returns
/// the resolved absolute URL on success.
///
/// `additionalReadRoots` widens the permitted area: a resolved path
/// that sits under `rootDirectory` OR under any of the additional
/// roots is accepted. This is how project-bound chats can address
/// files either through the symlink farm (`relative` form) or via
/// the real absolute path on the user's disk.
///
/// `checkResolvedTarget` opts into a second pass that resolves any
/// symlinks in the result and verifies the **real** path still falls
/// inside `rootDirectory` ∪ `additionalReadRoots`. Use it from
/// mutating tools (write, edit, apply_patch) so a sneaky symlink
/// inside the agent root can't be used to mutate a file outside the
/// trust boundary. Read-only tools generally leave it off: the
/// symlink farm relies on followed reads landing on the user's real
/// source, which is exactly what the second pass would reject if the
/// source path isn't listed in `additionalReadRoots` — and we want
/// reads to still succeed in that edge case rather than fail hard.
public func resolveInsideRoot(_ relative: String,
                              context: ToolContext,
                              checkResolvedTarget: Bool = false) throws -> URL {
    let resolved: URL
    if relative.hasPrefix("/") {
        resolved = URL(fileURLWithPath: relative).standardizedFileURL
    } else {
        resolved = context.rootDirectory
            .appendingPathComponent(relative)
            .standardizedFileURL
    }
    if !context.allowEscapingRoot {
        let resolvedPath = resolved.path
        let rootPath = context.rootDirectory.standardizedFileURL.path
        var permitted = pathIsWithin(resolvedPath, root: rootPath)
        if !permitted {
            for extra in context.additionalReadRoots {
                let extraPath = extra.standardizedFileURL.path
                if pathIsWithin(resolvedPath, root: extraPath) {
                    permitted = true
                    break
                }
            }
        }
        if !permitted {
            throw ToolError.permissionDenied(
                "path '\(relative)' resolves outside the agent's root")
        }
        if checkResolvedTarget {
            // Anti-escape: if any path component is a symlink that
            // points outside both `rootDirectory` and
            // `additionalReadRoots`, refuse the call. We only check
            // the real path here — the prefix check above is what
            // gates the "user typed an outside path" case.
            let realPath = (resolvedPath as NSString).resolvingSymlinksInPath
            if realPath != resolvedPath {
                var realPermitted = pathIsWithin(
                    realPath,
                    root: (rootPath as NSString).resolvingSymlinksInPath)
                if !realPermitted {
                    for extra in context.additionalReadRoots {
                        let extraReal = (extra.standardizedFileURL.path
                                         as NSString).resolvingSymlinksInPath
                        if pathIsWithin(realPath, root: extraReal) {
                            realPermitted = true
                            break
                        }
                    }
                }
                if !realPermitted {
                    throw ToolError.permissionDenied(
                        "path '\(relative)' resolves through a symlink to "
                        + "'\(realPath)', which is outside the agent's "
                        + "trust boundary")
                }
            }
        }
    }
    return resolved
}

/// True when `path` is `root` itself or a descendant of it. The
/// boundary check guards against the classic prefix-match bug
/// where `/foo/bar` would erroneously qualify as a child of
/// `/foo/ba`. Both inputs must already be standardised.
private func pathIsWithin(_ path: String, root: String) -> Bool {
    if path == root { return true }
    let rootWithSep = root.hasSuffix("/") ? root : root + "/"
    return path.hasPrefix(rootWithSep)
}

/// Decode an `NSError` raised by `Data(contentsOf:)` / `FileHandle`
/// when the macOS sandbox refused an `open()` through a symlink. The
/// signal we care about is the `NSFileReadNoPermissionError` (code
/// 257) inside `NSCocoaErrorDomain` — that's the EPERM the seatbelt
/// surfaces when the resolved target falls outside every active
/// security-scoped bookmark.
///
/// Returns the **resolved target's parent directory** when the error
/// matches. That's the granularity the user grants from
/// `NSOpenPanel`: granting access to the parent unlocks every
/// sibling link landing there, instead of nagging the user once per
/// file.
///
/// Returns nil for any other error so callers can keep their
/// existing fall-through behaviour for "real" not-found / I/O
/// failures.
public func sandboxBlockedSymlinkTarget(
    from error: Error,
    accessedFrom url: URL
) -> URL? {
    let ns = error as NSError
    guard ns.domain == NSCocoaErrorDomain,
          ns.code == NSFileReadNoPermissionError
    else { return nil }
    let resolved = (url.path as NSString).resolvingSymlinksInPath
    // No symlink involved (path == resolved): the EPERM is on the
    // path the user directly addressed, not on a target reached
    // through a link. The grant flow doesn't help there — return
    // nil so the caller raises the plain permission error.
    guard resolved != url.path else { return nil }
    let parent = (resolved as NSString).deletingLastPathComponent
    guard !parent.isEmpty, parent != "/" else { return nil }
    return URL(fileURLWithPath: parent)
}

/// Build the "macOS sandbox blocked the read through a symlink"
/// message for the model. Carries enough information that the user
/// can act on it without round-tripping through the chat — the
/// `relative` arg keeps the model talking in the same path it
/// addressed, while `resolved` and `grantParent` tell the user
/// where to point `NSOpenPanel`.
public func symlinkPermissionDeniedMessage(
    relative: String,
    resolved: URL,
    grantParent: URL
) -> String {
    "macOS sandbox blocked the read through a symlink. "
    + "'\(relative)' resolves to '\(resolved.path)'. "
    + "Grant access to '\(grantParent.path)' from "
    + "Settings → Project to make it readable."
}
