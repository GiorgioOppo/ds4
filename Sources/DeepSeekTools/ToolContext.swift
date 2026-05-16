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

    public init(rootDirectory: URL,
                allowEscapingRoot: Bool = false,
                mode: AgentMode = .build,
                permission: PermissionDelegate,
                environment: [String: String]? = nil,
                isCancelled: @escaping @Sendable () -> Bool = { false }) {
        self.rootDirectory = rootDirectory
        self.allowEscapingRoot = allowEscapingRoot
        self.mode = mode
        self.permission = permission
        self.environment = environment
        self.isCancelled = isCancelled
    }
}

/// Resolution of a path argument against the context root. Rejects
/// any path that escapes the root unless explicitly allowed. Returns
/// the resolved absolute URL on success.
public func resolveInsideRoot(_ relative: String,
                              context: ToolContext) throws -> URL {
    let resolved: URL
    if relative.hasPrefix("/") {
        resolved = URL(fileURLWithPath: relative).standardizedFileURL
    } else {
        resolved = context.rootDirectory
            .appendingPathComponent(relative)
            .standardizedFileURL
    }
    if !context.allowEscapingRoot {
        let rootPath = context.rootDirectory.standardizedFileURL.path
        if !resolved.path.hasPrefix(rootPath) {
            throw ToolError.permissionDenied(
                "path '\(relative)' resolves outside the agent's root")
        }
    }
    return resolved
}
