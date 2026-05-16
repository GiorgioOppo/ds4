import Foundation

/// Central pool of `Tool` instances. Owns:
///   1. Registration / lookup by name (tools must have unique names).
///   2. Plan-mode filtering — in `.plan`, only `.readOnly` and
///      `.planning` tools survive `availableSchemas(mode:)`.
///   3. Session-scoped cache of `alwaysAllow` decisions, keyed by
///      `(tool, category)`. The cache is per-registry-instance, so
///      hosts can scope it however they want (per-chat, per-session).
///
/// Concurrency: `actor` because the cache is mutable and dispatch can
/// happen from any caller. Adding tools must happen before the first
/// `run` of a session (registration mid-stream is allowed but the
/// model won't see the new tool until the next system block rebuild).
public actor ToolRegistry {
    private var tools: [String: Tool] = [:]
    private var sessionAllowCache: Set<String> = []

    public init() {}

    public func register(_ tool: Tool) {
        tools[tool.schema.name] = tool
    }

    public func registerAll(_ list: [Tool]) {
        for t in list { tools[t.schema.name] = t }
    }

    public func clear() {
        tools.removeAll()
        sessionAllowCache.removeAll()
    }

    public func names() -> [String] {
        Array(tools.keys).sorted()
    }

    public func schema(named name: String) -> ToolSchema? {
        tools[name]?.schema
    }

    /// Schemas eligible under the given mode. Use this to build the
    /// system block / `tools` array that the model sees. The order
    /// is sorted alphabetically by name for stable prompts.
    public func availableSchemas(mode: AgentMode) -> [ToolSchema] {
        tools.values
            .map(\.schema)
            .filter { isAllowed(category: $0.category, mode: mode) }
            .sorted { $0.name < $1.name }
    }

    /// Reset the per-session `alwaysAllow` cache. Hosts should call
    /// this on chat clear, agent switch, or "forget permissions".
    public func resetSessionCache() {
        sessionAllowCache.removeAll()
    }

    /// Look up `name`, validate against the active mode, ask the
    /// permission delegate if needed, then dispatch. Returns the
    /// tool's output, or a structured failure wrapped in
    /// `ToolOutput.error` so the model sees a usable message.
    public func dispatch(name: String,
                         input: [String: Any],
                         context: ToolContext) async -> ToolOutput {
        guard let tool = tools[name] else {
            return ToolOutput.error(.notFound("tool '\(name)' is not registered"))
        }
        if !isAllowed(category: tool.schema.category, mode: context.mode) {
            return ToolOutput.error(
                .denied(reason: "'\(name)' is not available in \(context.mode.displayName) mode")
            )
        }

        if categoryNeedsConsent(tool.schema.category) {
            let cacheKey = "\(name):\(tool.schema.category.rawValue)"
            if !sessionAllowCache.contains(cacheKey) {
                let request = PermissionRequest(
                    tool: name,
                    category: tool.schema.category,
                    summary: tool.permissionSummary(input: input),
                    detail: nil,
                    mode: context.mode
                )
                let decision = await context.permission.decide(request: request)
                switch decision {
                case .deny:
                    return ToolOutput.error(
                        .denied(reason: "user declined '\(name)'"))
                case .allowOnce:
                    break
                case .alwaysAllow:
                    sessionAllowCache.insert(cacheKey)
                }
            }
        }

        do {
            return try await tool.run(input: input, context: context)
        } catch let err as ToolError {
            return ToolOutput.error(err)
        } catch {
            return ToolOutput.error(.external(error.localizedDescription))
        }
    }

    /// Whether plan mode lets a category through. Kept here (not in
    /// `AgentMode`) so it lives next to the dispatcher that consults
    /// it.
    private func isAllowed(category: ToolCategory, mode: AgentMode) -> Bool {
        switch (mode, category) {
        case (.plan, .mutating), (.plan, .dangerous):
            return false
        default:
            return true
        }
    }

    private func categoryNeedsConsent(_ category: ToolCategory) -> Bool {
        switch category {
        case .readOnly, .planning: return false
        case .mutating, .dangerous, .network: return true
        }
    }
}
