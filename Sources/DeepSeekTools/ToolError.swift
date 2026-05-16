import Foundation

/// Structured failure surface for tool invocations. Distinct from
/// `Error.localizedDescription` because the model sees the
/// `wireMessage` body verbatim — making the error machine-friendly
/// helps the model self-correct (e.g. `notFound` vs `permissionDenied`
/// hint different next actions).
public enum ToolError: Error, Sendable, Equatable {
    case invalidInput(String)
    case notFound(String)
    case permissionDenied(String)
    case denied(reason: String)
    case timeout(after: TimeInterval)
    case spawnFailed(String)
    case external(String)
    case notImplemented(String)

    /// What gets returned to the model as the tool's textual result
    /// when the call fails. Prefixed so the model can parse the
    /// category programmatically.
    public var wireMessage: String {
        switch self {
        case .invalidInput(let m):     return "error: invalid_input: \(m)"
        case .notFound(let m):         return "error: not_found: \(m)"
        case .permissionDenied(let m): return "error: permission_denied: \(m)"
        case .denied(let m):           return "error: denied: \(m)"
        case .timeout(let s):          return "error: timeout: after \(s) s"
        case .spawnFailed(let m):      return "error: spawn_failed: \(m)"
        case .external(let m):         return "error: external: \(m)"
        case .notImplemented(let m):   return "error: not_implemented: \(m)"
        }
    }
}
