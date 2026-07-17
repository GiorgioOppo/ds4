import Foundation

/// Serializes long-lived users of the single in-process inference engine.
///
/// DwarfStar intentionally owns one engine: loading or retaining a second copy
/// can exhaust unified memory on low-RAM Macs.  A lease is therefore held for
/// the whole lifetime of an auto-tune, benchmark, or local HTTP server run.
/// Tokens make release idempotent and prevent stale asynchronous cleanup from
/// releasing a newer owner's lease.
@MainActor
final class EngineActivityGate {
    static let shared = EngineActivityGate()

    enum Owner: String, Equatable, Sendable {
        case autoTune
        case benchmark
        case server
        case expertBundleBuild

        var displayName: String {
            switch self {
            case .autoTune:  return "machine auto-tune"
            case .benchmark: return "benchmark"
            case .server:    return "local HTTP server"
            case .expertBundleBuild: return "expert-bundle build"
            }
        }
    }

    struct Lease: Equatable, Sendable {
        fileprivate let id: UUID
        let owner: Owner
    }

    private var activeLease: Lease?

    private init() {}

    var activeOwner: Owner? { activeLease?.owner }

    /// Acquires exclusive use of the engine, or returns nil when another owner
    /// is still active.  Re-entry is deliberately not implicit: every operation
    /// must retain and later release its own unique token.
    func acquire(_ owner: Owner) -> Lease? {
        guard activeLease == nil else { return nil }
        let lease = Lease(id: UUID(), owner: owner)
        activeLease = lease
        return lease
    }

    /// Releases only the exact lease supplied by `acquire`.  A late completion
    /// from a cancelled task is harmless after a newer owner has taken over.
    @discardableResult
    func release(_ lease: Lease) -> Bool {
        guard activeLease?.id == lease.id else { return false }
        activeLease = nil
        return true
    }

    func owns(_ lease: Lease) -> Bool {
        activeLease?.id == lease.id
    }
}
