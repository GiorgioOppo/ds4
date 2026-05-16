import Foundation

/// Coarse agent operating mode. Mirrors opencode's build/plan split.
///
/// `.build` lets every registered tool run after going through the
/// permission policy. `.plan` is the read-only stance: tools tagged
/// as `.mutating` or `.dangerous` are denied at registry-resolution
/// time, and tools tagged `.shell` always require an explicit
/// per-call confirmation regardless of any saved decision. The mode
/// is a property of the active agent (`AgentConfig.mode`) and is
/// honoured by both the local backend and the OpenRouter dispatch.
public enum AgentMode: String, Codable, Sendable, CaseIterable {
    case build
    case plan

    public var displayName: String {
        switch self {
        case .build: return "Build"
        case .plan:  return "Plan"
        }
    }

    public var summary: String {
        switch self {
        case .build:
            return "Full tool access. Mutations and shell use the permission policy."
        case .plan:
            return "Read-only exploration. Edits and patches are denied; shell prompts."
        }
    }
}
