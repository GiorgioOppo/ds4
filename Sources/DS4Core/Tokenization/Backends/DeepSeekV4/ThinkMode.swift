import Foundation

public enum ThinkMode: Sendable {
    case none, high, max
    public var enabled: Bool { self != .none }
}

/// Reasoning-effort prefix injected for Think Max (verbatim from ds4.c).
public let DS4ReasoningEffortMaxPrefix =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n" +
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n" +
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
