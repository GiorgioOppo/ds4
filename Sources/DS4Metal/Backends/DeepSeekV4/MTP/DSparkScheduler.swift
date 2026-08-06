import Foundation

/// Runtime proposal controls shared by the demo and GUI. Capping the emitted
/// prefix does not alter the official five-row support transformer; it only
/// bounds how many candidates reach the exact target verifier.
public struct DSparkProposalPolicy: Sendable, Equatable {
    public let confidenceThreshold: Float
    public let maxDraftTokens: Int

    public init(confidenceThreshold: Float = 0.7, maxDraftTokens: Int = 5) {
        self.confidenceThreshold = min(1, max(0, confidenceThreshold))
        self.maxDraftTokens = min(DSparkSupportModel.maxBlockSize,
                                  max(1, maxDraftTokens))
    }

    public static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Self {
        let confidence = environment["DS4_DSPARK_CONFIDENCE"]
            .flatMap(Float.init) ?? 0.7
        let drafts = environment["DS4_DSPARK_DRAFT_TOKENS"]
            .flatMap(Int.init) ?? 5
        return Self(confidenceThreshold: confidence, maxDraftTokens: drafts)
    }
}

/// Per-generation admission state matching upstream's speculative contract:
/// one authoritative target token is committed before the first support-model
/// proposal. The first scheduler decision still sees the original generation
/// budget (primer included), so a 10-token request is not incorrectly rejected
/// by a 10-token tail threshold after the primer has consumed one slot.
public struct DSparkGenerationGate: Sendable, Equatable {
    public private(set) var isTargetPrimed = false
    private var hasMadeSchedulingDecision = false

    public init() {}

    public mutating func noteTargetToken() {
        isTargetPrimed = true
    }

    /// Returns nil until the mandatory target primer has been committed.
    /// Thereafter the first call includes the consumed primer in the scheduler
    /// budget; later calls use the actual remaining-token count.
    public mutating func schedulingBudget(remainingTokens: Int) -> Int? {
        guard isTargetPrimed else { return nil }
        let remaining = max(0, remainingTokens)
        let primerCredit = hasMadeSchedulingDecision ? 0 : 1
        hasMadeSchedulingDecision = true
        return remaining > Int.max - primerCredit
            ? Int.max : remaining + primerCredit
    }
}

/// Deterministic, output-preserving admission policy for expensive DSpark
/// proposals. Skipped cycles still advance through the ordinary target
/// decoder, whose forward path maintains the private DSpark KV frontier.
public struct DSparkSchedulerPolicy: Sendable, Equatable {
    public var enabled: Bool
    public var window: Int
    public var skipCycles: Int
    public var slowSkipCycles: Int
    public var minimumAverageAccepted: Double
    public var noDraftSkipCycles: Int
    public var shortAcceptNoDraftSkipCycles: Int
    public var coldLowConfidenceSkipCycles: Int
    public var coldLowConfidenceThreshold: Float
    public var tailMinimumTokens: Int

    public init(
        enabled: Bool = true,
        window: Int = 4,
        skipCycles: Int = 2,
        slowSkipCycles: Int = 4,
        minimumAverageAccepted: Double = 1.5,
        noDraftSkipCycles: Int = 3,
        shortAcceptNoDraftSkipCycles: Int = 4,
        coldLowConfidenceSkipCycles: Int = 7,
        coldLowConfidenceThreshold: Float = 0.5,
        tailMinimumTokens: Int = 10
    ) {
        self.enabled = enabled
        self.window = max(1, window)
        self.skipCycles = max(0, skipCycles)
        self.slowSkipCycles = max(0, slowSkipCycles)
        self.minimumAverageAccepted = max(0, minimumAverageAccepted)
        self.noDraftSkipCycles = max(0, noDraftSkipCycles)
        self.shortAcceptNoDraftSkipCycles = max(0, shortAcceptNoDraftSkipCycles)
        self.coldLowConfidenceSkipCycles = max(0, coldLowConfidenceSkipCycles)
        self.coldLowConfidenceThreshold = min(1, max(0, coldLowConfidenceThreshold))
        self.tailMinimumTokens = max(0, tailMinimumTokens)
    }

    /// Defaults and environment names intentionally mirror upstream ds4.
    public static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Self {
        func integer(_ name: String, _ fallback: Int) -> Int {
            guard let value = environment[name], let parsed = Int(value), parsed >= 0
            else { return fallback }
            return parsed
        }
        func milli(_ name: String, _ fallback: Int) -> Double {
            Double(integer(name, fallback)) / 1_000
        }
        let enabled = environment["DS4_DSPARK_SCHEDULER"].map {
            $0.isEmpty || $0 != "0"
        } ?? true
        return Self(
            enabled: enabled,
            window: integer("DS4_DSPARK_SCHEDULER_WINDOW", 4),
            skipCycles: integer("DS4_DSPARK_SCHEDULER_SKIP", 2),
            slowSkipCycles: integer("DS4_DSPARK_SCHEDULER_SLOW_SKIP", 4),
            minimumAverageAccepted: milli(
                "DS4_DSPARK_SCHEDULER_MIN_AVG_MILLI", 1_500),
            noDraftSkipCycles: integer(
                "DS4_DSPARK_SCHEDULER_NO_DRAFT_SKIP", 3),
            shortAcceptNoDraftSkipCycles: integer(
                "DS4_DSPARK_SCHEDULER_SHORT_ACCEPT_NO_DRAFT_SKIP", 4),
            coldLowConfidenceSkipCycles: integer(
                "DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_SKIP", 7),
            coldLowConfidenceThreshold: Float(milli(
                "DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_MILLI", 500)),
            tailMinimumTokens: integer(
                "DS4_DSPARK_SCHEDULER_TAIL_MIN_TOKENS", 10))
    }
}

public enum DSparkScheduleDecision: Sendable, Equatable {
    case attempt
    case skipTail
    case skipBackoff(remaining: Int)

    public var shouldAttempt: Bool {
        if case .attempt = self { return true }
        return false
    }
}

public struct DSparkSchedulerSnapshot: Sendable, Equatable {
    public let attempts: Int
    public let proposals: Int
    public let accepted: Int
    public let emptyProposals: Int
    public let rejectedBlocks: Int
    public let tailSkips: Int
    public let backoffSkips: Int
    public let pendingBackoff: Int

    public var acceptanceRate: Double {
        proposals > 0 ? Double(accepted) / Double(proposals) : 0
    }

    public func delta(since previous: Self) -> Self {
        Self(
            attempts: max(0, attempts - previous.attempts),
            proposals: max(0, proposals - previous.proposals),
            accepted: max(0, accepted - previous.accepted),
            emptyProposals: max(0, emptyProposals - previous.emptyProposals),
            rejectedBlocks: max(0, rejectedBlocks - previous.rejectedBlocks),
            tailSkips: max(0, tailSkips - previous.tailSkips),
            backoffSkips: max(0, backoffSkips - previous.backoffSkips),
            pendingBackoff: pendingBackoff)
    }
}

/// Non-timing portion of the upstream scheduler. Decisions depend only on
/// proposal/acceptance outcomes, so the same greedy transcript is reproducible
/// across machines with different storage and GPU speeds.
public struct DSparkAdaptiveScheduler: Sendable {
    public let policy: DSparkSchedulerPolicy

    private var windowCycles = 0
    private var windowAccepted = 0
    private var windowEmpty = 0
    private var lifetimeAccepted = 0
    private var longAcceptSeen = false
    private var skipRemaining = 0

    private var totalAttempts = 0
    private var totalProposals = 0
    private var totalAccepted = 0
    private var totalEmpty = 0
    private var totalRejectedBlocks = 0
    private var totalTailSkips = 0
    private var totalBackoffSkips = 0

    public init(policy: DSparkSchedulerPolicy = .fromEnvironment()) {
        self.policy = policy
    }

    public mutating func decision(remainingTokens: Int) -> DSparkScheduleDecision {
        guard policy.enabled else { return .attempt }
        if policy.tailMinimumTokens > 0,
           remainingTokens < policy.tailMinimumTokens {
            totalTailSkips += 1
            return .skipTail
        }
        guard skipRemaining > 0 else { return .attempt }
        skipRemaining -= 1
        totalBackoffSkips += 1
        return .skipBackoff(remaining: skipRemaining)
    }

    /// Record one actual DSpark attempt. `proposedDrafts == 0` includes a
    /// confidence-gated empty block; verifier rejection is instead a non-empty
    /// proposal with zero accepted drafts.
    public mutating func note(
        proposedDrafts: Int,
        acceptedDrafts: Int,
        firstConfidence: Float?
    ) {
        let proposed = max(0, proposedDrafts)
        let accepted = min(proposed, max(0, acceptedDrafts))
        totalAttempts += 1
        totalProposals += proposed
        totalAccepted += accepted
        if proposed == 0 { totalEmpty += 1 }
        if proposed > 0, accepted == 0 { totalRejectedBlocks += 1 }

        guard policy.enabled else { return }
        windowCycles += 1
        windowAccepted += accepted
        if proposed == 0 { windowEmpty += 1 }
        lifetimeAccepted += accepted
        if accepted > 2 { longAcceptSeen = true }

        if proposed == 0 {
            var pause = policy.noDraftSkipCycles
            if lifetimeAccepted > 0, !longAcceptSeen {
                pause = max(pause, policy.shortAcceptNoDraftSkipCycles)
            } else if lifetimeAccepted == 0,
                      let confidence = firstConfidence,
                      confidence.isFinite,
                      confidence <= policy.coldLowConfidenceThreshold {
                pause = max(pause, policy.coldLowConfidenceSkipCycles)
            }
            skipRemaining = max(skipRemaining, pause)
        }

        guard windowCycles >= policy.window else { return }
        let average = Double(windowAccepted) / Double(windowCycles)
        let lowAcceptance = average < policy.minimumAverageAccepted
        let manyEmpty = windowEmpty * 2 >= windowCycles
        if lowAcceptance || manyEmpty {
            var pause = policy.skipCycles
            if manyEmpty { pause = max(pause, policy.slowSkipCycles) }
            skipRemaining = max(skipRemaining, pause)
        }
        windowCycles = 0
        windowAccepted = 0
        windowEmpty = 0
    }

    public var snapshot: DSparkSchedulerSnapshot {
        DSparkSchedulerSnapshot(
            attempts: totalAttempts,
            proposals: totalProposals,
            accepted: totalAccepted,
            emptyProposals: totalEmpty,
            rejectedBlocks: totalRejectedBlocks,
            tailSkips: totalTailSkips,
            backoffSkips: totalBackoffSkips,
            pendingBackoff: skipRemaining)
    }
}
