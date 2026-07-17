import Foundation

/// Runtime controls that the in-app machine tuner may explore.
///
/// Values are integers so boolean controls can use `0`/`1` and the same
/// configuration can be persisted or translated to environment variables by
/// the GUI adapter without losing type information in the search core.
public enum MachineAutoTuneKnob: String, CaseIterable, Codable, Sendable {
    case multiQuantCache
    case expertCacheSlots
    case expertCacheUniform
    case preadSplit
    case denseAhead
    case asyncFFN
    case expertLookahead
    case q8NSG
    case moeNSG
    case denseQ4NSG
}

/// Immutable-by-convention value object used by the coordinate search.
public struct MachineAutoTuneConfiguration: Codable, Sendable {
    public private(set) var settings: [MachineAutoTuneKnob: Int]

    public init(settings: [MachineAutoTuneKnob: Int] = [:]) {
        self.settings = settings
    }

    public func value(for knob: MachineAutoTuneKnob) -> Int? {
        settings[knob]
    }

    /// Returns a copy with one setting changed; the incumbent remains intact.
    public func setting(_ value: Int, for knob: MachineAutoTuneKnob) -> Self {
        var copy = self
        copy.settings[knob] = value
        return copy
    }
}

extension MachineAutoTuneConfiguration: Equatable {
    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.settings == rhs.settings
    }
}

extension MachineAutoTuneConfiguration: Hashable {
    public func hash(into hasher: inout Hasher) {
        // Dictionary iteration order is intentionally unspecified. A stable
        // knob order makes equal configurations hash from the same sequence.
        for knob in MachineAutoTuneKnob.allCases {
            guard let value = settings[knob] else { continue }
            hasher.combine(knob)
            hasher.combine(value)
        }
    }
}

public struct MachineAutoTuneParameter: Hashable, Codable, Sendable {
    public enum Search: String, Codable, Sendable {
        /// Try the next higher value first. A promotion locks that direction
        /// and keeps walking until the first rejected candidate. The opposite
        /// neighbour is tried only when the initial higher value does not win.
        case walk
        /// Measure the entire small hardware grid; useful for NSG and I/O queue
        /// depths, whose response is not expected to be monotonic.
        case sweep
    }

    public enum Quality: String, Codable, Sendable {
        /// Generated tokens and every captured full-logit frame must retain the
        /// exact finite count, raw-bit hash and ordered top-three tokens.
        case exact
    }

    public let knob: MachineAutoTuneKnob
    public let values: [Int]
    public let search: Search
    public let quality: Quality
    public let memoryRisk: Bool

    public init(
        knob: MachineAutoTuneKnob,
        values: [Int],
        search: Search,
        quality: Quality = .exact,
        memoryRisk: Bool = false
    ) {
        self.knob = knob
        self.values = values
        self.search = search
        self.quality = quality
        self.memoryRisk = memoryRisk
    }

    /// Safe, exact-quality default manifest. RAM only widens the slot-cache
    /// grid; runtime memory gates still decide whether a measured candidate is
    /// eligible for promotion.
    public static func standard(ramGB: Int) -> [MachineAutoTuneParameter] {
        let slotValues: [Int]
        switch max(1, ramGB) {
        case 96...:
            slotValues = [0, 8, 12, 16, 18, 20, 22, 24, 32, 48]
        case 48...:
            slotValues = [0, 8, 12, 16, 18, 20, 22, 24, 32]
        case 24...:
            slotValues = [0, 8, 12, 16, 18, 20, 22, 24]
        case 17...:
            slotValues = [0, 8, 12, 16, 18, 20, 22, 24]
        default:
            // 0 disables the cache; 8 is the runtime's minimum real pool (a
            // positive request below 8 is clamped to 8). These points are
            // essential escape hatches when a loaded 16 GB root is already
            // memory-constrained. Growth is still vetoed by live RAM guards.
            slotValues = [0, 8, 12, 16, 18, 20, 22, 24]
        }

        return [
            MachineAutoTuneParameter(
                knob: .multiQuantCache, values: [0, 1], search: .sweep
            ),
            MachineAutoTuneParameter(
                knob: .expertCacheSlots, values: slotValues,
                search: .walk, memoryRisk: true
            ),
            MachineAutoTuneParameter(
                knob: .expertCacheUniform, values: [0, 1], search: .sweep
            ),
            MachineAutoTuneParameter(
                knob: .preadSplit, values: Array(1...8), search: .sweep
            ),
            MachineAutoTuneParameter(
                knob: .denseAhead, values: [1, 2, 3],
                search: .walk, memoryRisk: true
            ),
            MachineAutoTuneParameter(
                knob: .asyncFFN, values: [0, 1], search: .walk
            ),
            MachineAutoTuneParameter(
                knob: .expertLookahead,
                values: [0, 2, 4, 6, 8, 10, 12], search: .sweep
            ),
            MachineAutoTuneParameter(
                knob: .moeNSG, values: Array(1...8), search: .sweep
            ),
            MachineAutoTuneParameter(
                knob: .denseQ4NSG, values: Array(1...8), search: .sweep
            ),
        ]
    }
}

/// Namespace retained for call sites that prefer a manifest-oriented API.
public enum MachineAutoTuneManifest {
    public static func standard(ramGB: Int) -> [MachineAutoTuneParameter] {
        MachineAutoTuneParameter.standard(ramGB: ramGB)
    }
}

/// Compact integrity summary for one bounded full-vocabulary logit frame.
public struct MachineAutoTuneQualityFrame: Codable, Equatable, Sendable {
    public let count: Int
    public let finiteCount: Int
    public let bitHash: String
    public let top3TokenIds: [Int]

    public init(count: Int, finiteCount: Int, bitHash: String, top3TokenIds: [Int]) {
        self.count = count
        self.finiteCount = finiteCount
        self.bitHash = bitHash
        self.top3TokenIds = top3TokenIds
    }
}

/// Exact-quality signature shared by the immutable root and every A/B run.
public struct MachineAutoTuneQualitySignature: Codable, Equatable, Sendable {
    public let generatedTokens: [Int]
    public let frames: [MachineAutoTuneQualityFrame]

    public init(generatedTokens: [Int], frames: [MachineAutoTuneQualityFrame]) {
        self.generatedTokens = generatedTokens
        self.frames = frames
    }
}

/// One complete warmed performance/quality measurement.
public struct MachineAutoTuneObservation: Equatable, Sendable {
    public let primaryTps: Double
    public let secondaryTps: Double
    public let stability: Double
    public let memoryFreePercent: Double
    public let swapoutMiB: Double
    public let quality: MachineAutoTuneQualitySignature

    public init(
        primaryTps: Double,
        secondaryTps: Double,
        stability: Double,
        memoryFreePercent: Double,
        swapoutMiB: Double,
        quality: MachineAutoTuneQualitySignature
    ) {
        self.primaryTps = primaryTps
        self.secondaryTps = secondaryTps
        self.stability = stability
        self.memoryFreePercent = memoryFreePercent
        self.swapoutMiB = swapoutMiB
        self.quality = quality
    }
}

/// Promotion thresholds shared by screening and order-balanced confirmation.
public struct MachineAutoTunePolicy: Hashable, Codable, Sendable {
    public let minimumGain: Double
    public let maximumScreeningRegression: Double
    public let maximumPairRegression: Double
    public let maximumSecondaryRegression: Double
    public let minimumStability: Double
    public let minimumMemoryFreePercent: Double
    public let maximumSwapoutMiB: Double

    public init(
        minimumGain: Double = 0.02,
        maximumScreeningRegression: Double = 0.03,
        maximumPairRegression: Double = 0.03,
        maximumSecondaryRegression: Double = 0.08,
        minimumStability: Double = 0.75,
        minimumMemoryFreePercent: Double = 10,
        maximumSwapoutMiB: Double = 128
    ) {
        self.minimumGain = minimumGain
        self.maximumScreeningRegression = maximumScreeningRegression
        self.maximumPairRegression = maximumPairRegression
        self.maximumSecondaryRegression = maximumSecondaryRegression
        self.minimumStability = minimumStability
        self.minimumMemoryFreePercent = minimumMemoryFreePercent
        self.maximumSwapoutMiB = maximumSwapoutMiB
    }

    /// Returns the same performance, stability and swap policy with a
    /// run-specific free-memory floor. This is intentionally a copy: the
    /// default 10% policy remains immutable and cannot be weakened globally by
    /// a constrained run.
    public func withMinimumMemoryFreePercent(_ percent: Double) -> Self {
        Self(
            minimumGain: minimumGain,
            maximumScreeningRegression: maximumScreeningRegression,
            maximumPairRegression: maximumPairRegression,
            maximumSecondaryRegression: maximumSecondaryRegression,
            minimumStability: minimumStability,
            minimumMemoryFreePercent: percent,
            maximumSwapoutMiB: maximumSwapoutMiB
        )
    }
}

/// Immutable memory envelope derived from the already loaded root engine.
///
/// A large model can legitimately leave less than the normal 10% headroom;
/// requiring 10% before the tuner starts makes that machine impossible to
/// tune because decoder geometry is available only after loading the model.
/// In that situation the root is treated as a known-loadable reference, while
/// the search is restricted to memory-neutral or memory-reducing candidates.
/// The effective observation floor remains anchored to that original root, so
/// repeated promotions cannot progressively consume more memory.
public struct MachineAutoTuneMemoryEnvelope: Hashable, Codable, Sendable {
    /// The already-running root only needs enough headroom to quiesce and tear
    /// down safely. The stricter 1.5 GiB/12% construction reserve is checked
    /// later, after that model has actually been released.
    public static let defaultHardReserveBytes: UInt64 = 536_870_912 // 512 MiB

    public let baselineFreePercent: Double
    public let physicalMemoryBytes: UInt64
    public let standardFloorPercent: Double
    public let hardReserveBytes: UInt64
    public let measurementTolerancePercent: Double

    public init(
        baselineFreePercent: Double,
        physicalMemoryBytes: UInt64,
        standardFloorPercent: Double = 10,
        hardReserveBytes: UInt64 = Self.defaultHardReserveBytes,
        measurementTolerancePercent: Double = 1
    ) {
        self.baselineFreePercent = baselineFreePercent
        self.physicalMemoryBytes = physicalMemoryBytes
        self.standardFloorPercent = standardFloorPercent
        self.hardReserveBytes = hardReserveBytes
        self.measurementTolerancePercent = measurementTolerancePercent
    }

    /// Inputs are validated separately so the envelope can be included in a
    /// crash report even when the counters themselves are invalid.
    public var isValid: Bool {
        baselineFreePercent.isFinite && (0...100).contains(baselineFreePercent) &&
            physicalMemoryBytes > 0 &&
            hardReserveBytes > 0 &&
            standardFloorPercent.isFinite && (0...100).contains(standardFloorPercent) &&
            measurementTolerancePercent.isFinite && measurementTolerancePercent >= 0 &&
            measurementTolerancePercent <= 100
    }

    public var hardReservePercent: Double {
        guard physicalMemoryBytes > 0 else { return .infinity }
        return Double(hardReserveBytes) / Double(physicalMemoryBytes) * 100
    }

    public var isConstrained: Bool {
        isValid && baselineFreePercent < standardFloorPercent
    }

    /// A loaded root below the byte reserve is not safe to reload repeatedly.
    /// Equality is accepted so the decision remains deterministic at the
    /// boundary.
    public var canStart: Bool {
        isValid && baselineFreePercent >= hardReservePercent
    }

    /// Normal runs retain the standard floor. Constrained runs use a fixed
    /// root-relative floor with one percentage point of default tolerance for
    /// VM accounting noise, but never cross the absolute byte reserve.
    public var effectiveMinimumFreePercent: Double {
        guard isValid else { return .infinity }
        guard isConstrained else { return standardFloorPercent }
        return max(hardReservePercent,
                   baselineFreePercent - measurementTolerancePercent)
    }

    /// Before constructing a candidate, constrained mode may only move toward
    /// equal or smaller known resident geometry. Runtime checks still enforce
    /// the effective floor after init and after warmup for allocations that
    /// are not captured by the estimate.
    public func allowsResidentDeltaBytes(_ deltaBytes: Int64) -> Bool {
        isValid && (!isConstrained || deltaBytes <= 0)
    }

    public func policy(
        from base: MachineAutoTunePolicy = MachineAutoTunePolicy()
    ) -> MachineAutoTunePolicy {
        base.withMinimumMemoryFreePercent(effectiveMinimumFreePercent)
    }
}

public struct MachineAutoTuneEvaluationResult: Equatable, Sendable {
    /// Candidate/incumbent ratio for each measured pair.
    public let primaryRatios: [Double]
    public let secondaryRatios: [Double]
    /// Geometric means. Screening has one pair, so its mean equals its ratio.
    public let balancedPrimaryRatio: Double?
    public let balancedSecondaryRatio: Double?
    public let transitionQualityExact: Bool
    public let cumulativeQualityExact: Bool
    public let reasons: [String]
    public let qualified: Bool

    public init(
        primaryRatios: [Double],
        secondaryRatios: [Double],
        balancedPrimaryRatio: Double?,
        balancedSecondaryRatio: Double?,
        transitionQualityExact: Bool,
        cumulativeQualityExact: Bool,
        reasons: [String],
        qualified: Bool
    ) {
        self.primaryRatios = primaryRatios
        self.secondaryRatios = secondaryRatios
        self.balancedPrimaryRatio = balancedPrimaryRatio
        self.balancedSecondaryRatio = balancedSecondaryRatio
        self.transitionQualityExact = transitionQualityExact
        self.cumulativeQualityExact = cumulativeQualityExact
        self.reasons = reasons
        self.qualified = qualified
    }
}

/// Symmetric repeatability check for two consecutive measurements of the same
/// configuration. Candidate screening is intentionally directional, while a
/// baseline must reject both regressions and large warm-up improvements.
public struct MachineAutoTuneRepeatabilityResult: Equatable, Sendable {
    public let forward: MachineAutoTuneEvaluationResult
    public let reverse: MachineAutoTuneEvaluationResult

    public init(
        forward: MachineAutoTuneEvaluationResult,
        reverse: MachineAutoTuneEvaluationResult
    ) {
        self.forward = forward
        self.reverse = reverse
    }

    public var qualified: Bool { forward.qualified && reverse.qualified }

    public var reasons: [String] {
        forward.reasons.map { "forward: \($0)" }
            + reverse.reasons.map { "reverse: \($0)" }
    }
}

public struct MachineAutoTuneNeighbor: Equatable, Sendable {
    public enum Direction: Int, Codable, Sendable {
        case downward = -1
        case upward = 1
    }

    public let value: Int
    public let direction: Direction

    public init(value: Int, direction: Direction) {
        self.value = value
        self.direction = direction
    }
}

/// Pure state machine for a sequential, directional hill climb.
///
/// It emits one candidate at a time. The first promotion locks the direction;
/// subsequent rejection stops the walk without testing values beyond the first
/// decline. Before a direction is locked, a rejected upward candidate permits
/// exactly one downward fallback. This avoids measuring both neighbours when
/// the first one already establishes the direction of improvement.
public struct MachineAutoTuneDirectionalWalk: Sendable {
    private let values: [Int]
    private var initialCandidates: [MachineAutoTuneNeighbor]
    private var pendingCandidate: MachineAutoTuneNeighbor?

    public private(set) var current: Int
    public private(set) var direction: MachineAutoTuneNeighbor.Direction?
    public private(set) var isStopped: Bool
    public let isValid: Bool

    public init(values: [Int], current: Int) {
        let strictlyIncreasing = zip(values, values.dropFirst()).allSatisfy {
            $0.0 < $0.1
        }
        let valid = !values.isEmpty &&
            strictlyIncreasing &&
            Set(values).count == values.count &&
            values.contains(current)

        self.values = values
        self.current = current
        self.direction = nil
        self.isValid = valid
        self.isStopped = !valid
        self.pendingCandidate = nil
        self.initialCandidates = valid
            ? MachineAutoTuneEvaluator.orderedNeighbors(values: values, current: current)
            : []
    }

    /// Returns at most one candidate until `recordResult` resolves it.
    public mutating func nextCandidate() -> MachineAutoTuneNeighbor? {
        guard isValid, !isStopped, pendingCandidate == nil else { return nil }

        let candidate: MachineAutoTuneNeighbor?
        if let direction {
            guard let index = values.firstIndex(of: current) else {
                isStopped = true
                return nil
            }
            let nextIndex = index + direction.rawValue
            guard values.indices.contains(nextIndex) else {
                isStopped = true
                return nil
            }
            candidate = MachineAutoTuneNeighbor(
                value: values[nextIndex],
                direction: direction
            )
        } else if initialCandidates.isEmpty {
            isStopped = true
            candidate = nil
        } else {
            candidate = initialCandidates.removeFirst()
        }

        pendingCandidate = candidate
        return candidate
    }

    /// Resolves the pending candidate. A qualified result advances and locks
    /// the walk direction. A rejection after the lock ends the parameter;
    /// rejection of the first upward probe leaves the downward fallback ready.
    @discardableResult
    public mutating func recordResult(qualified: Bool) -> Bool {
        guard let candidate = pendingCandidate else {
            isStopped = true
            return false
        }
        pendingCandidate = nil

        if qualified {
            current = candidate.value
            direction = candidate.direction
            initialCandidates.removeAll(keepingCapacity: false)
            return true
        }

        if direction != nil || initialCandidates.isEmpty {
            isStopped = true
        }
        return false
    }
}

/// Pure evaluator: no model, Metal device, process environment or persistence.
public enum MachineAutoTuneEvaluator {
    /// Exposes the same fail-closed structural quality validation used by every
    /// screen. The GUI uses it once for the immutable root, so malformed logits
    /// abort immediately instead of being mistaken for retryable warm-up noise.
    public static func qualityValidationReasons(
        _ signature: MachineAutoTuneQualitySignature,
        label: String = "quality"
    ) -> [String] {
        var reasons: [String] = []
        validateQuality(signature, label: label, reasons: &reasons)
        return reasons
    }

    /// Fail-closed eligibility check for a complete record-holder sample.
    /// Unlike a performance comparison this has no directional ratio: it only
    /// verifies the immutable quality root and every resource/stability gate.
    public static func observationValidationReasons(
        _ observation: MachineAutoTuneObservation,
        rootQuality: MachineAutoTuneQualitySignature,
        label: String = "observation",
        policy: MachineAutoTunePolicy = MachineAutoTunePolicy()
    ) -> [String] {
        var reasons: [String] = []
        validatePolicy(policy, reasons: &reasons)
        validateQuality(rootQuality, label: "root quality", reasons: &reasons)
        validateObservation(observation, label: label, policy: policy, reasons: &reasons)
        if observation.quality != rootQuality {
            reasons.append("\(label) quality differs from the immutable root")
        }
        return reasons
    }

    /// Selects one whole valid observation as the high-water reference. Metrics
    /// from different runs are never combined. Decode is authoritative; the
    /// remaining fields only make exact ties deterministic.
    public static func bestValidObservation(
        in observations: [MachineAutoTuneObservation],
        rootQuality: MachineAutoTuneQualitySignature,
        policy: MachineAutoTunePolicy = MachineAutoTunePolicy()
    ) -> MachineAutoTuneObservation? {
        observations
            .filter {
                observationValidationReasons(
                    $0,
                    rootQuality: rootQuality,
                    policy: policy
                ).isEmpty
            }
            .max(by: { recordRanksBefore($0, $1) })
    }

    /// Single-shot promotion used by the run-local record-holder tuner. Each
    /// unique configuration is benchmarked once; a candidate replaces the
    /// cached incumbent whenever its decode result is strictly higher and all
    /// exact-quality, prefill, stability, RAM and steady-swap safeguards pass.
    /// `minimumGain` intentionally belongs to the repeatable ABBA path: using
    /// it here would make the cached comparator differ from the highest valid
    /// single-shot benchmark already observed.
    public static func highWaterComparison(
        incumbent: MachineAutoTuneObservation,
        candidate: MachineAutoTuneObservation,
        rootQuality: MachineAutoTuneQualitySignature,
        policy: MachineAutoTunePolicy = MachineAutoTunePolicy()
    ) -> MachineAutoTuneEvaluationResult {
        var reasons: [String] = []
        validatePolicy(policy, reasons: &reasons)
        validateQuality(rootQuality, label: "root quality", reasons: &reasons)
        validateObservation(incumbent, label: "record", policy: policy, reasons: &reasons)
        validateObservation(candidate, label: "candidate", policy: policy, reasons: &reasons)

        let transitionExact = candidate.quality == incumbent.quality
        let cumulativeExact = candidate.quality == rootQuality
        if !transitionExact { reasons.append("quality is not exact across the record transition") }
        if !cumulativeExact { reasons.append("candidate quality differs from the immutable root") }

        let primary = safeRatio(candidate.primaryTps, incumbent.primaryTps)
        let secondary = safeRatio(candidate.secondaryTps, incumbent.secondaryTps)
        if let primary {
            if primary <= 1 {
                reasons.append(
                    "primary record ratio \(format(primary)) is not strictly above 1.0000"
                )
            }
        } else {
            reasons.append("primary record ratio is unavailable")
        }
        if let secondary {
            let floor = 1 - policy.maximumSecondaryRegression
            if secondary < floor {
                reasons.append("secondary ratio \(format(secondary)) is below \(format(floor))")
            }
        } else {
            reasons.append("secondary record ratio is unavailable")
        }

        return MachineAutoTuneEvaluationResult(
            primaryRatios: primary.map { [$0] } ?? [],
            secondaryRatios: secondary.map { [$0] } ?? [],
            balancedPrimaryRatio: primary,
            balancedSecondaryRatio: secondary,
            transitionQualityExact: transitionExact,
            cumulativeQualityExact: cumulativeExact,
            reasons: reasons,
            qualified: reasons.isEmpty
        )
    }

    /// Fast A→B screen. A candidate survives when it is not clearly slower;
    /// the 2% promotion threshold is reserved for a confirmation mode.
    public static func screening(
        incumbent: MachineAutoTuneObservation,
        candidate: MachineAutoTuneObservation,
        rootQuality: MachineAutoTuneQualitySignature,
        policy: MachineAutoTunePolicy = MachineAutoTunePolicy()
    ) -> MachineAutoTuneEvaluationResult {
        var reasons: [String] = []
        validatePolicy(policy, reasons: &reasons)
        validateQuality(rootQuality, label: "root quality", reasons: &reasons)
        validateObservation(incumbent, label: "incumbent", policy: policy, reasons: &reasons)
        validateObservation(candidate, label: "candidate", policy: policy, reasons: &reasons)

        let transitionExact = candidate.quality == incumbent.quality
        let cumulativeExact = candidate.quality == rootQuality
        if !transitionExact { reasons.append("quality is not exact across the A/B transition") }
        if !cumulativeExact { reasons.append("quality differs from the immutable root") }

        let primary = safeRatio(candidate.primaryTps, incumbent.primaryTps)
        let secondary = safeRatio(candidate.secondaryTps, incumbent.secondaryTps)
        if let primary {
            let floor = 1 - policy.maximumScreeningRegression
            if primary < floor {
                reasons.append("primary screening ratio \(format(primary)) is below \(format(floor))")
            }
        } else {
            reasons.append("primary screening ratio is unavailable")
        }
        if let secondary {
            let floor = 1 - policy.maximumSecondaryRegression
            if secondary < floor {
                reasons.append("secondary ratio \(format(secondary)) is below \(format(floor))")
            }
        } else {
            reasons.append("secondary screening ratio is unavailable")
        }

        return MachineAutoTuneEvaluationResult(
            primaryRatios: primary.map { [$0] } ?? [],
            secondaryRatios: secondary.map { [$0] } ?? [],
            balancedPrimaryRatio: primary,
            balancedSecondaryRatio: secondary,
            transitionQualityExact: transitionExact,
            cumulativeQualityExact: cumulativeExact,
            reasons: reasons,
            qualified: reasons.isEmpty
        )
    }

    /// Baseline repeatability is directional screening in both directions.
    /// This preserves every per-observation quality/RAM/swap/stability gate and
    /// bounds absolute throughput drift without inventing a second tolerance.
    public static func repeatability(
        first: MachineAutoTuneObservation,
        second: MachineAutoTuneObservation,
        rootQuality: MachineAutoTuneQualitySignature,
        policy: MachineAutoTunePolicy = MachineAutoTunePolicy()
    ) -> MachineAutoTuneRepeatabilityResult {
        MachineAutoTuneRepeatabilityResult(
            forward: screening(
                incumbent: first,
                candidate: second,
                rootQuality: rootQuality,
                policy: policy
            ),
            reverse: screening(
                incumbent: second,
                candidate: first,
                rootQuality: rootQuality,
                policy: policy
            )
        )
    }

    /// Order-balanced A→B, B→A confirmation. Ratios always use B/A, so the
    /// geometric mean removes multiplicative order bias without inverting the
    /// second pair accidentally.
    public static func abba(
        a1: MachineAutoTuneObservation,
        b1: MachineAutoTuneObservation,
        b2: MachineAutoTuneObservation,
        a2: MachineAutoTuneObservation,
        rootQuality: MachineAutoTuneQualitySignature,
        policy: MachineAutoTunePolicy = MachineAutoTunePolicy()
    ) -> MachineAutoTuneEvaluationResult {
        var reasons: [String] = []
        validatePolicy(policy, reasons: &reasons)
        validateQuality(rootQuality, label: "root quality", reasons: &reasons)
        for (label, observation) in [("A1", a1), ("B1", b1), ("B2", b2), ("A2", a2)] {
            validateObservation(observation, label: label, policy: policy, reasons: &reasons)
        }

        let transitionExact = b1.quality == a1.quality && b2.quality == a2.quality
        let cumulativeExact = b1.quality == rootQuality && b2.quality == rootQuality
        if !transitionExact { reasons.append("quality is not exact across both A/B transitions") }
        if !cumulativeExact { reasons.append("candidate quality differs from the immutable root") }

        let primaryRatios = compactRatios([
            safeRatio(b1.primaryTps, a1.primaryTps),
            safeRatio(b2.primaryTps, a2.primaryTps),
        ])
        let secondaryRatios = compactRatios([
            safeRatio(b1.secondaryTps, a1.secondaryTps),
            safeRatio(b2.secondaryTps, a2.secondaryTps),
        ])
        if primaryRatios.count != 2 { reasons.append("one or more primary pair ratios are unavailable") }
        if secondaryRatios.count != 2 { reasons.append("one or more secondary pair ratios are unavailable") }

        let primaryMean = geometricMean(primaryRatios, expectedCount: 2)
        let secondaryMean = geometricMean(secondaryRatios, expectedCount: 2)

        let pairFloor = 1 - policy.maximumPairRegression
        for (index, ratio) in primaryRatios.enumerated() where ratio < pairFloor {
            reasons.append("primary pair \(index + 1) ratio \(format(ratio)) is below \(format(pairFloor))")
        }
        if let primaryMean {
            let promotionFloor = 1 + policy.minimumGain
            if primaryMean < promotionFloor {
                reasons.append("balanced primary ratio \(format(primaryMean)) is below \(format(promotionFloor))")
            }
        } else {
            reasons.append("balanced primary ratio is unavailable")
        }
        if let secondaryMean {
            let floor = 1 - policy.maximumSecondaryRegression
            if secondaryMean < floor {
                reasons.append("balanced secondary ratio \(format(secondaryMean)) is below \(format(floor))")
            }
        } else {
            reasons.append("balanced secondary ratio is unavailable")
        }

        return MachineAutoTuneEvaluationResult(
            primaryRatios: primaryRatios,
            secondaryRatios: secondaryRatios,
            balancedPrimaryRatio: primaryMean,
            balancedSecondaryRatio: secondaryMean,
            transitionQualityExact: transitionExact,
            cumulativeQualityExact: cumulativeExact,
            reasons: reasons,
            qualified: reasons.isEmpty
        )
    }

    /// Immediate neighbours in the manifest's declared order. The upward side
    /// is intentionally tried first, while both directions remain visible to
    /// the caller before it commits to a directional walk.
    public static func orderedNeighbors(values: [Int], current: Int) -> [MachineAutoTuneNeighbor] {
        guard Set(values).count == values.count,
              let index = values.firstIndex(of: current) else { return [] }
        var result: [MachineAutoTuneNeighbor] = []
        if index + 1 < values.count {
            result.append(MachineAutoTuneNeighbor(value: values[index + 1], direction: .upward))
        }
        if index > values.startIndex {
            result.append(MachineAutoTuneNeighbor(value: values[index - 1], direction: .downward))
        }
        return result
    }

    /// Statistical median with the conventional average of the two central
    /// values for an even-sized sample. Empty or non-finite input is invalid.
    public static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty, values.allSatisfy(\.isFinite) else { return nil }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return sorted[middle - 1] + (sorted[middle] - sorted[middle - 1]) / 2
        }
        return sorted[middle]
    }

    private static func validatePolicy(
        _ policy: MachineAutoTunePolicy,
        reasons: inout [String]
    ) {
        let fractions = [
            policy.minimumGain,
            policy.maximumScreeningRegression,
            policy.maximumPairRegression,
            policy.maximumSecondaryRegression,
            policy.minimumStability,
            policy.minimumMemoryFreePercent,
            policy.maximumSwapoutMiB,
        ]
        guard fractions.allSatisfy(\.isFinite),
              policy.minimumGain >= 0,
              (0..<1).contains(policy.maximumScreeningRegression),
              (0..<1).contains(policy.maximumPairRegression),
              (0..<1).contains(policy.maximumSecondaryRegression),
              policy.minimumStability >= 0,
              (0...100).contains(policy.minimumMemoryFreePercent),
              policy.maximumSwapoutMiB >= 0 else {
            reasons.append("policy contains invalid thresholds")
            return
        }
    }

    private static func validateObservation(
        _ observation: MachineAutoTuneObservation,
        label: String,
        policy: MachineAutoTunePolicy,
        reasons: inout [String]
    ) {
        guard observation.primaryTps.isFinite, observation.primaryTps > 0,
              observation.secondaryTps.isFinite, observation.secondaryTps > 0,
              observation.stability.isFinite, observation.stability >= 0,
              observation.memoryFreePercent.isFinite,
              (0...100).contains(observation.memoryFreePercent),
              observation.swapoutMiB.isFinite, observation.swapoutMiB >= 0 else {
            reasons.append("\(label) contains non-finite or out-of-range metrics")
            return
        }
        if observation.stability < policy.minimumStability {
            reasons.append("\(label) stability \(format(observation.stability)) is below \(format(policy.minimumStability))")
        }
        if observation.memoryFreePercent < policy.minimumMemoryFreePercent {
            reasons.append("\(label) free memory \(format(observation.memoryFreePercent))% is below \(format(policy.minimumMemoryFreePercent))%")
        }
        if observation.swapoutMiB > policy.maximumSwapoutMiB {
            reasons.append("\(label) swapout \(format(observation.swapoutMiB)) MiB exceeds \(format(policy.maximumSwapoutMiB)) MiB")
        }
        validateQuality(observation.quality, label: "\(label) quality", reasons: &reasons)
    }

    private static func validateQuality(
        _ signature: MachineAutoTuneQualitySignature,
        label: String,
        reasons: inout [String]
    ) {
        guard !signature.frames.isEmpty else {
            reasons.append("\(label) has no captured frames")
            return
        }
        if signature.frames.count != signature.generatedTokens.count {
            reasons.append(
                "\(label) frame/token count mismatch " +
                "(\(signature.frames.count) frames, \(signature.generatedTokens.count) tokens)"
            )
        }

        let vocabularyCount = signature.frames[0].count
        for (index, frame) in signature.frames.enumerated() {
            let hashIsHex = frame.bitHash.count == 16 && frame.bitHash.unicodeScalars.allSatisfy {
                (48...57).contains($0.value) || (65...70).contains($0.value) || (97...102).contains($0.value)
            }
            let top3 = frame.top3TokenIds
            if frame.count <= 0 {
                reasons.append("\(label) frame \(index) has a non-positive vocabulary count")
            }
            if frame.count != vocabularyCount {
                reasons.append(
                    "\(label) frame \(index) vocabulary count \(frame.count) " +
                    "differs from \(vocabularyCount)"
                )
            }
            if frame.count < 3 || frame.finiteCount != frame.count || !hashIsHex ||
                top3.count != 3 || Set(top3).count != 3 ||
                top3.contains(where: { $0 < 0 || $0 >= frame.count }) {
                reasons.append("\(label) frame \(index) is malformed or contains non-finite logits")
            }
            if index < signature.generatedTokens.count,
               top3.first != signature.generatedTokens[index] {
                reasons.append(
                    "\(label) generated token \(index) is not the frame's greedy top-1 token"
                )
            }
        }
    }

    private static func safeRatio(_ numerator: Double, _ denominator: Double) -> Double? {
        guard numerator.isFinite, denominator.isFinite,
              numerator > 0, denominator > 0 else { return nil }
        let ratio = numerator / denominator
        return ratio.isFinite && ratio > 0 ? ratio : nil
    }

    private static func compactRatios(_ ratios: [Double?]) -> [Double] {
        ratios.compactMap { $0 }
    }

    private static func recordRanksBefore(
        _ lhs: MachineAutoTuneObservation,
        _ rhs: MachineAutoTuneObservation
    ) -> Bool {
        if lhs.primaryTps != rhs.primaryTps { return lhs.primaryTps < rhs.primaryTps }
        if lhs.secondaryTps != rhs.secondaryTps { return lhs.secondaryTps < rhs.secondaryTps }
        if lhs.stability != rhs.stability { return lhs.stability < rhs.stability }
        if lhs.memoryFreePercent != rhs.memoryFreePercent {
            return lhs.memoryFreePercent < rhs.memoryFreePercent
        }
        if lhs.swapoutMiB != rhs.swapoutMiB { return lhs.swapoutMiB > rhs.swapoutMiB }
        return false
    }

    private static func geometricMean(_ values: [Double], expectedCount: Int) -> Double? {
        guard values.count == expectedCount,
              values.allSatisfy({ $0.isFinite && $0 > 0 }) else { return nil }
        let meanLog = values.reduce(0) { $0 + log($1) } / Double(values.count)
        let result = exp(meanLog)
        return result.isFinite && result > 0 ? result : nil
    }

    private static func format(_ value: Double) -> String {
        String(format: "%.4f", value)
    }
}
