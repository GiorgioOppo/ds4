import Foundation

/// One transition of a monotonically increasing cumulative byte counter.
///
/// VM swapout counters are cumulative (normally since boot), so their raw
/// value is not a per-trial metric. A backwards transition must never be
/// silently interpreted as zero: it is either a credible integer wrap or an
/// invalid/reset sample.
public enum MachineAutoTuneCumulativeCounterDelta: Equatable, Sendable {
    case advanced(bytes: UInt64)
    case wrapped(bytes: UInt64)
    case reset

    /// The measured increment, or `nil` when the counter reset and the window
    /// cannot be evaluated safely.
    public var bytes: UInt64? {
        switch self {
        case .advanced(let bytes), .wrapped(let bytes): bytes
        case .reset: nil
        }
    }

    public var didWrap: Bool {
        if case .wrapped = self { return true }
        return false
    }

    public var didReset: Bool {
        if case .reset = self { return true }
        return false
    }
}

/// Split swapout accounting for one independently loaded tuning trial.
///
/// The three samples have deliberately different roles:
///
/// 1. `beforeLoadBytes` is captured immediately before model construction;
/// 2. `steadyStateStartBytes` is captured after init, warmup and any discarded
///    warm benchmark, immediately before the measured benchmark;
/// 3. `steadyStateEndBytes` is captured immediately after that benchmark and
///    before quiesce/teardown.
///
/// Loading may page out data on a memory-constrained machine even when decode
/// is stable. The load delta remains visible for diagnostics and load guards,
/// while only `policySwapoutMiB` should be compared with the per-observation
/// steady-state swap limit.
public struct MachineAutoTuneSwapWindows: Equatable, Sendable {
    public static let bytesPerMiB: Double = 1_048_576

    public let load: MachineAutoTuneCumulativeCounterDelta
    public let steadyState: MachineAutoTuneCumulativeCounterDelta

    public init(
        beforeLoadBytes: UInt64,
        steadyStateStartBytes: UInt64,
        steadyStateEndBytes: UInt64,
        counterMaximum: UInt64 = .max
    ) {
        load = Self.delta(
            from: beforeLoadBytes,
            to: steadyStateStartBytes,
            counterMaximum: counterMaximum
        )
        steadyState = Self.delta(
            from: steadyStateStartBytes,
            to: steadyStateEndBytes,
            counterMaximum: counterMaximum
        )
    }

    public var loadMiB: Double? {
        load.bytes.map { Double($0) / Self.bytesPerMiB }
    }

    public var steadyStateMiB: Double? {
        steadyState.bytes.map { Double($0) / Self.bytesPerMiB }
    }

    /// The swap value consumed by high-water, screening and ABBA policy.
    /// A reset inside the measured window returns `nil` and must fail closed.
    public var policySwapoutMiB: Double? { steadyStateMiB }

    /// Load plus steady-state swap, useful for diagnostics only. This is `nil`
    /// if either window reset or if their sum cannot be represented exactly.
    public var totalMiB: Double? {
        guard let loadBytes = load.bytes,
              let steadyBytes = steadyState.bytes else { return nil }
        let (total, overflow) = loadBytes.addingReportingOverflow(steadyBytes)
        guard !overflow else { return nil }
        return Double(total) / Self.bytesPerMiB
    }

    /// Computes a delta without assuming that a backwards sample means zero.
    /// A wrap is inferred only across the upper/lower quarters of the declared
    /// counter range. Other regressions are treated as reset/invalid because a
    /// reset and an arbitrary wrap are mathematically indistinguishable from
    /// two samples alone.
    public static func delta(
        from previous: UInt64,
        to current: UInt64,
        counterMaximum: UInt64 = .max
    ) -> MachineAutoTuneCumulativeCounterDelta {
        guard counterMaximum >= 3,
              previous <= counterMaximum,
              current <= counterMaximum else { return .reset }

        if current >= previous {
            return .advanced(bytes: current - previous)
        }

        let quarter = counterMaximum / 4
        let upperWrapBoundary = counterMaximum - quarter
        guard previous >= upperWrapBoundary,
              current <= quarter else { return .reset }

        let (partial, firstOverflow) =
            (counterMaximum - previous).addingReportingOverflow(current)
        let (wrappedDelta, secondOverflow) = partial.addingReportingOverflow(1)
        guard !firstOverflow, !secondOverflow else { return .reset }
        return .wrapped(bytes: wrappedDelta)
    }
}
