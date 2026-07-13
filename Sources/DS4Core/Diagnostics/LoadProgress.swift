import Foundation

/// Thread-safe model-load progress reporter. The engine's load path (decoder
/// factory, DenseStreamer, Q4 requant/cache) writes milestones and per-unit
/// advances; the UI polls `snapshot` (~10 Hz) and renders a determinate bar.
/// A singleton on purpose: one model loads at a time, and threading a callback
/// through every factory/init signature would touch a dozen call sites for the
/// same effect. Writers may call from concurrentPerform workers.
public final class LoadProgress: @unchecked Sendable {
    public static let shared = LoadProgress()

    private let lock = NSLock()
    private var fraction: Double = 0
    private var stage = ""
    private var spanBase: Double = 0
    private var spanEnd: Double = 0
    private var unitsDone = 0
    private var unitsTotal = 1

    /// Back to zero (call when a load starts).
    public func reset() { set(0, "") }

    /// Jump to an absolute fraction with a stage label.
    public func set(_ fraction: Double, _ stage: String) {
        lock.lock()
        self.fraction = fraction
        self.stage = stage
        self.spanBase = fraction
        self.spanEnd = fraction
        self.unitsDone = 0
        self.unitsTotal = 1
        lock.unlock()
    }

    /// Begin a stage that spans [from, to] over `units` equal steps.
    public func begin(_ stage: String, from: Double, to: Double, units: Int) {
        lock.lock()
        self.stage = stage
        self.fraction = from
        self.spanBase = from
        self.spanEnd = to
        self.unitsDone = 0
        self.unitsTotal = max(1, units)
        lock.unlock()
    }

    /// Complete one unit of the current stage (safe from worker threads).
    public func advance() {
        lock.lock()
        unitsDone = min(unitsTotal, unitsDone + 1)
        fraction = spanBase + (spanEnd - spanBase) * Double(unitsDone) / Double(unitsTotal)
        lock.unlock()
    }

    public var snapshot: (fraction: Double, stage: String) {
        lock.lock()
        defer { lock.unlock() }
        return (fraction, stage)
    }
}
