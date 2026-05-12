import Foundation

/// One incremental update from a running conversion. UI consumes
/// these via the `onProgress` closure passed to
/// `Converter.runQuantize` / `runDequantize`, formats them into a
/// progress bar + scrolling log.
public enum ConversionEvent: Sendable {
    /// Pre-flight: source enumerated.
    /// - inputShards: number of `.safetensors` files found in the
    ///   source dir.
    /// - inputBytes: total bytes summed across those shards.
    case discovered(inputShards: Int, inputBytes: UInt64)

    /// Per-tensor or per-shard incremental progress.
    /// - completed / total: numerator and denominator of the
    ///   progress bar (units are "tensors processed" or
    ///   "shards written" depending on phase).
    /// - phase: free-form short tag for the log ("Plan",
    ///   "Quantize", "Pack", "Write").
    /// - note: optional human-readable detail
    ///   (e.g. "layers.7.ffn.experts.42.w1 → I4").
    case progress(completed: Int, total: Int,
                   phase: String, note: String?)

    /// Human-readable log line, no progress implication. Mirrors a
    /// `print()` line from the legacy converter.
    case log(String)

    /// Conversion finished. `outputBytes` is the total size of all
    /// emitted shards (informational).
    case finished(outputBytes: UInt64)
}

/// Convenience wrapper so UI views can store the latest progress
/// in an @Published property without bookkeeping each event
/// separately.
public struct ConversionStatus: Sendable, Equatable {
    public var completed: Int = 0
    public var total: Int = 0
    public var phase: String = ""
    public var lastNote: String = ""
    public var logLines: [String] = []
    public var finishedAt: Date? = nil
    public var inputShards: Int = 0
    public var inputBytes: UInt64 = 0
    public var outputBytes: UInt64 = 0

    public init() {}

    public var fraction: Double {
        guard total > 0 else { return 0 }
        return min(1.0, Double(completed) / Double(total))
    }

    public mutating func apply(_ event: ConversionEvent) {
        switch event {
        case .discovered(let s, let b):
            inputShards = s
            inputBytes = b
        case .progress(let c, let t, let p, let n):
            completed = c; total = t; phase = p
            if let n { lastNote = n }
        case .log(let line):
            logLines.append(line)
            if logLines.count > 500 {
                logLines.removeFirst(logLines.count - 500)
            }
        case .finished(let bytes):
            finishedAt = Date()
            outputBytes = bytes
            completed = total
        }
    }
}

/// Error thrown by `Converter.run*` when cancellation is requested
/// mid-run via the provided cancellation token.
public struct ConversionCancelled: Error, CustomStringConvertible, LocalizedError {
    public var description: String { "Conversion was cancelled." }
    public var errorDescription: String? { description }
    public init() {}
}

/// Cooperative cancellation handle. Caller flips `isCancelled` to
/// true; the running conversion checks it between tensors and
/// throws `ConversionCancelled` at the next safe point.
public final class CancellationToken: @unchecked Sendable {
    private var _flag = false
    private let lock = NSLock()

    public init() {}

    public func cancel() {
        lock.lock(); defer { lock.unlock() }
        _flag = true
    }

    public var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return _flag
    }
}
