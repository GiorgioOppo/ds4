import Foundation

/// One incremental update emitted by `FineTuner.run`. UI consumers
/// keep the latest snapshot of `FineTuneStatus` and render a progress
/// bar + scrolling log on top of these events.
public enum FineTuneEvent: Sendable {
    /// Pre-flight: dataset enumerated, base model validated.
    /// - examples: number of training examples after tokenization.
    /// - bytes: total size of the dataset on disk.
    case discovered(examples: Int, bytes: UInt64)

    /// Per-step optimizer update. Drives the live progress bar.
    /// - step: current optimizer step (1-indexed).
    /// - total: total expected steps across all epochs.
    /// - epoch: 1-indexed epoch the step belongs to.
    /// - loss: training loss at this step.
    /// - learningRate: effective LR after schedule + warmup.
    case step(step: Int, total: Int, epoch: Int,
               loss: Double, learningRate: Double)

    /// Eval pass at the end of an epoch (or every N steps if
    /// configured).
    case eval(step: Int, loss: Double, perplexity: Double)

    /// Intermediate checkpoint persisted to disk.
    case checkpointSaved(path: URL, step: Int)

    /// Human-readable log line, no progress implication. Mirrors a
    /// `print()` from the underlying runner.
    case log(String)

    /// Training finished. `outputBytes` sums all shards in the output
    /// dir (informational).
    case finished(outputBytes: UInt64)
}

/// Snapshot of the latest progress, suitable for binding to a SwiftUI
/// view. Apply incoming events with `apply(_:)` to keep it up to
/// date without bookkeeping each variant separately.
public struct FineTuneStatus: Sendable, Equatable {
    public var step: Int = 0
    public var totalSteps: Int = 0
    public var currentEpoch: Int = 0
    public var lastLoss: Double = 0
    public var lastEvalLoss: Double? = nil
    public var lastPerplexity: Double? = nil
    public var lastLearningRate: Double = 0
    public var examples: Int = 0
    public var inputBytes: UInt64 = 0
    public var outputBytes: UInt64 = 0
    public var checkpoints: [URL] = []
    public var logLines: [String] = []
    public var finishedAt: Date? = nil

    public init() {}

    public var fraction: Double {
        guard totalSteps > 0 else { return 0 }
        return min(1.0, Double(step) / Double(totalSteps))
    }

    public mutating func apply(_ event: FineTuneEvent) {
        switch event {
        case .discovered(let n, let b):
            examples = n
            inputBytes = b
        case .step(let s, let t, let e, let l, let lr):
            step = s
            totalSteps = t
            currentEpoch = e
            lastLoss = l
            lastLearningRate = lr
        case .eval(_, let l, let ppl):
            lastEvalLoss = l
            lastPerplexity = ppl
        case .checkpointSaved(let url, _):
            checkpoints.append(url)
        case .log(let line):
            logLines.append(line)
            if logLines.count > 1000 {
                logLines.removeFirst(logLines.count - 1000)
            }
        case .finished(let bytes):
            outputBytes = bytes
            finishedAt = Date()
            step = totalSteps
        }
    }
}

/// Thrown by `FineTuner.run` when the caller requests cancellation
/// mid-run. The runner checks the token between optimizer steps and
/// throws at the next safe point.
public struct FineTuneCancelled: Error, CustomStringConvertible, LocalizedError {
    public var description: String { "Fine-tuning was cancelled." }
    public var errorDescription: String? { description }
    public init() {}
}

/// Thrown by the stub runner when no native training backend is
/// available. The UI surfaces this as a friendly explanation rather
/// than a generic error toast.
public struct FineTuneNotImplemented: Error, CustomStringConvertible, LocalizedError {
    public let detail: String
    public var description: String { detail }
    public var errorDescription: String? { detail }
    public init(_ detail: String) { self.detail = detail }
}

/// Cooperative cancellation handle. The caller flips `isCancelled` to
/// true; the running trainer checks it between steps and throws
/// `FineTuneCancelled` at the next safe point. Mirrors the
/// `CancellationToken` in DeepSeekConverter — kept separate to avoid
/// a target-level coupling between the converter and the trainer.
public final class TrainingCancellationToken: @unchecked Sendable {
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
