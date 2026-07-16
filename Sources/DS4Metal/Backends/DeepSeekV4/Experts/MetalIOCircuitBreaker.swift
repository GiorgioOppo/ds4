import Foundation

/// Aggregates MetalIO timings before deciding that the backend is slow.
/// Per-command bandwidth is misleading for the common 1-expert (~7 MB) fill:
/// fixed command-buffer latency can make it look slower than pread even when
/// MetalIO is healthy. Two slow 64 MiB windows are enough to detect a sustained
/// regression without reacting to one small or temporarily delayed batch.
struct MetalIOCircuitBreaker {
    enum Decision {
        case keep
        case slowWindow(gbs: Double, count: Int)
        case disable(gbs: Double)
    }

    let minimumGBs: Double
    let windowBytes: Int
    let slowWindowsRequired: Int
    private(set) var accumulatedBytes = 0
    private(set) var accumulatedSeconds = 0.0
    private(set) var consecutiveSlowWindows = 0

    init(minimumGBs: Double, windowBytes: Int = 64 << 20, slowWindowsRequired: Int = 2) {
        self.minimumGBs = minimumGBs
        self.windowBytes = max(1, windowBytes)
        self.slowWindowsRequired = max(1, slowWindowsRequired)
    }

    mutating func record(bytes: Int, seconds: Double) -> Decision {
        guard bytes > 0, seconds > 0 else { return .keep }
        accumulatedBytes += bytes
        accumulatedSeconds += seconds
        guard accumulatedBytes >= windowBytes else { return .keep }

        let gbs = Double(accumulatedBytes) / accumulatedSeconds / 1e9
        accumulatedBytes = 0
        accumulatedSeconds = 0
        if gbs >= minimumGBs {
            consecutiveSlowWindows = 0
            return .keep
        }
        consecutiveSlowWindows += 1
        if consecutiveSlowWindows >= slowWindowsRequired {
            return .disable(gbs: gbs)
        }
        return .slowWindow(gbs: gbs, count: consecutiveSlowWindows)
    }
}
