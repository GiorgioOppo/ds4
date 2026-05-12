import Foundation

/// Strategy `WeightLoader` uses to bring each safetensors shard into a
/// `storageModeShared` MTLBuffer:
///   - `.preload`: open + `read(2)` the whole file into a freshly
///     allocated MTLBuffer. Removes per-tensor page faults during
///     inference but spikes RSS by `totalBytes`. Done in parallel.
///   - `.mmap`: `mmap(PROT_READ, MAP_PRIVATE)` + `MTLBuffer(bytesNoCopy:)`.
///     Lower RSS at idle but the OS pages on demand, which under
///     pressure means jetsam. Done sequentially (mmap is just VM mapping,
///     parallelism wouldn't help).
public enum LoadStrategy: String, Sendable {
    case preload
    case mmap
}

/// Output of `LoadPlan.decide`: the strategy plus everything the
/// caller needs to log a useful diagnosis.
public struct LoadPlan: Sendable {
    public let strategy: LoadStrategy
    /// Sorted by lastPathComponent.
    public let shards: [(url: URL, byteCount: UInt64)]
    public let totalBytes: UInt64
    public let maxShardBytes: UInt64
    public let availableRAM: UInt64
    public let physicalRAM: UInt64
    public let mtlWorkingSet: UInt64
    public let cores: Int
    /// One-line "why this strategy" for the log.
    public let reason: String

    /// Multi-line stderr-bound summary. Always ends with `\n`.
    public func summary() -> String {
        let gib = 1024.0 * 1024.0 * 1024.0
        func g(_ b: UInt64) -> String { String(format: "%.2f GB", Double(b) / gib) }
        return """
        system: \(g(physicalRAM)) physical / \(g(availableRAM)) available / \
        \(cores) cores / GPU rec. working-set \(g(mtlWorkingSet))
        checkpoint: \(shards.count) shards, \(g(totalBytes)) total, largest \(g(maxShardBytes))
        strategy: \(strategy.rawValue) (\(reason))

        """
    }
}

public enum LoadStrategyError: Error, CustomStringConvertible, LocalizedError {
    /// The biggest single shard exceeds the process's available RAM.
    /// Re-shard the checkpoint (lower `--shard-size-gb` in the
    /// converter) or run on a host with more memory.
    case shardTooLarge(maxShard: UInt64, available: UInt64, shardURL: URL)
    /// `--load-strategy` got something other than auto/preload/mmap.
    case unknownOverride(String)

    public var description: String {
        let gib = 1024.0 * 1024.0 * 1024.0
        switch self {
        case let .shardTooLarge(maxShard, available, shardURL):
            return """
            largest shard \(shardURL.lastPathComponent) is \
            \(String(format: "%.2f GB", Double(maxShard) / gib)) but only \
            \(String(format: "%.2f GB", Double(available) / gib)) of RAM \
            is available to this process. Free memory (close other apps), \
            re-shard the checkpoint with a smaller --shard-size-gb in the \
            converter, or run on a host with more RAM.
            """
        case let .unknownOverride(s):
            return "unknown --load-strategy value: \(s) (expected auto|preload|mmap)"
        }
    }

    /// Forwarded to `(self as NSError).localizedDescription`, which is
    /// what `main.swift`'s `catch` block prints. Without this
    /// conformance Foundation falls back to the generic
    /// "The operation couldn't be completed. (DeepSeekKit.LoadStrategyError error N.)"
    /// wrapper that hides our human-readable text.
    public var errorDescription: String? { description }
}

extension LoadPlan {
    /// Aggressive preload threshold (numerator/denominator pair to
    /// stay in integer arithmetic): copy everything into RAM if the
    /// total checkpoint size is at most 80% of what's currently free.
    /// User-locked design choice — see plan file.
    static let preloadThresholdNumerator: UInt64 = 4
    static let preloadThresholdDenominator: UInt64 = 5

    /// `total ≤ 0.80 × available`, computed without floats so the
    /// boundary is exactly representable in tests.
    static func totalFitsPreloadCap(total: UInt64, available: UInt64) -> Bool {
        // total * 5 ≤ available * 4  ⇔  total ≤ available * (4/5)
        total.multipliedReportingOverflow(by: preloadThresholdDenominator).0
            <= available.multipliedReportingOverflow(by: preloadThresholdNumerator).0
    }

    /// Picks `.preload` vs `.mmap`, validates the hard-error guard.
    ///
    /// - Parameters:
    ///   - modelDir: directory containing `.safetensors` shards.
    ///   - override: nil/"auto" → automatic; "preload"/"mmap" → forced
    ///     (the hard-error refuse-to-start still applies in both cases).
    public static func decide(modelDir: URL, override: String?) throws -> LoadPlan {
        let shards = try WeightLoader.discoverShards(in: modelDir)
        let total = shards.reduce(0 as UInt64) { $0 + $1.byteCount }
        let maxShard = shards.map(\.byteCount).max() ?? 0
        let avail = SystemProbe.processAvailableRAM()
        let phys = SystemProbe.physicalRAM()
        let mtl = SystemProbe.mtlRecommendedWorkingSet()
        let cores = SystemProbe.cpuCount()

        // Hard guard: the biggest single shard must fit at once, even
        // with mmap. Demand-paging a >RAM shard works for a moment but
        // thrashes during inference and risks jetsam.
        if avail > 0, maxShard > avail {
            let biggest = shards.max(by: { $0.byteCount < $1.byteCount })!.url
            throw LoadStrategyError.shardTooLarge(
                maxShard: maxShard, available: avail, shardURL: biggest)
        }

        let (strategy, reason) = try Self.pickStrategy(
            total: total, available: avail, override: override)

        return LoadPlan(
            strategy: strategy,
            shards: shards.map { ($0.url, $0.byteCount) },
            totalBytes: total,
            maxShardBytes: maxShard,
            availableRAM: avail,
            physicalRAM: phys,
            mtlWorkingSet: mtl,
            cores: cores,
            reason: reason)
    }

    /// Pure-function strategy chooser, shared by `decide` and
    /// `decideForTesting`. Returns the strategy plus the human-readable
    /// reason for the log.
    static func pickStrategy(total: UInt64, available: UInt64,
                              override: String?) throws -> (LoadStrategy, String) {
        switch override?.lowercased() {
        case nil, "", "auto":
            if available == 0 {
                return (.mmap, "auto: available-RAM probe unavailable, defaulting to mmap")
            }
            if totalFitsPreloadCap(total: total, available: available) {
                return (.preload, "auto: total fits under 80% of available RAM")
            }
            return (.mmap, "auto: preload would exceed 80% of available RAM")
        case "preload":
            return (.preload, "forced by --load-strategy")
        case "mmap":
            return (.mmap, "forced by --load-strategy")
        case let other?:
            throw LoadStrategyError.unknownOverride(other)
        }
    }

    /// Test-only constructor: build a plan without touching the disk.
    /// Used by `LoadStrategyTests` to validate the decision matrix.
    public static func decideForTesting(
        shards: [(url: URL, byteCount: UInt64)],
        availableRAM: UInt64,
        override: String? = nil
    ) throws -> LoadPlan {
        let total = shards.reduce(0 as UInt64) { $0 + $1.byteCount }
        let maxShard = shards.map(\.byteCount).max() ?? 0
        if availableRAM > 0, maxShard > availableRAM {
            let biggest = shards.max(by: { $0.byteCount < $1.byteCount })!.url
            throw LoadStrategyError.shardTooLarge(
                maxShard: maxShard, available: availableRAM, shardURL: biggest)
        }
        let (strategy, reason) = try pickStrategy(
            total: total, available: availableRAM, override: override)
        return LoadPlan(
            strategy: strategy, shards: shards,
            totalBytes: total, maxShardBytes: maxShard,
            availableRAM: availableRAM, physicalRAM: 0, mtlWorkingSet: 0,
            cores: 0, reason: reason)
    }
}
