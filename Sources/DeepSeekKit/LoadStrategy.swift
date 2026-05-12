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
    /// The biggest single shard exceeds the conservative shard cap
    /// (70% of available RAM by default). Refusing here leaves
    /// headroom for the OS file cache and other processes; without it
    /// page-faulting during inference can starve the system.
    case shardTooLarge(maxShard: UInt64, available: UInt64,
                        capFraction: Double, shardURL: URL)
    /// Total checkpoint dwarfs available RAM by more than the
    /// oversubscription multiplier (25× by default). Even MoE models
    /// with sparse activation can grind a small-RAM host to a halt
    /// when the working set is dozens of times what's resident.
    case totalTooLarge(total: UInt64, available: UInt64, multiplier: Double)
    /// `--load-strategy` got something other than auto/preload/mmap.
    case unknownOverride(String)

    public var description: String {
        let gib = 1024.0 * 1024.0 * 1024.0
        switch self {
        case let .shardTooLarge(maxShard, available, capFraction, shardURL):
            let capGB = Double(available) / gib * capFraction
            return """
            largest shard \(shardURL.lastPathComponent) is \
            \(String(format: "%.2f GB", Double(maxShard) / gib)) which exceeds the \
            conservative cap of \(String(format: "%.2f GB", capGB)) \
            (\(String(format: "%.0f", capFraction * 100))% of \
            \(String(format: "%.2f GB", Double(available) / gib)) available). \
            Free memory (close other apps, run `sudo purge`), re-shard the \
            checkpoint with a smaller --shard-size-gb in the converter, \
            or pass --force-load to bypass.
            """
        case let .totalTooLarge(total, available, multiplier):
            let totalGB = Double(total) / gib
            let availGB = Double(available) / gib
            return """
            checkpoint is \(String(format: "%.2f GB", totalGB)) but only \
            \(String(format: "%.2f GB", availGB)) of RAM is available — \
            oversubscription ratio \(String(format: "%.1fx", totalGB / availGB)) \
            exceeds the safety multiplier \
            (\(String(format: "%.0fx", multiplier))). Mmap'ing a checkpoint \
            this much larger than RAM tends to thrash the system. Run on a \
            host with more RAM, re-quantize to a smaller dtype, or pass \
            --force-load if you accept the risk.
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

    /// Conservative RAM policy applied at pre-flight. Defaults are
    /// tuned for "don't lock up a small-RAM Mac under load":
    ///   - shardCapFraction = 0.7 leaves a 30% headroom for the OS
    ///     file cache and other processes when mmap-ing a hot shard.
    ///   - totalOversubMultiplier = 25 refuses checkpoints that are
    ///     more than 25× available RAM — even sparse MoE inference
    ///     tends to thrash above that ratio.
    /// Both are bypassed when `forceLoad: true` (the `--force-load`
    /// CLI flag).
    public static let defaultShardCapFraction: Double = 0.7
    public static let defaultTotalOversubMultiplier: Double = 25.0

    /// `total ≤ 0.80 × available`, computed without floats so the
    /// boundary is exactly representable in tests.
    static func totalFitsPreloadCap(total: UInt64, available: UInt64) -> Bool {
        // total * 5 ≤ available * 4  ⇔  total ≤ available * (4/5)
        total.multipliedReportingOverflow(by: preloadThresholdDenominator).0
            <= available.multipliedReportingOverflow(by: preloadThresholdNumerator).0
    }

    /// Picks `.preload` vs `.mmap`, validates the conservative
    /// pre-flight guards.
    ///
    /// - Parameters:
    ///   - modelDir: directory containing `.safetensors` shards.
    ///   - override: nil/"auto" → automatic; "preload"/"mmap" → forced.
    ///   - forceLoad: bypass the RAM-safety refusals (shardTooLarge,
    ///     totalTooLarge). The strategy decision still runs; only the
    ///     two refusal guards are skipped.
    ///   - shardCapFraction: max shard size as a fraction of available
    ///     RAM. Default 0.7; pass 1.0 to match the pre-conservative
    ///     behaviour. Ignored when forceLoad=true.
    ///   - totalOversubMultiplier: cap on `total/available`. Default
    ///     25.0; pass .infinity to disable.
    public static func decide(modelDir: URL,
                               override: String?,
                               forceLoad: Bool = false,
                               shardCapFraction: Double = defaultShardCapFraction,
                               totalOversubMultiplier: Double = defaultTotalOversubMultiplier
    ) throws -> LoadPlan {
        let shards = try WeightLoader.discoverShards(in: modelDir)
        let total = shards.reduce(0 as UInt64) { $0 + $1.byteCount }
        let maxShard = shards.map(\.byteCount).max() ?? 0
        let avail = SystemProbe.processAvailableRAM()
        let phys = SystemProbe.physicalRAM()
        let mtl = SystemProbe.mtlRecommendedWorkingSet()
        let cores = SystemProbe.cpuCount()

        if !forceLoad, avail > 0 {
            // Guard 1: largest shard must fit with headroom. Without
            // it the kernel can be forced to evict the file cache (or
            // worse, page anonymous memory to swap) at every fault on
            // a hot tensor.
            let shardCap = UInt64(Double(avail) * shardCapFraction)
            if maxShard > shardCap {
                let biggest = shards.max(by: { $0.byteCount < $1.byteCount })!.url
                throw LoadStrategyError.shardTooLarge(
                    maxShard: maxShard, available: avail,
                    capFraction: shardCapFraction, shardURL: biggest)
            }
            // Guard 2: catch the wildly-oversubscribed case (e.g.
            // 277 GB INT8 V4-Flash on a 16 GB Mac with 7 GB free →
            // ratio 39×, well above 25×). The shard cap alone would
            // pass these because individual shards are small, but the
            // working set across a forward still thrashes.
            if Double(total) > Double(avail) * totalOversubMultiplier {
                throw LoadStrategyError.totalTooLarge(
                    total: total, available: avail,
                    multiplier: totalOversubMultiplier)
            }
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
        override: String? = nil,
        forceLoad: Bool = false,
        shardCapFraction: Double = defaultShardCapFraction,
        totalOversubMultiplier: Double = defaultTotalOversubMultiplier
    ) throws -> LoadPlan {
        let total = shards.reduce(0 as UInt64) { $0 + $1.byteCount }
        let maxShard = shards.map(\.byteCount).max() ?? 0
        if !forceLoad, availableRAM > 0 {
            let shardCap = UInt64(Double(availableRAM) * shardCapFraction)
            if maxShard > shardCap {
                let biggest = shards.max(by: { $0.byteCount < $1.byteCount })!.url
                throw LoadStrategyError.shardTooLarge(
                    maxShard: maxShard, available: availableRAM,
                    capFraction: shardCapFraction, shardURL: biggest)
            }
            if Double(total) > Double(availableRAM) * totalOversubMultiplier {
                throw LoadStrategyError.totalTooLarge(
                    total: total, available: availableRAM,
                    multiplier: totalOversubMultiplier)
            }
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
