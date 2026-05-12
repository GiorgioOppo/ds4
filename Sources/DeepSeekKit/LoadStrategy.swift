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
    /// Standard mmap; OS pages on demand, eviction is OS-driven.
    case mmap
    /// Mmap + cooperative streaming hints. Same byte-level
    /// representation as `.mmap` (every shard mmapped at load time)
    /// but `WeightLoader.prefetchLayer(K+1)` /
    /// `releaseLayer(K-1)` run between blocks of
    /// `Transformer.forward`, using `madvise(MADV_WILLNEED /
    /// MADV_DONTNEED)` to steer the kernel's LRU. Trades per-token
    /// latency (cold pages need refaulting on the next token) for
    /// a constant-bounded working set that won't freeze the OS on
    /// wildly oversubscribed checkpoints.
    case streaming
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
    ///
    /// `availableRAM` here is the *effective unified-memory budget*
    /// the guards used (`min(processAvailable, physical × 0.6)`),
    /// not raw "free + inactive". On Apple Silicon the GPU and CPU
    /// share the same pages, so the GPU rec. working-set is the
    /// upper bound MTL prefers — NOT additional memory.
    public func summary() -> String {
        let gib = 1024.0 * 1024.0 * 1024.0
        func g(_ b: UInt64) -> String { String(format: "%.2f GB", Double(b) / gib) }
        let ratio = availableRAM == 0 ? 0
            : Double(totalBytes) / Double(availableRAM)
        return """
        system: \(g(physicalRAM)) unified (CPU + GPU share this pool)
                \(g(availableRAM)) effective budget for this process
                \(cores) cores · GPU rec. working-set \(g(mtlWorkingSet)) (same pool)
        checkpoint: \(shards.count) shards, \(g(totalBytes)) total, largest \(g(maxShardBytes))
        oversubscription: \(String(format: "%.1f×", ratio)) of effective budget
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
    /// Retained for API compatibility; no longer thrown by
    /// `decide()` since pickStrategy downgrades to `.streaming`.
    case totalTooLarge(total: UInt64, available: UInt64, multiplier: Double)
    /// Projected KV cache size at the configured `max_seq_len` ×
    /// `max_batch_size` exceeds the unified-memory budget.
    /// Streaming can't help: KV caches are real `storageModeShared`
    /// MTLBuffers (not mmap'd file pages), the GPU writes to them
    /// during forward, so every byte stays resident. Lower
    /// `max_position_embeddings` / `max_batch_size` in config.json.
    case kvCacheTooLarge(projected: UInt64, available: UInt64,
                          maxSeqLen: Int, maxBatchSize: Int)
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
        case let .kvCacheTooLarge(projected, available, maxSeq, maxBatch):
            return """
            projected KV cache is \
            \(String(format: "%.2f GB", Double(projected) / gib)) at \
            max_position_embeddings=\(maxSeq), max_batch_size=\(maxBatch), \
            but only \(String(format: "%.2f GB", Double(available) / gib)) of unified \
            memory is available to this process. KV caches are dense MTLBuffers, \
            not mmap'd file pages — streaming doesn't help and force-load won't \
            either. Edit config.json:
              jq '.max_position_embeddings = 4096 | .max_batch_size = 1' \
                  <model-dir>/config.json > /tmp/c.json && mv /tmp/c.json <model-dir>/config.json
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

    /// Conservative RAM policy applied at pre-flight. Tightened in
    /// the unified-memory-aware revision after a user report that
    /// 25×-oversubscription froze a 16 GB Mac:
    ///   - shardCapFraction = 0.5 leaves a 50% headroom for the OS
    ///     file cache, the GUI compositor, and the GPU's
    ///     command-buffer working set when mmap-ing a hot shard.
    ///   - totalOversubMultiplier = 10 refuses checkpoints more
    ///     than 10× the effective unified-memory budget. Even
    ///     sparse-MoE inference at ~6 GB active footprint thrashes
    ///     above that ratio because every fresh token activates
    ///     slightly different experts and the kernel pages them in
    ///     against the GUI's pages.
    ///   - both are evaluated against
    ///     `SystemProbe.effectiveProcessBudget` (which is
    ///     `min(available, physical × 0.6)`), NOT raw
    ///     `processAvailableRAM`. On Apple Silicon the "available"
    ///     pages include inactive file cache and other apps' working
    ///     sets the kernel doesn't want to evict.
    /// Both bypassed when `forceLoad: true`.
    public static let defaultShardCapFraction: Double = 0.5
    public static let defaultTotalOversubMultiplier: Double = 10.0

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
        // Unified-memory-aware effective budget. On Apple Silicon
        // the GPU and CPU share physical pages, so `processAvailable`
        // alone (free + inactive + speculative) is optimistic.
        let budget = SystemProbe.effectiveProcessBudget()

        if !forceLoad, budget > 0 {
            // Guard 1: largest shard must fit with headroom. Without
            // it the kernel is forced to evict the file cache or the
            // GUI's resident pages on every fresh hot-tensor fault.
            // Streaming doesn't help here — a single shard is still
            // mmap'd contiguously, and the GPU has to read all of it
            // during the layer that owns it.
            let shardCap = UInt64(Double(budget) * shardCapFraction)
            if maxShard > shardCap {
                let biggest = shards.max(by: { $0.byteCount < $1.byteCount })!.url
                throw LoadStrategyError.shardTooLarge(
                    maxShard: maxShard, available: budget,
                    capFraction: shardCapFraction, shardURL: biggest)
            }
            // Guard 2: wildly-oversubscribed total. The
            // pickStrategy below DOWNGRADES this into a `.streaming`
            // strategy instead of refusing, so the model can still
            // load — but only after the user (or auto) accepted
            // that they're past the shard cap is the only hard line.
        }

        let (strategy, reason) = try Self.pickStrategy(
            total: total, available: budget,
            totalOversubMultiplier: totalOversubMultiplier,
            override: override)

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
                              totalOversubMultiplier: Double = defaultTotalOversubMultiplier,
                              override: String?) throws -> (LoadStrategy, String) {
        switch override?.lowercased() {
        case nil, "", "auto":
            if available == 0 {
                return (.mmap, "auto: available-RAM probe unavailable, defaulting to mmap")
            }
            // Wildly oversubscribed → streaming with madvise hints
            // instead of refusing or freezing.
            if Double(total) > Double(available) * totalOversubMultiplier {
                let ratio = Double(total) / Double(available)
                return (.streaming, String(format:
                    "auto: total is %.1f× effective budget (cap %.0f×) — streaming with per-layer madvise hints",
                    ratio, totalOversubMultiplier))
            }
            if totalFitsPreloadCap(total: total, available: available) {
                return (.preload, "auto: total fits under 80% of available RAM")
            }
            return (.mmap, "auto: preload would exceed 80% of available RAM")
        case "preload":
            return (.preload, "forced by --load-strategy")
        case "mmap":
            return (.mmap, "forced by --load-strategy")
        case "streaming":
            return (.streaming, "forced by --load-strategy")
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
            // total-oversub no longer throws — auto picks .streaming
            // when ratio exceeds the multiplier.
        }
        let (strategy, reason) = try pickStrategy(
            total: total, available: availableRAM,
            totalOversubMultiplier: totalOversubMultiplier,
            override: override)
        return LoadPlan(
            strategy: strategy, shards: shards,
            totalBytes: total, maxShardBytes: maxShard,
            availableRAM: availableRAM, physicalRAM: 0, mtlWorkingSet: 0,
            cores: 0, reason: reason)
    }
}
