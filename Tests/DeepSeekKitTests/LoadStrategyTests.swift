import XCTest
@testable import DeepSeekKit

final class LoadStrategyTests: XCTestCase {

    private func u(_ name: String) -> URL {
        URL(fileURLWithPath: "/tmp/\(name).safetensors")
    }

    /// total ≤ 80% of available → preload.
    func testAutoPicksPreloadWhenComfortablyFits() throws {
        // 1 GB shard, 10 GB available → 1 / 10 = 10%, well under 80%.
        let plan = try LoadPlan.decideForTesting(
            shards: [(u("a"), 1 * 1024 * 1024 * 1024)],
            availableRAM: 10 * 1024 * 1024 * 1024,
            override: nil)
        XCTAssertEqual(plan.strategy, .preload)
        XCTAssertTrue(plan.reason.contains("auto"))
    }

    /// total > 80% of available → mmap.
    func testAutoFallsBackToMmapWhenAboveThreshold() throws {
        // 9 GB total, 10 GB available → 90%, above the 80% cap.
        let plan = try LoadPlan.decideForTesting(
            shards: [
                (u("a"), 5 * 1024 * 1024 * 1024),
                (u("b"), 4 * 1024 * 1024 * 1024),
            ],
            availableRAM: 10 * 1024 * 1024 * 1024,
            override: nil)
        XCTAssertEqual(plan.strategy, .mmap)
        XCTAssertTrue(plan.reason.contains("80%"))
    }

    /// Borderline: total exactly == 80% of available → still preload
    /// (threshold is inclusive on the safe side). Uses forceLoad so
    /// the conservative shard cap (0.7) doesn't intercept first.
    func testAutoExactlyAtThresholdPicksPreload() throws {
        let plan = try LoadPlan.decideForTesting(
            shards: [(u("a"), 8 * 1024 * 1024 * 1024)],
            availableRAM: 10 * 1024 * 1024 * 1024,
            override: nil,
            forceLoad: true)
        XCTAssertEqual(plan.strategy, .preload)
    }

    /// `largest_shard > 0.7 × available` is a hard error regardless of total.
    func testHardErrorWhenShardExceedsConservativeCap() {
        // 12 GB shard, 16 GB available, cap = 0.7 × 16 = 11.2 GB → refuse.
        XCTAssertThrowsError(
            try LoadPlan.decideForTesting(
                shards: [(u("big"), 12 * 1024 * 1024 * 1024)],
                availableRAM: 16 * 1024 * 1024 * 1024,
                override: nil)
        ) { error in
            guard case LoadStrategyError.shardTooLarge(let maxShard, let avail, let cap, let url) = error else {
                XCTFail("expected .shardTooLarge, got \(error)")
                return
            }
            XCTAssertEqual(maxShard, 12 * 1024 * 1024 * 1024)
            XCTAssertEqual(avail, 16 * 1024 * 1024 * 1024)
            XCTAssertEqual(cap, 0.7, accuracy: 1e-9)
            XCTAssertEqual(url.lastPathComponent, "big.safetensors")
        }
    }

    /// Even `--load-strategy preload` doesn't bypass the conservative cap.
    func testHardErrorWinsOverPreloadOverride() {
        XCTAssertThrowsError(
            try LoadPlan.decideForTesting(
                shards: [(u("big"), 12 * 1024 * 1024 * 1024)],
                availableRAM: 16 * 1024 * 1024 * 1024,
                override: "preload")
        ) { error in
            guard case LoadStrategyError.shardTooLarge = error else {
                XCTFail("expected .shardTooLarge, got \(error)")
                return
            }
        }
    }

    /// Even `--load-strategy mmap` doesn't bypass the conservative cap.
    func testHardErrorWinsOverMmapOverride() {
        XCTAssertThrowsError(
            try LoadPlan.decideForTesting(
                shards: [(u("big"), 12 * 1024 * 1024 * 1024)],
                availableRAM: 16 * 1024 * 1024 * 1024,
                override: "mmap")
        )
    }

    /// total > 25× available → .totalTooLarge even when individual shards fit.
    func testHardErrorWhenTotalOversubscriptionTooHigh() {
        // 50 × 4 GB shards = 200 GB total, 4 GB available → 50× oversub.
        let shards = (0..<50).map { (u("s\($0)"), 4 * 1024 * 1024 * 1024 as UInt64) }
        XCTAssertThrowsError(
            try LoadPlan.decideForTesting(
                shards: shards,
                availableRAM: 4 * 1024 * 1024 * 1024,
                override: nil)
        ) { error in
            guard case LoadStrategyError.shardTooLarge = error else {
                // First trip should be shardTooLarge: 4 GB shard > 0.7 × 4 GB = 2.8 GB.
                // The total-cap is checked after; this is fine.
                if case LoadStrategyError.totalTooLarge = error { return }
                XCTFail("expected shardTooLarge or totalTooLarge, got \(error)")
                return
            }
        }
    }

    /// The total-cap fires independently when shards are individually small.
    func testHardErrorWhenTotalOversubscriptionAlone() {
        // 100 × 60 MB shards = ~5.86 GB total, 100 MB available → ~60× oversub.
        // Each shard alone (60 MB) is under the 70% cap (70 MB) so guard 1
        // passes; guard 2 fires because 60× > 25× multiplier.
        let shardBytes: UInt64 = 60 * 1024 * 1024
        let avail: UInt64 = 100 * 1024 * 1024
        let shards = (0..<100).map { (u("s\($0)"), shardBytes) }
        XCTAssertThrowsError(
            try LoadPlan.decideForTesting(
                shards: shards, availableRAM: avail, override: nil)
        ) { error in
            guard case LoadStrategyError.totalTooLarge(let total, let av, let mult) = error else {
                XCTFail("expected .totalTooLarge, got \(error)")
                return
            }
            XCTAssertEqual(total, UInt64(100) * shardBytes)
            XCTAssertEqual(av, avail)
            XCTAssertEqual(mult, 25.0, accuracy: 1e-9)
        }
    }

    /// `--force-load` skips both refusals.
    func testForceLoadBypassesBothGuards() throws {
        // 12 GB shard on 16 GB available — would refuse, but forceLoad=true.
        let plan = try LoadPlan.decideForTesting(
            shards: [(u("big"), 12 * 1024 * 1024 * 1024)],
            availableRAM: 16 * 1024 * 1024 * 1024,
            override: "mmap",
            forceLoad: true)
        XCTAssertEqual(plan.strategy, .mmap)
        XCTAssertEqual(plan.maxShardBytes, 12 * 1024 * 1024 * 1024)
    }

    /// Custom thresholds let advanced callers loosen or tighten.
    func testCustomShardCapFractionLoosens() throws {
        // 12 GB / 16 GB available = 75%. Default 0.7 refuses; 0.8 passes.
        let plan = try LoadPlan.decideForTesting(
            shards: [(u("a"), 12 * 1024 * 1024 * 1024)],
            availableRAM: 16 * 1024 * 1024 * 1024,
            override: "mmap",
            shardCapFraction: 0.8)
        XCTAssertEqual(plan.strategy, .mmap)
    }

    func testOverridePreloadForcesPreloadWhenItFits() throws {
        // 5 GB shard, 10 GB available — 5 < 0.7 × 10 = 7, passes cap.
        let plan = try LoadPlan.decideForTesting(
            shards: [(u("a"), 5 * 1024 * 1024 * 1024)],
            availableRAM: 10 * 1024 * 1024 * 1024,
            override: "preload")
        XCTAssertEqual(plan.strategy, .preload)
        XCTAssertTrue(plan.reason.contains("forced"))
    }

    func testOverrideMmapForcesMmapEvenWhenPreloadWouldFit() throws {
        let plan = try LoadPlan.decideForTesting(
            shards: [(u("a"), 1 * 1024 * 1024 * 1024)],
            availableRAM: 10 * 1024 * 1024 * 1024,
            override: "mmap")
        XCTAssertEqual(plan.strategy, .mmap)
        XCTAssertTrue(plan.reason.contains("forced"))
    }

    func testUnknownOverrideThrows() {
        XCTAssertThrowsError(
            try LoadPlan.decideForTesting(
                shards: [(u("a"), 1)],
                availableRAM: 100,
                override: "swap")
        ) { error in
            guard case LoadStrategyError.unknownOverride(let s) = error else {
                XCTFail("expected .unknownOverride, got \(error)")
                return
            }
            XCTAssertEqual(s, "swap")
        }
    }

    /// `availableRAM == 0` means the probe failed; we fall back to
    /// mmap and skip the hard cap (we can't validate it).
    func testZeroAvailableSkipsHardCapAndPicksMmap() throws {
        let plan = try LoadPlan.decideForTesting(
            shards: [(u("a"), 1 * 1024 * 1024 * 1024)],
            availableRAM: 0,
            override: nil)
        XCTAssertEqual(plan.strategy, .mmap)
        XCTAssertTrue(plan.reason.contains("probe unavailable"))
    }

    /// Aggregates total and max correctly across shards.
    func testTotalAndMaxBookkeeping() throws {
        let plan = try LoadPlan.decideForTesting(
            shards: [(u("a"), 100), (u("b"), 250), (u("c"), 80)],
            availableRAM: 100_000,
            override: nil)
        XCTAssertEqual(plan.totalBytes, 430)
        XCTAssertEqual(plan.maxShardBytes, 250)
    }

    /// `discoverShards` filters out LFS-pointer stubs (< 1 KiB) and
    /// non-`.safetensors` files in the same directory.
    func testDiscoverShardsFiltersLfsPointersAndOtherFiles() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("ds-loadstrategy-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        // Real shard: 4 KiB of zeros.
        let real = tmp.appendingPathComponent("a.safetensors")
        try Data(count: 4096).write(to: real)
        // LFS-pointer stub: < 1 KiB.
        let stub = tmp.appendingPathComponent("b.safetensors")
        try "version https://git-lfs...".data(using: .utf8)!.write(to: stub)
        // Unrelated file.
        let other = tmp.appendingPathComponent("README.txt")
        try Data("hi".utf8).write(to: other)

        let found = try WeightLoader.discoverShards(in: tmp)
        XCTAssertEqual(found.count, 1)
        XCTAssertEqual(found[0].url.lastPathComponent, "a.safetensors")
        XCTAssertEqual(found[0].byteCount, 4096)
    }

    func testSummaryFormatIsParseable() throws {
        let plan = try LoadPlan.decideForTesting(
            shards: [(u("a"), 1 * 1024 * 1024 * 1024)],
            availableRAM: 10 * 1024 * 1024 * 1024,
            override: nil)
        let s = plan.summary()
        XCTAssertTrue(s.contains("strategy: preload"))
        XCTAssertTrue(s.contains("checkpoint: 1 shards"))
        XCTAssertTrue(s.hasSuffix("\n"))
    }
}
