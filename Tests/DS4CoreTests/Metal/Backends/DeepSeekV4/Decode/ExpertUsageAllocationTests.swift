import XCTest
@testable import DS4Metal

/// Pure policy tests: no Metal device or model file is required.
final class ExpertUsageAllocationTests: XCTestCase {
    private func recordUniform(_ usage: ExpertUsageStats, layer: Int, experts: Int) {
        usage.record(layer: layer, ids: (0..<experts).map(Int32.init))
    }

    private func recordCounts(_ counts: [Int], usage: ExpertUsageStats, layer: Int) {
        for (id, count) in counts.enumerated() {
            usage.record(layer: layer, ids: Array(repeating: Int32(id), count: count))
        }
    }

    func testByteBudgetIsRespectedAndBypassLayersAreFiltered() {
        let usage = ExpertUsageStats(nLayers: 3, nExperts: 4)
        recordUniform(usage, layer: 0, experts: 4)
        recordUniform(usage, layer: 1, experts: 4)
        // Layer 2 has plenty of history, but it is deliberately not cacheable.
        for _ in 0..<10 { recordUniform(usage, layer: 2, experts: 4) }

        let bytes = [0: 10, 1: 20, 2: 1]
        let alloc = usage.slotAllocation(budgetBytes: 60,
                                         bytesPerSlot: bytes,
                                         cacheableLayers: [0, 1],
                                         floor: 1, cap: 4)

        XCTAssertEqual(alloc?[0], 4)
        XCTAssertEqual(alloc?[1], 1)
        XCTAssertNil(alloc?[2])
        let used = (alloc ?? [:]).reduce(0) { $0 + $1.value * (bytes[$1.key] ?? 0) }
        XCTAssertLessThanOrEqual(used, 60)
    }

    func testMarginalScoreDoesNotPenalizeLargerRecordsTwice() {
        let usage = ExpertUsageStats(nLayers: 2, nExperts: 4)
        recordCounts([10, 10, 10, 10], usage: usage, layer: 0) // next share 0.25
        recordCounts([40, 30, 20, 10], usage: usage, layer: 1) // next share 0.30

        // After the one-slot floor, only one 20-byte Q4 slot (or two 10-byte
        // IQ2 slots) fits. The Q4 slot wins because its marginal route share is
        // higher; dividing by record size would incorrectly choose layer 0.
        let alloc = usage.slotAllocation(budgetBytes: 50,
                                         bytesPerSlot: [0: 10, 1: 20],
                                         floor: 1, cap: 4)
        XCTAssertEqual(alloc?[0], 1)
        XCTAssertEqual(alloc?[1], 2)
    }

    func testByteBalancedFallbackPreservesLegacyMixedCacheBudget() {
        var bytes: [Int: Int] = [:]
        for layer in 0..<37 { bytes[layer] = 10 }
        for layer in 37..<43 { bytes[layer] = 20 }
        // The old implementation actually allocated only the 37 global-class
        // layers. Turning on six Q4 pools must remain inside that same RAM.
        let budget = 22 * 37 * 10
        let alloc = ExpertUsageStats.byteBalancedSlotAllocation(
            budgetBytes: budget, bytesPerSlot: bytes, floor: 8, cap: 64
        )

        XCTAssertNotNil(alloc)
        let cheap = (0..<37).compactMap { alloc?[$0] }
        let expensive = (37..<43).compactMap { alloc?[$0] }
        XCTAssertGreaterThanOrEqual(cheap.min() ?? 0, 19)
        XCTAssertLessThanOrEqual(cheap.max() ?? .max, 20)
        XCTAssertEqual(Set(expensive), [9])
        let used = (alloc ?? [:]).reduce(0) { $0 + $1.value * (bytes[$1.key] ?? 0) }
        XCTAssertLessThanOrEqual(used, budget)
    }

    func testFixedFloorLayerDoesNotConsumeMarginalBudget() {
        let usage = ExpertUsageStats(nLayers: 2, nExperts: 4)
        recordUniform(usage, layer: 0, experts: 4)
        recordUniform(usage, layer: 1, experts: 4)

        let alloc = usage.slotAllocation(budgetBytes: 60,
                                         bytesPerSlot: [0: 10, 1: 10],
                                         fixedFloorLayers: [0],
                                         floor: 1, cap: 3)

        XCTAssertEqual(alloc?[0], 1)
        XCTAssertEqual(alloc?[1], 3)
    }

    func testBypassHistoryCannotSatisfyCacheableReadiness() {
        let usage = ExpertUsageStats(nLayers: 2, nExperts: 4)
        usage.record(layer: 0, ids: [0])
        for _ in 0..<20 { recordUniform(usage, layer: 1, experts: 4) }

        XCTAssertNil(usage.slotAllocation(budgetBytes: 80,
                                          bytesPerSlot: [0: 10, 1: 10],
                                          cacheableLayers: [0],
                                          floor: 1, cap: 4))
    }

    func testPartialHistoryNeverReturnsAPartialAllLayerPlan() {
        let usage = ExpertUsageStats(nLayers: 2, nExperts: 4)
        recordUniform(usage, layer: 0, experts: 4)
        // Layer 1 is expected/cacheable but has not been observed yet. A map
        // containing only layer 0 would make the cache fall back to its large
        // uniform slot count on layer 1 and violate the byte budget.
        XCTAssertNil(usage.slotAllocation(budgetBytes: 80,
                                          bytesPerSlot: [0: 10, 1: 10],
                                          cacheableLayers: [0, 1],
                                          floor: 1, cap: 4))
    }

    func testByteBudgetMustCoverMandatoryFloor() {
        let usage = ExpertUsageStats(nLayers: 2, nExperts: 4)
        recordUniform(usage, layer: 0, experts: 4)
        recordUniform(usage, layer: 1, experts: 4)

        XCTAssertNil(usage.slotAllocation(budgetBytes: 59,
                                          bytesPerSlot: [0: 10, 1: 20],
                                          floor: 2, cap: 4))
    }

    func testLegacySlotAllocatorPreservesTotalSlotCount() {
        let usage = ExpertUsageStats(nLayers: 2, nExperts: 4)
        for _ in 0..<2 {
            recordUniform(usage, layer: 0, experts: 4)
            recordUniform(usage, layer: 1, experts: 4)
        }

        let alloc = usage.slotAllocation(base: 3, floor: 1, cap: 4)
        XCTAssertEqual(alloc?.values.reduce(0, +), 6)
        XCTAssertGreaterThanOrEqual(alloc?[0] ?? 0, 1)
        XCTAssertGreaterThanOrEqual(alloc?[1] ?? 0, 1)
    }
}
