import XCTest
import DS4Core
@testable import DS4Metal

/// The slot cache's contract: a hit serves byte-identical records to a fresh
/// read (logits invariance by construction), eviction is LRU, the current
/// batch is pinned against itself, and undersized budgets are refused at
/// creation. All fixtures are synthetic pattern files — no device, no model.
final class GLM52ExpertSlotCacheTests: XCTestCase {
    private let expertCount: UInt64 = 16

    private func patternByte(_ i: Int) -> UInt8 {
        UInt8(truncatingIfNeeded: i &* 31 &+ (i >> 8) &+ 7)
    }

    private func descriptor(name: String, type: UInt32, dims: [UInt64],
                            offset: UInt64) -> GLM52WeightDescriptor {
        GLM52WeightDescriptor(
            name: name, type: type, dims: dims, absOffset: offset,
            bytes: GGUF.tensorNBytes(type: type, elements: dims.reduce(1, *))!)
    }

    private func routedWeights() -> GLM52RoutedExpertWeights {
        let gate = descriptor(name: "blk.3.ffn_gate_exps.weight",
                              type: GLM52TensorSchema.q4_K,
                              dims: [256, 256, expertCount], offset: 4_096)
        let up = descriptor(name: "blk.3.ffn_up_exps.weight",
                            type: GLM52TensorSchema.q4_K,
                            dims: [256, 256, expertCount],
                            offset: gate.absOffset + gate.bytes + 4_096)
        let down = descriptor(name: "blk.3.ffn_down_exps.weight",
                              type: GLM52TensorSchema.q6_K,
                              dims: [256, 256, expertCount],
                              offset: up.absOffset + up.bytes + 4_096)
        return GLM52RoutedExpertWeights(gate: gate, up: up, down: down)
    }

    private func makeFixture() throws -> (reader: GLM52PayloadReader,
                                          planner: GLM52ExpertStreamPlanner) {
        let weights = routedWeights()
        let byteCount = Int(weights.down.absOffset + weights.down.bytes)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("glm52-slotcache-\(UUID().uuidString).bin")
        var data = Data(count: byteCount)
        data.withUnsafeMutableBytes { raw in
            for i in 0..<byteCount { raw[i] = patternByte(i) }
        }
        try data.write(to: url)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: url)
        }
        return (try GLM52PayloadReader(path: url.path),
                try GLM52ExpertStreamPlanner(layer: 3, weights: routedWeights()))
    }

    private func freshRecords(_ reader: GLM52PayloadReader,
                              _ plan: GLM52ExpertStreamPlan) throws
        -> (packed: [UInt8], layout: GLM52ExpertPackedRecordLayout) {
        let layout = try reader.packedLayout(of: plan)
        var packed = [UInt8](repeating: 0, count: layout.totalBytes)
        _ = try packed.withUnsafeMutableBytes {
            try reader.read(plan: plan, into: $0)
        }
        return (packed, layout)
    }

    func testHitsServeByteIdenticalRecords() throws {
        let (reader, planner) = try makeFixture()
        let selection: [UInt32] = [7, 0, 15, 3, 8, 1, 14, 2]
        let plan = try planner.plan(selectedExperts: selection)
        let (fresh, freshLayout) = try freshRecords(reader, plan)
        let cache = try GLM52ExpertSlotCache(
            reader: reader, slotCount: 8,
            slotBytes: freshLayout.recordBytes)

        for pass in 0..<2 {
            try cache.withRecords(plan: plan) { records, layout in
                XCTAssertEqual(records.count, selection.count)
                XCTAssertEqual(layout.recordBytes, freshLayout.recordBytes)
                for (rank, record) in records.enumerated() {
                    let start = freshLayout.recordOffset(rank: rank)
                    XCTAssertEqual(
                        Array(record),
                        Array(fresh[start..<start + layout.recordBytes]),
                        "pass \(pass) rank \(rank) bytes diverge")
                }
            }
        }
        XCTAssertEqual(cache.stats.misses, 8)
        XCTAssertEqual(cache.stats.hits, 8)
        XCTAssertEqual(cache.stats.evictions, 0)
    }

    func testLRUEvictionAndRefill() throws {
        let (reader, planner) = try makeFixture()
        let planA = try planner.plan(selectedExperts: [0, 1, 2, 3, 4, 5, 6, 7])
        let planB = try planner.plan(selectedExperts: [8, 9, 10, 11, 12, 13, 14, 15])
        let layout = try reader.packedLayout(of: planA)
        let cache = try GLM52ExpertSlotCache(
            reader: reader, slotCount: 8, slotBytes: layout.recordBytes)

        try cache.withRecords(plan: planA) { _, _ in }
        try cache.withRecords(plan: planB) { _, _ in }   // evicts every A slot
        try cache.withRecords(plan: planA) { _, _ in }   // misses again

        XCTAssertEqual(cache.stats.misses, 24)
        XCTAssertEqual(cache.stats.hits, 0)
        XCTAssertEqual(cache.stats.evictions, 16)
    }

    func testPartialOverlapKeepsRecentExperts() throws {
        let (reader, planner) = try makeFixture()
        let layout = try reader.packedLayout(
            of: try planner.plan(selectedExperts: [0, 1, 2, 3, 4, 5, 6, 7]))
        // 12 slots: batch B (8) evicts only the 4 least-recent A experts.
        let cache = try GLM52ExpertSlotCache(
            reader: reader, slotCount: 12, slotBytes: layout.recordBytes)

        try cache.withRecords(
            plan: planner.plan(selectedExperts: [0, 1, 2, 3, 4, 5, 6, 7])) { _, _ in }
        try cache.withRecords(
            plan: planner.plan(selectedExperts: [8, 9, 10, 11, 12, 13, 14, 15])) { _, _ in }
        XCTAssertEqual(cache.stats.evictions, 4)

        // Experts 4..7 were touched later than 0..3, so they survived.
        try cache.withRecords(
            plan: planner.plan(selectedExperts: [4, 5, 6, 7, 8, 9, 10, 11])) { _, _ in }
        XCTAssertEqual(cache.stats.hits, 8)
        XCTAssertEqual(cache.stats.misses, 16)
    }

    func testBudgetFloorAndBatchGuards() throws {
        let (reader, planner) = try makeFixture()
        let plan = try planner.plan(selectedExperts: [0, 1, 2, 3, 4, 5, 6, 7])
        let layout = try reader.packedLayout(of: plan)

        XCTAssertThrowsError(try GLM52ExpertSlotCache(
            reader: reader, slotCount: 7, slotBytes: layout.recordBytes)) {
            XCTAssertEqual($0 as? GLM52ExpertSlotCacheError,
                           .budgetBelowSelectionWorkingSet(slots: 7, required: 8))
        }
        XCTAssertThrowsError(try GLM52ExpertSlotCache(
            reader: reader,
            byteBudget: 7 * layout.recordBytes,
            slotBytes: layout.recordBytes)) {
            XCTAssertEqual($0 as? GLM52ExpertSlotCacheError,
                           .budgetBelowSelectionWorkingSet(slots: 7, required: 8))
        }

        // A cache sized for a narrower selection refuses a wider batch.
        let narrow = try GLM52ExpertSlotCache(
            reader: reader, slotCount: 4, slotBytes: layout.recordBytes,
            selectionWidth: 4)
        XCTAssertThrowsError(try narrow.withRecords(plan: plan) { _, _ in }) {
            XCTAssertEqual($0 as? GLM52ExpertSlotCacheError,
                           .batchLargerThanCache(batch: 8, slots: 4))
        }

        // A slot smaller than the record is refused before any read.
        let tiny = try GLM52ExpertSlotCache(
            reader: reader, slotCount: 8, slotBytes: 16)
        XCTAssertThrowsError(try tiny.withRecords(plan: plan) { _, _ in }) {
            XCTAssertEqual($0 as? GLM52ExpertSlotCacheError,
                           .recordLargerThanSlot(needed: layout.recordBytes,
                                                 slotBytes: 16))
        }
    }
}
