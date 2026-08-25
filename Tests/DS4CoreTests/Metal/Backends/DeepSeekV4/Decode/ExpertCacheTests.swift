import XCTest
import Foundation
@testable import DS4Metal

/// Stage D (expert-cache): validates that running the routed MoE matvec over a
/// PACKED buffer of only the K selected experts (ids remapped to 0..<K) gives the
/// EXACT same result as running over the full expert set with the real ids. This
/// is the core of the expert-cache: stream ~K/256 of each layer's expert weight.
final class ExpertCacheTests: XCTestCase {
    func testFrequencyAwareVictimProtectsHistoricalHotExpert() {
        let owners: [Int32] = [10, 20, 30]
        let uses: [UInt64] = [1, 5, 9]
        let ranks: [Int32: Int] = [10: 0, 20: 1, 30: 50]

        XCTAssertEqual(
            ExpertSlotCache.chooseVictim(owner: owners, lastUse: uses, hotRank: ranks,
                                         now: 10, lastDemand: 0, speculative: false,
                                         frequencyAware: false),
            0, "pure LRU evicts the oldest slot even when it is historically hottest")
        XCTAssertEqual(
            ExpertSlotCache.chooseVictim(owner: owners, lastUse: uses, hotRank: ranks,
                                         now: 10, lastDemand: 0, speculative: false,
                                         frequencyAware: true),
            2, "hot eviction preserves the hot expert and removes the coldest rank")
    }

    func testVictimPoliciesPreserveSafetyAndPreferFreeSlots() {
        let owners: [Int32] = [10, -1, 30]
        let uses: [UInt64] = [4, 0, 7]
        for frequencyAware in [false, true] {
            XCTAssertEqual(
                ExpertSlotCache.chooseVictim(owner: owners, lastUse: uses, hotRank: [10: 0],
                                             now: 9, lastDemand: 7, speculative: true,
                                             frequencyAware: frequencyAware),
                1)
        }
    }

    func testReuseAwareVictimUsesRecentCadence() {
        let owners: [Int32] = [10, 20, 30]
        let uses: [UInt64] = [10, 12, 14]
        let gaps: [Int32: UInt64] = [10: 2, 20: 20, 30: 3]
        XCTAssertEqual(
            ExpertSlotCache.chooseVictim(owner: owners, lastUse: uses, hotRank: [:],
                                         reuseGap: gaps, now: 15, lastDemand: 0,
                                         speculative: false, frequencyAware: false,
                                         reuseAware: true),
            1, "the expert whose cadence predicts the farthest next reuse is evicted")
        XCTAssertEqual(
            ExpertSlotCache.chooseVictim(owner: owners, lastUse: uses, hotRank: [:],
                                         reuseGap: [10: 2, 20: 20], now: 15,
                                         lastDemand: 0, speculative: false,
                                         frequencyAware: false, reuseAware: true),
            2, "an expert never reused is evicted before one with observed cadence")
    }

    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testPackedExpertsMatchFull() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xEC0
        func nb() -> UInt8 { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return UInt8(truncatingIfNeeded: seed >> 40) }
        func rf() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let inDim = 512, outDim = 128, nExperts = 8
        let rowBytes = (inDim / 256) * 144
        let expertBytes = rowBytes * outDim

        var full = [UInt8](repeating: 0, count: expertBytes * nExperts)
        var off = 0
        for _ in 0..<(nExperts * outDim * (inDim / 256)) {
            let d = Float16(abs(rf()) * 0.05), dmin = Float16(abs(rf()) * 0.02)
            withUnsafeBytes(of: d.bitPattern.littleEndian) { full[off] = $0[0]; full[off+1] = $0[1] }
            withUnsafeBytes(of: dmin.bitPattern.littleEndian) { full[off+2] = $0[0]; full[off+3] = $0[1] }
            for i in 0..<12 { full[off+4+i] = nb() }
            for i in 0..<128 { full[off+16+i] = nb() }
            off += 144
        }
        var activation = [Float](repeating: 0, count: inDim)
        for i in 0..<inDim { activation[i] = rf() }

        let ids: [Int32] = [3, 0, 7, 5, 1, 6]
        let gpuFull = try rt.moeMatvecQ4_K(experts: full, expertIds: ids, activation: activation,
                                           nExperts: nExperts, inDim: inDim, outDim: outDim)

        // Pack only the selected experts (mirrors GGUFWeights.gatherExperts).
        var packed = [UInt8](repeating: 0, count: ids.count * expertBytes)
        for (i, e) in ids.enumerated() {
            let src = Int(e) * expertBytes
            for b in 0..<expertBytes { packed[i * expertBytes + b] = full[src + b] }
        }
        let remapped: [Int32] = Array(0..<Int32(ids.count))
        let gpuPacked = try rt.moeMatvecQ4_K(experts: packed, expertIds: remapped, activation: activation,
                                             nExperts: ids.count, inDim: inDim, outDim: outDim)

        XCTAssertEqual(gpuFull.count, gpuPacked.count)
        for i in 0..<gpuFull.count {
            XCTAssertEqual(gpuFull[i], gpuPacked[i], "packed vs full expert \(i)")
        }
    }

    private final class Flag: @unchecked Sendable { var on = false }
    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0
        func bump() { lock.lock(); value += 1; lock.unlock() }
        func read() -> Int { lock.lock(); defer { lock.unlock() }; return value }
    }

    private final class FillGate: @unchecked Sendable {
        private let lock = NSLock()
        private var blocking = false
        let started = DispatchSemaphore(value: 0)
        let proceed = DispatchSemaphore(value: 0)

        func enable() { lock.lock(); blocking = true; lock.unlock() }
        func disable() { lock.lock(); blocking = false; lock.unlock() }
        func waitIfEnabled() {
            lock.lock(); let shouldBlock = blocking; lock.unlock()
            guard shouldBlock else { return }
            started.signal()
            proceed.wait()
        }
    }

    func testAsyncAcquireExposesAndProtectsResidentSubset() throws {
        let rt = try makeRuntime()
        let gate = FillGate(), expertBytes = 64
        let cache = ExpertSlotCache(
            slotsPerLayer: 8, bytesPerExpert: expertBytes * 3,
            makePool: { slots in
                (gate: try GPUTensor.zerosBytes(rt, byteLength: slots * expertBytes),
                 up: try GPUTensor.zerosBytes(rt, byteLength: slots * expertBytes),
                 down: try GPUTensor.zerosBytes(rt, byteLength: slots * expertBytes))
            },
            fill: { layer, id, pool, slot in
                gate.waitIfEnabled()
                let p = pool.gate.buffer.contents()
                    .advanced(by: pool.gate.byteOffset + slot * expertBytes)
                p.storeBytes(of: Int32(layer * 1000) + id, as: Int32.self)
            })

        _ = try cache.acquire(layer: 2, ids: [1, 2, 3, 4, 5, 6])
        gate.enable()
        let pending = cache.acquireAsync(layer: 2, ids: [1, 2, 3, 4, 5, 7])
        XCTAssertEqual(pending.residentMask, 0b01_1111)
        XCTAssertNotNil(pending.residentPool)
        XCTAssertTrue(pending.residentSlots.prefix(5).allSatisfy { $0 >= 0 })
        XCTAssertEqual(pending.residentSlots[5], -1)
        XCTAssertEqual(gate.started.wait(timeout: .now() + 2), .success,
                       "the missing expert must start off-thread")

        gate.proceed.signal()
        let (pool, slots) = try pending.join()
        XCTAssertTrue(slots.allSatisfy { $0 >= 0 })
        let p = pool.gate.buffer.contents()
            .advanced(by: pool.gate.byteOffset + Int(slots[5]) * expertBytes)
        XCTAssertEqual(p.load(as: Int32.self), 2007)
        XCTAssertEqual(cache.hits, 5)
        XCTAssertEqual(cache.misses, 7)

        // The joined acquire remains leased until its GPU consumer completes:
        // a delayed look-ahead must not write another slot of the same resource.
        gate.disable()
        cache.prefill(layer: 2, ids: [8])
        XCTAssertEqual(cache.prefilled, 0)
        pending.release()
        cache.prefill(layer: 2, ids: [8])
        XCTAssertEqual(cache.prefilled, 1)
    }

    func testSynchronousAcquireLeaseProtectsPoolUntilGPUCompletion() throws {
        let rt = try makeRuntime()
        let expertBytes = 64
        let cache = ExpertSlotCache(
            slotsPerLayer: 8, bytesPerExpert: expertBytes * 3,
            makePool: { slots in
                (gate: try GPUTensor.zerosBytes(rt, byteLength: slots * expertBytes),
                 up: try GPUTensor.zerosBytes(rt, byteLength: slots * expertBytes),
                 down: try GPUTensor.zerosBytes(rt, byteLength: slots * expertBytes))
            },
            fill: { _, id, pool, slot in
                pool.gate.buffer.contents()
                    .advanced(by: pool.gate.byteOffset + slot * expertBytes)
                    .storeBytes(of: id, as: Int32.self)
            })

        let acquired = try cache.acquireLeased(layer: 5, ids: [1, 2, 3, 4, 5, 6])
        XCTAssertTrue(acquired.slots.allSatisfy { $0 >= 0 })
        cache.prefill(layer: 5, ids: [7])
        XCTAssertEqual(cache.prefilled, 0,
                       "look-ahead must not write the pool while Metal may read it")

        acquired.lease.release() // mirrors the command-buffer completion handler
        cache.prefill(layer: 5, ids: [7])
        XCTAssertEqual(cache.prefilled, 1)
    }

    func testDuplicateExpertIdsShareOneSlotAndOneFill() throws {
        let rt = try makeRuntime()
        let expertBytes = 64
        let fills = Counter()
        let cache = ExpertSlotCache(
            slotsPerLayer: 8, bytesPerExpert: expertBytes * 3,
            makePool: { slots in
                (gate: try GPUTensor.zerosBytes(rt, byteLength: slots * expertBytes),
                 up: try GPUTensor.zerosBytes(rt, byteLength: slots * expertBytes),
                 down: try GPUTensor.zerosBytes(rt, byteLength: slots * expertBytes))
            },
            fill: { _, id, pool, slot in
                fills.bump()
                pool.gate.buffer.contents()
                    .advanced(by: pool.gate.byteOffset + slot * expertBytes)
                    .storeBytes(of: id, as: Int32.self)
            })

        _ = try cache.acquire(layer: 1, ids: [1]) // materialize the pool
        cache.prefill(layer: 1, ids: [7, 7, 8, 7])
        XCTAssertEqual(cache.prefilled, 2, "only unique missing experts require I/O")

        let acquired = try cache.acquire(layer: 1, ids: [7, 7, 8, 7])
        XCTAssertEqual(acquired.slots[0], acquired.slots[1])
        XCTAssertEqual(acquired.slots[0], acquired.slots[3])
        XCTAssertNotEqual(acquired.slots[0], acquired.slots[2])
        XCTAssertEqual(fills.read(), 3, "one initial fill plus two unique prefetch fills")
        XCTAssertEqual(cache.misses, 1)
        XCTAssertEqual(cache.hits, 4)
    }

    func testSlotCacheUsesBatchFillWhenAvailable() throws {
        let rt = try makeRuntime()
        let expertBytes = 64
        let singles = Counter(), batches = Counter()
        let cache = ExpertSlotCache(
            slotsPerLayer: 8, bytesPerExpert: expertBytes * 3,
            makePool: { slots in
                (gate: try GPUTensor.zerosBytes(rt, byteLength: slots * expertBytes),
                 up: try GPUTensor.zerosBytes(rt, byteLength: slots * expertBytes),
                 down: try GPUTensor.zerosBytes(rt, byteLength: slots * expertBytes))
            },
            fill: { _, _, _, _ in singles.bump() },
            fillBatch: { layer, pairs, pool in
                batches.bump()
                for pair in pairs {
                    let p = pool.gate.buffer.contents()
                        .advanced(by: pool.gate.byteOffset + pair.slot * expertBytes)
                    p.storeBytes(of: Int32(layer * 1000) + pair.id, as: Int32.self)
                }
            })

        let ids: [Int32] = [2, 4, 6, 8, 10, 12]
        let (pool, slots) = try cache.acquire(layer: 5, ids: ids)
        XCTAssertEqual(batches.read(), 1)
        XCTAssertEqual(singles.read(), 0)
        for (j, id) in ids.enumerated() {
            let p = pool.gate.buffer.contents()
                .advanced(by: pool.gate.byteOffset + Int(slots[j]) * expertBytes)
            XCTAssertEqual(p.load(as: Int32.self), 5000 + id)
        }
    }

    func testSlotCacheLayerAwareGeometryAndByteCounters() throws {
        let rt = try makeRuntime()
        let smallSlab = 64, largeSlab = 128
        func slab(_ layer: Int) -> Int { layer == 0 ? smallSlab : largeSlab }
        let cache = ExpertSlotCache(
            slotsPerLayer: 22,
            bytesPerExpert: smallSlab * 3,
            bytesPerExpertForLayer: { slab($0) * 3 },
            expertStrideForLayer: { slab($0) * 3 },
            supportsLayer: { $0 == 0 || $0 == 1 },
            makePoolForLayer: { layer, slots in
                let b = slab(layer), record = b * 3
                let all = try GPUTensor.zerosBytes(rt, byteLength: slots * record)
                let up = GPUTensor(buffer: all.buffer, byteLength: slots * record - b,
                                   count: slots * record - b, byteOffset: b)
                let down = GPUTensor(buffer: all.buffer, byteLength: slots * record - 2 * b,
                                     count: slots * record - 2 * b, byteOffset: 2 * b)
                return (gate: all, up: up, down: down)
            },
            fill: { layer, id, pool, slot in
                let p = pool.gate.buffer.contents()
                    .advanced(by: pool.gate.byteOffset + slot * pool.bytesPerExpert)
                p.storeBytes(of: Int32(layer * 1000) + id, as: Int32.self)
            },
            slotsPlan: { [0: 9, 1: 8] })

        let ids: [Int32] = [1, 2, 3, 4, 5, 6]
        let (small, _) = try cache.acquire(layer: 0, ids: ids)
        let (large, _) = try cache.acquire(layer: 1, ids: ids)
        _ = try cache.acquire(layer: 1, ids: ids)

        XCTAssertTrue(cache.supports(layer: 1))
        XCTAssertFalse(cache.supports(layer: 2))
        XCTAssertEqual(small.bytesPerExpert, smallSlab * 3)
        XCTAssertEqual(large.bytesPerExpert, largeSlab * 3)
        XCTAssertEqual(small.expertStride, smallSlab * 3)
        XCTAssertEqual(large.expertStride, largeSlab * 3)
        XCTAssertEqual(cache.configuredSlots(layer: 0), 9)
        XCTAssertEqual(cache.configuredSlots(layer: 1), 8)
        XCTAssertEqual(cache.configuredSlots(layer: 2), 8, "partial plans fail closed at the floor")
        XCTAssertEqual(cache.misses, 12)
        XCTAssertEqual(cache.hits, 6)
        XCTAssertEqual(cache.missBytes, 6 * smallSlab * 3 + 6 * largeSlab * 3)
        XCTAssertEqual(cache.hitBytes, 6 * largeSlab * 3)
        XCTAssertEqual(cache.allocatedSlotsByLayer, [0: 9, 1: 8])
        XCTAssertEqual(cache.allocatedBytesByLayer[0], 9 * smallSlab * 3)
        XCTAssertEqual(cache.allocatedBytesByLayer[1], 8 * largeSlab * 3)
    }

    func testSlotCacheAccountsForSynchronousWarmFillSeparately() throws {
        let rt = try makeRuntime()
        let slab = 64
        let cache = ExpertSlotCache(
            slotsPerLayer: 8, bytesPerExpert: slab * 3,
            makePool: { slots in
                (gate: try GPUTensor.zerosBytes(rt, byteLength: slots * slab),
                 up: try GPUTensor.zerosBytes(rt, byteLength: slots * slab),
                 down: try GPUTensor.zerosBytes(rt, byteLength: slots * slab))
            },
            fill: { _, _, _, _ in },
            warm: { _ in [1, 2, 3] })

        _ = try cache.acquire(layer: 0, ids: [1, 4])

        XCTAssertEqual(cache.warmed, 3)
        XCTAssertEqual(cache.warmedBytes, 3 * slab * 3)
        XCTAssertEqual(cache.hits, 1, "warm residency is still a demand hit")
        XCTAssertEqual(cache.misses, 1, "warm fills are not mislabeled as demand misses")
        XCTAssertEqual(cache.prefilled, 0, "synchronous warm fill is not look-ahead")
    }

    /// ExpertSlotCache look-ahead: prefill(ids) fills the pool off the decode
    /// path (counted as `prefilled`, not misses); the demand acquire then
    /// reports HITS with the prefilled bytes in the right slots. A prefill
    /// error is swallowed — the batch's slots stay free and the demand
    /// acquire refills them (and reports the miss).
    func testSlotCachePrefillMakesHits() throws {
        let rt = try makeRuntime()
        let expertBytes = 64
        let fail = Flag()
        let cache = ExpertSlotCache(
            slotsPerLayer: 8, bytesPerExpert: expertBytes * 3,
            makePool: { slots in
                (gate: try GPUTensor.zerosBytes(rt, byteLength: slots * expertBytes),
                 up: try GPUTensor.zerosBytes(rt, byteLength: slots * expertBytes),
                 down: try GPUTensor.zerosBytes(rt, byteLength: slots * expertBytes))
            },
            fill: { layer, id, pool, slot in
                if fail.on { throw GGUFWeights.LoadError.message("synthetic fill failure") }
                let p = pool.gate.buffer.contents()
                    .advanced(by: pool.gate.byteOffset + slot * expertBytes)
                p.storeBytes(of: Int32(layer * 1000) + id, as: Int32.self)
            })
        let ids: [Int32] = [11, 22, 33, 44, 55, 66]
        // Pool creation (and its warm fill) is a DEMAND-path job: a prefill
        // before any acquire on the layer is a no-op.
        cache.prefill(layer: 3, ids: ids)
        XCTAssertEqual(cache.prefilled, 0)
        _ = try cache.acquire(layer: 3, ids: [99])       // creates the pool (1 miss)
        XCTAssertEqual(cache.misses, 1)
        cache.prefill(layer: 3, ids: ids)
        XCTAssertEqual(cache.prefilled, 6)
        let (pool, slots) = try cache.acquire(layer: 3, ids: ids)
        XCTAssertEqual(cache.hits, 6)                    // all prefilled -> all hits
        XCTAssertEqual(cache.misses, 1)                  // unchanged
        for (j, id) in ids.enumerated() {
            let p = pool.gate.buffer.contents()
                .advanced(by: pool.gate.byteOffset + Int(slots[j]) * expertBytes)
            XCTAssertEqual(p.load(as: Int32.self), 3000 + id, "slot content for expert \(id)")
        }
        // Failing prefill: swallowed, slots left free, demand acquire refills.
        _ = try cache.acquire(layer: 4, ids: [1])        // creates layer 4's pool (1 miss)
        fail.on = true
        cache.prefill(layer: 4, ids: [7])                // fill throws -> batch abandoned
        fail.on = false
        XCTAssertEqual(cache.prefilled, 6)               // nothing added
        let (pool4, s4) = try cache.acquire(layer: 4, ids: [7])
        XCTAssertEqual(cache.misses, 3)                  // demand refilled expert 7
        let p4 = pool4.gate.buffer.contents()
            .advanced(by: pool4.gate.byteOffset + Int(s4[0]) * expertBytes)
        XCTAssertEqual(p4.load(as: Int32.self), 4007)
    }
}
