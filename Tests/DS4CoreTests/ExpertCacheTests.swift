import XCTest
import Foundation
@testable import DS4Metal

/// Stage D (expert-cache): validates that running the routed MoE matvec over a
/// PACKED buffer of only the K selected experts (ids remapped to 0..<K) gives the
/// EXACT same result as running over the full expert set with the real ids. This
/// is the core of the expert-cache: stream ~K/256 of each layer's expert weight.
final class ExpertCacheTests: XCTestCase {
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
