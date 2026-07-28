import XCTest
import Foundation
import DS4Core
@testable import DS4Metal

/// GPU/CPU parity for the Laguna decode kernels: every dispatch is judged
/// against the scalar oracles in `Reference/`. Skipped where Metal is
/// unavailable.
final class LagunaKernelsTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    /// Deterministic pseudo-random floats in roughly [-1, 1].
    private struct Generator {
        var seed: UInt64
        mutating func next() -> Float {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(Int32.max)
        }
        mutating func vector(_ count: Int) -> [Float] {
            (0..<count).map { _ in next() }
        }
    }

    /// Small head counts on the production 128-dimension heads: the decode
    /// attention kernel is specialized for headDim 128.
    private func fullSpec(cacheCapacity: Int = 64) -> LagunaAttentionSpec {
        LagunaAttentionSpec(
            headCount: 2, kvHeadCount: 1, headDim: 128, rotationDims: 64,
            cacheCapacity: cacheCapacity,
            ropeFrequencyBase: 500_000, ropeFrequencyScale: 1 / 32,
            extrapolationFactor: 1, attentionFactor: 1,
            betaFast: 32, betaSlow: 1, ropeOriginalContext: 8_192,
            rmsEpsilon: 1e-6
        )
    }

    private func slidingSpec(cacheCapacity: Int = 4) -> LagunaAttentionSpec {
        LagunaAttentionSpec(
            headCount: 4, kvHeadCount: 2, headDim: 128, rotationDims: 128,
            cacheCapacity: cacheCapacity,
            ropeFrequencyBase: 10_000, ropeFrequencyScale: 1,
            extrapolationFactor: 0, attentionFactor: 1,
            betaFast: 0, betaSlow: 0, ropeOriginalContext: 262_144,
            rmsEpsilon: 1e-6
        )
    }

    private func assertClose(_ got: [Float], _ expected: [Float],
                             _ label: String,
                             tolerance: Float = 2e-3) {
        XCTAssertEqual(got.count, expected.count, label)
        for i in 0..<min(got.count, expected.count) {
            XCTAssertEqual(got[i], expected[i],
                           accuracy: tolerance + abs(expected[i]) * tolerance,
                           "\(label)[\(i)]")
        }
    }

    // MARK: Norm + RoPE

    func testQKHeadRMSNormRopeMatchesTheOracleOnBothBlockKinds() throws {
        let runtime = try makeRuntime()
        var generator = Generator(seed: 0x4C41_4755_4E41_0001)

        for (name, spec) in [("full", fullSpec()), ("sliding", slidingSpec())] {
            for position in [0, 77] {
                let query = generator.vector(spec.queryWidth)
                let key = generator.vector(spec.keyValueWidth)
                let queryWeight = generator.vector(spec.headDim).map { $0 + 1.5 }
                let keyWeight = generator.vector(spec.headDim).map { $0 + 1.5 }

                let gpu = try runtime.lagunaQKHeadRMSNormRope(
                    query: query, key: key,
                    queryWeight: queryWeight, keyWeight: keyWeight,
                    spec: spec, position: position
                )
                let cpuQuery = try LagunaLayerReference.projectionHeadRMSNormRope(
                    query, weight: queryWeight, headCount: spec.headCount,
                    spec: spec, position: position
                )
                let cpuKey = try LagunaLayerReference.projectionHeadRMSNormRope(
                    key, weight: keyWeight, headCount: spec.kvHeadCount,
                    spec: spec, position: position
                )
                assertClose(gpu.query, cpuQuery, "\(name)@\(position).q")
                assertClose(gpu.key, cpuKey, "\(name)@\(position).k")
            }
        }
    }

    // MARK: KV store

    func testStoreKVWritesF16RowsIntoTheRing() throws {
        let runtime = try makeRuntime()
        let spec = slidingSpec(cacheCapacity: 4)
        let cache = try runtime.lagunaKVCache(capacity: spec.cacheCapacity,
                                              rowWidth: spec.keyValueWidth)
        var generator = Generator(seed: 0x4C41_4755_4E41_0002)
        var rows: [[Float]] = []
        for position in 0...5 {
            let row = generator.vector(spec.keyValueWidth)
            rows.append(row)
            try runtime.lagunaStoreKV(keyRow: row, valueRow: row,
                                      cache: cache, position: position)
        }

        let bits = cache.keyBits()
        // With capacity 4, rows 0 and 1 now hold positions 4 and 5.
        for (slot, position) in [(0, 4), (1, 5), (2, 2), (3, 3)] {
            for column in 0..<spec.keyValueWidth {
                XCTAssertEqual(
                    bits[slot * spec.keyValueWidth + column],
                    Half.bits(rows[position][column]),
                    "slot \(slot) column \(column)"
                )
            }
        }
    }

    // MARK: Decode attention

    private func attentionParity(spec: LagunaAttentionSpec, positions: Int,
                                 seed: UInt64, label: String) throws {
        let runtime = try makeRuntime()
        let gpuCache = try runtime.lagunaKVCache(capacity: spec.cacheCapacity,
                                                 rowWidth: spec.keyValueWidth)
        let cpuCache = LagunaReferenceKVCache(capacity: spec.cacheCapacity,
                                              rowWidth: spec.keyValueWidth)
        var generator = Generator(seed: seed)
        for position in 0..<positions {
            let keyRow = generator.vector(spec.keyValueWidth)
            let valueRow = generator.vector(spec.keyValueWidth)
            try runtime.lagunaStoreKV(keyRow: keyRow, valueRow: valueRow,
                                      cache: gpuCache, position: position)
            cpuCache.store(position: position, keyRow: keyRow,
                           valueRow: valueRow)
        }

        let position = positions - 1
        let query = generator.vector(spec.queryWidth)
        let gate = generator.vector(spec.headCount).map { $0 * 3 }
        let gpu = try runtime.lagunaAttentionDecode(
            query: query, gate: gate, cache: gpuCache,
            position: position, spec: spec
        )
        let cpu = try LagunaLayerReference.attend(
            query: query, gate: gate, cache: cpuCache,
            position: position, spec: spec
        )
        assertClose(gpu, cpu, label)
    }

    func testDecodeAttentionMatchesTheOracle() throws {
        try attentionParity(spec: fullSpec(cacheCapacity: 64), positions: 9,
                            seed: 0x4C41_4755_4E41_0003, label: "short")
    }

    func testDecodeAttentionMatchesTheOracleAfterTheRingWraps() throws {
        try attentionParity(spec: slidingSpec(cacheCapacity: 4), positions: 7,
                            seed: 0x4C41_4755_4E41_0004, label: "wrapped")
    }

    func testDecodeAttentionSplitPathMatchesTheOracleOnLongWindows() throws {
        // key_count > 256 activates the eight-way split reduction.
        try attentionParity(spec: fullSpec(cacheCapacity: 512), positions: 300,
                            seed: 0x4C41_4755_4E41_0005, label: "split")
    }
}
