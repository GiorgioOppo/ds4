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

    private func q8Row(_ values: [Int8]) -> [UInt8] {
        precondition(values.count.isMultiple(of: 32))
        var bytes: [UInt8] = []
        bytes.reserveCapacity(values.count / 32 * 34)
        for block in stride(from: 0, to: values.count, by: 32) {
            withUnsafeBytes(of: Half.bits(1).littleEndian) {
                bytes.append(contentsOf: $0)
            }
            for value in values[block..<block + 32] {
                bytes.append(UInt8(bitPattern: value))
            }
        }
        return bytes
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

    func testQKHeadRMSNormRopeBatchMatchesPerTokenOracle() throws {
        let runtime = try makeRuntime()
        var generator = Generator(seed: 0x4C41_4755_4E41_000F)
        for (name, spec) in [
            ("full", fullSpec()),
            ("sliding", slidingSpec()),
        ] {
            let count = 3
            let position = 73
            let queries = (0..<count).map { _ in
                generator.vector(spec.queryWidth)
            }
            let keys = (0..<count).map { _ in
                generator.vector(spec.keyValueWidth)
            }
            let queryWeight = generator.vector(spec.headDim).map { $0 + 1.5 }
            let keyWeight = generator.vector(spec.headDim).map { $0 + 1.5 }
            let gpu = try runtime.lagunaQKHeadRMSNormRopeBatch(
                queries: queries, keys: keys,
                queryWeight: queryWeight, keyWeight: keyWeight,
                spec: spec, position: position)

            for row in 0..<count {
                let expectedQuery =
                    try LagunaLayerReference.projectionHeadRMSNormRope(
                        queries[row], weight: queryWeight,
                        headCount: spec.headCount, spec: spec,
                        position: position + row)
                let expectedKey =
                    try LagunaLayerReference.projectionHeadRMSNormRope(
                        keys[row], weight: keyWeight,
                        headCount: spec.kvHeadCount, spec: spec,
                        position: position + row)
                assertClose(
                    gpu.queries[row], expectedQuery,
                    "\(name).batch\(row).q")
                assertClose(
                    gpu.keys[row], expectedKey,
                    "\(name).batch\(row).k")
            }
        }
    }

    func testFusedRopeStoreVariantsMatchOracleAndWriteTheRing() throws {
        let runtime = try makeRuntime()
        var generator = Generator(seed: 0x4C41_4755_4E41_0013)
        for (name, spec) in [
            ("full", fullSpec(cacheCapacity: 8)),
            ("sliding", slidingSpec(cacheCapacity: 4)),
        ] {
            let position = spec.cacheCapacity + 2
            let query = generator.vector(spec.queryWidth)
            let key = generator.vector(spec.keyValueWidth)
            let value = generator.vector(spec.keyValueWidth)
            let queryWeight = generator.vector(spec.headDim).map { $0 + 1.5 }
            let keyWeight = generator.vector(spec.headDim).map { $0 + 1.5 }
            let expectedQuery =
                try LagunaLayerReference.projectionHeadRMSNormRope(
                    query, weight: queryWeight,
                    headCount: spec.headCount, spec: spec,
                    position: position)
            let expectedKey =
                try LagunaLayerReference.projectionHeadRMSNormRope(
                    key, weight: keyWeight,
                    headCount: spec.kvHeadCount, spec: spec,
                    position: position)
            for simd in [false, true] {
                let mode = simd ? "simd" : "legacy"
                let cache = try runtime.lagunaKVCache(
                    capacity: spec.cacheCapacity,
                    rowWidth: spec.keyValueWidth)
                let got = try runtime.lagunaQKHeadRMSNormRopeStore(
                    query: query, key: key, value: value,
                    queryWeight: queryWeight, keyWeight: keyWeight,
                    spec: spec, cache: cache, position: position,
                    simd: simd)
                assertClose(
                    got.query, expectedQuery, "\(name).\(mode).q")
                assertClose(
                    got.key, expectedKey, "\(name).\(mode).k")

                let row = position % spec.cacheCapacity
                let keyBits = cache.keyBits()
                let valueBits = cache.valueBits()
                for column in 0..<spec.keyValueWidth {
                    let index = row * spec.keyValueWidth + column
                    XCTAssertEqual(
                        keyBits[index], Half.bits(expectedKey[column]),
                        "\(name).\(mode).cache.k[\(column)]")
                    XCTAssertEqual(
                        valueBits[index], Half.bits(value[column]),
                        "\(name).\(mode).cache.v[\(column)]")
                }
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

    // MARK: Prefill attention

    private func prefillAttentionParity(
        spec: LagunaAttentionSpec,
        seed: UInt64,
        label: String
    ) throws {
        let runtime = try makeRuntime()
        let gpuCache = try runtime.lagunaKVCache(
            capacity: spec.cacheCapacity, rowWidth: spec.keyValueWidth)
        let cpuCache = LagunaReferenceKVCache(
            capacity: spec.cacheCapacity, rowWidth: spec.keyValueWidth)
        var generator = Generator(seed: seed)

        // Populate and wrap the ring before the batch. The early batch rows
        // still need old slots that later rows will overwrite on commit.
        let position = spec.cacheCapacity + 1
        for oldPosition in 0..<position {
            let key = generator.vector(spec.keyValueWidth)
            let value = generator.vector(spec.keyValueWidth)
            try runtime.lagunaStoreKV(
                keyRow: key, valueRow: value,
                cache: gpuCache, position: oldPosition)
            cpuCache.store(
                position: oldPosition, keyRow: key, valueRow: value)
        }

        let count = min(3, spec.cacheCapacity)
        let queries = (0..<count).map { _ in
            generator.vector(spec.queryWidth)
        }
        let gates = (0..<count).map { _ in
            generator.vector(spec.headCount).map { $0 * 3 }
        }
        let keys = (0..<count).map { _ in
            generator.vector(spec.keyValueWidth)
        }
        let values = (0..<count).map { _ in
            generator.vector(spec.keyValueWidth)
        }

        var expected: [[Float]] = []
        for row in 0..<count {
            cpuCache.store(
                position: position + row,
                keyRow: keys[row], valueRow: values[row])
            expected.append(try LagunaLayerReference.attend(
                query: queries[row], gate: gates[row], cache: cpuCache,
                position: position + row, spec: spec))
        }
        let got = try runtime.lagunaAttentionPrefill(
            queries: queries, gates: gates,
            keyRows: keys, valueRows: values,
            cache: gpuCache, position: position, spec: spec)

        XCTAssertEqual(got.count, count)
        for row in 0..<count {
            assertClose(got[row], expected[row], "\(label).row\(row)")
        }
        let gpuKeyBits = gpuCache.keyBits()
        for i in 0..<gpuKeyBits.count {
            XCTAssertEqual(
                gpuKeyBits[i], Half.bits(cpuCache.keys[i]),
                "\(label).cache[\(i)]")
        }
    }

    func testPrefillAttentionMatchesSequentialOracleAcrossRingWrap()
        throws {
        try prefillAttentionParity(
            spec: slidingSpec(cacheCapacity: 4),
            seed: 0x4C41_4755_4E41_0010,
            label: "generic")

        let gqa6 = LagunaAttentionSpec(
            headCount: 6, kvHeadCount: 1,
            headDim: 128, rotationDims: 64,
            cacheCapacity: 4,
            ropeFrequencyBase: 500_000, ropeFrequencyScale: 1 / 32,
            extrapolationFactor: 1, attentionFactor: 1,
            betaFast: 32, betaSlow: 1,
            ropeOriginalContext: 8_192, rmsEpsilon: 1e-6)
        try prefillAttentionParity(
            spec: gqa6,
            seed: 0x4C41_4755_4E41_0011,
            label: "gqa6")

        let gqa3 = LagunaAttentionSpec(
            headCount: 3, kvHeadCount: 1,
            headDim: 128, rotationDims: 128,
            cacheCapacity: 4,
            ropeFrequencyBase: 10_000, ropeFrequencyScale: 1,
            extrapolationFactor: 0, attentionFactor: 1,
            betaFast: 0, betaSlow: 0,
            ropeOriginalContext: 262_144, rmsEpsilon: 1e-6)
        try prefillAttentionParity(
            spec: gqa3,
            seed: 0x4C41_4755_4E41_0014,
            label: "gqa3")
    }

    // MARK: Routed MoE prefill

    func testMoEPrefillExpertBatchMatchesLegacyWithinRoundingTolerance()
        throws {
        let runtime = try makeRuntime()
        let inputWidth = 128
        let expertWidth = 64
        let tokenCount = 5 // Crosses the four-token kernel tile.
        var state: UInt64 = 0x4C41_4755_4E41_0012
        func nextInt8() -> Int8 {
            state = state &* 6_364_136_223_846_793_005
                &+ 1_442_695_040_888_963_407
            return Int8(Int(truncatingIfNeeded: state >> 40) % 5 - 2)
        }
        func matrix(rowCount: Int, columnCount: Int) -> [UInt8] {
            (0..<rowCount).flatMap { _ in
                q8Row((0..<columnCount).map { _ in nextInt8() })
            }
        }

        let gate = matrix(rowCount: expertWidth, columnCount: inputWidth)
        let up = matrix(rowCount: expertWidth, columnCount: inputWidth)
        let down = matrix(rowCount: inputWidth, columnCount: expertWidth)
        var inputs: [[Float]] = []
        for token in 0..<tokenCount {
            var row: [Float] = []
            for column in 0..<inputWidth {
                let bucket = (column + token * 3) % 7
                row.append(Float(bucket) * 0.25 - 0.75)
            }
            inputs.append(row)
        }
        let routeWeights: [Float] = [0.5, 0.4, 0.3, 0.2, 0.55]
        let batched = try runtime.lagunaMoEPrefillExpert(
            inputs: inputs, routeWeights: routeWeights,
            gateRows: gate, upRows: up, downRows: down,
            weightType: GLM52TensorSchema.q8_0,
            expertWidth: expertWidth)

        for token in 0..<tokenCount {
            let mid = try runtime.glm52MoEPairSwiGLU(
                input: inputs[token], gateRows: gate, upRows: up,
                weightType: GLM52TensorSchema.q8_0,
                hiddenWidth: expertWidth,
                routeWeight: routeWeights[token])
            let legacy = try runtime.glm52MoEDown(
                mid: mid, downRows: down,
                weightType: GLM52TensorSchema.q8_0,
                outputWidth: inputWidth)
            // The validation wrapper uses the scalar reference kernels,
            // whereas the batched path uses simdgroup reductions. Their
            // association differs by a few ULPs.
            assertClose(
                batched[token], legacy, "token \(token)",
                tolerance: 3e-5)
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

    func testDecodeAttentionGroupedSplitKMatchesTheOracle() throws {
        let spec = LagunaAttentionSpec(
            headCount: 6, kvHeadCount: 1,
            headDim: 128, rotationDims: 64,
            cacheCapacity: 512,
            ropeFrequencyBase: 500_000, ropeFrequencyScale: 1 / 32,
            extrapolationFactor: 1, attentionFactor: 1,
            betaFast: 32, betaSlow: 1,
            ropeOriginalContext: 8_192, rmsEpsilon: 1e-6)
        let runtime = try makeRuntime()
        let gpuCache = try runtime.lagunaKVCache(
            capacity: spec.cacheCapacity, rowWidth: spec.keyValueWidth)
        let cpuCache = LagunaReferenceKVCache(
            capacity: spec.cacheCapacity, rowWidth: spec.keyValueWidth)
        var generator = Generator(seed: 0x4C41_4755_4E41_0013)
        let positions = 385
        for position in 0..<positions {
            let key = generator.vector(spec.keyValueWidth)
            let value = generator.vector(spec.keyValueWidth)
            try runtime.lagunaStoreKV(
                keyRow: key, valueRow: value,
                cache: gpuCache, position: position)
            cpuCache.store(
                position: position, keyRow: key, valueRow: value)
        }
        let query = generator.vector(spec.queryWidth)
        let gate = generator.vector(spec.headCount).map { $0 * 3 }
        let got = try runtime.lagunaAttentionDecodeGQA3Split(
            query: query, gate: gate, cache: gpuCache,
            position: positions - 1, spec: spec)
        let expected = try LagunaLayerReference.attend(
            query: query, gate: gate, cache: cpuCache,
            position: positions - 1, spec: spec)
        assertClose(
            got, expected, "grouped-split", tolerance: 3e-3)
    }
}
