import DS4Core
import XCTest
@testable import DS4Metal

/// The GPU decode composition must match `GLM52DecodeCPUReference` on the
/// dequantized weights, position by position: outputs within float tolerance,
/// selections exactly, and the F16 caches within F16 tolerance. The attention
/// and indexer kernels fix the v5_2 head geometry, so those dimensions are
/// real; the embedding width, Q-LoRA rank and FFN widths shrink to keep the
/// fixture affordable. The quantized bytes are the source of truth for both
/// sides, exactly like the layer-executor suite.
final class GLM52DecodeExecutorTests: XCTestCase {
    private static func geometry(topK: Int) -> GLM52DecodeGeometry {
        GLM52DecodeGeometry(
            layer: GLM52LayerGeometry(
                embeddingWidth: 256, headCount: 64, kvLoraRank: 512,
                ropeDimension: 64, valueDimension: 256,
                denseHiddenWidth: 128, expertHiddenWidth: 128,
                expertsUsed: 8),
            qLoraRank: 128, nopeDimension: 192,
            indexerHeadCount: 32, indexerHeadDimension: 128,
            indexerRotationDimension: 64, indexerTopK: topK)
    }

    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    // MARK: - Fixtures

    private struct Generator {
        var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func float(_ scale: Float) -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return (Float(state >> 33) / Float(1 << 30) - 1.0) * scale
        }
    }

    private static func floats(_ count: Int, seed: UInt64,
                               scale: Float) -> [Float] {
        var generator = Generator(seed: seed)
        return (0..<count).map { _ in generator.float(scale) }
    }

    private static func quantQ8(_ row: [Float]) -> [UInt8] {
        var out: [UInt8] = []
        out.reserveCapacity((row.count / 32) * 34)
        var b = 0
        while b < row.count {
            let block = Array(row[b..<b + 32])
            let amax = block.map { abs($0) }.max() ?? 0
            let d = amax / 127.0
            withUnsafeBytes(of: Half.bits(d).littleEndian) {
                out.append(contentsOf: $0)
            }
            for x in block {
                out.append(UInt8(bitPattern: Int8(
                    clamping: d != 0 ? Int((x / d).rounded()) : 0)))
            }
            b += 32
        }
        return out
    }

    private static func dequantQ8(_ bytes: [UInt8]) -> [Float] {
        var out = [Float](repeating: 0, count: (bytes.count / 34) * 32)
        bytes.withUnsafeBytes {
            Quantize.dequantQ8_0($0.baseAddress!, count: out.count, into: &out)
        }
        return out
    }

    private struct Fixture {
        let attention: GLM52QuantizedDecodeAttention
        let attentionF32: GLM52DecodeAttentionWeightsF32
        let indexer: GLM52QuantizedDecodeIndexer
        let indexerF32: GLM52DecodeIndexerWeightsF32
    }

    /// Built once per process: ~22M dequantized weights dominate the cost.
    private static let fixture: Fixture = {
        let g = geometry(topK: 1)
        let layer = g.layer
        let headsWidth = layer.headCount * layer.valueDimension
        let attnNorm = floats(layer.embeddingWidth, seed: 1, scale: 0.5)
            .map { $0 + 1.5 }
        let qANorm = floats(g.qLoraRank, seed: 2, scale: 0.5)
            .map { $0 + 1.5 }
        let kvANorm = floats(layer.kvLoraRank, seed: 3, scale: 0.5)
            .map { $0 + 1.5 }
        let qA = quantQ8(floats(g.qLoraRank * layer.embeddingWidth,
                                seed: 4, scale: 0.1))
        let qB = quantQ8(floats(g.queryWidth * g.qLoraRank,
                                seed: 5, scale: 0.1))
        let kvA = quantQ8(floats(layer.kvRawWidth * layer.embeddingWidth,
                                 seed: 6, scale: 0.1))
        let keyB = quantQ8(floats(
            layer.headCount * layer.kvLoraRank * g.nopeDimension,
            seed: 7, scale: 0.05))
        let valueB = quantQ8(floats(headsWidth * layer.kvLoraRank,
                                    seed: 8, scale: 0.05))
        let attnOutput = quantQ8(floats(layer.embeddingWidth * headsWidth,
                                        seed: 9, scale: 0.05))
        let indexerKey = quantQ8(floats(
            g.indexerHeadDimension * layer.embeddingWidth,
            seed: 10, scale: 0.1))
        let indexerQueryB = quantQ8(floats(
            g.indexerQueryWidth * g.qLoraRank, seed: 11, scale: 0.1))
        let keyNorm = floats(g.indexerHeadDimension, seed: 12, scale: 0.4)
            .map { $0 + 1.2 }
        let keyNormBias = floats(g.indexerHeadDimension, seed: 13, scale: 0.2)
        let proj = floats(g.indexerHeadCount * layer.embeddingWidth,
                          seed: 14, scale: 0.3)
        return Fixture(
            attention: GLM52QuantizedDecodeAttention(
                attnNorm: attnNorm, qA: qA, qANorm: qANorm, qB: qB,
                kvA: kvA, kvANorm: kvANorm, keyB: keyB, valueB: valueB,
                attnOutput: attnOutput),
            attentionF32: GLM52DecodeAttentionWeightsF32(
                attnNorm: attnNorm, qA: dequantQ8(qA), qANorm: qANorm,
                qB: dequantQ8(qB), kvA: dequantQ8(kvA), kvANorm: kvANorm,
                keyB: dequantQ8(keyB), valueB: dequantQ8(valueB),
                attnOutput: dequantQ8(attnOutput)),
            indexer: GLM52QuantizedDecodeIndexer(
                key: indexerKey, keyNorm: keyNorm,
                keyNormBias: keyNormBias, queryB: indexerQueryB, proj: proj),
            indexerF32: GLM52DecodeIndexerWeightsF32(
                key: dequantQ8(indexerKey), keyNorm: keyNorm,
                keyNormBias: keyNormBias, queryB: dequantQ8(indexerQueryB),
                proj: proj))
    }()

    private func input(position: Int) -> [Float] {
        Self.floats(Self.geometry(topK: 1).layer.embeddingWidth,
                    seed: 5_000 + UInt64(position), scale: 0.5)
    }

    private func assertClose(_ got: [Float], _ expected: [Float],
                             label: String, tolerance: Float = 2e-3,
                             file: StaticString = #filePath,
                             line: UInt = #line) {
        XCTAssertEqual(got.count, expected.count, file: file, line: line)
        for i in 0..<got.count {
            XCTAssertEqual(got[i], expected[i],
                           accuracy: tolerance + abs(expected[i]) * tolerance,
                           "\(label) diverges at \(i)", file: file, line: line)
        }
    }

    private func assertCachesMatch(_ caches: GLM52DecodeCaches,
                                   _ oracle: GLM52DecodeCacheStateF32,
                                   position: Int,
                                   file: StaticString = #filePath,
                                   line: UInt = #line) {
        XCTAssertEqual(caches.compactBits.count, oracle.compactRows.count,
                       file: file, line: line)
        for i in 0..<caches.compactBits.count {
            XCTAssertEqual(Half.float(caches.compactBits[i]),
                           oracle.compactRows[i],
                           accuracy: 2e-3 + abs(oracle.compactRows[i]) * 2e-3,
                           "compact cache diverges at \(i), position \(position)",
                           file: file, line: line)
        }
        XCTAssertEqual(caches.indexerKeyBits.count, oracle.indexerKeys.count,
                       file: file, line: line)
        for i in 0..<caches.indexerKeyBits.count {
            XCTAssertEqual(Half.float(caches.indexerKeyBits[i]),
                           oracle.indexerKeys[i],
                           accuracy: 2e-3 + abs(oracle.indexerKeys[i]) * 2e-3,
                           "indexer cache diverges at \(i), position \(position)",
                           file: file, line: line)
        }
    }

    // MARK: - Tests

    func testFillRangeDecodeMatchesOracleAcrossPositions() throws {
        let runtime = try makeRuntime()
        let geometry = Self.geometry(topK: 2_048)
        var caches = GLM52DecodeCaches()
        var oracleCache = GLM52DecodeCacheStateF32()

        for position in 0..<2 {
            let x = input(position: position)
            let gpu = try runtime.glm52DecodeAttention(
                geometry: geometry, input: x, attention: Self.fixture.attention,
                indexer: Self.fixture.indexer, reusedSelection: nil,
                caches: &caches, position: position)
            let oracle = try GLM52DecodeCPUReference.decodeAttention(
                geometry: geometry, input: x,
                weights: Self.fixture.attentionF32,
                indexer: Self.fixture.indexerF32, reusedSelection: nil,
                cache: &oracleCache, position: position)

            XCTAssertEqual(gpu.selection.map(Int.init), oracle.selection,
                           "selection diverges at position \(position)")
            XCTAssertEqual(gpu.selection.map(Int.init),
                           Array(0...position),
                           "fill range must cover the causal range")
            assertClose(gpu.output, oracle.output,
                        label: "attention position \(position)")
            assertCachesMatch(caches, oracleCache, position: position)
        }
    }

    func testTopKDecodeAndIndexShareMatchOracle() throws {
        let runtime = try makeRuntime()
        let geometry = Self.geometry(topK: 2)
        var caches = GLM52DecodeCaches()
        var oracleCache = GLM52DecodeCacheStateF32()
        var lastSelection: (gpu: [UInt32], oracle: [Int])?

        for position in 0..<4 {
            let x = input(position: position)
            let gpu = try runtime.glm52DecodeAttention(
                geometry: geometry, input: x, attention: Self.fixture.attention,
                indexer: Self.fixture.indexer, reusedSelection: nil,
                caches: &caches, position: position)
            let oracle = try GLM52DecodeCPUReference.decodeAttention(
                geometry: geometry, input: x,
                weights: Self.fixture.attentionF32,
                indexer: Self.fixture.indexerF32, reusedSelection: nil,
                cache: &oracleCache, position: position)

            XCTAssertEqual(gpu.selection.map(Int.init), oracle.selection,
                           "selection diverges at position \(position)")
            if position >= 2 {
                // visible > topK: the scored branch ran on both sides.
                XCTAssertEqual(gpu.selection.count, 2)
            }
            assertClose(gpu.output, oracle.output,
                        label: "top-k position \(position)")
            lastSelection = (gpu.selection, oracle.selection)
        }
        assertCachesMatch(caches, oracleCache, position: 3)

        // IndexShare at the final position: fresh per-layer caches, the
        // full-indexer selection reused verbatim, no indexer keys stored.
        let selection = try XCTUnwrap(lastSelection)
        var sharedCaches = GLM52DecodeCaches()
        var sharedOracleCache = GLM52DecodeCacheStateF32()
        for position in 0..<3 {
            // Reach position 3 with causal fill selections on both sides.
            let x = input(position: position)
            _ = try runtime.glm52DecodeAttention(
                geometry: geometry, input: x, attention: Self.fixture.attention,
                indexer: nil, reusedSelection: (0...UInt32(position)).map { $0 },
                caches: &sharedCaches, position: position)
            _ = try GLM52DecodeCPUReference.decodeAttention(
                geometry: geometry, input: x,
                weights: Self.fixture.attentionF32,
                indexer: nil, reusedSelection: Array(0...position),
                cache: &sharedOracleCache, position: position)
        }
        let x = input(position: 3)
        let gpuShared = try runtime.glm52DecodeAttention(
            geometry: geometry, input: x, attention: Self.fixture.attention,
            indexer: nil, reusedSelection: selection.gpu,
            caches: &sharedCaches, position: 3)
        let oracleShared = try GLM52DecodeCPUReference.decodeAttention(
            geometry: geometry, input: x,
            weights: Self.fixture.attentionF32,
            indexer: nil, reusedSelection: selection.oracle,
            cache: &sharedOracleCache, position: 3)
        XCTAssertTrue(sharedCaches.indexerKeyBits.isEmpty)
        XCTAssertTrue(sharedOracleCache.indexerKeys.isEmpty)
        assertClose(gpuShared.output, oracleShared.output,
                    label: "IndexShare position 3")
    }

    func testDecodeLayerMatchesOracleWithDenseFFN() throws {
        let runtime = try makeRuntime()
        let geometry = Self.geometry(topK: 2_048)
        let layer = geometry.layer
        let ffnNorm = Self.floats(layer.embeddingWidth, seed: 20, scale: 0.5)
            .map { $0 + 1.5 }
        let gateQ = Self.quantQ8(Self.floats(
            layer.denseHiddenWidth * layer.embeddingWidth,
            seed: 21, scale: 0.1))
        let upQ = Self.quantQ8(Self.floats(
            layer.denseHiddenWidth * layer.embeddingWidth,
            seed: 22, scale: 0.1))
        let downQ = Self.quantQ8(Self.floats(
            layer.embeddingWidth * layer.denseHiddenWidth,
            seed: 23, scale: 0.1))

        var caches = GLM52DecodeCaches()
        var oracleCache = GLM52DecodeCacheStateF32()
        let x = input(position: 0)
        let gpu = try runtime.glm52DecodeLayer(
            geometry: geometry, input: x, attention: Self.fixture.attention,
            indexer: Self.fixture.indexer, reusedSelection: nil,
            ffnNorm: ffnNorm,
            ffn: .dense(gate: gateQ, up: upQ, down: downQ),
            caches: &caches, position: 0)
        let oracle = try GLM52DecodeCPUReference.decodeLayer(
            geometry: geometry, input: x,
            attention: Self.fixture.attentionF32,
            indexer: Self.fixture.indexerF32, reusedSelection: nil,
            ffnNorm: ffnNorm,
            ffn: .dense(gate: Self.dequantQ8(gateQ),
                        up: Self.dequantQ8(upQ),
                        down: Self.dequantQ8(downQ)),
            cache: &oracleCache, position: 0)

        XCTAssertNil(gpu.routing)
        XCTAssertEqual(gpu.selection.map(Int.init), oracle.selection)
        assertClose(gpu.output, oracle.output, label: "decode layer")
    }

    func testValidationRejectsWrongGeometryAndContracts() throws {
        let runtime = try makeRuntime()
        var caches = GLM52DecodeCaches()
        let x = input(position: 0)

        // Non-v5_2 attention geometry is refused before any dispatch.
        var wrong = Self.geometry(topK: 8)
        wrong = GLM52DecodeGeometry(
            layer: GLM52LayerGeometry(
                embeddingWidth: 256, headCount: 2, kvLoraRank: 512,
                ropeDimension: 64, valueDimension: 256,
                denseHiddenWidth: 128, expertHiddenWidth: 128,
                expertsUsed: 8),
            qLoraRank: wrong.qLoraRank, nopeDimension: wrong.nopeDimension,
            indexerHeadCount: wrong.indexerHeadCount,
            indexerHeadDimension: wrong.indexerHeadDimension,
            indexerRotationDimension: wrong.indexerRotationDimension,
            indexerTopK: wrong.indexerTopK)
        XCTAssertThrowsError(try runtime.glm52DecodeAttention(
            geometry: wrong, input: x, attention: Self.fixture.attention,
            indexer: Self.fixture.indexer, reusedSelection: nil,
            caches: &caches, position: 0))

        // IndexShare without a selection, and a full layer with one.
        XCTAssertThrowsError(try runtime.glm52DecodeAttention(
            geometry: Self.geometry(topK: 8), input: x,
            attention: Self.fixture.attention, indexer: nil,
            reusedSelection: nil, caches: &caches, position: 0))
        XCTAssertThrowsError(try runtime.glm52DecodeAttention(
            geometry: Self.geometry(topK: 8), input: x,
            attention: Self.fixture.attention, indexer: Self.fixture.indexer,
            reusedSelection: [0], caches: &caches, position: 0))

        // Cache row count must match the position.
        var stale = GLM52DecodeCaches()
        stale.compactBits = [UInt16](repeating: 0, count: 576)
        XCTAssertThrowsError(try runtime.glm52DecodeAttention(
            geometry: Self.geometry(topK: 8), input: x,
            attention: Self.fixture.attention, indexer: Self.fixture.indexer,
            reusedSelection: nil, caches: &stale, position: 0))
    }
}
