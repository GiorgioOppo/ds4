import DS4Core
import XCTest
@testable import DS4Metal

/// The resident decode graph must reproduce the per-dispatch executor —
/// which is itself judged by `GLM52DecodeCPUReference` — position by
/// position: selections exactly, outputs and resident F16 caches within the
/// tolerance of the only intentional arithmetic difference (the GPU
/// float-reduction RMSNorm replacing the CPU Double-accumulation glue).
/// Quantized bytes only: no dequantized oracle run here.
final class GLM52DecodeGraphTests: XCTestCase {
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

    /// Quantized-only fixture, built once per process.
    private static let fixture: (attention: GLM52QuantizedDecodeAttention,
                                 indexer: GLM52QuantizedDecodeIndexer) = {
        let g = geometry(topK: 1)
        let layer = g.layer
        let headsWidth = layer.headCount * layer.valueDimension
        return (
            GLM52QuantizedDecodeAttention(
                attnNorm: floats(layer.embeddingWidth, seed: 31, scale: 0.5)
                    .map { $0 + 1.5 },
                qA: quantQ8(floats(g.qLoraRank * layer.embeddingWidth,
                                   seed: 32, scale: 0.1)),
                qANorm: floats(g.qLoraRank, seed: 33, scale: 0.5)
                    .map { $0 + 1.5 },
                qB: quantQ8(floats(g.queryWidth * g.qLoraRank,
                                   seed: 34, scale: 0.1)),
                kvA: quantQ8(floats(layer.kvRawWidth * layer.embeddingWidth,
                                    seed: 35, scale: 0.1)),
                kvANorm: floats(layer.kvLoraRank, seed: 36, scale: 0.5)
                    .map { $0 + 1.5 },
                keyB: quantQ8(floats(
                    layer.headCount * layer.kvLoraRank * g.nopeDimension,
                    seed: 37, scale: 0.05)),
                valueB: quantQ8(floats(headsWidth * layer.kvLoraRank,
                                       seed: 38, scale: 0.05)),
                attnOutput: quantQ8(floats(layer.embeddingWidth * headsWidth,
                                           seed: 39, scale: 0.05))),
            GLM52QuantizedDecodeIndexer(
                key: quantQ8(floats(
                    g.indexerHeadDimension * layer.embeddingWidth,
                    seed: 40, scale: 0.1)),
                keyNorm: floats(g.indexerHeadDimension, seed: 41, scale: 0.4)
                    .map { $0 + 1.2 },
                keyNormBias: floats(g.indexerHeadDimension, seed: 42,
                                    scale: 0.2),
                queryB: quantQ8(floats(g.indexerQueryWidth * g.qLoraRank,
                                       seed: 43, scale: 0.1)),
                proj: floats(g.indexerHeadCount * layer.embeddingWidth,
                             seed: 44, scale: 0.3)))
    }()

    private func input(position: Int) -> [Float] {
        Self.floats(Self.geometry(topK: 1).layer.embeddingWidth,
                    seed: 7_000 + UInt64(position), scale: 0.5)
    }

    private func assertClose(_ got: [Float], _ expected: [Float],
                             label: String, tolerance: Float = 1e-3,
                             file: StaticString = #filePath,
                             line: UInt = #line) {
        XCTAssertEqual(got.count, expected.count, file: file, line: line)
        for i in 0..<got.count {
            XCTAssertEqual(got[i], expected[i],
                           accuracy: tolerance + abs(expected[i]) * tolerance,
                           "\(label) diverges at \(i)", file: file, line: line)
        }
    }

    private func assertBitsClose(_ got: [UInt16], _ expected: [UInt16],
                                 label: String,
                                 file: StaticString = #filePath,
                                 line: UInt = #line) {
        XCTAssertEqual(got.count, expected.count, file: file, line: line)
        for i in 0..<got.count {
            let g = Half.float(got[i])
            let e = Half.float(expected[i])
            XCTAssertEqual(g, e, accuracy: 2e-3 + abs(e) * 2e-3,
                           "\(label) cache diverges at \(i)",
                           file: file, line: line)
        }
    }

    func testRMSNormKernelMatchesOracle() throws {
        let runtime = try makeRuntime()
        for width in [256, 300] {
            let values = Self.floats(width, seed: UInt64(width), scale: 0.7)
            let weight = Self.floats(width, seed: UInt64(width) + 1,
                                     scale: 0.5).map { $0 + 1.5 }
            let gpu = try runtime.glm52RMSNorm(values: values, weight: weight)
            let oracle = try GLM52FFNCPUReference.rmsNorm(
                values, weight: weight)
            assertClose(gpu, oracle, label: "rmsNorm width \(width)",
                        tolerance: 1e-4)
        }
        XCTAssertThrowsError(try runtime.glm52RMSNorm(
            values: [1, 2], weight: [1]))
    }

    func testF32MatvecKernelMatchesOracle() throws {
        let runtime = try makeRuntime()
        let rows = Self.floats(32 * 256, seed: 61, scale: 0.4)
        let input = Self.floats(256, seed: 62, scale: 0.4)
        let gpu = try runtime.glm52MatvecF32(rows: rows, input: input,
                                             rowCount: 32)
        let oracle = try GLM52FFNCPUReference.matvec(
            rows: rows, input: input, rowCount: 32)
        assertClose(gpu, oracle, label: "f32 matvec", tolerance: 1e-4)
        XCTAssertThrowsError(try runtime.glm52MatvecF32(
            rows: [1, 2], input: [1], rowCount: 3))
    }

    func testResidentFillRangeMatchesPerDispatchExecutor() throws {
        let runtime = try makeRuntime()
        let geometry = Self.geometry(topK: 2_048)
        let resident = try GLM52ResidentDecodeWeights(
            runtime: runtime, geometry: geometry,
            attention: Self.fixture.attention, indexer: Self.fixture.indexer)
        let caches = try GLM52ResidentDecodeCaches(
            runtime: runtime, geometry: geometry, capacity: 8,
            fullIndexer: true)
        var executorCaches = GLM52DecodeCaches()

        for position in 0..<3 {
            let x = input(position: position)
            let graph = try runtime.glm52ResidentDecodeAttention(
                weights: resident, caches: caches, input: x,
                reusedSelection: nil, position: position)
            let executor = try runtime.glm52DecodeAttention(
                geometry: geometry, input: x,
                attention: Self.fixture.attention,
                indexer: Self.fixture.indexer, reusedSelection: nil,
                caches: &executorCaches, position: position)

            XCTAssertEqual(graph.selection, executor.selection,
                           "selection diverges at position \(position)")
            assertClose(graph.output, executor.output,
                        label: "resident position \(position)")
        }
        XCTAssertEqual(caches.rows, 3)
        assertBitsClose(caches.compactSnapshot(),
                        executorCaches.compactBits, label: "compact")
        assertBitsClose(caches.indexerKeySnapshot(),
                        executorCaches.indexerKeyBits, label: "indexer")
    }

    func testResidentTopKAndIndexShareMatchExecutor() throws {
        let runtime = try makeRuntime()
        let geometry = Self.geometry(topK: 2)
        let fullResident = try GLM52ResidentDecodeWeights(
            runtime: runtime, geometry: geometry,
            attention: Self.fixture.attention, indexer: Self.fixture.indexer)
        let fullCaches = try GLM52ResidentDecodeCaches(
            runtime: runtime, geometry: geometry, capacity: 8,
            fullIndexer: true)
        let sharedResident = try GLM52ResidentDecodeWeights(
            runtime: runtime, geometry: geometry,
            attention: Self.fixture.attention, indexer: nil)
        let sharedCaches = try GLM52ResidentDecodeCaches(
            runtime: runtime, geometry: geometry, capacity: 8,
            fullIndexer: false)
        var executorFullCaches = GLM52DecodeCaches()
        var executorSharedCaches = GLM52DecodeCaches()

        for position in 0..<4 {
            let x = input(position: position)
            let graph = try runtime.glm52ResidentDecodeAttention(
                weights: fullResident, caches: fullCaches, input: x,
                reusedSelection: nil, position: position)
            let executor = try runtime.glm52DecodeAttention(
                geometry: geometry, input: x,
                attention: Self.fixture.attention,
                indexer: Self.fixture.indexer, reusedSelection: nil,
                caches: &executorFullCaches, position: position)
            XCTAssertEqual(graph.selection, executor.selection,
                           "full selection diverges at position \(position)")
            if position >= 2 {
                XCTAssertEqual(graph.selection.count, 2,
                               "top-k branch expected from position 2")
            }
            assertClose(graph.output, executor.output,
                        label: "full position \(position)")

            // IndexShare siblings reuse the (identical) selection verbatim.
            let graphShared = try runtime.glm52ResidentDecodeAttention(
                weights: sharedResident, caches: sharedCaches, input: x,
                reusedSelection: graph.selection, position: position)
            let executorShared = try runtime.glm52DecodeAttention(
                geometry: geometry, input: x,
                attention: Self.fixture.attention, indexer: nil,
                reusedSelection: executor.selection,
                caches: &executorSharedCaches, position: position)
            XCTAssertEqual(graphShared.selection, executorShared.selection)
            assertClose(graphShared.output, executorShared.output,
                        label: "shared position \(position)")
        }
        XCTAssertTrue(sharedCaches.indexerKeySnapshot().isEmpty)
        assertBitsClose(fullCaches.compactSnapshot(),
                        executorFullCaches.compactBits, label: "full compact")
        assertBitsClose(fullCaches.indexerKeySnapshot(),
                        executorFullCaches.indexerKeyBits,
                        label: "full indexer")
    }

    func testResidentContractErrors() throws {
        let runtime = try makeRuntime()
        let geometry = Self.geometry(topK: 2_048)
        let resident = try GLM52ResidentDecodeWeights(
            runtime: runtime, geometry: geometry,
            attention: Self.fixture.attention, indexer: Self.fixture.indexer)
        let x = input(position: 0)

        // Position must match the live row count; capacity is a hard bound.
        let tiny = try GLM52ResidentDecodeCaches(
            runtime: runtime, geometry: geometry, capacity: 1,
            fullIndexer: true)
        XCTAssertThrowsError(try runtime.glm52ResidentDecodeAttention(
            weights: resident, caches: tiny, input: x,
            reusedSelection: nil, position: 1))
        _ = try runtime.glm52ResidentDecodeAttention(
            weights: resident, caches: tiny, input: x,
            reusedSelection: nil, position: 0)
        XCTAssertThrowsError(try runtime.glm52ResidentDecodeAttention(
            weights: resident, caches: tiny, input: input(position: 1),
            reusedSelection: nil, position: 1))

        // Full-indexer weights over IndexShare caches (and vice versa).
        let sharedCaches = try GLM52ResidentDecodeCaches(
            runtime: runtime, geometry: geometry, capacity: 4,
            fullIndexer: false)
        XCTAssertThrowsError(try runtime.glm52ResidentDecodeAttention(
            weights: resident, caches: sharedCaches, input: x,
            reusedSelection: nil, position: 0))
        let sharedResident = try GLM52ResidentDecodeWeights(
            runtime: runtime, geometry: geometry,
            attention: Self.fixture.attention, indexer: nil)
        XCTAssertThrowsError(try runtime.glm52ResidentDecodeAttention(
            weights: sharedResident, caches: sharedCaches, input: x,
            reusedSelection: nil, position: 0))

        // A reused selection outside the visible range is refused.
        XCTAssertThrowsError(try runtime.glm52ResidentDecodeAttention(
            weights: sharedResident, caches: sharedCaches, input: x,
            reusedSelection: [4], position: 0))
    }
}
