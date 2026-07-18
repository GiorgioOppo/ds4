import DS4Core
import XCTest
@testable import DS4Metal

/// The resident multi-layer decode forward — real IndexShare policy on
/// absolute layer indices, resident FFN stages and the resident output head
/// — must match the per-dispatch composition (glm52DecodeLayer per layer,
/// manual selection threading, per-dispatch output head), which is itself
/// anchored to GLM52DecodeCPUReference. Stack: layer 0 (full indexer,
/// dense FFN), layer 6 (full, sparse) and layer 7 (IndexShare, sparse).
final class GLM52ResidentForwardTests: XCTestCase {
    private static let vocabulary = 64

    private static func geometry() -> GLM52DecodeGeometry {
        GLM52DecodeGeometry(
            layer: GLM52LayerGeometry(
                embeddingWidth: 256, headCount: 64, kvLoraRank: 512,
                ropeDimension: 64, valueDimension: 256,
                denseHiddenWidth: 128, expertHiddenWidth: 128,
                expertsUsed: 8),
            qLoraRank: 128, nopeDimension: 192,
            indexerHeadCount: 32, indexerHeadDimension: 128,
            indexerRotationDimension: 64, indexerTopK: 2_048)
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

    private static func attentionFixture(seed: UInt64)
        -> GLM52QuantizedDecodeAttention {
        let g = geometry()
        let layer = g.layer
        let headsWidth = layer.headCount * layer.valueDimension
        return GLM52QuantizedDecodeAttention(
            attnNorm: floats(layer.embeddingWidth, seed: seed, scale: 0.5)
                .map { $0 + 1.5 },
            qA: quantQ8(floats(g.qLoraRank * layer.embeddingWidth,
                               seed: seed &+ 1, scale: 0.1)),
            qANorm: floats(g.qLoraRank, seed: seed &+ 2, scale: 0.5)
                .map { $0 + 1.5 },
            qB: quantQ8(floats(g.queryWidth * g.qLoraRank,
                               seed: seed &+ 3, scale: 0.1)),
            kvA: quantQ8(floats(layer.kvRawWidth * layer.embeddingWidth,
                                seed: seed &+ 4, scale: 0.1)),
            kvANorm: floats(layer.kvLoraRank, seed: seed &+ 5, scale: 0.5)
                .map { $0 + 1.5 },
            keyB: quantQ8(floats(
                layer.headCount * layer.kvLoraRank * g.nopeDimension,
                seed: seed &+ 6, scale: 0.05)),
            valueB: quantQ8(floats(headsWidth * layer.kvLoraRank,
                                   seed: seed &+ 7, scale: 0.05)),
            attnOutput: quantQ8(floats(layer.embeddingWidth * headsWidth,
                                       seed: seed &+ 8, scale: 0.05)))
    }

    private static func indexerFixture(seed: UInt64)
        -> GLM52QuantizedDecodeIndexer {
        let g = geometry()
        return GLM52QuantizedDecodeIndexer(
            key: quantQ8(floats(
                g.indexerHeadDimension * g.layer.embeddingWidth,
                seed: seed, scale: 0.1)),
            keyNorm: floats(g.indexerHeadDimension, seed: seed &+ 1,
                            scale: 0.4).map { $0 + 1.2 },
            keyNormBias: floats(g.indexerHeadDimension, seed: seed &+ 2,
                                scale: 0.2),
            queryB: quantQ8(floats(g.indexerQueryWidth * g.qLoraRank,
                                   seed: seed &+ 3, scale: 0.1)),
            proj: floats(g.indexerHeadCount * g.layer.embeddingWidth,
                         seed: seed &+ 4, scale: 0.3))
    }

    private static func expert(seed: UInt64) -> GLM52QuantizedExpert {
        let layer = geometry().layer
        let embed = layer.embeddingWidth
        return GLM52QuantizedExpert(
            gateUpType: GLM52TensorSchema.q8_0,
            downType: GLM52TensorSchema.q8_0,
            gate: quantQ8(floats(layer.expertHiddenWidth * embed,
                                 seed: seed, scale: 0.1)),
            up: quantQ8(floats(layer.expertHiddenWidth * embed,
                               seed: seed &+ 1, scale: 0.1)),
            down: quantQ8(floats(embed * layer.expertHiddenWidth,
                                 seed: seed &+ 2, scale: 0.1)))
    }

    private static func sparseFFN(seed: UInt64) -> GLM52QuantizedLayerFFN {
        let layer = geometry().layer
        return .sparse(
            routerRows: floats(
                GLM52RouterReference.expertCount * layer.embeddingWidth,
                seed: seed, scale: 0.3),
            routerBias: floats(GLM52RouterReference.expertCount,
                               seed: seed &+ 1, scale: 0.1),
            sharedGate: quantQ8(floats(
                layer.expertHiddenWidth * layer.embeddingWidth,
                seed: seed &+ 2, scale: 0.1)),
            sharedUp: quantQ8(floats(
                layer.expertHiddenWidth * layer.embeddingWidth,
                seed: seed &+ 3, scale: 0.1)),
            sharedDown: quantQ8(floats(
                layer.embeddingWidth * layer.expertHiddenWidth,
                seed: seed &+ 4, scale: 0.1)),
            expertProvider: { expert(seed: 90_000 &+ UInt64($0) &* 11) })
    }

    private static func denseFFN(seed: UInt64) -> GLM52QuantizedLayerFFN {
        let layer = geometry().layer
        return .dense(
            gate: quantQ8(floats(
                layer.denseHiddenWidth * layer.embeddingWidth,
                seed: seed, scale: 0.1)),
            up: quantQ8(floats(
                layer.denseHiddenWidth * layer.embeddingWidth,
                seed: seed &+ 1, scale: 0.1)),
            down: quantQ8(floats(
                layer.embeddingWidth * layer.denseHiddenWidth,
                seed: seed &+ 2, scale: 0.1)))
    }

    private func assertClose(_ got: [Float], _ expected: [Float],
                             label: String, tolerance: Float,
                             file: StaticString = #filePath,
                             line: UInt = #line) {
        XCTAssertEqual(got.count, expected.count, file: file, line: line)
        for i in 0..<got.count {
            XCTAssertEqual(got[i], expected[i],
                           accuracy: tolerance + abs(expected[i]) * tolerance,
                           "\(label) diverges at \(i)", file: file, line: line)
        }
    }

    func testResidentForwardMatchesPerDispatchComposition() throws {
        let runtime = try makeRuntime()
        let g = Self.geometry()
        let layer = g.layer
        let attention = Self.attentionFixture(seed: 100)
        let indexer0 = Self.indexerFixture(seed: 200)
        let indexer6 = Self.indexerFixture(seed: 300)
        let dense0 = Self.denseFFN(seed: 400)
        let sparse6 = Self.sparseFFN(seed: 500)
        let sparse7 = Self.sparseFFN(seed: 600)
        let ffnNorm0 = Self.floats(layer.embeddingWidth, seed: 700,
                                   scale: 0.5).map { $0 + 1.5 }
        let ffnNorm6 = Self.floats(layer.embeddingWidth, seed: 701,
                                   scale: 0.5).map { $0 + 1.5 }
        let ffnNorm7 = Self.floats(layer.embeddingWidth, seed: 702,
                                   scale: 0.5).map { $0 + 1.5 }
        let outputNorm = Self.floats(layer.embeddingWidth, seed: 800,
                                     scale: 0.5).map { $0 + 1.5 }
        let outputHeadQ = Self.quantQ8(Self.floats(
            Self.vocabulary * layer.embeddingWidth, seed: 801, scale: 0.1))

        func residentLayer(index: Int,
                           indexer: GLM52QuantizedDecodeIndexer?,
                           ffnNorm: [Float],
                           ffn: GLM52QuantizedLayerFFN) throws
            -> GLM52ResidentStackLayer {
            GLM52ResidentStackLayer(
                index: index,
                weights: try GLM52ResidentDecodeWeights(
                    runtime: runtime, geometry: g, attention: attention,
                    indexer: indexer),
                ffn: try GLM52ResidentFFN(
                    runtime: runtime, geometry: g, ffnNorm: ffnNorm,
                    ffn: ffn),
                caches: try GLM52ResidentDecodeCaches(
                    runtime: runtime, geometry: g, capacity: 4,
                    fullIndexer: indexer != nil))
        }
        let stack = [
            try residentLayer(index: 0, indexer: indexer0,
                              ffnNorm: ffnNorm0, ffn: dense0),
            try residentLayer(index: 6, indexer: indexer6,
                              ffnNorm: ffnNorm6, ffn: sparse6),
            try residentLayer(index: 7, indexer: nil,
                              ffnNorm: ffnNorm7, ffn: sparse7),
        ]
        let head = try GLM52ResidentOutputHead(
            runtime: runtime, geometry: g, outputNorm: outputNorm,
            outputHead: outputHeadQ, vocabularySize: Self.vocabulary)

        var caches0 = GLM52DecodeCaches()
        var caches6 = GLM52DecodeCaches()
        var caches7 = GLM52DecodeCaches()

        for position in 0..<2 {
            let embedded = Self.floats(layer.embeddingWidth,
                                       seed: 9_000 + UInt64(position),
                                       scale: 0.5)
            let resident = try runtime.glm52ResidentDecodeForward(
                layers: stack, outputHead: head,
                embeddedToken: embedded, position: position)

            // Per-dispatch composition with manual IndexShare threading.
            let step0 = try runtime.glm52DecodeLayer(
                geometry: g, input: embedded, attention: attention,
                indexer: indexer0, reusedSelection: nil, ffnNorm: ffnNorm0,
                ffn: dense0, caches: &caches0, position: position)
            let step6 = try runtime.glm52DecodeLayer(
                geometry: g, input: step0.output, attention: attention,
                indexer: indexer6, reusedSelection: nil, ffnNorm: ffnNorm6,
                ffn: sparse6, caches: &caches6, position: position)
            let step7 = try runtime.glm52DecodeLayer(
                geometry: g, input: step6.output, attention: attention,
                indexer: nil, reusedSelection: step6.selection,
                ffnNorm: ffnNorm7, ffn: sparse7, caches: &caches7,
                position: position)
            let normalized = try GLM52FFNCPUReference.rmsNorm(
                step7.output, weight: outputNorm)
            let logits = try runtime.glm52OutputHeadLogits(
                normalized: normalized, headRows: outputHeadQ,
                weightType: GLM52TensorSchema.q8_0,
                vocabularySize: Self.vocabulary)

            XCTAssertEqual(resident.selections[0], step0.selection)
            XCTAssertEqual(resident.selections[6], step6.selection)
            XCTAssertEqual(resident.selections[7], step6.selection,
                           "IndexShare layer 7 must reuse layer 6's selection")
            XCTAssertEqual(resident.routings[6]?.selected,
                           step6.routing?.selected,
                           "router selection diverges at position \(position)")
            XCTAssertNil(resident.routings[0])
            assertClose(resident.logits, logits,
                        label: "logits position \(position)",
                        tolerance: 5e-3)
        }
    }

    func testForwardPolicyContractErrors() throws {
        let runtime = try makeRuntime()
        let g = Self.geometry()
        let attention = Self.attentionFixture(seed: 100)
        let head = try GLM52ResidentOutputHead(
            runtime: runtime, geometry: g,
            outputNorm: Self.floats(g.layer.embeddingWidth, seed: 800,
                                    scale: 0.5).map { $0 + 1.5 },
            outputHead: Self.quantQ8(Self.floats(
                Self.vocabulary * g.layer.embeddingWidth,
                seed: 801, scale: 0.1)),
            vocabularySize: Self.vocabulary)
        let embedded = Self.floats(g.layer.embeddingWidth, seed: 9_000,
                                   scale: 0.5)

        // Policy says layer 7 is IndexShare: full-indexer weights refused.
        let fullAtSeven = GLM52ResidentStackLayer(
            index: 7,
            weights: try GLM52ResidentDecodeWeights(
                runtime: runtime, geometry: g, attention: attention,
                indexer: Self.indexerFixture(seed: 200)),
            ffn: try GLM52ResidentFFN(
                runtime: runtime, geometry: g,
                ffnNorm: Self.floats(g.layer.embeddingWidth, seed: 700,
                                     scale: 0.5).map { $0 + 1.5 },
                ffn: Self.sparseFFN(seed: 500)),
            caches: try GLM52ResidentDecodeCaches(
                runtime: runtime, geometry: g, capacity: 4,
                fullIndexer: true))
        XCTAssertThrowsError(try runtime.glm52ResidentDecodeForward(
            layers: [fullAtSeven], outputHead: head,
            embeddedToken: embedded, position: 0))

        // An IndexShare layer with no preceding full-indexer source.
        let orphanShared = GLM52ResidentStackLayer(
            index: 7,
            weights: try GLM52ResidentDecodeWeights(
                runtime: runtime, geometry: g, attention: attention,
                indexer: nil),
            ffn: try GLM52ResidentFFN(
                runtime: runtime, geometry: g,
                ffnNorm: Self.floats(g.layer.embeddingWidth, seed: 700,
                                     scale: 0.5).map { $0 + 1.5 },
                ffn: Self.sparseFFN(seed: 500)),
            caches: try GLM52ResidentDecodeCaches(
                runtime: runtime, geometry: g, capacity: 4,
                fullIndexer: false))
        XCTAssertThrowsError(try runtime.glm52ResidentDecodeForward(
            layers: [orphanShared], outputHead: head,
            embeddedToken: embedded, position: 0))
    }
}
