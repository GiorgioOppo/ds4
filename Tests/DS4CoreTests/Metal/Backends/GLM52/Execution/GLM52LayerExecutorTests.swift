import DS4Core
import XCTest
@testable import DS4Metal

/// The GPU layer composition must match `GLM52LayerCPUReference` on the
/// dequantized weights: dense and sparse layers, routing equality, and the
/// multi-layer forward with the output head. Same fixture discipline as the
/// MoE suite — the quantized bytes are the source of truth for both sides.
final class GLM52LayerExecutorTests: XCTestCase {
    private let geometry = GLM52LayerGeometry(
        embeddingWidth: 256, headCount: 2, kvLoraRank: 32, ropeDimension: 32,
        valueDimension: 16, denseHiddenWidth: 96, expertHiddenWidth: 256,
        expertsUsed: 8)

    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    // MARK: - Fixtures

    private struct Generator {
        var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func byte() -> UInt8 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return UInt8(truncatingIfNeeded: state >> 33)
        }
        mutating func float(_ scale: Float) -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return (Float(state >> 33) / Float(1 << 30) - 1.0) * scale
        }
    }

    private func floats(_ count: Int, seed: UInt64, scale: Float) -> [Float] {
        var generator = Generator(seed: seed)
        return (0..<count).map { _ in generator.float(scale) }
    }

    private func quantQ8(_ row: [Float]) -> [UInt8] {
        var out: [UInt8] = []
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

    private func dequantQ8(_ bytes: [UInt8]) -> [Float] {
        var out = [Float](repeating: 0, count: (bytes.count / 34) * 32)
        bytes.withUnsafeBytes {
            Quantize.dequantQ8_0($0.baseAddress!, count: out.count, into: &out)
        }
        return out
    }

    /// Synthetic Q4_K rows (random fields, small sane f16 scales).
    private func syntheticQ4KRows(rows: Int, width: Int,
                                  seed: UInt64) -> [UInt8] {
        var generator = Generator(seed: seed)
        var bytes = [UInt8]()
        for _ in 0..<(rows * width / 256) {
            let d = Half.bits(0.001 + abs(generator.float(0.02)))
            let dmin = Half.bits(0.001 + abs(generator.float(0.02)))
            withUnsafeBytes(of: d.littleEndian) { bytes.append(contentsOf: $0) }
            withUnsafeBytes(of: dmin.littleEndian) { bytes.append(contentsOf: $0) }
            for _ in 0..<(12 + 128) { bytes.append(generator.byte()) }
        }
        return bytes
    }

    private func dequantQ4K(_ bytes: [UInt8], count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        bytes.withUnsafeBytes {
            Quantize.dequantQ4_K($0.baseAddress!, count: count, into: &out)
        }
        return out
    }

    private struct AttentionFixture {
        let quantized: GLM52QuantizedLayerAttention
        let dequantized: GLM52LayerAttentionWeightsF32
    }

    private func attentionFixture(seed: UInt64) -> AttentionFixture {
        let g = geometry
        let headsWidth = g.headCount * g.valueDimension
        let attnNorm = floats(g.embeddingWidth, seed: seed, scale: 0.5)
            .map { $0 + 1.5 }
        let kvANorm = floats(g.kvLoraRank, seed: seed &+ 1, scale: 0.5)
            .map { $0 + 1.5 }
        let kvAQ = quantQ8(floats(g.kvRawWidth * g.embeddingWidth,
                                  seed: seed &+ 2, scale: 0.1))
        let valueBQ = quantQ8(floats(headsWidth * g.kvLoraRank,
                                     seed: seed &+ 3, scale: 0.1))
        let attnOutputQ = quantQ8(floats(g.embeddingWidth * headsWidth,
                                         seed: seed &+ 4, scale: 0.1))
        return AttentionFixture(
            quantized: GLM52QuantizedLayerAttention(
                attnNorm: attnNorm, kvA: kvAQ, kvANorm: kvANorm,
                valueB: valueBQ, attnOutput: attnOutputQ),
            dequantized: GLM52LayerAttentionWeightsF32(
                attnNorm: attnNorm, kvA: dequantQ8(kvAQ), kvANorm: kvANorm,
                valueB: dequantQ8(valueBQ), attnOutput: dequantQ8(attnOutputQ)))
    }

    private func quantizedExpert(id: UInt32) -> GLM52QuantizedExpert {
        let g = geometry
        let seed = 5_000 + UInt64(id) * 7
        return GLM52QuantizedExpert(
            gateUpType: GLM52TensorSchema.q4_K,
            downType: GLM52TensorSchema.q4_K,
            gate: syntheticQ4KRows(rows: g.expertHiddenWidth,
                                   width: g.embeddingWidth, seed: seed),
            up: syntheticQ4KRows(rows: g.expertHiddenWidth,
                                 width: g.embeddingWidth, seed: seed + 1),
            down: syntheticQ4KRows(rows: g.embeddingWidth,
                                   width: g.expertHiddenWidth, seed: seed + 2))
    }

    private func dequantizedExpert(id: UInt32) -> GLM52ExpertWeights {
        let g = geometry
        let q = quantizedExpert(id: id)
        return GLM52ExpertWeights(
            gate: dequantQ4K(q.gate, count: g.expertHiddenWidth * g.embeddingWidth),
            up: dequantQ4K(q.up, count: g.expertHiddenWidth * g.embeddingWidth),
            down: dequantQ4K(q.down, count: g.embeddingWidth * g.expertHiddenWidth))
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

    // MARK: - Tests

    func testDenseLayerMatchesOracle() throws {
        let runtime = try makeRuntime()
        let g = geometry
        let x = floats(g.embeddingWidth, seed: 11, scale: 0.5)
        let attention = attentionFixture(seed: 20)
        let ffnNorm = floats(g.embeddingWidth, seed: 30, scale: 0.5)
            .map { $0 + 1.5 }
        let gateQ = quantQ8(floats(g.denseHiddenWidth * g.embeddingWidth,
                                   seed: 31, scale: 0.1))
        let upQ = quantQ8(floats(g.denseHiddenWidth * g.embeddingWidth,
                                 seed: 32, scale: 0.1))
        let downQ = quantQ8(floats(g.embeddingWidth * g.denseHiddenWidth,
                                   seed: 33, scale: 0.1))

        let (gpu, routing) = try runtime.glm52FirstTokenLayer(
            geometry: g, input: x, attention: attention.quantized,
            ffnNorm: ffnNorm,
            ffn: .dense(gate: gateQ, up: upQ, down: downQ))
        XCTAssertNil(routing)

        let (oracle, _) = try GLM52LayerCPUReference.firstTokenLayer(
            geometry: g, input: x, attention: attention.dequantized,
            ffnNorm: ffnNorm,
            ffn: .dense(gate: dequantQ8(gateQ), up: dequantQ8(upQ),
                        down: dequantQ8(downQ)))
        assertClose(gpu, oracle, label: "dense layer")
    }

    func testSparseLayerMatchesOracleAndRouting() throws {
        let runtime = try makeRuntime()
        let g = geometry
        let x = floats(g.embeddingWidth, seed: 41, scale: 0.5)
        let attention = attentionFixture(seed: 50)
        let ffnNorm = floats(g.embeddingWidth, seed: 60, scale: 0.5)
            .map { $0 + 1.5 }
        let routerRows = floats(
            GLM52RouterReference.expertCount * g.embeddingWidth,
            seed: 61, scale: 0.2)
        let routerBias = floats(GLM52RouterReference.expertCount,
                                seed: 62, scale: 0.01)
        let sharedGateQ = quantQ8(floats(g.expertHiddenWidth * g.embeddingWidth,
                                         seed: 63, scale: 0.1))
        let sharedUpQ = quantQ8(floats(g.expertHiddenWidth * g.embeddingWidth,
                                       seed: 64, scale: 0.1))
        let sharedDownQ = quantQ8(floats(g.embeddingWidth * g.expertHiddenWidth,
                                         seed: 65, scale: 0.1))

        let (gpu, gpuRouting) = try runtime.glm52FirstTokenLayer(
            geometry: g, input: x, attention: attention.quantized,
            ffnNorm: ffnNorm,
            ffn: .sparse(routerRows: routerRows, routerBias: routerBias,
                         sharedGate: sharedGateQ, sharedUp: sharedUpQ,
                         sharedDown: sharedDownQ,
                         expertProvider: { self.quantizedExpert(id: $0) }))

        let (oracle, oracleRouting) = try GLM52LayerCPUReference.firstTokenLayer(
            geometry: g, input: x, attention: attention.dequantized,
            ffnNorm: ffnNorm,
            ffn: .sparse(routerRows: routerRows, routerBias: routerBias,
                         shared: GLM52ExpertWeights(
                             gate: dequantQ8(sharedGateQ),
                             up: dequantQ8(sharedUpQ),
                             down: dequantQ8(sharedDownQ)),
                         expertProvider: {
                             self.dequantizedExpert(id: $0)
                         }))

        // Same selection and weights: both sides route on identically
        // computed logits, and the GPU router equals the CPU oracle.
        XCTAssertEqual(try XCTUnwrap(gpuRouting).selected,
                       try XCTUnwrap(oracleRouting).selected)
        assertClose(try XCTUnwrap(gpuRouting).weights,
                    try XCTUnwrap(oracleRouting).weights,
                    label: "router weights", tolerance: 1e-4)
        assertClose(gpu, oracle, label: "sparse layer", tolerance: 4e-3)
    }

    func testForwardWithOutputHeadMatchesOracleChain() throws {
        let runtime = try makeRuntime()
        let g = geometry
        let vocabulary = 64
        let embedded = floats(g.embeddingWidth, seed: 71, scale: 0.5)
        let dense = attentionFixture(seed: 80)
        let sparse = attentionFixture(seed: 90)
        let denseFFNNorm = floats(g.embeddingWidth, seed: 100, scale: 0.5)
            .map { $0 + 1.5 }
        let sparseFFNNorm = floats(g.embeddingWidth, seed: 101, scale: 0.5)
            .map { $0 + 1.5 }
        let gateQ = quantQ8(floats(g.denseHiddenWidth * g.embeddingWidth,
                                   seed: 102, scale: 0.1))
        let upQ = quantQ8(floats(g.denseHiddenWidth * g.embeddingWidth,
                                 seed: 103, scale: 0.1))
        let downQ = quantQ8(floats(g.embeddingWidth * g.denseHiddenWidth,
                                   seed: 104, scale: 0.1))
        let routerRows = floats(
            GLM52RouterReference.expertCount * g.embeddingWidth,
            seed: 105, scale: 0.2)
        let routerBias = floats(GLM52RouterReference.expertCount,
                                seed: 106, scale: 0.01)
        let sharedGateQ = quantQ8(floats(g.expertHiddenWidth * g.embeddingWidth,
                                         seed: 107, scale: 0.1))
        let sharedUpQ = quantQ8(floats(g.expertHiddenWidth * g.embeddingWidth,
                                       seed: 108, scale: 0.1))
        let sharedDownQ = quantQ8(floats(g.embeddingWidth * g.expertHiddenWidth,
                                         seed: 109, scale: 0.1))
        let outputNorm = floats(g.embeddingWidth, seed: 110, scale: 0.5)
            .map { $0 + 1.5 }
        let headQ = quantQ8(floats(vocabulary * g.embeddingWidth,
                                   seed: 111, scale: 0.1))

        let gpu = try runtime.glm52FirstTokenForward(
            geometry: g, embeddedToken: embedded,
            layers: [
                (attention: dense.quantized, ffnNorm: denseFFNNorm,
                 ffn: .dense(gate: gateQ, up: upQ, down: downQ)),
                (attention: sparse.quantized, ffnNorm: sparseFFNNorm,
                 ffn: .sparse(routerRows: routerRows, routerBias: routerBias,
                              sharedGate: sharedGateQ, sharedUp: sharedUpQ,
                              sharedDown: sharedDownQ,
                              expertProvider: { self.quantizedExpert(id: $0) })),
            ],
            outputNorm: outputNorm, outputHead: headQ,
            vocabularySize: vocabulary)

        let hidden = try GLM52LayerCPUReference.firstTokenForward(
            geometry: g, embeddedToken: embedded,
            layers: [
                (attention: dense.dequantized, ffnNorm: denseFFNNorm,
                 ffn: .dense(gate: dequantQ8(gateQ), up: dequantQ8(upQ),
                             down: dequantQ8(downQ))),
                (attention: sparse.dequantized, ffnNorm: sparseFFNNorm,
                 ffn: .sparse(routerRows: routerRows, routerBias: routerBias,
                              shared: GLM52ExpertWeights(
                                  gate: dequantQ8(sharedGateQ),
                                  up: dequantQ8(sharedUpQ),
                                  down: dequantQ8(sharedDownQ)),
                              expertProvider: {
                                  self.dequantizedExpert(id: $0)
                              })),
            ])
        let oracle = try GLM52FFNCPUReference.outputHead(
            hidden: hidden, normWeight: outputNorm,
            head: dequantQ8(headQ), vocabularySize: vocabulary)
        assertClose(gpu, oracle, label: "forward + output head",
                    tolerance: 5e-3)
    }

    func testRejectsWrongInputWidth() throws {
        let runtime = try makeRuntime()
        let attention = attentionFixture(seed: 120)
        XCTAssertThrowsError(try runtime.glm52FirstTokenLayer(
            geometry: geometry, input: [1, 2, 3],
            attention: attention.quantized,
            ffnNorm: [Float](repeating: 1, count: geometry.embeddingWidth),
            ffn: .dense(gate: [], up: [], down: [])))
    }
}
