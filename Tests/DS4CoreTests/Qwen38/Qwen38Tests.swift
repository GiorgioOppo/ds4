import Foundation
import XCTest
@testable import DS4Core
@testable import DS4Metal

final class Qwen38Tests: XCTestCase {
    private func temporary(_ suffix: String = ".gguf") -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("qwen38-test-" + UUID().uuidString + suffix)
    }

    func testMetadataAndLayerSchedule() throws {
        let url = temporary(); defer { try? FileManager.default.removeItem(at: url) }
        try Qwen38Fixture.write(path: url.path)
        let c = try Qwen38Configuration(model: GGUFModel(path: url.path, metalMapping: false))
        XCTAssertEqual(c.layers, 8); XCTAssertEqual(c.embedding, 64); XCTAssertEqual(c.vocabulary, 248320)
        XCTAssertEqual(c.ngramRows, 127)
        XCTAssertEqual((0..<8).filter { !c.isLinear(layer: $0) }, [3, 7])
        XCTAssertEqual(c.linearQKVDimension, 320); XCTAssertEqual(c.hcDimension, 256)
        XCTAssertEqual(c.pleMultipliers[0], UInt64.max - 4)
    }

    func testRejectsUnsupportedMetadataBeforeGPU() throws {
        let url = temporary(); defer { try? FileManager.default.removeItem(at: url) }
        for override in [("qwen4exp.embedding_length", GGUFMetadataValue.uint32(2561)),
                         ("qwen4exp.expert_used_count", .uint32(11)),
                         ("qwen4exp.context_length", .uint32(0)),
                         ("qwen4exp.nextn_predict_layers", .uint64(UInt64.max)),
                         ("qwen4exp.ple.head_vocab_sizes", .array(elementType: .uint32, elements: (0..<16).map { _ in .uint32(0) }))] {
            try Qwen38Fixture.write(path: url.path, override: override)
            XCTAssertThrowsError(try Qwen38Configuration(model: GGUFModel(path: url.path, metalMapping: false)))
        }
    }

    func testMissingWeightsRejectedWithoutConstructingRuntime() throws {
        let url = temporary(); defer { try? FileManager.default.removeItem(at: url) }
        try Qwen38Fixture.write(path: url.path)
        let model = try GGUFModel(path: url.path, metalMapping: false)
        let c = try Qwen38Configuration(model: model)
        XCTAssertThrowsError(try Qwen38Weights(model: model, configuration: c))
    }

    func testNgramUnsignedOverflowAndEOSBoundary() {
        let multipliers: [UInt64] = [UInt64.max - 4, 0xdeadbeef12345679, 0xfedcba9876543211]
        let offsets = (0..<16).map { UInt32($0 * 8) }, sizes = [UInt32](repeating: 7, count: 16)
        var state = Qwen38NgramState()
        let expected: [[UInt32]] = [
            [2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122],
            [6, 14, 22, 30, 38, 46, 54, 62, 69, 77, 85, 93, 101, 109, 117, 125],
            [4, 12, 20, 28, 36, 44, 52, 60, 65, 73, 81, 89, 97, 105, 113, 121],
            [4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124]
        ]
        for (token, answer) in zip([123, 456, 248044, 789], expected) {
            XCTAssertEqual(state.rows(token: token, multipliers: multipliers, offsets: offsets, vocabularies: sizes), answer)
        }
        var reset = Qwen38NgramState()
        XCTAssertEqual(reset.rows(token: 789, multipliers: multipliers, offsets: offsets, vocabularies: sizes), expected[3])
    }

    func testQuantizedRowStridesIncludeQ2PaddingAndMXFP4() throws {
        XCTAssertEqual(try Qwen38Weights.rowBytes(type: 10, width: 768), 252)
        XCTAssertThrowsError(try Qwen38Weights.rowBytes(type: 10, width: 640))
        XCTAssertEqual(try Qwen38Weights.rowBytes(type: 39, width: 640), 340)
        XCTAssertEqual(try Qwen38Weights.rowBytes(type: 16, width: 2560), 660)
        XCTAssertThrowsError(try Qwen38Weights.rowBytes(type: 142, width: 2560))
        XCTAssertThrowsError(try Qwen38Weights.rowBytes(type: 0, width: Int.max))
    }

    func testStableActiveExpertRemappingAndBounds() throws {
        let route = try Qwen38Model.compactExperts([7, 2, 4, 2, 9, 7], slots: 3, experts: 10)
        XCTAssertEqual(route.ids, [7, 2, 4, 9]); XCTAssertEqual(route.remapped, [0, 1, 2, 1, 3, 0])
        XCTAssertThrowsError(try Qwen38Model.compactExperts([1, 1], slots: 2, experts: 10))
        XCTAssertThrowsError(try Qwen38Model.compactExperts([-1, 2], slots: 2, experts: 10))
        XCTAssertThrowsError(try Qwen38Model.compactExperts([10, 2], slots: 2, experts: 10))
    }

    func testDenseSparseBoundaryKeepsIncompleteTail() {
        XCTAssertEqual(Qwen38Model.denseRows(position: 0, count: 32, blockBudget: 2), 11)
        XCTAssertEqual(Qwen38Model.denseRows(position: 8, count: 4, blockBudget: 2), 3)
        XCTAssertEqual(Qwen38Model.denseRows(position: 11, count: 4, blockBudget: 2), 0)
        XCTAssertEqual(Qwen38Model.denseRows(position: 2048, count: 32, blockBudget: 512), 3)
    }

    func testEmbeddingBF16AndQ4NibbleOrder() throws {
        let bf: [UInt16] = [0x3f80, 0xbf00, 0x0001]
        let decoded = try bf.withUnsafeBytes { try Qwen38Model.embeddingRow($0.baseAddress!, width: 3, type: 30) }
        XCTAssertEqual(decoded, [1, -0.5, Float(bitPattern: 0x00010000)])
        var q4 = [UInt8](repeating: 0xF0, count: 18); q4[0] = 0; q4[1] = 0x3c
        let row = try q4.withUnsafeBytes { try Qwen38Model.embeddingRow($0.baseAddress!, width: 32, type: 2) }
        XCTAssertEqual(Array(row[0..<16]), [Float](repeating: -8, count: 16))
        XCTAssertEqual(Array(row[16..<32]), [Float](repeating: 7, count: 16))
    }

    func testSyntheticFullGraphChunkingResetAndCancellation() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["DS4_TEST_QWEN38_GPU"] == "1", "Run explicitly in the serialized GPU validation job")
        let url = temporary(); defer { try? FileManager.default.removeItem(at: url) }
        try Qwen38Fixture.write(path: url.path, full: true)
        let model = try Qwen38Model(model: GGUFModel(path: url.path), contextSize: 64)
        let tokens = Array(1...16)
        let batched = try model.evaluate(tokens: tokens, cancelled: { false })
        XCTAssertEqual(model.position, 16); XCTAssertEqual(batched.count, 248320)
        XCTAssertTrue(batched.allSatisfy(\.isFinite)); XCTAssertGreaterThan(batched.map(abs).max()!, 1e-4)
        try model.reset()
        var sequential = [Float]()
        for token in tokens { sequential = try model.evaluate(tokens: [token], cancelled: { false }) }
        let maxError = zip(batched, sequential).map { abs($0 - $1) }.max()!
        XCTAssertLessThan(maxError, 0.001, "Causal batch/decode mismatch: \(maxError)")
        try model.reset()
        XCTAssertEqual(try model.evaluate(tokens: tokens, cancelled: { false }), batched)
        XCTAssertThrowsError(try model.evaluate(tokens: [248320], cancelled: { false }))
        XCTAssertEqual(model.position, 16)
        XCTAssertThrowsError(try model.evaluate(tokens: [Int](repeating: 1, count: 49), cancelled: { false }))
        let gate = Qwen38CancellationGate()
        XCTAssertThrowsError(try model.evaluate(tokens: [17, 18], cancelled: { gate.next() }))
        XCTAssertThrowsError(try model.evaluate(tokens: [17], cancelled: { false }))
        try model.reset()
        XCTAssertEqual(try model.evaluate(tokens: tokens, cancelled: { false }), batched)
    }
}

private final class Qwen38CancellationGate: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    func next() -> Bool { lock.lock(); defer { lock.unlock() }; count += 1; return count >= 4 }
}
