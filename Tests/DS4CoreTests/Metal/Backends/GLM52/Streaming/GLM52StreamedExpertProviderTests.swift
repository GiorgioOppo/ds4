import XCTest
import DS4Core
@testable import DS4Metal

/// The per-expert streaming provider must slice the packed gate|up|down
/// record into byte-exact expert weights, hit the slot cache on repeats,
/// refuse types without a validated kernel, and the reader's ranged read
/// must serve exact sub-slices. Synthetic pattern files only — no device.
final class GLM52StreamedExpertProviderTests: XCTestCase {
    private let expertCount: UInt64 = 8

    private func patternByte(_ i: Int) -> UInt8 {
        UInt8(truncatingIfNeeded: i &* 131 &+ (i >> 7) &+ 3)
    }

    private func descriptor(name: String, type: UInt32, dims: [UInt64],
                            offset: UInt64) -> GLM52WeightDescriptor {
        GLM52WeightDescriptor(
            name: name, type: type, dims: dims, absOffset: offset,
            bytes: GGUF.tensorNBytes(type: type, elements: dims.reduce(1, *))!)
    }

    private func routedWeights(gateUpType: UInt32 = GLM52TensorSchema.q4_K,
                               downType: UInt32 = GLM52TensorSchema.q6_K)
        -> GLM52RoutedExpertWeights {
        let gate = descriptor(name: "blk.5.ffn_gate_exps.weight",
                              type: gateUpType,
                              dims: [256, 512, expertCount], offset: 1_024)
        let up = descriptor(name: "blk.5.ffn_up_exps.weight",
                            type: gateUpType,
                            dims: [256, 512, expertCount],
                            offset: gate.absOffset + gate.bytes + 512)
        let down = descriptor(name: "blk.5.ffn_down_exps.weight",
                              type: downType,
                              dims: [512, 256, expertCount],
                              offset: up.absOffset + up.bytes + 512)
        return GLM52RoutedExpertWeights(gate: gate, up: up, down: down)
    }

    private func makeReader() throws -> GLM52PayloadReader {
        let weights = routedWeights()
        let byteCount = Int(weights.down.absOffset + weights.down.bytes)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("glm52-provider-\(UUID().uuidString).bin")
        var data = Data(count: byteCount)
        data.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            for i in 0..<byteCount { raw[i] = patternByte(i) }
        }
        try data.write(to: url)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: url)
        }
        return try GLM52PayloadReader(path: url.path)
    }

    private func expectedBytes(_ descriptor: GLM52WeightDescriptor,
                               expert: Int) -> [UInt8] {
        let perExpert = Int(descriptor.bytes / expertCount)
        let start = Int(descriptor.absOffset) + expert * perExpert
        return (start..<start + perExpert).map(patternByte)
    }

    func testExpertRecordsAreByteExactAndCached() throws {
        let reader = try makeReader()
        let weights = routedWeights()
        let provider = try GLM52StreamedExpertProvider(
            reader: reader, layer: 5, weights: weights, slotCount: 8)

        for id in [UInt32(3), 0, 7] {
            let expert = try provider.expert(id)
            XCTAssertEqual(expert.gateUpType, GLM52TensorSchema.q4_K)
            XCTAssertEqual(expert.downType, GLM52TensorSchema.q6_K)
            XCTAssertEqual(expert.gate,
                           expectedBytes(weights.gate, expert: Int(id)))
            XCTAssertEqual(expert.up,
                           expectedBytes(weights.up, expert: Int(id)))
            XCTAssertEqual(expert.down,
                           expectedBytes(weights.down, expert: Int(id)))
        }
        XCTAssertEqual(provider.stats.misses, 3)
        _ = try provider.expert(3)
        XCTAssertEqual(provider.stats.hits, 1)
        XCTAssertEqual(provider.stats.misses, 3)
    }

    func testUnsupportedRoutedTypeIsRefusedAtLoad() throws {
        let reader = try makeReader()
        // F32 stands in for any type without a validated routed kernel
        // (IQ2_XXS graduated into the supported set with its kernel).
        XCTAssertThrowsError(try GLM52StreamedExpertProvider(
            reader: reader, layer: 5,
            weights: routedWeights(downType: GLM52TensorSchema.f32),
            slotCount: 8)) {
            XCTAssertEqual(
                $0 as? GLM52StreamedExpertProviderError,
                .unsupportedExpertType(layer: 5,
                                       type: GLM52TensorSchema.f32))
        }
    }

    func testRangedDescriptorReadServesExactSlices() throws {
        let reader = try makeReader()
        let gate = routedWeights().gate

        let slice = try reader.bytes(of: gate, byteOffset: 100, byteCount: 40)
        XCTAssertEqual(slice, (0..<40).map {
            patternByte(Int(gate.absOffset) + 100 + $0)
        })
        // The range is bounded by the DESCRIPTOR, not just the file.
        XCTAssertThrowsError(try reader.bytes(
            of: gate, byteOffset: gate.bytes, byteCount: 1))
        XCTAssertThrowsError(try reader.bytes(
            of: gate, byteOffset: 0, byteCount: gate.bytes + 1))
        XCTAssertThrowsError(try reader.bytes(
            of: gate, byteOffset: 10, byteCount: 0))
    }

    func testGreedyDecodingArgmaxAndLoop() throws {
        // Ties prefer the lower token id; empty logits yield nil.
        XCTAssertEqual(GLM52GreedyDecoding.argmax([0.5, 2.0, 2.0, -1]), 1)
        XCTAssertEqual(GLM52GreedyDecoding.argmax([-3, -1, -2]), 1)
        XCTAssertNil(GLM52GreedyDecoding.argmax([]))

        // Scripted forward: token t leads to logits peaking at t+1.
        var stepped: [Int32] = []
        func step(_ token: Int32) throws -> [Float] {
            stepped.append(token)
            var logits = [Float](repeating: 0, count: 6)
            logits[Int(token) + 1] = 1
            return logits
        }
        let generated = try GLM52GreedyDecoding.generate(
            logitsAfterPrompt: [0, 0, 1, 0, 0, 0],
            maxNewTokens: 3, endTokens: [], step: step)
        XCTAssertEqual(generated, [2, 3, 4])
        XCTAssertEqual(stepped, [2, 3])   // the last token is never stepped

        // End token stops the loop and is included in the output.
        stepped = []
        let ended = try GLM52GreedyDecoding.generate(
            logitsAfterPrompt: [0, 0, 1, 0, 0, 0],
            maxNewTokens: 10, endTokens: [3], step: step)
        XCTAssertEqual(ended, [2, 3])
        XCTAssertEqual(stepped, [2])
    }
}
