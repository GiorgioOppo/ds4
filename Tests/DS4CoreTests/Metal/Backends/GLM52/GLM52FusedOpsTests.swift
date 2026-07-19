import DS4Core
import XCTest
@testable import DS4Metal

/// Le fusioni ausiliarie del decode: argmax greedy sul device (due stadi,
/// pareggi all'indice più basso come la CPU) e la variante ENCODE del
/// top-k dell'indexer, giudicata contro il wrapper standalone già
/// validato (stessi kernel, stesso staging).
final class GLM52FusedOpsTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    func testDeviceArgmaxMatchesCPUIncludingTies() throws {
        let runtime = try makeRuntime()
        var state: UInt64 = 3
        func nextFloat() -> Float {
            state = state &* 6_364_136_223_846_793_005
                &+ 1_442_695_040_888_963_407
            return Float(Int(truncatingIfNeeded: state >> 33) % 1000) / 250
        }
        // Dimensione da head vero (non multipla dei chunk) + pareggio
        // artificiale: il massimo compare DUE volte, deve vincere il primo.
        var values = (0..<151_251).map { _ in nextFloat() }
        let peak = (values.max() ?? 0) + 1
        values[70_007] = peak
        values[130_501] = peak
        // Head sintetico minimo: vocab = count, embed piccolo — qui si prova
        // SOLO la coppia di kernel argmax, con i logits scritti direttamente.
        let logits = try runtime.glm52GraphBuffer(values)
        let partials = GLM52ResidentOutputHead.argmaxPartials
        let partialValues = try runtime.glm52GraphOutputBuffer(
            floats: partials)
        let partialIndices = try runtime.glm52GraphOutputBuffer(
            floats: partials)
        let result = try runtime.glm52GraphOutputBuffer(floats: 1)
        guard let commandBuffer = runtime.queue.makeCommandBuffer() else {
            throw MetalError.bufferAlloc
        }
        let chunk = (values.count + partials - 1) / partials
        try runtime.glm52GraphEncode(
            into: commandBuffer,
            pipelineName: "kernel_glm52_argmax_partial_f32",
            arguments: [UInt32(values.count), UInt32(chunk), 0, 0],
            buffers: [logits, partialValues, partialIndices],
            threadgroups: MTLSize(width: partials, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1),
            threadgroupMemoryLength: 256 * 2 * MemoryLayout<Float>.stride)
        try runtime.glm52GraphEncode(
            into: commandBuffer,
            pipelineName: "kernel_glm52_argmax_final_f32",
            arguments: [UInt32(partials), 0, 0, 0],
            buffers: [partialValues, partialIndices, result],
            threadgroups: MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1),
            threadgroupMemoryLength: 64 * 2 * MemoryLayout<Float>.stride)
        try runtime.glm52GraphCommit(commandBuffer)
        let device = result.contents()
            .bindMemory(to: Int32.self, capacity: 1).pointee
        XCTAssertEqual(device, 70_007, "pareggio: vince l'indice più basso")
    }

    func testEncodedTopKMatchesStandaloneWrapper() throws {
        let runtime = try makeRuntime()
        var state: UInt64 = 11
        func nextFloat() -> Float {
            state = state &* 6_364_136_223_846_793_005
                &+ 1_442_695_040_888_963_407
            return Float(Int(truncatingIfNeeded: state >> 30) % 100_000)
        }
        // Score DISTINTI (il tie-break del sort bitonico non è quello
        // dell'oracle CPU — stessa premessa delle fixture esistenti).
        let rowCount = 3000
        let topK = 512
        var seen = Set<Float>()
        let scores: [Float] = (0..<rowCount).map { _ in
            var v = nextFloat()
            while seen.contains(v) { v += 0.5 }
            seen.insert(v)
            return v
        }
        let expected = try runtime.glm52IndexerTopK(
            scores: scores, rowCount: rowCount, tokenCount: 1, topK: topK)

        let scoreBuffer = try runtime.glm52GraphBuffer(scores)
        let output = try runtime.glm52GraphOutputBuffer(floats: topK)
        let scratch = try runtime.glm52GraphOutputBuffer(
            floats: 2 * rowCount)
        guard let commandBuffer = runtime.queue.makeCommandBuffer() else {
            throw MetalError.bufferAlloc
        }
        try runtime.glm52EncodeIndexerTopK(
            into: commandBuffer, scores: scoreBuffer, rowCount: rowCount,
            topK: topK, output: output, sortScratch: scratch)
        try runtime.glm52GraphCommit(commandBuffer)
        let pointer = output.contents()
            .bindMemory(to: UInt32.self, capacity: topK)
        XCTAssertEqual(Array(UnsafeBufferPointer(start: pointer,
                                                 count: topK)), expected)
    }
}
