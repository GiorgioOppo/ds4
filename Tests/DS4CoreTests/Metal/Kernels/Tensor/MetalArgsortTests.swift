import XCTest
@testable import DS4Metal

/// Phase 9 (router/indexer): validates the real metal/argsort.metal kernel
/// (kernel_argsort_f32_i32_desc) single-pass top-k vs a CPU descending sort.
final class MetalArgsortTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/argsort.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testArgsortTopKDesc() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xA50A
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        // 256 experts, top-8 (DeepSeek-style routing dims).
        let n = 256, topK = 8
        var scores = [Float](repeating: 0, count: n)
        // distinct values to avoid tie ambiguity
        var used = Set<Int32>()
        for i in 0..<n {
            var v = rndF()
            var bits = v.bitPattern
            while used.contains(Int32(bitPattern: bits)) { v = rndF(); bits = v.bitPattern }
            used.insert(Int32(bitPattern: bits))
            scores[i] = v
        }

        let gpu = try rt.argsortTopKDesc(scores, n: n, topK: topK)
        XCTAssertEqual(gpu.count, topK)

        let refOrder = (0..<n).sorted { scores[$0] > scores[$1] }
        for k in 0..<topK {
            XCTAssertEqual(Int(gpu[k]), refOrder[k], "top-\(k) index mismatch")
        }
    }

    func testStreamingTop512MatchesExactCPUOrderForWidePrefill() throws {
        let rt: MetalRuntime
        do { rt = try MetalRuntime() }
        catch { throw XCTSkip("Metal unavailable: \(error)") }

        let nComp = 1_280
        let nTokens = 32
        let topK = 512
        var scores = [Float](repeating: 0, count: nComp * nTokens)
        for token in 0..<nTokens {
            for column in 0..<nComp {
                // Unique finite values with a token-dependent permutation,
                // including negatives, pin both ordering and index recovery.
                let rank = (column * 733 + token * 197) % nComp
                scores[token * nComp + column] = Float(rank) - 700.25
            }
        }

        let scoreTensor = try GPUTensor.floats(rt, scores)
        let selected = try GPUTensor.zerosBytes(
            rt, byteLength: nTokens * topK * 4)
        // The streaming path must not touch global scratch.
        let scratch = try GPUTensor.zerosBytes(rt, byteLength: 4)
        let context = GraphContext(rt)
        try context.begin()
        try context.indexerTopKIndicesBatch(
            scores: scoreTensor, nComp: nComp, nTokens: nTokens,
            topK: topK, out: selected, scratch: scratch)
        context.commit()

        let pointer = selected.buffer.contents()
            .advanced(by: selected.byteOffset)
            .bindMemory(to: Int32.self, capacity: nTokens * topK)
        let gpu = Array(UnsafeBufferPointer(
            start: pointer, count: nTokens * topK))
        for token in 0..<nTokens {
            let base = token * nComp
            let expected = (0..<nComp).sorted {
                let lhs = scores[base + $0]
                let rhs = scores[base + $1]
                return lhs == rhs ? $0 < $1 : lhs > rhs
            }.prefix(topK).map(Int32.init)
            XCTAssertEqual(Array(gpu[token * topK..<(token + 1) * topK]),
                           expected, "token \(token)")
        }
    }
}
