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
}
