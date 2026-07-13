import XCTest
@testable import DS4Metal

/// Phase 9: validates the Swift get-rows (embedding gather) dispatch against the
/// expected half->float row copy.
final class MetalGetRowsTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/get_rows.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testGetRowsF16() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x7777
        func rnd() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30)
        }
        let nVocab = 200, nEmbd = 4096
        var table = [UInt16](repeating: 0, count: nVocab * nEmbd)
        for i in 0..<table.count { table[i] = Float16(rnd()).bitPattern }

        for id in [0, 1, 99, 199] {
            let gpu = try rt.getRowsF16(table: table, id: id, nEmbd: nEmbd, nVocab: nVocab)
            XCTAssertEqual(gpu.count, nEmbd)
            for i in 0..<nEmbd {
                let expected = Float(Float16(bitPattern: table[id * nEmbd + i]))
                XCTAssertEqual(gpu[i], expected, "get_rows id=\(id) i=\(i)")
            }
        }
    }

    func testGetRowsF32() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x3939
        func rnd() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28)
        }
        let nVocab = 128, nEmbd = 2048
        var table = [Float](repeating: 0, count: nVocab * nEmbd)
        for i in 0..<table.count { table[i] = rnd() }

        for id in [0, 1, 64, 127] {
            let gpu = try rt.getRowsF32(table: table, id: id, nEmbd: nEmbd, nVocab: nVocab)
            XCTAssertEqual(gpu.count, nEmbd)
            for i in 0..<nEmbd {
                XCTAssertEqual(gpu[i], table[id * nEmbd + i], "get_rows_f32 id=\(id) i=\(i)")
            }
        }
    }
}
