import XCTest
@testable import DS4Metal

/// Phase 9 / Stage A7: validates the minor leftover kernels — cpy f32->f32
/// (flash ring copy) and get_rows i32 — vs trivial references.
final class MetalA7Tests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/cpy.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testCpyF32F32() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xA701
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }
        let n = 1000
        var x = [Float](repeating: 0, count: n)
        for i in 0..<n { x[i] = rndF() }
        let out = try rt.cpyF32toF32(x)
        XCTAssertEqual(out, x)
    }

    func testGetRowsI32() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xA702
        func nextI() -> Int32 { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Int32(truncatingIfNeeded: seed >> 32) }
        let nVocab = 100, nEmbd = 256
        var table = [Int32](repeating: 0, count: nVocab * nEmbd)
        for i in 0..<table.count { table[i] = nextI() }
        for id in [0, 1, 50, 99] {
            let row = try rt.getRowsI32(table: table, id: id, nEmbd: nEmbd, nVocab: nVocab)
            XCTAssertEqual(row, Array(table[id*nEmbd..<(id+1)*nEmbd]), "get_rows_i32 id=\(id)")
        }
    }
}
