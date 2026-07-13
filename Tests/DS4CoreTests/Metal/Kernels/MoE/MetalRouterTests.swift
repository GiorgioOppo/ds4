import XCTest
@testable import DS4Metal

/// Phase 9 / Stage A5: validates the real dsv4_misc.metal router kernels
/// (kernel_dsv4_router_finalize_one top-6, kernel_dsv4_router_weights_one) vs CPU.
final class MetalRouterTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_misc.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testRouterSelectAndWeights() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x2071
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        // 256 distinct positive probs.
        var probs = [Float](repeating: 0, count: 256)
        var used = Set<UInt32>()
        for i in 0..<256 {
            var v = abs(rndF()) + 0.001
            while used.contains(v.bitPattern) { v = abs(rndF()) + 0.001 }
            used.insert(v.bitPattern); probs[i] = v
        }

        let sel = try rt.routerFinalizeTop6(probs: probs)
        XCTAssertEqual(sel.count, 6)
        let refTop6 = Array((0..<256).sorted { probs[$0] > probs[$1] }.prefix(6))
        XCTAssertEqual(Set(sel.map { Int($0) }), Set(refTop6), "router top-6 set mismatch: \(sel) vs \(refTop6)")

        let w = try rt.routerWeights(probs: probs, selected: sel)
        var sum: Float = 0
        for i in 0..<6 { sum += probs[Int(sel[i])] }
        sum = max(sum, 6.103515625e-5)
        for i in 0..<6 {
            let ref = probs[Int(sel[i])] / sum * 1.5
            XCTAssertEqual(w[i], ref, accuracy: max(abs(ref),1)*1e-5, "router weight \(i)")
        }
    }
}
