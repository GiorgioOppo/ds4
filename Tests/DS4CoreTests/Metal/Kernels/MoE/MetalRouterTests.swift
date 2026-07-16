import XCTest
@testable import DS4Metal

/// Phase 9 / Stage A5: validates the real dsv4_misc.metal router kernels
/// (kernel_dsv4_router_finalize_one top-6, kernel_dsv4_router_weights_one) vs CPU.
final class MetalRouterTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private func assertRouter(nExperts: Int, expertWeightScale: Float,
                              file: StaticString = #filePath, line: UInt = #line) throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x2071
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        var probs = [Float](repeating: 0, count: nExperts)
        var used = Set<UInt32>()
        for i in 0..<nExperts {
            var v = abs(rndF()) + 0.001
            while used.contains(v.bitPattern) { v = abs(rndF()) + 0.001 }
            used.insert(v.bitPattern); probs[i] = v
        }

        let sel = try rt.routerFinalizeTop6(probs: probs, nExperts: nExperts)
        XCTAssertEqual(sel.count, 6, file: file, line: line)
        let refTop6 = Array((0..<nExperts).sorted { probs[$0] > probs[$1] }.prefix(6))
        XCTAssertEqual(Set(sel.map { Int($0) }), Set(refTop6),
                       "router top-6 set mismatch: \(sel) vs \(refTop6)", file: file, line: line)

        let w = try rt.routerWeights(probs: probs, selected: sel,
                                     nExperts: nExperts,
                                     expertWeightScale: expertWeightScale)
        var sum: Float = 0
        for i in 0..<6 { sum += probs[Int(sel[i])] }
        sum = max(sum, 6.103515625e-5)
        for i in 0..<6 {
            let ref = probs[Int(sel[i])] / sum * expertWeightScale
            XCTAssertEqual(w[i], ref, accuracy: max(abs(ref),1)*1e-5,
                           "router weight \(i)", file: file, line: line)
        }
    }

    func testFlashRouterSelectAndWeights() throws {
        try assertRouter(nExperts: 256, expertWeightScale: 1.5)
    }

    func testProRouterSelectAndWeights() throws {
        try assertRouter(nExperts: 384, expertWeightScale: 2.5)
    }

    func testLegacyFlashOverloadsMatchExplicitAPIBitExactly() throws {
        let rt = try makeRuntime()
        let probs = (0..<256).map { Float(($0 * 73) % 257 + 1) / 257 }

        let legacySelected = try rt.routerFinalizeTop6(probs: probs)
        let explicitSelected = try rt.routerFinalizeTop6(probs: probs, nExperts: 256)
        XCTAssertEqual(legacySelected, explicitSelected)

        let legacyWeights = try rt.routerWeights(probs: probs, selected: legacySelected)
        let explicitWeights = try rt.routerWeights(probs: probs, selected: explicitSelected,
                                                   nExperts: 256, expertWeightScale: 1.5)
        for i in 0..<6 {
            XCTAssertEqual(legacyWeights[i].bitPattern, explicitWeights[i].bitPattern,
                           "Flash compatibility weight \(i)")
        }
    }
}
