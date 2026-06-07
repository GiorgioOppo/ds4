import XCTest
import Foundation
@testable import DS4Metal

/// Stage C: validates the router sub-block — logits -> softplus -> sqrt (unary
/// chain) -> top-6 select -> weight normalize — composed in one command buffer.
final class GraphRouterTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_misc.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testRouterSubBlock() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xC0DE12
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) * 4 }

        var logits = [Float](repeating: 0, count: 256)
        for i in 0..<256 { logits[i] = rndF() }

        let ctx = GraphContext(rt)
        let lt = try GPUTensor.floats(rt, logits)
        let sp = try GPUTensor.zeros(rt, floatCount: 256)
        let probs = try GPUTensor.zeros(rt, floatCount: 256)
        let selected = try GPUTensor.zerosBytes(rt, byteLength: 6 * 4)
        let weights = try GPUTensor.zeros(rt, floatCount: 6)

        try ctx.begin()
        try ctx.unary(lt, op: .softplus, out: sp, width: 256)
        try ctx.unary(sp, op: .sqrt, out: probs, width: 256)
        try ctx.routerFinalizeTop6(probs: probs, selected: selected)
        try ctx.routerWeights(probs: probs, selected: selected, weights: weights)
        ctx.commit()

        // probs ~ sqrt(softplus(logits))
        let gp = probs.floatArray(256)
        for i in 0..<256 {
            let ref = (logf(1 + expf(logits[i]))).squareRoot()
            XCTAssertEqual(gp[i], ref, accuracy: max(abs(ref),1)*1e-3, "probs \(i)")
        }
        // selected = top-6 of probs
        let sp32 = selected.buffer.contents().bindMemory(to: Int32.self, capacity: 6)
        let sel = Array(UnsafeBufferPointer(start: sp32, count: 6)).map { Int($0) }
        let refTop6 = Set((0..<256).sorted { gp[$0] > gp[$1] }.prefix(6))
        XCTAssertEqual(Set(sel), refTop6, "router top6 \(sel)")
        // weights
        let gw = weights.floatArray(6)
        var s: Float = 0; for i in sel { s += gp[i] }; s = max(s, 6.103515625e-5)
        for i in 0..<6 {
            let ref = gp[sel[i]] / s * 1.5
            XCTAssertEqual(gw[i], ref, accuracy: max(abs(ref),1)*1e-4, "weight \(i)")
        }
    }
}
