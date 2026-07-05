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

    /// exp_probs_b: the bias shifts SELECTION only; the weights normalize the
    /// UNBIASED probs of the selected experts (ds4.c layer_topk_selected_
    /// experts_from_probs).
    func testRouterBiasAffectsSelectionOnly() throws {
        let rt = try makeRuntime()
        // Ramp probs (unbiased top-6 = 250..255) + a bias that promotes 0..5.
        let probs = [Float](repeating: 0.5, count: 256).enumerated().map { i, p in p + Float(i) * 1e-4 }
        var bias = [Float](repeating: 0, count: 256)
        for i in 0..<6 { bias[i] = 10 }   // push experts 0..5 to the top

        let ctx = GraphContext(rt)
        let pt = try GPUTensor.floats(rt, probs)
        let bt = try GPUTensor.floats(rt, bias)
        let selected = try GPUTensor.zerosBytes(rt, byteLength: 6 * 4)
        let weights = try GPUTensor.zeros(rt, floatCount: 6)
        try ctx.begin()
        try ctx.routerFinalizeTop6(probs: pt, selected: selected, bias: bt)
        try ctx.routerWeights(probs: pt, selected: selected, weights: weights)
        ctx.commit()

        let sp32 = selected.buffer.contents().bindMemory(to: Int32.self, capacity: 6)
        let sel = Array(UnsafeBufferPointer(start: sp32, count: 6)).map { Int($0) }
        XCTAssertEqual(Set(sel), Set(0..<6), "biased selection \(sel)")
        // Weights come from the UNBIASED probs.
        let gw = weights.floatArray(6)
        var s: Float = 0; for i in sel { s += probs[i] }; s = max(s, 6.103515625e-5)
        for i in 0..<6 {
            XCTAssertEqual(gw[i], probs[sel[i]] / s * 1.5, accuracy: 1e-5, "biased weight \(i)")
        }
    }

    /// ffn_gate_tid2eid: hash mode copies row min(token, rows-1) of the I32
    /// [6 x rows] table and ignores the probs entirely (ds4.c layer_hash_
    /// selected_experts; weights still normalize probs of the hash-selected).
    func testRouterHashModeSelectsTableRow() throws {
        let rt = try makeRuntime()
        let rows = 8
        var table = [Int32](repeating: 0, count: 6 * rows)
        for r in 0..<rows { for i in 0..<6 { table[r * 6 + i] = Int32(r * 10 + i) } }
        let probs = (0..<256).map { Float($0) * 1e-3 }

        let ctx = GraphContext(rt)
        let pt = try GPUTensor.floats(rt, probs)
        let ht = try GPUTensor.bytes(rt, table.withUnsafeBytes { Array($0) }, elementCount: table.count)
        let selected = try GPUTensor.zerosBytes(rt, byteLength: 6 * 4)
        try ctx.begin()
        try ctx.routerFinalizeTop6(probs: pt, selected: selected, hashTable: ht, hashRows: rows, token: 3)
        ctx.commit()
        let sp32 = selected.buffer.contents().bindMemory(to: Int32.self, capacity: 6)
        XCTAssertEqual(Array(UnsafeBufferPointer(start: sp32, count: 6)), [30, 31, 32, 33, 34, 35])

        // Token beyond the table clamps to the last row (kernel min(token, rows-1)).
        let ctx2 = GraphContext(rt)
        try ctx2.begin()
        try ctx2.routerFinalizeTop6(probs: pt, selected: selected, hashTable: ht, hashRows: rows, token: 1000)
        ctx2.commit()
        XCTAssertEqual(Array(UnsafeBufferPointer(start: sp32, count: 6)), [70, 71, 72, 73, 74, 75])
    }
}
