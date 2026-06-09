import XCTest
@testable import DS4Metal

/// Equivalence test for the FUSED decode MoE path: pair_swiglu (gate+up+SwiGLU·w)
/// and down_sum6 (down projection + sum over 6 slots) must reproduce the
/// validated 5-dispatch path on identical random quantized experts.
/// Uses the user's model scheme: iq2_xxs gate/up + q2_K down.
final class MetalMoEFusedDecodeTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/moe.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private var seed: UInt64 = 0xFEED
    private func nextByte() -> UInt8 { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return UInt8(truncatingIfNeeded: seed >> 40) }
    private func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

    /// Random iq2_xxs experts (66 B / 256 elems) with a finite small `d` (half @0).
    private func makeIQ2XXSExperts(_ count: Int, rows: Int, inDim: Int) -> [UInt8] {
        let nblk = inDim / 256, rowBytes = nblk * 66
        var w = [UInt8](repeating: 0, count: count * rows * rowBytes)
        var off = 0
        for _ in 0..<(count * rows * nblk) {
            w[off] = 0x00; w[off + 1] = 0x30                       // d = 0.125
            for i in 2..<66 { w[off + i] = nextByte() }            // qs
            off += 66
        }
        return w
    }

    /// Random q2_K experts (84 B / 256 elems) with finite d/dmin (halfs @80/@82).
    private func makeQ2KExperts(_ count: Int, rows: Int, inDim: Int) -> [UInt8] {
        let nblk = inDim / 256, rowBytes = nblk * 84
        var w = [UInt8](repeating: 0, count: count * rows * rowBytes)
        var off = 0
        for _ in 0..<(count * rows * nblk) {
            for i in 0..<80 { w[off + i] = nextByte() }            // scales + qs
            w[off + 80] = 0x00; w[off + 81] = 0x30                 // d    = 0.125
            w[off + 82] = 0x00; w[off + 83] = 0x2C                 // dmin = 0.0625
            off += 84
        }
        return w
    }

    func testFusedMatchesUnfused() throws {
        let rt = try makeRuntime()
        let nEmbd = 512, ffn = 256, nExperts = 8, k = 6
        let clamp: Float = 10.0

        let gateW = makeIQ2XXSExperts(nExperts, rows: ffn, inDim: nEmbd)
        let upW = makeIQ2XXSExperts(nExperts, rows: ffn, inDim: nEmbd)
        let downW = makeQ2KExperts(nExperts, rows: nEmbd, inDim: ffn)
        let x = (0..<nEmbd).map { _ in rndF() }
        let rw = (0..<k).map { _ in abs(rndF()) + 0.1 }
        let ids: [Int32] = [0, 2, 4, 5, 6, 7]

        let gate = try GPUTensor.bytes(rt, gateW, elementCount: gateW.count)
        let up = try GPUTensor.bytes(rt, upW, elementCount: upW.count)
        let down = try GPUTensor.bytes(rt, downW, elementCount: downW.count)
        let act = try GPUTensor.floats(rt, x)
        let weights = try GPUTensor.floats(rt, rw)
        let idsBuf = try GPUTensor.bytes(rt, ids.withUnsafeBytes { Array($0) }, elementCount: k)

        // A: validated 5-dispatch path.
        let gate6 = try GPUTensor.zeros(rt, floatCount: k * ffn)
        let up6 = try GPUTensor.zeros(rt, floatCount: k * ffn)
        let mid6 = try GPUTensor.zeros(rt, floatCount: k * ffn)
        let down6 = try GPUTensor.zeros(rt, floatCount: k * nEmbd)
        let routedA = try GPUTensor.zeros(rt, floatCount: nEmbd)
        let a = GraphContext(rt); try a.begin()
        try a.moeMatvecID(.iq2_xxs, experts: gate, ids: idsBuf, activation: act, out: gate6, k: k, inDim: nEmbd, outDim: ffn, perExpertAct: false)
        try a.moeMatvecID(.iq2_xxs, experts: up, ids: idsBuf, activation: act, out: up6, k: k, inDim: nEmbd, outDim: ffn, perExpertAct: false)
        try a.moeSwiGLUWeight(gate: gate6, up: up6, weights: weights, mid: mid6, width: ffn, rows: k, clampValue: clamp)
        try a.moeMatvecID(.q2_K, experts: down, ids: idsBuf, activation: mid6, out: down6, k: k, inDim: ffn, outDim: nEmbd, perExpertAct: true)
        try a.moeSum6(experts: down6, out: routedA, width: nEmbd)
        a.commit()

        // B: fused path (pair_swiglu + down_sum6).
        let gS = try GPUTensor.zeros(rt, floatCount: k * ffn)
        let uS = try GPUTensor.zeros(rt, floatCount: k * ffn)
        let midB = try GPUTensor.zeros(rt, floatCount: k * ffn)
        let routedB = try GPUTensor.zeros(rt, floatCount: nEmbd)
        let b = GraphContext(rt); try b.begin()
        try b.moePairSwiGLU(.iq2_xxs, gateExp: gate, upExp: up, ids: idsBuf, activation: act,
                            weights: weights, gateScratch: gS, upScratch: uS, mid: midB,
                            k: k, inDim: nEmbd, outDim: ffn, clamp: clamp)
        try b.moeDownSum6(.q2_K, experts: down, ids: idsBuf, mid: midB, out: routedB,
                          inDim: ffn, outDim: nEmbd)
        b.commit()

        // mid must match (same dots, same activation math)…
        let mA = mid6.floatArray(k * ffn), mB = midB.floatArray(k * ffn)
        for i in 0..<(k * ffn) {
            XCTAssertEqual(mA[i], mB[i], accuracy: max(1e-4, abs(mA[i]) * 1e-3), "mid[\(i)]")
        }
        // …and so must the routed output.
        let rA = routedA.floatArray(nEmbd), rB = routedB.floatArray(nEmbd)
        for i in 0..<nEmbd {
            XCTAssertEqual(rA[i], rB[i], accuracy: max(1e-3, abs(rA[i]) * 1e-3), "routed[\(i)]")
        }
    }
}
