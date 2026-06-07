import XCTest
import Foundation
@testable import DS4Metal

/// Stage C: validates the routed-MoE FFN block — gate/up (shared-activation
/// kernel_mul_mv_id_q4_K) -> swiglu_weight -> down (per-expert activation, ne11=K)
/// -> sum6 — chained on GPUTensors, vs a CPU reference dequantizing the same Q4_K
/// bytes. This confirms the per-expert-activation down path (the open question).
final class GraphMoETests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/moe.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private func half(_ w: [UInt8], _ o: Int) -> Float { Float(Float16(bitPattern: UInt16(w[o]) | (UInt16(w[o+1])<<8))) }
    private func scaleMinK4(_ j: Int, _ s: [UInt8], _ base: Int) -> (Float, Float) {
        if j < 4 { return (Float(s[base+j] & 63), Float(s[base+j+4] & 63)) }
        let d = (s[base+j+4] & 0xF) | ((s[base+j-4] >> 6) << 4)
        let m = (s[base+j+4] >> 4) | ((s[base+j] >> 6) << 4)
        return (Float(d), Float(m))
    }
    private func dequantBlock(_ w: [UInt8], _ bb: Int) -> [Float] {
        let d = half(w, bb), dmin = half(w, bb+2); let sb = bb+4; var q = bb+16
        var out = [Float](repeating: 0, count: 256); var oi = 0, isb = 0, j = 0
        while j < 256 {
            let (s1,m1) = scaleMinK4(isb, w, sb); let d1 = d*s1, mm1 = dmin*m1
            let (s2,m2) = scaleMinK4(isb+1, w, sb); let d2 = d*s2, mm2 = dmin*m2
            for l in 0..<32 { out[oi] = d1*Float(w[q+l] & 0xF) - mm1; oi += 1 }
            for l in 0..<32 { out[oi] = d2*Float(w[q+l] >> 4) - mm2; oi += 1 }
            q += 32; isb += 2; j += 64
        }
        return out
    }
    private func matRow(_ experts: [UInt8], _ e: Int, _ o: Int, _ act: [Float], _ inDim: Int, _ rowBytes: Int, _ expertBytes: Int) -> Float {
        let nblk = inDim/256; let rowBase = e*expertBytes + o*rowBytes; var acc: Float = 0
        for blk in 0..<nblk { let dq = dequantBlock(experts, rowBase+blk*144); for i in 0..<256 { acc += dq[i]*act[blk*256+i] } }
        return acc
    }

    func testRoutedMoEBlock() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xC0DE60
        func nb() -> UInt8 { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return UInt8(truncatingIfNeeded: seed >> 40) }
        func rf() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let embDim = 512, ffn = 512, nExp = 8, k = 6
        let rowBytes = (embDim/256)*144, expertBytes = rowBytes*ffn       // gate/up: ffn rows of embDim
        let downRowBytes = (ffn/256)*144, downExpertBytes = downRowBytes*embDim

        func randExperts(_ rows: Int, _ inDim: Int) -> [UInt8] {
            let rb = (inDim/256)*144; var bytes = [UInt8](repeating: 0, count: rb*rows*nExp); var off = 0
            for _ in 0..<(nExp*rows*(inDim/256)) {
                let d = Float16(abs(rf())*0.05), dmin = Float16(abs(rf())*0.02)
                withUnsafeBytes(of: d.bitPattern.littleEndian) { bytes[off] = $0[0]; bytes[off+1] = $0[1] }
                withUnsafeBytes(of: dmin.bitPattern.littleEndian) { bytes[off+2] = $0[0]; bytes[off+3] = $0[1] }
                for i in 0..<12 { bytes[off+4+i] = nb() }
                for i in 0..<128 { bytes[off+16+i] = nb() }
                off += 144
            }
            return bytes
        }
        let gateB = randExperts(ffn, embDim), upB = randExperts(ffn, embDim), downB = randExperts(embDim, ffn)
        var normed = [Float](repeating: 0, count: embDim); for i in 0..<embDim { normed[i] = rf() }
        let ids: [Int32] = [3, 0, 7, 5, 1, 6]
        var rw = [Float](repeating: 0, count: k); for i in 0..<k { rw[i] = abs(rf())+0.1 }

        let ctx = GraphContext(rt)
        let gE = try GPUTensor.bytes(rt, gateB, elementCount: gateB.count)
        let uE = try GPUTensor.bytes(rt, upB, elementCount: upB.count)
        let dE = try GPUTensor.bytes(rt, downB, elementCount: downB.count)
        let idsT = try GPUTensor.bytes(rt, ids.withUnsafeBytes { Array($0) }, elementCount: k)
        let normT = try GPUTensor.floats(rt, normed)
        let rwT = try GPUTensor.floats(rt, rw)
        let gate6 = try GPUTensor.zeros(rt, floatCount: k*ffn)
        let up6 = try GPUTensor.zeros(rt, floatCount: k*ffn)
        let mid6 = try GPUTensor.zeros(rt, floatCount: k*ffn)
        let down6 = try GPUTensor.zeros(rt, floatCount: k*embDim)
        let routed = try GPUTensor.zeros(rt, floatCount: embDim)

        try ctx.begin()
        try ctx.moeMatvecQ4K(experts: gE, ids: idsT, activation: normT, out: gate6, k: k, inDim: embDim, outDim: ffn, perExpertAct: false)
        try ctx.moeMatvecQ4K(experts: uE, ids: idsT, activation: normT, out: up6, k: k, inDim: embDim, outDim: ffn, perExpertAct: false)
        try ctx.moeSwiGLUWeight(gate: gate6, up: up6, weights: rwT, mid: mid6, width: ffn, rows: k)
        try ctx.moeMatvecQ4K(experts: dE, ids: idsT, activation: mid6, out: down6, k: k, inDim: ffn, outDim: embDim, perExpertAct: true)
        try ctx.moeSum6(experts: down6, out: routed, width: embDim)
        ctx.commit()

        // CPU reference
        var gate = [[Float]](repeating: [Float](repeating: 0, count: ffn), count: k)
        var up = gate, mid = gate
        for s in 0..<k { let e = Int(ids[s]); for o in 0..<ffn {
            gate[s][o] = matRow(gateB, e, o, normed, embDim, rowBytes, expertBytes)
            up[s][o]   = matRow(upB, e, o, normed, embDim, rowBytes, expertBytes)
            mid[s][o]  = (gate[s][o]/(1+expf(-gate[s][o]))) * up[s][o] * rw[s]
        } }
        var refRouted = [Float](repeating: 0, count: embDim)
        for s in 0..<k { let e = Int(ids[s]); for o in 0..<embDim {
            refRouted[o] += matRow(downB, e, o, mid[s], ffn, downRowBytes, downExpertBytes)
        } }
        let got = routed.floatArray(embDim)
        var maxRel: Float = 0
        for o in 0..<embDim { maxRel = max(maxRel, abs(got[o]-refRouted[o]) / max(abs(refRouted[o]), 0.5)) }
        XCTAssertLessThan(maxRel, 1e-2, "routed MoE block max rel \(maxRel)")
    }
}
