import XCTest
import Foundation
@testable import DS4Metal

/// Stage D (expert-cache integration): validates that the split decode layer
/// [decodeRoute -> read the 6 selected ids -> gather only those 6 experts ->
/// decodeExperts(packed, ids 0..<6)] produces the SAME outHc as the resident
/// decodeLayer (full 256 experts, one command buffer). Proves the expert-cache
/// streaming path is numerically identical to the all-resident path.
final class ExpertCacheLayerTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_hc.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private var seed: UInt64 = 0xECA7
    private func rf() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }
    private func nb() -> UInt8 { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return UInt8(truncatingIfNeeded: seed >> 40) }
    private func q8(_ rows: Int, _ cols: Int) -> [UInt8] {
        var out = [UInt8](); for _ in 0..<rows { var c = 0; while c < cols {
            var amax: Float = 0; var blk = [Float](repeating: 0, count: 32)
            for i in 0..<32 { blk[i] = rf(); amax = max(amax, abs(blk[i])) }
            let dq = amax/127.0
            withUnsafeBytes(of: Float16(dq).bitPattern.littleEndian) { out.append($0[0]); out.append($0[1]) }
            for i in 0..<32 { out.append(UInt8(bitPattern: Int8(clamping: dq != 0 ? Int((blk[i]/dq).rounded()) : 0))) }
            c += 32 } }
        return out
    }
    // random block_q4_K experts: e experts of (rows x inDim) -> e*rows*(inDim/256)*144 bytes
    private func q4kExperts(_ e: Int, _ rows: Int, _ inDim: Int) -> [UInt8] {
        let nblk = e * rows * (inDim/256)
        var b = [UInt8](repeating: 0, count: nblk*144); var off = 0
        for _ in 0..<nblk {
            let d = Float16(abs(rf())*0.05), dm = Float16(abs(rf())*0.02)
            withUnsafeBytes(of: d.bitPattern.littleEndian) { b[off] = $0[0]; b[off+1] = $0[1] }
            withUnsafeBytes(of: dm.bitPattern.littleEndian) { b[off+2] = $0[0]; b[off+3] = $0[1] }
            for i in 0..<12 { b[off+4+i] = nb() }
            for i in 0..<128 { b[off+16+i] = nb() }
            off += 144
        }
        return b
    }

    func testExpertCacheLayerMatchesResident() throws {
        let rt = try makeRuntime()
        let d = DSV4Dims(nEmbd: 512, nHC: 4, headDim: 512, nHead: 2, qRank: 256, qDim: 1024,
                         sharedFfn: 512, nExperts: 256, expertFfn: 256, k: 6, nRot: 64, vocab: 1024, nOutGroup: 2, nLoraO: 128)
        let hcDim = d.nHC * d.nEmbd
        let nKeys = 32, pos = 0

        func tf(_ n: Int) throws -> GPUTensor { var a = [Float](repeating: 0, count: n); for i in 0..<n { a[i] = rf() }; return try GPUTensor.floats(rt, a) }
        func tf16(_ n: Int) throws -> GPUTensor { var a = [UInt16](repeating: 0, count: n); for i in 0..<n { a[i] = Float16(rf()).bitPattern }; return try GPUTensor.bytes(rt, a.withUnsafeBytes { Array($0) }, elementCount: n) }
        func tq8(_ r: Int, _ c: Int) throws -> GPUTensor { try GPUTensor.bytes(rt, q8(r, c), elementCount: r*c) }

        let gateB = q4kExperts(d.nExperts, d.expertFfn, d.nEmbd)
        let upB = q4kExperts(d.nExperts, d.expertFfn, d.nEmbd)
        let downB = q4kExperts(d.nExperts, d.nEmbd, d.expertFfn)
        let w = try LayerWeights(
            hcAttnFn: tf16(24*hcDim), attnScale: tf(3), attnBase: tf(24), attnNorm: tf(d.nEmbd),
            qA: tq8(d.qRank, d.nEmbd), qANorm: tf(d.qRank), qB: tq8(d.qDim, d.qRank), kvW: tq8(d.headDim, d.nEmbd),
            kvNorm: tf(d.headDim), attnSinks: tf(d.nHead), attnOutA: tq8(d.nOutGroup*d.nLoraO, d.attnGroupDim), attnOut: tq8(d.nEmbd, d.attnLowDim), hcFfnFn: tf16(24*hcDim), ffnScale: tf(3),
            ffnBase: tf(24), ffnNorm: tf(d.nEmbd), sharedGate: tq8(d.sharedFfn, d.nEmbd), sharedUp: tq8(d.sharedFfn, d.nEmbd),
            sharedDown: tq8(d.nEmbd, d.sharedFfn), routerW: tq8(d.nExperts, d.nEmbd),
            expGate: try GPUTensor.bytes(rt, gateB, elementCount: gateB.count),
            expUp: try GPUTensor.bytes(rt, upB, elementCount: upB.count),
            expDown: try GPUTensor.bytes(rt, downB, elementCount: downB.count))

        let rope = RopeParams(nCtxOrig: 4096, freqBase: 10000, freqScale: 1, extFactor: 0, attnFactor: 1, betaFast: 32, betaSlow: 1)
        let curHc = try tf(hcDim)

        // Resident path.
        let sR = try DecodeScratch(rt, d, maxKeys: nKeys)
        let rawR = try GPUTensor.zeros(rt, floatCount: nKeys * d.headDim)
        let outR = try GPUTensor.zeros(rt, floatCount: hcDim)
        let cr = GraphContext(rt); try cr.begin()
        try cr.decodeLayer(curHc: curHc, w: w, s: sR, d: d, rope: rope, rawCache: rawR, nKeys: nKeys, pos: pos, outHc: outR, rmsEps: 1e-5, hcEps: 1e-3)
        cr.commit()
        let resident = outR.floatArray(hcDim)

        // Expert-cache path: route -> read ids -> gather 6 -> experts.
        let sC = try DecodeScratch(rt, d, maxKeys: nKeys)
        let rawC = try GPUTensor.zeros(rt, floatCount: nKeys * d.headDim)
        let outC = try GPUTensor.zeros(rt, floatCount: hcDim)
        let c1 = GraphContext(rt); try c1.begin()
        try c1.decodeRoute(curHc: curHc, w: w, s: sC, d: d, rope: rope, rawCache: rawC, nKeys: nKeys, pos: pos, rmsEps: 1e-5, hcEps: 1e-3)
        c1.commit()
        let selPtr = sC.selected.buffer.contents().bindMemory(to: Int32.self, capacity: d.k)
        let ids = Array(UnsafeBufferPointer(start: selPtr, count: d.k))

        func pack(_ full: [UInt8], _ rows: Int, _ inDim: Int) -> [UInt8] {
            let eb = rows * (inDim/256) * 144
            var out = [UInt8](repeating: 0, count: ids.count * eb)
            for (i, e) in ids.enumerated() { for b in 0..<eb { out[i*eb+b] = full[Int(e)*eb+b] } }
            return out
        }
        let pg = try GPUTensor.bytes(rt, pack(gateB, d.expertFfn, d.nEmbd), elementCount: 1)
        let pu = try GPUTensor.bytes(rt, pack(upB, d.expertFfn, d.nEmbd), elementCount: 1)
        let pd = try GPUTensor.bytes(rt, pack(downB, d.nEmbd, d.expertFfn), elementCount: 1)
        let idsPacked = try GPUTensor.bytes(rt, (Array(0..<Int32(d.k))).withUnsafeBytes { Array($0) }, elementCount: d.k)

        let c2 = GraphContext(rt); try c2.begin()
        try c2.decodeExperts(w: w, s: sC, d: d, gateExp: pg, upExp: pu, downExp: pd, ids: idsPacked, outHc: outC)
        c2.commit()
        let cached = outC.floatArray(hcDim)

        var maxAbs: Float = 0
        for i in 0..<hcDim { maxAbs = max(maxAbs, abs(resident[i] - cached[i])) }
        XCTAssertLessThan(maxAbs, 1e-3, "expert-cache vs resident outHc max abs diff \(maxAbs)")
    }
}
