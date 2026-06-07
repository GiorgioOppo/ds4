import XCTest
import Foundation
@testable import DS4Metal

/// Stage C: validates the shared-expert FFN decode-layer block
/// (pre-norm -> SwiGLU MLP over Q8_0 weights -> residual) composed on GPUTensors
/// in one command buffer, vs a CPU reference reading the same quantized bytes.
final class GraphFFNTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dense.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private func quantQ8(_ row: [Float]) -> [UInt8] {
        var out: [UInt8] = []
        var b = 0
        while b < row.count {
            let blk = Array(row[b..<b+32])
            let amax = blk.map { abs($0) }.max() ?? 0
            let d = amax / 127.0
            withUnsafeBytes(of: Float16(d).bitPattern.littleEndian) { out.append(contentsOf: $0) }
            for x in blk { out.append(UInt8(bitPattern: Int8(clamping: d != 0 ? Int((x/d).rounded()) : 0))) }
            b += 32
        }
        return out
    }
    private func deqRow(_ w: [UInt8], _ base: Int, _ inDim: Int) -> [Float] {
        var r = [Float](repeating: 0, count: inDim)
        let nb = inDim / 32
        for blk in 0..<nb {
            let o = base + blk*34
            let d = Float(Float16(bitPattern: UInt16(w[o]) | (UInt16(w[o+1])<<8)))
            for i in 0..<32 { r[blk*32+i] = Float(Int8(bitPattern: w[o+2+i])) * d }
        }
        return r
    }
    private func mat(_ w: [UInt8], _ x: [Float], _ inDim: Int, _ outDim: Int) -> [Float] {
        let rb = (inDim/32)*34
        var o = [Float](repeating: 0, count: outDim)
        for r in 0..<outDim {
            let dq = deqRow(w, r*rb, inDim)
            var acc: Float = 0
            for i in 0..<inDim { acc += dq[i]*x[i] }
            o[r] = acc
        }
        return o
    }

    func testFFNBlock() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xCFFA
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let inDim = 1024, ffnDim = 2048
        let eps: Float = 1e-5
        var x = [Float](repeating: 0, count: inDim)
        var nw = [Float](repeating: 0, count: inDim)
        for i in 0..<inDim { x[i] = rndF(); nw[i] = rndF() }
        func qWeight(_ rows: Int, _ cols: Int) -> [UInt8] {
            var bytes: [UInt8] = []
            for _ in 0..<rows { var row = [Float](repeating: 0, count: cols); for i in 0..<cols { row[i] = rndF() }; bytes += quantQ8(row) }
            return bytes
        }
        let gateB = qWeight(ffnDim, inDim), upB = qWeight(ffnDim, inDim), downB = qWeight(inDim, ffnDim)

        let ctx = GraphContext(rt)
        let xt = try GPUTensor.floats(rt, x), nwt = try GPUTensor.floats(rt, nw)
        let gW = try GPUTensor.bytes(rt, gateB, elementCount: ffnDim*inDim)
        let uW = try GPUTensor.bytes(rt, upB, elementCount: ffnDim*inDim)
        let dW = try GPUTensor.bytes(rt, downB, elementCount: inDim*ffnDim)
        let normed = try GPUTensor.zeros(rt, floatCount: inDim)
        let gate = try GPUTensor.zeros(rt, floatCount: ffnDim)
        let up = try GPUTensor.zeros(rt, floatCount: ffnDim)
        let mid = try GPUTensor.zeros(rt, floatCount: ffnDim)
        let down = try GPUTensor.zeros(rt, floatCount: inDim)
        let out = try GPUTensor.zeros(rt, floatCount: inDim)

        try ctx.begin()
        try ctx.ffnBlock(x: xt, normWeight: nwt, gateW: gW, upW: uW, downW: dW,
                         normed: normed, gate: gate, up: up, mid: mid, down: down, out: out,
                         inDim: inDim, ffnDim: ffnDim, eps: eps)
        ctx.commit()

        // CPU reference
        var ss: Float = 0; for i in 0..<inDim { ss += x[i]*x[i] }
        let scale = 1.0 / (ss/Float(inDim) + eps).squareRoot()
        var nrm = [Float](repeating: 0, count: inDim); for i in 0..<inDim { nrm[i] = x[i]*scale*nw[i] }
        let g = mat(gateB, nrm, inDim, ffnDim), u = mat(upB, nrm, inDim, ffnDim)
        var m = [Float](repeating: 0, count: ffnDim)
        for i in 0..<ffnDim { m[i] = (g[i]/(1+expf(-g[i]))) * u[i] }
        let d = mat(downB, m, ffnDim, inDim)
        let got = out.floatArray(inDim)
        var maxRel: Float = 0
        for i in 0..<inDim { maxRel = max(maxRel, abs(got[i] - (x[i]+d[i])) / max(abs(x[i]+d[i]), 0.1)) }
        XCTAssertLessThan(maxRel, 5e-3, "FFN block max rel \(maxRel)")
    }
}
