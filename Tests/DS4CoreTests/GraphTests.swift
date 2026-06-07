import XCTest
import Foundation
@testable import DS4Metal

/// Stage C: validates the first graph fragments — token embedding (get_rows -> HC
/// repeat) and the output head (final RMSNorm -> vocab matmul) — composed on
/// GPUTensors in a single command buffer, vs CPU references.
final class GraphTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/norm.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testOutputHead() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xC001
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 29) }

        let inDim = 2048, vocab = 4096
        let eps: Float = 1e-5
        var hidden = [Float](repeating: 0, count: inDim)
        var nw = [Float](repeating: 0, count: inDim)
        var ow = [Float](repeating: 0, count: vocab * inDim)
        for i in 0..<inDim { hidden[i] = rndF(); nw[i] = rndF() }
        for i in 0..<ow.count { ow[i] = rndF() }

        let ctx = GraphContext(rt)
        let ht = try GPUTensor.floats(rt, hidden)
        let nwt = try GPUTensor.floats(rt, nw)
        let owt = try GPUTensor.floats(rt, ow)
        let normed = try GPUTensor.zeros(rt, floatCount: inDim)
        let logits = try GPUTensor.zeros(rt, floatCount: vocab)

        try ctx.begin()
        try ctx.outputHead(hidden: ht, normWeight: nwt, outWeight: owt, normed: normed, logits: logits,
                           inDim: inDim, vocab: vocab, eps: eps)
        ctx.commit()

        var ss: Float = 0
        for i in 0..<inDim { ss += hidden[i]*hidden[i] }
        let scale = 1.0 / (ss/Float(inDim) + eps).squareRoot()
        var nrm = [Float](repeating: 0, count: inDim)
        for i in 0..<inDim { nrm[i] = hidden[i]*scale*nw[i] }
        let got = logits.floatArray(vocab)
        var maxRel: Float = 0
        for r in 0..<vocab {
            var acc: Float = 0
            for i in 0..<inDim { acc += ow[r*inDim+i]*nrm[i] }
            maxRel = max(maxRel, abs(got[r]-acc)/max(abs(acc),1))
        }
        XCTAssertLessThan(maxRel, 2e-3, "output head max rel \(maxRel)")
    }

    func testEmbedTokenHC() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xC002
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let nVocab = 128, nEmbd = 1024, nHC = 4, token = 77
        var table = [UInt16](repeating: 0, count: nVocab * nEmbd)
        for i in 0..<table.count { table[i] = Float16(rndF()).bitPattern }

        let ctx = GraphContext(rt)
        let tt = try GPUTensor.bytes(rt, table.withUnsafeBytes { Array($0) }, elementCount: nVocab*nEmbd)
        let embd = try GPUTensor.zeros(rt, floatCount: nEmbd)
        let hc = try GPUTensor.zeros(rt, floatCount: nHC * nEmbd)
        try ctx.begin()
        try ctx.embedTokenHC(table: tt, token: token, embd: embd, hc: hc, nEmbd: nEmbd, nVocab: nVocab, nHC: nHC)
        ctx.commit()

        let got = hc.floatArray(nHC * nEmbd)
        for h in 0..<nHC {
            for e in 0..<nEmbd {
                let expected = Float(Float16(bitPattern: table[token*nEmbd+e]))
                XCTAssertEqual(got[h*nEmbd+e], expected, "embed hc=\(h) e=\(e)")
            }
        }
    }
}
