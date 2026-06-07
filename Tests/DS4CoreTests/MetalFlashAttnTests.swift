import XCTest
import Foundation
@testable import DS4Metal

/// Phase 9 / Stage A3: validates the real flash_attn.metal decode kernels
/// (kernel_flash_attn_ext_vec_f16_dk512_dv512 + _reduce) vs a CPU softmax
/// attention reference. K==V==latent (MLA), all keys visible, scale=1/sqrt(512).
final class MetalFlashAttnTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/flash_attn.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testFlashDecodeMatchesCPU() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xF1A5
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let headDim = 512, nHead = 4, nKeys = 64
        var q = [Float](repeating: 0, count: nHead * headDim)
        var kv = [Float](repeating: 0, count: nKeys * headDim)
        for i in 0..<q.count { q[i] = rndF() }
        for i in 0..<kv.count { kv[i] = rndF() }

        let gpu = try rt.flashAttnDecode(q: q, kv: kv, nHead: nHead, nKeys: nKeys)
        XCTAssertEqual(gpu.count, nHead * headDim)

        // CPU reference: K=V=kv as F16 (kernel reads F16). out[h] = softmax(scale*Qh·K) · V.
        var kvH = [Float](repeating: 0, count: kv.count)
        for i in 0..<kv.count { kvH[i] = Float(Float16(kv[i])) }
        let scale = 1.0 / Float(headDim).squareRoot()

        var maxRel: Float = 0
        for h in 0..<nHead {
            var s = [Float](repeating: 0, count: nKeys)
            var m = -Float.infinity
            for k in 0..<nKeys {
                var dot: Float = 0
                for d in 0..<headDim { dot += q[h*headDim+d] * kvH[k*headDim+d] }
                s[k] = dot * scale
                m = max(m, s[k])
            }
            var sum: Float = 0
            for k in 0..<nKeys { s[k] = expf(s[k] - m); sum += s[k] }
            var out = [Float](repeating: 0, count: headDim)
            for k in 0..<nKeys {
                let w = s[k] / sum
                for d in 0..<headDim { out[d] += w * kvH[k*headDim+d] }
            }
            for d in 0..<headDim {
                maxRel = max(maxRel, abs(gpu[h*headDim+d] - out[d]) / max(abs(out[d]), 0.05))
            }
        }
        XCTAssertLessThan(maxRel, 2e-2, "flash decode max rel err \(maxRel)")
    }
}
