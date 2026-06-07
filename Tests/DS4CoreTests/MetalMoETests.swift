import XCTest
@testable import DS4Metal

/// Phase 9 (MoE): validates the expert-routed Q8_0 matvec (real
/// kernel_mul_mv_id_q8_0_f32) against a CPU reference that gathers the selected
/// expert and does the Q8_0 dot.
final class MetalMoETests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/moe.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private func quantizeQ8_0(_ row: [Float]) -> [UInt8] {
        var out: [UInt8] = []
        var b = 0
        while b < row.count {
            let block = row[b..<b+32]
            let amax = block.map { abs($0) }.max() ?? 0
            let d = amax / 127.0
            withUnsafeBytes(of: Float16(d).bitPattern.littleEndian) { out.append(contentsOf: $0) }
            for x in block { out.append(UInt8(bitPattern: Int8(clamping: d != 0 ? Int((x / d).rounded()) : 0))) }
            b += 32
        }
        return out
    }

    private func q8Dot(_ weight: [UInt8], rowBase: Int, activation: [Float], inDim: Int) -> Float {
        let nblk = inDim / 32
        var acc: Float = 0
        for blk in 0..<nblk {
            let base = rowBase + blk * 34
            let d = Float(Float16(bitPattern: UInt16(weight[base]) | (UInt16(weight[base+1]) << 8)))
            var sumq: Float = 0
            for i in 0..<32 { sumq += Float(Int8(bitPattern: weight[base+2+i])) * activation[blk*32+i] }
            acc += sumq * d
        }
        return acc
    }

    func testMoEQ8_0MatvecRoutes() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xB0BA
        func rnd() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30)
        }
        let inDim = 512, outDim = 256, nExperts = 8
        let rowBytes = (inDim / 32) * 34
        let expertBytes = rowBytes * outDim

        var experts = [UInt8](); experts.reserveCapacity(expertBytes * nExperts)
        for _ in 0..<nExperts {
            for _ in 0..<outDim {
                var row = [Float](repeating: 0, count: inDim)
                for i in 0..<inDim { row[i] = rnd() }
                experts.append(contentsOf: quantizeQ8_0(row))
            }
        }
        var activation = [Float](repeating: 0, count: inDim)
        for i in 0..<inDim { activation[i] = rnd() }

        let ids: [Int32] = [3, 0, 7, 5, 1, 6]   // selected experts for the token
        let gpu = try rt.moeMatvecQ8_0(experts: experts, expertIds: ids, activation: activation,
                                       nExperts: nExperts, inDim: inDim, outDim: outDim)
        XCTAssertEqual(gpu.count, ids.count * outDim)

        var maxRel: Float = 0
        for (slot, e) in ids.enumerated() {
            for o in 0..<outDim {
                let ref = q8Dot(experts, rowBase: Int(e) * expertBytes + o * rowBytes,
                                activation: activation, inDim: inDim)
                let got = gpu[slot * outDim + o]
                maxRel = max(maxRel, abs(got - ref) / max(abs(ref), 1.0))
            }
        }
        XCTAssertLessThan(maxRel, 1e-3, "MoE Q8_0 routed matvec max rel err \(maxRel)")
    }
}
