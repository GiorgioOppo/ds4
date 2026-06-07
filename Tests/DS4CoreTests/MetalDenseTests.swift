import XCTest
@testable import DS4Metal

/// Phase 9: validates the Swift Metal Q8_0 matvec dispatch against a CPU
/// reference that consumes the same quantized bytes. The kernel is the unchanged
/// metal/dense.metal; this proves the Swift dispatch (args layout, function
/// constant, grid/threadgroup) drives it correctly.
final class MetalDenseTests: XCTestCase {

    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dense.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    /// Quantize a row of floats (length multiple of 32) to Q8_0 blocks (34B each).
    private func quantizeQ8_0(_ row: [Float]) -> [UInt8] {
        var out: [UInt8] = []
        var b = 0
        while b < row.count {
            let block = row[b..<b+32]
            let amax = block.map { abs($0) }.max() ?? 0
            let d = amax / 127.0
            let dh = Float16(d)
            withUnsafeBytes(of: dh.bitPattern.littleEndian) { out.append(contentsOf: $0) }
            for x in block {
                let q = d != 0 ? Int((x / d).rounded()) : 0
                out.append(UInt8(bitPattern: Int8(clamping: q)))
            }
            b += 32
        }
        return out
    }

    /// CPU reference: same math as kernel_mul_mv_q8_0_f32, same bytes.
    private func referenceMatmul(weight: [UInt8], activation: [Float],
                                 inDim: Int, outDim: Int) -> [Float] {
        let nblocks = inDim / 32
        let rowBytes = nblocks * 34
        var out = [Float](repeating: 0, count: outDim)
        for r in 0..<outDim {
            var acc: Float = 0
            for blk in 0..<nblocks {
                let base = r * rowBytes + blk * 34
                let dbits = UInt16(weight[base]) | (UInt16(weight[base + 1]) << 8)
                let d = Float(Float16(bitPattern: dbits))
                var sumq: Float = 0
                for i in 0..<32 {
                    let q = Int8(bitPattern: weight[base + 2 + i])
                    sumq += Float(q) * activation[blk * 32 + i]
                }
                acc += sumq * d
            }
            out[r] = acc
        }
        return out
    }

    func testQ8_0MatvecMatchesReference() throws {
        let rt = try makeRuntime()
        // Deterministic pseudo-random inputs.
        var seed: UInt64 = 0xABCDEF
        func rnd() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30)  // ~[-2,2]
        }

        for (inDim, outDim) in [(512, 300), (1024, 256), (2048, 1), (128, 4097)] {
            var weightF = [Float](repeating: 0, count: inDim * outDim)
            for i in 0..<weightF.count { weightF[i] = rnd() }
            var activation = [Float](repeating: 0, count: inDim)
            for i in 0..<inDim { activation[i] = rnd() }

            // Quantize per output row.
            var weight: [UInt8] = []
            weight.reserveCapacity((inDim / 32) * 34 * outDim)
            for r in 0..<outDim {
                weight.append(contentsOf: quantizeQ8_0(Array(weightF[r*inDim..<(r+1)*inDim])))
            }

            let gpu = try rt.matmulQ8_0(weight: weight, activation: activation, inDim: inDim, outDim: outDim)
            let cpu = referenceMatmul(weight: weight, activation: activation, inDim: inDim, outDim: outDim)

            XCTAssertEqual(gpu.count, outDim)
            var maxRel: Float = 0
            for r in 0..<outDim {
                let denom = max(abs(cpu[r]), 1.0)
                maxRel = max(maxRel, abs(gpu[r] - cpu[r]) / denom)
            }
            XCTAssertLessThan(maxRel, 1e-3, "Q8_0 matvec \(inDim)x\(outDim) max rel err \(maxRel)")
        }
    }

    func testF16MatvecMatchesReference() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x13579B
        func rnd() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30)
        }

        for (inDim, outDim) in [(512, 300), (1024, 512), (4096, 512), (128, 257)] {
            var weight = [UInt16](repeating: 0, count: inDim * outDim)
            var weightF = [Float](repeating: 0, count: inDim * outDim)
            for i in 0..<weight.count {
                let h = Float16(rnd())
                weight[i] = h.bitPattern
                weightF[i] = Float(h)   // exact half value, as the kernel sees it
            }
            var activation = [Float](repeating: 0, count: inDim)
            for i in 0..<inDim { activation[i] = rnd() }

            let gpu = try rt.matmulF16(weight: weight, activation: activation, inDim: inDim, outDim: outDim)

            // CPU reference: dot of half-precision weights with f32 activation.
            var cpu = [Float](repeating: 0, count: outDim)
            for r in 0..<outDim {
                var acc: Float = 0
                for i in 0..<inDim { acc += weightF[r * inDim + i] * activation[i] }
                cpu[r] = acc
            }

            XCTAssertEqual(gpu.count, outDim)
            var maxRel: Float = 0
            for r in 0..<outDim {
                maxRel = max(maxRel, abs(gpu[r] - cpu[r]) / max(abs(cpu[r]), 1.0))
            }
            XCTAssertLessThan(maxRel, 2e-3, "F16 matvec \(inDim)x\(outDim) max rel err \(maxRel)")
        }
    }

    func testF32MatvecMatchesReference() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x2468AC
        func rnd() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30)
        }

        for (inDim, outDim) in [(512, 300), (4096, 1024), (2048, 1), (128, 4097)] {
            var weight = [Float](repeating: 0, count: inDim * outDim)
            for i in 0..<weight.count { weight[i] = rnd() }
            var activation = [Float](repeating: 0, count: inDim)
            for i in 0..<inDim { activation[i] = rnd() }

            let gpu = try rt.matmulF32(weight: weight, activation: activation, inDim: inDim, outDim: outDim)

            var cpu = [Float](repeating: 0, count: outDim)
            for r in 0..<outDim {
                var acc: Float = 0
                for i in 0..<inDim { acc += weight[r * inDim + i] * activation[i] }
                cpu[r] = acc
            }

            XCTAssertEqual(gpu.count, outDim)
            var maxRel: Float = 0
            for r in 0..<outDim {
                maxRel = max(maxRel, abs(gpu[r] - cpu[r]) / max(abs(cpu[r]), 1.0))
            }
            XCTAssertLessThan(maxRel, 1e-4, "F32 matvec \(inDim)x\(outDim) max rel err \(maxRel)")
        }
    }

    func testF16PairMatvecMatchesReference() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x1F2E3D
        func rnd() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30)
        }

        for (inDim, outDim) in [(512, 300), (4096, 512), (1024, 256)] {
            var wa = [UInt16](repeating: 0, count: inDim * outDim)
            var wb = [UInt16](repeating: 0, count: inDim * outDim)
            var waF = [Float](repeating: 0, count: inDim * outDim)
            var wbF = [Float](repeating: 0, count: inDim * outDim)
            for i in 0..<wa.count {
                let ha = Float16(rnd()), hb = Float16(rnd())
                wa[i] = ha.bitPattern; waF[i] = Float(ha)
                wb[i] = hb.bitPattern; wbF[i] = Float(hb)
            }
            var activation = [Float](repeating: 0, count: inDim)
            for i in 0..<inDim { activation[i] = rnd() }

            let (ga, gb) = try rt.matmulF16Pair(weightA: wa, weightB: wb, activation: activation, inDim: inDim, outDim: outDim)

            var ca = [Float](repeating: 0, count: outDim), cb = [Float](repeating: 0, count: outDim)
            for r in 0..<outDim {
                var accA: Float = 0, accB: Float = 0
                for i in 0..<inDim {
                    accA += waF[r * inDim + i] * activation[i]
                    accB += wbF[r * inDim + i] * activation[i]
                }
                ca[r] = accA; cb[r] = accB
            }

            var maxRel: Float = 0
            for r in 0..<outDim {
                maxRel = max(maxRel, abs(ga[r] - ca[r]) / max(abs(ca[r]), 1.0))
                maxRel = max(maxRel, abs(gb[r] - cb[r]) / max(abs(cb[r]), 1.0))
            }
            XCTAssertLessThan(maxRel, 2e-3, "F16 pair matvec \(inDim)x\(outDim) max rel err \(maxRel)")
        }
    }
}
