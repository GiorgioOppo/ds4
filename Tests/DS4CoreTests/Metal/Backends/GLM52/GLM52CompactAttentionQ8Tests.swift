import DS4Core
import XCTest
@testable import DS4Metal

/// The Q8_0 projection kernels must match the F32 oracle evaluated on the
/// DEQUANTIZED weights: quantization error is part of the fixture, never of
/// the kernel. Same skip-without-device policy as the other Metal suites.
final class GLM52CompactAttentionQ8Tests: XCTestCase {
    private let geometry = GLM52AttentionGeometry.v5_2
    private let rowCount = 48
    private let selection: [UInt32] = [7, 0, 33, 12, 45, 3, 21, 9]

    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    /// Q8_0 quantizer shared by the DeepSeek kernel tests (GraphFFNTests):
    /// per 32-block, d = amax/127 as f16 bits, then rounded clamped int8.
    private func quantQ8(_ row: [Float]) -> [UInt8] {
        var out: [UInt8] = []
        var b = 0
        while b < row.count {
            let block = Array(row[b..<b + 32])
            let amax = block.map { abs($0) }.max() ?? 0
            let d = amax / 127.0
            withUnsafeBytes(of: Half.bits(d).littleEndian) {
                out.append(contentsOf: $0)
            }
            for x in block {
                out.append(UInt8(bitPattern: Int8(
                    clamping: d != 0 ? Int((x / d).rounded()) : 0)))
            }
            b += 32
        }
        return out
    }

    /// Dequantize the SAME bytes the GPU reads — the comparison baseline.
    private func dequantQ8(_ bytes: [UInt8]) -> [Float] {
        var out: [Float] = []
        out.reserveCapacity((bytes.count / 34) * 32)
        var b = 0
        while b < bytes.count {
            let d = Half.float(UInt16(bytes[b]) | (UInt16(bytes[b + 1]) << 8))
            for i in 0..<32 {
                out.append(d * Float(Int8(bitPattern: bytes[b + 2 + i])))
            }
            b += 34
        }
        return out
    }

    private struct Fixture {
        let query: [Float]
        let keyBQ8: [UInt8]
        let keyBDequant: [Float]
        let valueBQ8: [UInt8]
        let valueBDequant: [Float]
        let cacheBits: [UInt16]
        let cacheAsFloat: [Float]
    }

    private func fixture() -> Fixture {
        var seed: UInt64 = 0x474C4D35_32513830
        func next() -> Float {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int32(truncatingIfNeeded: seed >> 32)) / Float(Int32.max)
        }
        let g = geometry
        let query = (0..<g.headCount * g.qkDimension).map { _ in next() * 0.25 }
        let keyB = (0..<g.headCount * g.kvLoraRank * g.nopeDimension).map {
            _ in next() * 0.05
        }
        let valueB = (0..<g.headCount * g.valueDimension * g.kvLoraRank).map {
            _ in next() * 0.05
        }
        let cacheBits = (0..<rowCount * g.cacheRowWidth).map {
            _ in Half.bits(next() * 0.25)
        }
        let keyBQ8 = quantQ8(keyB)
        let valueBQ8 = quantQ8(valueB)
        return Fixture(
            query: query,
            keyBQ8: keyBQ8,
            keyBDequant: dequantQ8(keyBQ8),
            valueBQ8: valueBQ8,
            valueBDequant: dequantQ8(valueBQ8),
            cacheBits: cacheBits,
            cacheAsFloat: cacheBits.map(Half.float)
        )
    }

    func testQ8RowByteSizesMatchGGUFLayout() {
        XCTAssertEqual(MetalRuntime.glm52Q8RowBytes(geometry.nopeDimension), 204)
        XCTAssertEqual(MetalRuntime.glm52Q8RowBytes(geometry.kvLoraRank), 544)
    }

    func testQKLowRankQ8MatchesDequantizedScalarAbsorb() throws {
        let runtime = try makeRuntime()
        let f = fixture()
        let g = geometry

        let gpu = try runtime.glm52QKLowRankQ8(query: f.query, keyBQ8: f.keyBQ8)
        XCTAssertEqual(gpu.count, g.headCount * g.kvLoraRank)

        for head in [0, 17, g.headCount - 1] {
            for j in [0, 255, g.kvLoraRank - 1] {
                var expected: Float = 0
                let rowBase = (head * g.kvLoraRank + j) * g.nopeDimension
                for i in 0..<g.nopeDimension {
                    expected += f.query[head * g.qkDimension + i] *
                        f.keyBDequant[rowBase + i]
                }
                XCTAssertEqual(gpu[head * g.kvLoraRank + j], expected,
                               accuracy: 1e-4,
                               "Q8 q_low diverges at head \(head) j \(j)")
            }
        }
    }

    func testValueProjectQ8MatchesDequantizedScalarDot() throws {
        let runtime = try makeRuntime()
        let f = fixture()
        let g = geometry
        let attnLora = (0..<g.headCount * g.kvLoraRank).map {
            Float($0 % 89) * 0.01 - 0.4
        }

        let gpu = try runtime.glm52ValueProjectQ8(
            attnLora: attnLora, valueBQ8: f.valueBQ8)
        XCTAssertEqual(gpu.count, g.headCount * g.valueDimension)

        for head in [0, 31, g.headCount - 1] {
            for d in [0, 100, g.valueDimension - 1] {
                var expected: Float = 0
                let rowBase = (head * g.valueDimension + d) * g.kvLoraRank
                for j in 0..<g.kvLoraRank {
                    expected += f.valueBDequant[rowBase + j] *
                        attnLora[head * g.kvLoraRank + j]
                }
                XCTAssertEqual(gpu[head * g.valueDimension + d], expected,
                               accuracy: 1e-4,
                               "Q8 value projection diverges at head \(head) d \(d)")
            }
        }
    }

    func testChainedQ8MatchesOracleOnDequantizedWeights() throws {
        let runtime = try makeRuntime()
        let f = fixture()

        let gpu = try runtime.glm52CompactAttentionQ8(
            query: f.query,
            keyBQ8: f.keyBQ8,
            valueBQ8: f.valueBQ8,
            cacheBits: f.cacheBits,
            selection: selection)
        let oracle = try GLM52AttentionCPUReference.absorbed(
            geometry: geometry,
            query: f.query,
            keyB: f.keyBDequant,
            valueB: f.valueBDequant,
            cache: f.cacheAsFloat,
            selection: selection.map(Int.init))

        XCTAssertEqual(gpu.count, oracle.count)
        for i in 0..<gpu.count {
            XCTAssertEqual(gpu[i], oracle[i], accuracy: 2e-3,
                           "chained Q8 attention diverges from the oracle at \(i)")
        }
    }

    func testQ8ValidationRejectsWrongByteCounts() throws {
        let runtime = try makeRuntime()
        let f = fixture()

        XCTAssertThrowsError(try runtime.glm52QKLowRankQ8(
            query: f.query, keyBQ8: Array(f.keyBQ8.dropLast())))
        XCTAssertThrowsError(try runtime.glm52ValueProjectQ8(
            attnLora: [Float](repeating: 0,
                              count: geometry.headCount * geometry.kvLoraRank),
            valueBQ8: Array(f.valueBQ8.dropLast())))
        XCTAssertThrowsError(try runtime.glm52CompactAttentionQ8(
            query: Array(f.query.dropLast()),
            keyBQ8: f.keyBQ8,
            valueBQ8: f.valueBQ8,
            cacheBits: f.cacheBits,
            selection: selection))
    }
}
