import DS4Core
import XCTest
@testable import DS4Metal

/// The routed-expert kernels must match the FFN oracle evaluated on the
/// DEQUANTIZED weights (the `Quantize` references decode the same bytes the
/// GPU reads). Q4_K fixtures come from the real quantizer; Q2_K/Q5_K/Q6_K
/// blocks are synthesized field by field — any byte pattern is decodable, so
/// the bytes themselves are the fixture. Skips without a Metal device.
final class GLM52MoETests: XCTestCase {
    private let width = 256      // one super-block per row
    private let hidden = 256

    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    // MARK: - Fixtures

    private struct Generator {
        var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func byte() -> UInt8 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return UInt8(truncatingIfNeeded: state >> 33)
        }
        mutating func float(_ scale: Float) -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return (Float(state >> 33) / Float(1 << 30) - 1.0) * scale
        }
        mutating func smallHalfBits() -> UInt16 {
            Half.bits(0.001 + abs(float(0.02)))
        }
    }

    /// One synthetic super-block for `type` (84/176/210 bytes): random field
    /// bytes with small sane f16 scales.
    private func syntheticBlock(type: UInt32,
                                generator: inout Generator) -> [UInt8] {
        var block = [UInt8]()
        func appendBytes(_ count: Int) {
            for _ in 0..<count { block.append(generator.byte()) }
        }
        func appendHalf(_ bits: UInt16) {
            withUnsafeBytes(of: bits.littleEndian) { block.append(contentsOf: $0) }
        }
        switch type {
        case GLM52TensorSchema.q2_K:
            appendBytes(16 + 64)
            appendHalf(generator.smallHalfBits())
            appendHalf(generator.smallHalfBits())
        case GLM52TensorSchema.q4_K:
            appendHalf(generator.smallHalfBits())
            appendHalf(generator.smallHalfBits())
            appendBytes(12 + 128)
        case GLM52TensorSchema.q5_K:
            appendHalf(generator.smallHalfBits())
            appendHalf(generator.smallHalfBits())
            appendBytes(12 + 32 + 128)
        case GLM52TensorSchema.q6_K:
            appendBytes(128 + 64 + 16)
            appendHalf(generator.smallHalfBits())
        default:
            XCTFail("unsupported synthetic type \(type)")
        }
        return block
    }

    private func syntheticRows(type: UInt32, rows: Int,
                               seed: UInt64) -> [UInt8] {
        var generator = Generator(seed: seed)
        var bytes = [UInt8]()
        for _ in 0..<rows {
            bytes += syntheticBlock(type: type, generator: &generator)
        }
        return bytes
    }

    private func dequantize(_ bytes: [UInt8], type: UInt32,
                            count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        bytes.withUnsafeBytes { raw in
            switch type {
            case GLM52TensorSchema.q2_K:
                Quantize.dequantQ2_K(raw.baseAddress!, count: count, into: &out)
            case GLM52TensorSchema.q4_K:
                Quantize.dequantQ4_K(raw.baseAddress!, count: count, into: &out)
            case GLM52TensorSchema.q5_K:
                Quantize.dequantQ5_K(raw.baseAddress!, count: count, into: &out)
            case GLM52TensorSchema.q6_K:
                Quantize.dequantQ6_K(raw.baseAddress!, count: count, into: &out)
            default:
                XCTFail("unsupported dequant type \(type)")
            }
        }
        return out
    }

    private func input(seed: UInt64) -> [Float] {
        var generator = Generator(seed: seed)
        return (0..<width).map { _ in generator.float(0.2) }
    }

    private func assertClose(_ got: [Float], _ expected: [Float],
                             label: String,
                             file: StaticString = #filePath,
                             line: UInt = #line) {
        XCTAssertEqual(got.count, expected.count, file: file, line: line)
        for i in 0..<got.count {
            XCTAssertEqual(got[i], expected[i],
                           accuracy: 1e-3 + abs(expected[i]) * 1e-3,
                           "\(label) diverges at \(i)",
                           file: file, line: line)
        }
    }

    // MARK: - Stage tests

    func testPairSwiGLUQ4KMatchesOracleOnDequantizedWeights() throws {
        let runtime = try makeRuntime()
        var generator = Generator(seed: 5)
        let gateF32 = (0..<hidden * width).map { _ in generator.float(0.05) }
        let upF32 = (0..<hidden * width).map { _ in generator.float(0.05) }
        var gateQ = [UInt8](repeating: 0, count: (hidden * width / 256) * 144)
        var upQ = gateQ
        gateF32.withUnsafeBufferPointer { src in
            gateQ.withUnsafeMutableBytes {
                Quantize.quantizeQ4_K(src.baseAddress!, count: gateF32.count,
                                      into: $0.baseAddress!)
            }
        }
        upF32.withUnsafeBufferPointer { src in
            upQ.withUnsafeMutableBytes {
                Quantize.quantizeQ4_K(src.baseAddress!, count: upF32.count,
                                      into: $0.baseAddress!)
            }
        }
        let x = input(seed: 6)
        let routeWeight: Float = 1.25

        let gpu = try runtime.glm52MoEPairSwiGLU(
            input: x, gateRows: gateQ, upRows: upQ,
            weightType: GLM52TensorSchema.q4_K,
            hiddenWidth: hidden, routeWeight: routeWeight)

        let gateOut = try GLM52FFNCPUReference.matvec(
            rows: dequantize(gateQ, type: GLM52TensorSchema.q4_K,
                             count: hidden * width),
            input: x, rowCount: hidden)
        let upOut = try GLM52FFNCPUReference.matvec(
            rows: dequantize(upQ, type: GLM52TensorSchema.q4_K,
                             count: hidden * width),
            input: x, rowCount: hidden)
        let expected = (0..<hidden).map {
            GLM52FFNCPUReference.silu(gateOut[$0]) * upOut[$0] * routeWeight
        }
        assertClose(gpu, expected, label: "Q4_K pair SwiGLU")
    }

    func testPairSwiGLUQ5KMatchesOracleOnDequantizedWeights() throws {
        let runtime = try makeRuntime()
        let gateQ = syntheticRows(type: GLM52TensorSchema.q5_K,
                                  rows: hidden, seed: 11)
        let upQ = syntheticRows(type: GLM52TensorSchema.q5_K,
                                rows: hidden, seed: 12)
        let x = input(seed: 13)

        let gpu = try runtime.glm52MoEPairSwiGLU(
            input: x, gateRows: gateQ, upRows: upQ,
            weightType: GLM52TensorSchema.q5_K,
            hiddenWidth: hidden, routeWeight: 1)

        let gateOut = try GLM52FFNCPUReference.matvec(
            rows: dequantize(gateQ, type: GLM52TensorSchema.q5_K,
                             count: hidden * width),
            input: x, rowCount: hidden)
        let upOut = try GLM52FFNCPUReference.matvec(
            rows: dequantize(upQ, type: GLM52TensorSchema.q5_K,
                             count: hidden * width),
            input: x, rowCount: hidden)
        let expected = (0..<hidden).map {
            GLM52FFNCPUReference.silu(gateOut[$0]) * upOut[$0]
        }
        assertClose(gpu, expected, label: "Q5_K pair SwiGLU")
    }

    func testDownQ6KMatchesOracleMatvec() throws {
        let runtime = try makeRuntime()
        let downQ = syntheticRows(type: GLM52TensorSchema.q6_K,
                                  rows: width, seed: 21)
        var generator = Generator(seed: 22)
        let mid = (0..<hidden).map { _ in generator.float(0.3) }

        let gpu = try runtime.glm52MoEDown(
            mid: mid, downRows: downQ,
            weightType: GLM52TensorSchema.q6_K, outputWidth: width)

        let expected = try GLM52FFNCPUReference.matvec(
            rows: dequantize(downQ, type: GLM52TensorSchema.q6_K,
                             count: width * hidden),
            input: mid, rowCount: width)
        assertClose(gpu, expected, label: "Q6_K down")
    }

    func testChainedRoutedFFNMatchesOracle() throws {
        let runtime = try makeRuntime()
        let geometry = GLM52FFNGeometry(
            embeddingWidth: width,
            denseHiddenWidth: 256,
            expertHiddenWidth: hidden,
            expertsUsed: 2)
        let weights: [Float] = [0.75, 1.75]
        var quantized = [GLM52QuantizedExpert]()
        var dequantized = [GLM52ExpertWeights]()
        for e in 0..<2 {
            let seed = UInt64(31 + e * 10)
            let gate = syntheticRows(type: GLM52TensorSchema.q2_K,
                                     rows: hidden, seed: seed)
            let up = syntheticRows(type: GLM52TensorSchema.q2_K,
                                   rows: hidden, seed: seed + 1)
            let down = syntheticRows(type: GLM52TensorSchema.q4_K,
                                     rows: width, seed: seed + 2)
            quantized.append(GLM52QuantizedExpert(
                gateUpType: GLM52TensorSchema.q2_K,
                downType: GLM52TensorSchema.q4_K,
                gate: gate, up: up, down: down))
            dequantized.append(GLM52ExpertWeights(
                gate: dequantize(gate, type: GLM52TensorSchema.q2_K,
                                 count: hidden * width),
                up: dequantize(up, type: GLM52TensorSchema.q2_K,
                               count: hidden * width),
                down: dequantize(down, type: GLM52TensorSchema.q4_K,
                                 count: width * hidden)))
        }
        let x = input(seed: 41)

        let gpu = try runtime.glm52RoutedFFN(
            input: x, experts: quantized, weights: weights,
            hiddenWidth: hidden)
        let oracle = try GLM52FFNCPUReference.routedFFN(
            geometry: geometry, input: x,
            experts: dequantized, weights: weights)
        assertClose(gpu, oracle, label: "chained routed FFN")
    }

    func testValidationRejectsBadTypesAndSizes() throws {
        let runtime = try makeRuntime()
        let x = input(seed: 51)
        let rows = syntheticRows(type: GLM52TensorSchema.q5_K,
                                 rows: hidden, seed: 52)

        XCTAssertThrowsError(try runtime.glm52MoEPairSwiGLU(
            input: x, gateRows: rows, upRows: rows, weightType: 8,
            hiddenWidth: hidden, routeWeight: 1))
        XCTAssertThrowsError(try runtime.glm52MoEPairSwiGLU(
            input: x, gateRows: Array(rows.dropLast()), upRows: rows,
            weightType: GLM52TensorSchema.q5_K,
            hiddenWidth: hidden, routeWeight: 1))
        XCTAssertThrowsError(try runtime.glm52MoEDown(
            mid: [1, 2, 3], downRows: rows,
            weightType: GLM52TensorSchema.q5_K, outputWidth: width))
        XCTAssertThrowsError(try runtime.glm52RoutedFFN(
            input: x, experts: [], weights: [], hiddenWidth: hidden))
    }
}
