import DS4Core
import XCTest
@testable import DS4Metal

/// I due kernel MoE BATCHED (tutti gli esperti in due dispatch) giudicati
/// contro un oracle CPU in float su record Q8_0 sintetici: mid per esperto
/// (silu(gate·x)·(up·x)·peso) e accumulo dei contributi down su hidden.
/// Q8_0 con d=1: la dequant è esatta e l'oracle non dipende dal dequant
/// CPU di altri formati.
final class GLM52MoEBatchTests: XCTestCase {
    private let hiddenWidth = 64
    private let inputWidth = 128
    private let expertCount = 3

    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    /// Una riga Q8_0 con d=1.0: dequant esatta ai valori int8.
    private func q8Row(_ values: [Int8]) -> [UInt8] {
        precondition(values.count % 32 == 0)
        var bytes: [UInt8] = []
        for block in stride(from: 0, to: values.count, by: 32) {
            withUnsafeBytes(of: Half.bits(1.0).littleEndian) {
                bytes.append(contentsOf: $0)
            }
            for value in values[block..<block + 32] {
                bytes.append(UInt8(bitPattern: value))
            }
        }
        return bytes
    }

    private func silu(_ x: Float) -> Float { x / (1 + exp(-x)) }

    func testBatchedMoEMatchesCPUOracle() throws {
        let runtime = try makeRuntime()
        var state: UInt64 = 7
        func nextInt8() -> Int8 {
            state = state &* 6_364_136_223_846_793_005
                &+ 1_442_695_040_888_963_407
            return Int8(truncatingIfNeeded: Int8(
                clamping: Int(truncatingIfNeeded: state >> 40) % 5 - 2))
        }
        // Record contigui gate|up|down per esperto, come l'arena.
        var gates: [[Int8]] = []
        var ups: [[Int8]] = []
        var downs: [[Int8]] = []
        var records: [UInt8] = []
        var offsets: [Int] = []
        let gateBytes = hiddenWidth * (inputWidth / 32 * 34)
        let upBytes = gateBytes
        for _ in 0..<expertCount {
            offsets.append(records.count)
            let gate = (0..<hiddenWidth * inputWidth).map { _ in nextInt8() }
            let up = (0..<hiddenWidth * inputWidth).map { _ in nextInt8() }
            let down = (0..<inputWidth * hiddenWidth).map { _ in nextInt8() }
            gates.append(gate); ups.append(up); downs.append(down)
            for row in 0..<hiddenWidth {
                records += q8Row(Array(
                    gate[row * inputWidth..<(row + 1) * inputWidth]))
            }
            for row in 0..<hiddenWidth {
                records += q8Row(Array(
                    up[row * inputWidth..<(row + 1) * inputWidth]))
            }
            for row in 0..<inputWidth {
                records += q8Row(Array(
                    down[row * hiddenWidth..<(row + 1) * hiddenWidth]))
            }
        }
        let x = (0..<inputWidth).map { Float($0 % 7) * 0.25 - 0.75 }
        let weights: [Float] = [0.5, 0.3, 0.2]
        let hidden0 = (0..<inputWidth).map { Float($0 % 5) * 0.1 }

        // Oracle CPU in float.
        var mids = [[Float]]()
        for e in 0..<expertCount {
            var mid = [Float](repeating: 0, count: hiddenWidth)
            for r in 0..<hiddenWidth {
                var g: Float = 0, u: Float = 0
                for k in 0..<inputWidth {
                    g += Float(gates[e][r * inputWidth + k]) * x[k]
                    u += Float(ups[e][r * inputWidth + k]) * x[k]
                }
                mid[r] = silu(g) * u * weights[e]
            }
            mids.append(mid)
        }
        var expected = hidden0
        for e in 0..<expertCount {
            for r in 0..<inputWidth {
                var acc: Float = 0
                for k in 0..<hiddenWidth {
                    acc += Float(downs[e][r * hiddenWidth + k]) * mids[e][k]
                }
                expected[r] += acc
            }
        }

        // GPU: gli stessi due dispatch del decode.
        let recordsBuffer = try runtime.glm52GraphBuffer(records)
        let xBuffer = try runtime.glm52GraphBuffer(x)
        let midsBuffer = try runtime.glm52GraphOutputBuffer(
            floats: 8 * hiddenWidth)
        let hiddenBuffer = try runtime.glm52GraphBuffer(hidden0)
        let staged = GLM52StagedExpertSelection(
            buffer: recordsBuffer, recordOffsets: offsets,
            gateBytes: gateBytes, upBytes: upBytes,
            downBytes: inputWidth * (hiddenWidth / 32 * 34),
            gateUpType: GLM52TensorSchema.q8_0,
            downType: GLM52TensorSchema.q8_0)
        guard let commandBuffer = runtime.queue.makeCommandBuffer() else {
            throw MetalError.bufferAlloc
        }
        try runtime.glm52EncodeMoEBatch(
            into: commandBuffer, staged: staged, weights: weights,
            input: xBuffer, mids: midsBuffer, accumulate: hiddenBuffer,
            hiddenWidth: hiddenWidth, inputWidth: inputWidth)
        try runtime.glm52GraphCommit(commandBuffer)

        let gpuMids = runtime.glm52GraphReadback(midsBuffer,
                                                 count: 8 * hiddenWidth)
        for e in 0..<expertCount {
            for r in 0..<hiddenWidth {
                XCTAssertEqual(gpuMids[e * hiddenWidth + r], mids[e][r],
                               accuracy: 2e-3, "mid esperto \(e) riga \(r)")
            }
        }
        let gpuHidden = runtime.glm52GraphReadback(hiddenBuffer,
                                                   count: inputWidth)
        for r in 0..<inputWidth {
            XCTAssertEqual(gpuHidden[r], expected[r], accuracy: 5e-2,
                           "hidden riga \(r)")
        }
    }
}
