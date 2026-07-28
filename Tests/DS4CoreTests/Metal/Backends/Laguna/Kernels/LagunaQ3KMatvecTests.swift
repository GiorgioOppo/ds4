import XCTest
import Foundation
import Metal
import DS4Core
@testable import DS4Metal

/// GPU/CPU parity for the Q3_K dot helpers added for the Laguna mixed
/// RoutedQ2_K-Last27Q3_K file. There is no Q3_K encoder in the port, so the
/// fixtures are raw random blocks (every bit pattern is a valid Q3_K payload)
/// with an explicitly sanitized F16 scale, judged against a scalar CPU
/// dequantization that mirrors the llama.cpp layout independently.
final class LagunaQ3KMatvecTests: XCTestCase {
    private let q3KType: UInt32 = 11
    private let blockBytes = 110
    private let blockElems = 256

    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private struct Generator {
        var seed: UInt64
        mutating func nextByte() -> UInt8 {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return UInt8(truncatingIfNeeded: seed >> 33)
        }
        mutating func nextFloat() -> Float {
            Float(Int8(bitPattern: nextByte())) / 128
        }
    }

    /// Random Q3_K rows with a finite, small `d` written explicitly (random
    /// bits could otherwise encode NaN/Inf halves).
    private func randomRows(rows: Int, width: Int,
                            generator: inout Generator) -> [UInt8] {
        let blocks = width / blockElems
        var bytes = [UInt8](repeating: 0, count: rows * blocks * blockBytes)
        for index in bytes.indices { bytes[index] = generator.nextByte() }
        for row in 0..<rows {
            for block in 0..<blocks {
                let base = (row * blocks + block) * blockBytes
                let scale = Half.bits(0.01 + abs(generator.nextFloat()) * 0.05)
                bytes[base + 108] = UInt8(scale & 0xFF)
                bytes[base + 109] = UInt8(scale >> 8)
            }
        }
        return bytes
    }

    /// Scalar CPU reference: the exact llama.cpp Q3_K layout — hmask[32],
    /// qs[64] (2-bit planes), scales[12] (16 signed 6-bit values), d.
    private func cpuDequantRow(_ bytes: ArraySlice<UInt8>, width: Int) -> [Float] {
        var out = [Float](repeating: 0, count: width)
        let base = bytes.startIndex
        for superblock in 0..<(width / blockElems) {
            let b = base + superblock * blockBytes
            let d = Half.float(UInt16(bytes[b + 108]) | (UInt16(bytes[b + 109]) << 8))
            func scale(_ k: Int) -> Int {
                let low = k < 8
                    ? Int(bytes[b + 96 + k] & 0x0F)
                    : Int(bytes[b + 96 + k - 8] >> 4)
                let high = (Int(bytes[b + 96 + 8 + (k % 4)]) >> (2 * (k / 4))) & 3
                return (low | (high << 4)) - 32
            }
            for j in 0..<8 {
                let plane = j % 4
                let qBase = (j / 4) * 32
                let shift = plane * 2
                let mbit = UInt8(1 << ((j / 4) * 4 + plane))
                let xBase = superblock * 256 + (j / 4) * 128 + plane * 32
                for l in 0..<32 {
                    let q = Int((bytes[b + 32 + qBase + l] >> shift) & 3)
                    let v = q - ((bytes[b + l] & mbit) != 0 ? 0 : 4)
                    let s = scale((j / 4) * 8 + plane * 2 + l / 16)
                    out[xBase + l] = d * Float(s) * Float(v)
                }
            }
        }
        return out
    }

    private func cpuMatvec(rows: [UInt8], rowCount: Int, width: Int,
                           x: [Float]) -> [Float] {
        let rowBytes = width / blockElems * blockBytes
        return (0..<rowCount).map { row in
            let start = row * rowBytes
            let decoded = cpuDequantRow(rows[start..<start + rowBytes],
                                        width: width)
            return zip(decoded, x).reduce(0) { $0 + $1.0 * $1.1 }
        }
    }

    /// Dispatch the simdgroup variant the engine uses
    /// (`kernel_glm52_moe_down_sg`), which exercises the per-32-group dot.
    private func gpuMatvecSimdgroup(runtime: MetalRuntime, rows: [UInt8],
                                    rowCount: Int, width: Int,
                                    x: [Float]) throws -> [Float] {
        guard let rowsBuffer = runtime.device.makeBuffer(
                  bytes: rows, length: rows.count, options: .storageModeShared
              ),
              let inputBuffer = runtime.device.makeBuffer(
                  bytes: x, length: x.count * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let outBuffer = runtime.device.makeBuffer(
                  length: rowCount * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ) else {
            throw MetalError.bufferAlloc
        }
        var arguments = [UInt32](repeating: 0, count: 4)
        arguments[0] = q3KType
        arguments[1] = UInt32(rowCount)
        arguments[2] = UInt32(width)

        let pipeline = try runtime.pipeline("kernel_glm52_moe_down_sg")
        guard let commandBuffer = runtime.queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.bufferAlloc
        }
        encoder.setComputePipelineState(pipeline)
        arguments.withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 0)
        }
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(rowsBuffer, offset: 0, index: 2)
        encoder.setBuffer(outBuffer, offset: 0, index: 3)
        let simdgroups = 4
        encoder.dispatchThreadgroups(
            MTLSize(width: (rowCount + simdgroups - 1) / simdgroups,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: simdgroups, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
        let pointer = outBuffer.contents().bindMemory(to: Float.self,
                                                      capacity: rowCount)
        return Array(UnsafeBufferPointer(start: pointer, count: rowCount))
    }

    private func assertClose(_ got: [Float], _ expected: [Float],
                             _ label: String) {
        XCTAssertEqual(got.count, expected.count, label)
        for i in 0..<min(got.count, expected.count) {
            XCTAssertEqual(got[i], expected[i],
                           accuracy: 2e-3 + abs(expected[i]) * 2e-3,
                           "\(label)[\(i)]")
        }
    }

    func testQ3KRowDotMatchesTheCPUReference() throws {
        let runtime = try makeRuntime()
        var generator = Generator(seed: 0x4C41_4755_5133_0001)
        for width in [256, 512] {
            let rowCount = 6
            let rows = randomRows(rows: rowCount, width: width,
                                  generator: &generator)
            let x = (0..<width).map { _ in generator.nextFloat() }
            let cpu = cpuMatvec(rows: rows, rowCount: rowCount,
                                width: width, x: x)
            // Scalar row-dot path (kernel_glm52_moe_down).
            let gpuRow = try runtime.glm52MoEDown(
                mid: x, downRows: rows, weightType: q3KType,
                outputWidth: rowCount
            )
            assertClose(gpuRow, cpu, "row@\(width)")
            // Simdgroup group-dot path (the engine's dispatch).
            let gpuGroup = try gpuMatvecSimdgroup(
                runtime: runtime, rows: rows, rowCount: rowCount,
                width: width, x: x
            )
            assertClose(gpuGroup, cpu, "sg@\(width)")
        }
    }

    func testQ3KRowBytesFollowTheGGUFTypeTable() {
        XCTAssertEqual(MetalRuntime.glm52KQuantRowBytes(type: q3KType,
                                                        width: 1_024),
                       4 * 110)
        XCTAssertNil(MetalRuntime.glm52KQuantRowBytes(type: q3KType,
                                                      width: 100))
    }
}
