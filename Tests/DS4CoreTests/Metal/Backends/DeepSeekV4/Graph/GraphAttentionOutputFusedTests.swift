import Foundation
import Metal
import XCTest
import DS4Core
@testable import DS4Metal

/// The fusion may remove a dispatch, but may not change either the projected
/// row or the four residual streams. Use the shipped library, including its
/// normal math mode; a shader compilation error must fail, rather than skip.
final class GraphAttentionOutputFusedTests: XCTestCase {
    private func runtime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("No Metal device") }
        catch MetalError.noQueue { throw XCTSkip("No Metal queue") }
    }

    func testEnvironmentSelectsQ8ByDefaultAndRequiresQ4OptIn() throws {
        let key = "DS4_FUSED_ATTN_OUT_HC"
        let original = ProcessInfo.processInfo.environment[key]
        defer {
            if let original { setenv(key, original, 1) } else { unsetenv(key) }
            GraphContext.refreshQ8NSG()
        }
        let rt = try runtime()
        for value: String? in [nil, "0", "1"] {
            if let value { setenv(key, value, 1) } else { unsetenv(key) }
            GraphContext.refreshQ8NSG()
            for q4 in [false, true] {
                let f = try Fixture(rt, inDim: 512, outDim: 64, q4: q4)
                let before = [f.block, f.out].map(bytes)
                let graph = GraphContext(rt)
                try graph.begin()
                let used = try graph.attentionOutputHCFused(
                    weight: f.weight, x: f.x, blockOut: f.block,
                    residual: f.residual, split: f.split, out: f.out,
                    inDim: 512, outDim: 64, q4: q4)
                graph.commit()
                XCTAssertNil(graph.lastError)
                let expected = value == "1" || (value == nil && !q4)
                XCTAssertEqual(used, expected, "env=\(value ?? "unset"), q4=\(q4)")
                if expected {
                    XCTAssertTrue(f.out.floatArray().allSatisfy(\.isFinite))
                    XCTAssertTrue(f.out.floatArray().contains { $0 != 0 })
                } else {
                    XCTAssertEqual([f.block, f.out].map(bytes), before)
                }
            }
        }
    }

    func testQ4AndQ8MatchUnfusedBitsWithOffsetsAndTailRows() throws {
        let rt = try runtime()
        for q4 in [true, false] {
            // 66 rows exercises inactive Q4 simdgroups; 4096 input columns
            // exercise multiple iterations of every Q8 K-split worker.
            for (k, n) in [(512, 66), (4096, 128)] {
                let f = try Fixture(rt, inDim: k, outDim: n, q4: q4)
                let originalInputs = f.inputs.map(bytes)
                let refBlock = try padded(rt, payload: [UInt8](repeating: 0, count: n * 4))
                let refOut = try padded(rt, payload: [UInt8](repeating: 0, count: n * 16))
                let graph = GraphContext(rt)
                try graph.begin()
                if q4 {
                    try graph.matmulQ4_K(weight: f.weight, x: f.x, out: refBlock,
                                        inDim: k, outDim: n)
                } else {
                    try graph.matmulQ8_0(weight: f.weight, x: f.x, out: refBlock,
                                        inDim: k, outDim: n)
                }
                try graph.hcExpand4(blockOut: refBlock, residual: f.residual,
                                    post: f.split, comb: f.split, blockAdd: nil,
                                    out: refOut, nEmbd: n, nTokens: 1,
                                    postByteOffset: 16, combByteOffset: 32)
                let encoded = try graph.attentionOutputHCFused(
                    weight: f.weight, x: f.x, blockOut: f.block,
                    residual: f.residual, split: f.split, out: f.out,
                    inDim: k, outDim: n, q4: q4, enabled: true)
                XCTAssertTrue(encoded, "Expected supported Q\(q4 ? 4 : 8) \(k)→\(n) fusion")
                graph.commit()
                XCTAssertNil(graph.lastError)
                XCTAssertTrue(refOut.floatArray().allSatisfy(\.isFinite))
                XCTAssertEqual(f.block.floatArray().map(\.bitPattern),
                               refBlock.floatArray().map(\.bitPattern), "Projected row")
                XCTAssertEqual(f.out.floatArray().map(\.bitPattern),
                               refOut.floatArray().map(\.bitPattern), "HC streams")
                XCTAssertEqual(f.inputs.map(bytes), originalInputs, "Read-only input changed")
                assertPadding(f.block)
                assertPadding(f.out)
            }
        }
    }

    func testRejectedInputsDoNotRequireAnEncoderOrMutateBuffers() throws {
        let rt = try runtime()
        let f = try Fixture(rt, inDim: 512, outDim: 64, q4: true)
        let graph = GraphContext(rt) // Deliberately do not begin().
        let initial = (f.inputs + [f.block, f.out]).map(bytes)
        func attempt(weight: GPUTensor? = nil, x: GPUTensor? = nil,
                     block: GPUTensor? = nil, out: GPUTensor? = nil,
                     k: Int = 512, n: Int = 64, nHC: Int = 4,
                     enabled: Bool = true) throws -> Bool {
            try graph.attentionOutputHCFused(
                weight: weight ?? f.weight, x: x ?? f.x, blockOut: block ?? f.block,
                residual: f.residual, split: f.split, out: out ?? f.out,
                inDim: k, outDim: n, q4: true, nHC: nHC, enabled: enabled)
        }
        XCTAssertFalse(try attempt(enabled: false))
        XCTAssertFalse(try attempt(nHC: 3))
        XCTAssertFalse(try attempt(n: 63))
        XCTAssertFalse(try attempt(k: 511))
        XCTAssertFalse(try attempt(k: Int.max))
        XCTAssertFalse(try attempt(x: view(f.x, offset: f.x.byteOffset + 4)))
        XCTAssertFalse(try attempt(weight: view(f.weight, offset: f.weight.byteOffset + 2)))
        XCTAssertFalse(try attempt(x: view(f.x, length: 511 * 4)))
        XCTAssertFalse(try attempt(x: view(f.x, offset: -16)))
        XCTAssertFalse(try attempt(x: view(f.x, length: f.x.buffer.length)))
        XCTAssertFalse(try attempt(block: view(f.residual, length: 64 * 4)))
        XCTAssertFalse(try attempt(out: f.residual))
        XCTAssertFalse(try attempt(out: view(f.weight, length: 64 * 16)))
        XCTAssertFalse(try attempt(block: view(f.out, length: 64 * 4)))
        let untrackedBuffer = try XCTUnwrap(rt.device.makeBuffer(
            length: f.x.byteLength, options: [.storageModeShared, .hazardTrackingModeUntracked]))
        let untrackedX = GPUTensor(buffer: untrackedBuffer,
                                  byteLength: f.x.byteLength, count: f.x.count)
        XCTAssertFalse(try attempt(x: untrackedX))
        XCTAssertEqual((f.inputs + [f.block, f.out]).map(bytes), initial)
    }

    func testDisjointOutputViewsOfOneBufferAreSupported() throws {
        let rt = try runtime()
        let f = try Fixture(rt, inDim: 512, outDim: 64, q4: false)
        let slab = try GPUTensor.zerosBytes(rt, byteLength: 64 * 20)
        let block = slab.subview(byteOffset: 0, byteLength: 64 * 4, count: 64)
        let out = slab.subview(byteOffset: 64 * 4, byteLength: 64 * 16, count: 256)
        let graph = GraphContext(rt)
        try graph.begin()
        XCTAssertTrue(try graph.attentionOutputHCFused(
            weight: f.weight, x: f.x, blockOut: block, residual: f.residual,
            split: f.split, out: out, inDim: 512, outDim: 64, q4: false,
            enabled: true))
        graph.commit()
        XCTAssertNil(graph.lastError)
        XCTAssertTrue(out.floatArray().allSatisfy(\.isFinite))
        XCTAssertTrue(out.floatArray().contains { $0 != 0 })
    }

    private struct Fixture {
        let weight, x, residual, split, block, out: GPUTensor
        var inputs: [GPUTensor] { [weight, x, residual, split] }

        init(_ rt: MetalRuntime, inDim: Int, outDim: Int, q4: Bool) throws {
            // The old Q4 matvec reads complete row pairs for inactive groups
            // in the last threadgroup. Pad only this reference-test fixture
            // up to 16 rows (the largest supported NSG is 8).
            let allocatedRows = (outDim + 15) / 16 * 16
            let values = (0..<(inDim * allocatedRows)).map {
                Float(($0 &* 71 &+ $0 / 257 &* 13) % 509 - 254) / 32768
            }
            var weights = [UInt8](repeating: 0,
                count: values.count / (q4 ? 256 : 32) * (q4 ? 144 : 34))
            weights.withUnsafeMutableBytes { dst in
                if q4 {
                    values.withUnsafeBufferPointer {
                        Quantize.quantizeQ4_K($0.baseAddress!, count: values.count,
                                               into: dst.baseAddress!)
                    }
                } else {
                    let halves = values.map { Float16($0).bitPattern }
                    halves.withUnsafeBytes {
                        Quantize.quantizeF16Q8_0($0.baseAddress!, count: values.count,
                                                 into: dst.baseAddress!)
                    }
                }
            }
            weight = try GraphAttentionOutputFusedTests.padded(rt, payload: weights)
            x = try GraphAttentionOutputFusedTests.paddedFloats(rt,
                (0..<inDim).map { Float(($0 * 31) % 127 - 63) / 67 })
            residual = try GraphAttentionOutputFusedTests.paddedFloats(rt,
                (0..<(outDim * 4)).map { Float(($0 * 47) % 251 - 125) / 113 })
            split = try GraphAttentionOutputFusedTests.paddedFloats(rt,
                (0..<24).map { Float(($0 * 17) % 43 - 21) / 23 })
            block = try GraphAttentionOutputFusedTests.padded(rt,
                payload: [UInt8](repeating: 0, count: outDim * 4))
            out = try GraphAttentionOutputFusedTests.padded(rt,
                payload: [UInt8](repeating: 0, count: outDim * 16))
        }
    }

    private func view(_ t: GPUTensor, offset: Int? = nil, length: Int? = nil) -> GPUTensor {
        GPUTensor(buffer: t.buffer, byteLength: length ?? t.byteLength,
                  count: t.count, byteOffset: offset ?? t.byteOffset)
    }

    private static func padded(_ rt: MetalRuntime, payload: [UInt8]) throws -> GPUTensor {
        let all = [UInt8](repeating: 0xA5, count: 64) + payload + [UInt8](repeating: 0xA5, count: 64)
        let allocation = try GPUTensor.bytes(rt, all, elementCount: all.count)
        return allocation.subview(byteOffset: 64, byteLength: payload.count, count: payload.count / 4)
    }

    private static func paddedFloats(_ rt: MetalRuntime, _ values: [Float]) throws -> GPUTensor {
        try values.withUnsafeBytes { try padded(rt, payload: Array($0)) }
    }

    private func padded(_ rt: MetalRuntime, payload: [UInt8]) throws -> GPUTensor {
        try Self.padded(rt, payload: payload)
    }

    private func bytes(_ t: GPUTensor) -> [UInt8] {
        Array(UnsafeRawBufferPointer(start: t.buffer.contents(), count: t.buffer.length))
    }

    private func assertPadding(_ t: GPUTensor, file: StaticString = #filePath, line: UInt = #line) {
        let all = bytes(t)
        XCTAssertTrue(all[..<t.byteOffset].allSatisfy { $0 == 0xA5 }, file: file, line: line)
        XCTAssertTrue(all[(t.byteOffset + t.byteLength)...].allSatisfy { $0 == 0xA5 }, file: file, line: line)
    }
}
