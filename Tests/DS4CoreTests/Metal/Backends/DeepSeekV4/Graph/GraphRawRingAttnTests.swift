import XCTest
import Foundation
@testable import DS4Metal

/// Regression coverage for the decode raw-KV ring.  Once the 128-row physical
/// ring wraps, `flashAttnCore` must gather the logical SWA window into its F16
/// scratch in chronological order and attend over exactly those rows.
final class GraphRawRingAttnTests: XCTestCase {
    private let headDim = 512

    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() } // embedded kernels only
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testWrappedRawRingStagesF16InLogicalOrderAndMatchesCPUAttention() throws {
        let rt = try makeRuntime()
        let capacity = 128
        let nKeys = 47
        let rawStartRow = 111 // 17 rows at the tail, then 30 at the head.
        let nHead = 2

        // Give every physical element a deterministic value.  The range is
        // deliberately small enough to remain finite and meaningful in F16.
        var physical = [Float](repeating: 0, count: capacity * headDim)
        for row in 0..<capacity {
            for column in 0..<headDim {
                let code = (row * 521 + column * 17) % 8191
                physical[row * headDim + column] = Float(code - 4095) / 257.0
            }
        }

        var logical = [Float](repeating: 0, count: nKeys * headDim)
        for row in 0..<nKeys {
            let physicalRow = (rawStartRow + row) % capacity
            logical.replaceSubrange(
                row * headDim..<(row + 1) * headDim,
                with: physical[physicalRow * headDim..<(physicalRow + 1) * headDim]
            )
        }

        var query = [Float](repeating: 0, count: nHead * headDim)
        for index in query.indices {
            query[index] = Float((index * 29 + 7) % 127 - 63) / 128.0
        }

        let scratch = GraphContext.flashScratchBytes(nHead: nHead, nKeys: nKeys)
        let q = try GPUTensor.floats(rt, query)
        let raw = try GPUTensor.floats(rt, physical)
        let kvF16 = try GPUTensor.zerosBytes(rt, byteLength: scratch.kvF16)
        let mask = try GPUTensor.zerosBytes(rt, byteLength: scratch.mask)
        let sinks = try GPUTensor.zerosBytes(rt, byteLength: scratch.sinks)
        let pad = try GPUTensor.zerosBytes(rt, byteLength: scratch.pad)
        let tmp = try GPUTensor.zerosBytes(rt, byteLength: scratch.tmp)
        let heads = try GPUTensor.zeros(rt, floatCount: nHead * headDim)

        let graph = GraphContext(rt)
        try graph.begin()
        try graph.flashAttnCore(
            q: q,
            kvF32: raw,
            kvF16: kvF16,
            mask: mask,
            sinks: sinks,
            pad: pad,
            tmp: tmp,
            heads: heads,
            nHead: nHead,
            nKeys: nKeys,
            rawStartRow: rawStartRow
        )
        graph.commit()
        XCTAssertNil(graph.lastError)

        // Inspect the intermediate staging buffer so this catches both an
        // incorrect wrap span and an incorrect chronological reorder.  An
        // attention-only assertion could not detect permutations of the same
        // unmasked key set.
        let staged = (kvF16.buffer.contents() + kvF16.byteOffset)
            .bindMemory(to: UInt16.self, capacity: logical.count)
        for index in logical.indices {
            XCTAssertEqual(
                staged[index],
                Float16(logical[index]).bitPattern,
                "F16 raw-ring staging mismatch at logical element \(index)"
            )
        }

        // Also validate the end-to-end attention result against the same
        // F16-rounded logical window consumed by the GPU kernel.
        let keys = logical.map { Float(Float16($0)) }
        let expected = cpuAttention(query: query, keys: keys, nHead: nHead, nKeys: nKeys)
        let actual = heads.floatArray(nHead * headDim)
        var maxRelativeError: Float = 0
        for index in actual.indices {
            let denominator = max(abs(expected[index]), 0.05)
            maxRelativeError = max(maxRelativeError, abs(actual[index] - expected[index]) / denominator)
        }
        XCTAssertLessThan(maxRelativeError, 2e-2, "wrapped raw-ring attention max relative error \(maxRelativeError)")
    }

    private func cpuAttention(query: [Float], keys: [Float], nHead: Int, nKeys: Int) -> [Float] {
        let scale = 1.0 / Float(headDim).squareRoot()
        var output = [Float](repeating: 0, count: nHead * headDim)

        for head in 0..<nHead {
            var scores = [Float](repeating: 0, count: nKeys)
            var maximum = -Float.infinity
            for key in 0..<nKeys {
                var dot: Float = 0
                for column in 0..<headDim {
                    dot += query[head * headDim + column] * keys[key * headDim + column]
                }
                scores[key] = dot * scale
                maximum = max(maximum, scores[key])
            }

            var sum: Float = 0
            for key in 0..<nKeys {
                scores[key] = expf(scores[key] - maximum)
                sum += scores[key]
            }
            for key in 0..<nKeys {
                let weight = scores[key] / sum
                for column in 0..<headDim {
                    output[head * headDim + column] += weight * keys[key * headDim + column]
                }
            }
        }
        return output
    }
}
