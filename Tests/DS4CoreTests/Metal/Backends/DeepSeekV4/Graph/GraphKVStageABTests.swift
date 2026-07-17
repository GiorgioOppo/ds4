import XCTest
@testable import DS4Metal

/// Exact A/B coverage for the decode KV staging graph.  The optimized path is
/// allowed to change dispatch count only: staged F16 bits, partial-block pad and
/// final FlashAttention output must equal the historical generic graph.
final class GraphKVStageABTests: XCTestCase {
    private let headDim = 512

    private struct Result {
        let staged: [UInt16]
        let pad: [UInt8]
        let heads: [UInt32]
    }

    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testWrappedRawAndCompressedStageMatchesBaselineBitForBit() throws {
        let rt = try makeRuntime()
        let baseline = try run(rt: rt, rawCapacity: 128, rawStart: 111,
                               nRaw: 47, nComp: 5,
                               fused: false, vector: false)
        let vector = try run(rt: rt, rawCapacity: 128, rawStart: 111,
                             nRaw: 47, nComp: 5,
                             fused: false, vector: true)
        let fused = try run(rt: rt, rawCapacity: 128, rawStart: 111,
                            nRaw: 47, nComp: 5,
                            fused: true, vector: true)

        assertExact(vector, baseline, label: "vector-copy fallback")
        assertExact(fused, baseline, label: "fused circular+compressed stage")
    }

    func testFullFinalBlockStageMatchesBaselineBitForBit() throws {
        let rt = try makeRuntime()
        let baseline = try run(rt: rt, rawCapacity: 64, rawStart: 9,
                               nRaw: 27, nComp: 5,
                               fused: false, vector: false)
        let fused = try run(rt: rt, rawCapacity: 64, rawStart: 9,
                            nRaw: 27, nComp: 5,
                            fused: true, vector: true)
        assertExact(fused, baseline, label: "fused no-pad stage")
    }

    private func assertExact(_ actual: Result, _ expected: Result,
                             label: String, file: StaticString = #filePath,
                             line: UInt = #line) {
        XCTAssertEqual(actual.staged, expected.staged,
                       "\(label): staged F16 differs", file: file, line: line)
        XCTAssertEqual(actual.pad, expected.pad,
                       "\(label): pad bytes differ", file: file, line: line)
        XCTAssertEqual(actual.heads, expected.heads,
                       "\(label): attention output differs", file: file, line: line)
    }

    private func run(rt: MetalRuntime, rawCapacity: Int, rawStart: Int,
                     nRaw: Int, nComp: Int,
                     fused: Bool, vector: Bool) throws -> Result {
        let nHead = 2
        let total = nRaw + nComp
        var raw = [Float](repeating: 0, count: rawCapacity * headDim)
        var comp = [Float](repeating: 0, count: nComp * headDim)
        var query = [Float](repeating: 0, count: nHead * headDim)

        for i in raw.indices {
            raw[i] = Float((i &* 37 &+ 11) % 8191 - 4095) / 509.0
        }
        for i in comp.indices {
            comp[i] = Float((i &* 53 &+ 19) % 4093 - 2046) / 383.0
        }
        for i in query.indices {
            query[i] = Float((i &* 29 &+ 7) % 257 - 128) / 257.0
        }

        let scratch = GraphContext.flashScratchBytes(nHead: nHead, nKeys: total)
        let q = try GPUTensor.floats(rt, query)
        let rawTensor = try GPUTensor.floats(rt, raw)
        let compTensor = try GPUTensor.floats(rt, comp)
        let kvF16 = try GPUTensor.zerosBytes(rt, byteLength: scratch.kvF16)
        let mask = try GPUTensor.zerosBytes(rt, byteLength: scratch.mask)
        let sinks = try GPUTensor.zerosBytes(rt, byteLength: scratch.sinks)
        let pad = try GPUTensor.zerosBytes(rt, byteLength: scratch.pad)
        let tmp = try GPUTensor.zerosBytes(rt, byteLength: scratch.tmp)
        let heads = try GPUTensor.zeros(rt, floatCount: nHead * headDim)

        // Raw rows are visible.  Alternate compressed rows exercise bitwise
        // propagation of both zero and -inf masks into the partial pad block.
        let maskBits = (mask.buffer.contents() + mask.byteOffset)
            .bindMemory(to: UInt16.self, capacity: total)
        let negativeInfinity = Float16(-Float.infinity).bitPattern
        for i in 0..<nRaw { maskBits[i] = 0 }
        for i in 0..<nComp { maskBits[nRaw + i] = i.isMultiple(of: 2) ? 0 : negativeInfinity }

        let graph = GraphContext(rt)
        try graph.begin()
        try graph.flashAttnCore(q: q, kvF32: rawTensor, kvF16: kvF16,
                                mask: mask, sinks: sinks, pad: pad, tmp: tmp,
                                heads: heads, nHead: nHead, nKeys: nRaw,
                                rawStartRow: rawStart, hasSinks: false,
                                comp: compTensor, nComp: nComp,
                                fusedStage: fused, vectorCopy: vector)
        graph.commit()
        if let error = graph.lastError { throw error }

        let stagedPointer = (kvF16.buffer.contents() + kvF16.byteOffset)
            .bindMemory(to: UInt16.self, capacity: total * headDim)
        let staged = Array(UnsafeBufferPointer(start: stagedPointer,
                                               count: total * headDim))
        let padPointer = (pad.buffer.contents() + pad.byteOffset)
            .bindMemory(to: UInt8.self, capacity: scratch.pad)
        let padBytes = Array(UnsafeBufferPointer(start: padPointer, count: scratch.pad))
        let headBits = heads.floatArray(nHead * headDim).map(\.bitPattern)
        return Result(staged: staged, pad: padBytes, heads: headBits)
    }
}
