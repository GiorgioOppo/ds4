import Foundation
import Metal
import XCTest
import DS4Core
@testable import DS4Metal

final class GLM53KernelTests: XCTestCase {
    private func runtime() throws -> MetalRuntime {
        do { return try MetalRuntime(additionalSources: GLM53KernelSources.all) }
        catch MetalError.noDevice { throw XCTSkip("No Metal device") }
        catch MetalError.noQueue { throw XCTSkip("No Metal queue") }
    }
    private func values(_ n: Int, _ seed: Int = 0, _ scale: Float = 0.01) -> [Float] {
        (0..<n).map { Float(($0 * 13 + seed * 7) % 127 - 63) * scale }
    }
    private func finish(_ graph: GraphContext) throws {
        graph.commit()
        if let error = graph.lastError { throw error }
    }
    private func near(_ got: [Float], _ want: [Float], _ label: String,
                      absolute: Float = 0.0001, relative: Float = 0.0001,
                      file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(got.count, want.count, file: file, line: line)
        for i in got.indices {
            guard got[i].isFinite, want[i].isFinite,
                  abs(got[i] - want[i]) <= absolute + relative * abs(want[i]) else {
                XCTFail("\(label)[\(i)]: \(got[i]) != \(want[i])", file: file, line: line); return
            }
        }
    }

    func testKDAChunkedPrefillMatchesDecodeAndCPUIncludingPersistentState() throws {
        let rt = try runtime(), heads = 2, n = 5, width = heads * 128
        let rawQ = values(n * width, 1), rawK = values(n * width, 2), rawV = values(n * width, 3)
        let rawD = values(n * width, 4), rawB = values(n * heads, 5), rawG = values(n * width, 6)
        let cq = values(width * 4, 7, 0.003), ck = values(width * 4, 8, 0.003), cv = values(width * 4, 9, 0.003)
        let a = values(heads, 10), dt = values(width, 11), norm = values(128, 12, 0.002).map { $0 + 1 }
        let qConv = try GPUTensor.floats(rt, cq), kConv = try GPUTensor.floats(rt, ck), vConv = try GPUTensor.floats(rt, cv)
        let aLog = try GPUTensor.floats(rt, a), dtBias = try GPUTensor.floats(rt, dt), normGPU = try GPUTensor.floats(rt, norm)
        func run(chunks: [Int]) throws -> (out: [Float], state: [Float], conv: [Float]) {
            let state = try GPUTensor.zeros(rt, floatCount: heads * 128 * 128)
            let conv = try GPUTensor.zeros(rt, floatCount: 9 * width)
            var result: [Float] = [], offset = 0
            for count in chunks {
                func input(_ x: [Float], _ stride: Int) throws -> GPUTensor {
                    try .floats(rt, Array(x[(offset * stride)..<((offset + count) * stride)]))
                }
                let q = try input(rawQ, width), k = try input(rawK, width), v = try input(rawV, width), decay = try input(rawD, width)
                let beta = try input(rawB, heads), gate = try input(rawG, width)
                let output = try GPUTensor.zeros(rt, floatCount: count * width)
                let graph = GraphContext(rt); try graph.begin()
                try graph.glm53KDA(q: q, k: k, v: v, decay: decay, beta: beta, gate: gate,
                    qConv: qConv, kConv: kConv, vConv: vConv, aLog: aLog, dtBias: dtBias, norm: normGPU,
                    convolutionState: conv, recurrentState: state, out: output, heads: heads, rows: count)
                try finish(graph)
                result += output.floatArray()
                offset += count
            }
            return (result, state.floatArray(), conv.floatArray())
        }
        let decode = try run(chunks: Array(repeating: 1, count: n))
        let batch = try run(chunks: [n]), resumed = try run(chunks: [2, 3])
        near(batch.out, decode.out, "batch/decode")
        near(batch.state, decode.state, "batch/decode state")
        XCTAssertEqual(batch.conv, decode.conv)
        near(resumed.out, decode.out, "resumed/decode")
        near(resumed.state, decode.state, "resumed/decode state")

        // Independent scalar recurrence: no GPU-produced intermediate enters
        // the oracle. Weight and activation dimensions are deliberately small.
        var state = [Float](repeating: 0, count: heads * 128 * 128)
        var histories = [[Float]](repeating: Array(repeating: 0, count: 3 * width), count: 3)
        var expected = [Float](repeating: 0, count: n * width)
        func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }
        for t in 0..<n {
            var transformed = [[Float]](repeating: Array(repeating: 0, count: width), count: 3)
            for (channel, pair) in [(rawQ, cq), (rawK, ck), (rawV, cv)].enumerated() {
                for d in 0..<width {
                    var sum: Float = 0
                    for h in 0..<3 { sum += histories[channel][h * width + d] * pair.1[d * 4 + h] }
                    sum += pair.0[t * width + d] * pair.1[d * 4 + 3]
                    transformed[channel][d] = sum * sigmoid(sum)
                    histories[channel][d] = histories[channel][width + d]
                    histories[channel][width + d] = histories[channel][2 * width + d]
                    histories[channel][2 * width + d] = pair.0[t * width + d]
                }
            }
            for h in 0..<heads {
                let base = h * 128
                let qsum = (0..<128).reduce(Float(0)) { $0 + transformed[0][base + $1] * transformed[0][base + $1] }
                let ksum = (0..<128).reduce(Float(0)) { $0 + transformed[1][base + $1] * transformed[1][base + $1] }
                let qs = 1 / sqrt(qsum + 1e-6) / sqrt(Float(128)), ks = 1 / sqrt(ksum + 1e-6)
                let beta = sigmoid(rawB[t * heads + h])
                var out = [Float](repeating: 0, count: 128)
                for v in 0..<128 {
                    let sb = (h * 128 + v) * 128
                    var prediction: Float = 0
                    for k in 0..<128 {
                        let decay = exp(-5 * sigmoid(exp(a[h]) * (rawD[t * width + base + k] + dt[base + k])))
                        state[sb + k] *= decay
                        prediction += state[sb + k] * transformed[1][base + k] * ks
                    }
                    let delta = (transformed[2][base + v] - prediction) * beta
                    for k in 0..<128 {
                        state[sb + k] += transformed[1][base + k] * ks * delta
                        out[v] += state[sb + k] * transformed[0][base + k] * qs
                    }
                }
                let scale = 1 / sqrt(out.reduce(Float(0)) { $0 + $1 * $1 } / 128 + 1e-5)
                for d in 0..<128 { expected[t * width + base + d] = out[d] * scale * norm[d] * sigmoid(rawG[t * width + base + d]) }
            }
        }
        near(batch.out, expected, "CPU output")
        near(batch.state, state, "CPU recurrent state")
        XCTAssertEqual(batch.conv, histories.flatMap { $0 })
    }

    func testBF16ProjectionMatvecAndTiledPrefillAgainstCPU() throws {
        let rt = try runtime(), columns = 64, width = 65
        let bits = values(columns * width, 3).map { UInt16($0.bitPattern >> 16) }
        let weight = try bits.withUnsafeBytes { try GPUTensor.raw(rt, ptr: $0.baseAddress!, byteLength: $0.count, elementCount: bits.count) }
        for rows in [1, 9, 32] {
            let x = values(rows * columns, 4)
            let input = try GPUTensor.floats(rt, x), output = try GPUTensor.zeros(rt, floatCount: rows * width)
            let graph = GraphContext(rt); try graph.begin()
            try graph.glm53Projection(weight, type: 30, input: input, output: output, columns: columns, width: width, rows: rows)
            try finish(graph)
            var want = [Float](repeating: 0, count: rows * width)
            for r in 0..<rows { for o in 0..<width {
                var sum: Double = 0
                for k in 0..<columns {
                    let w = Float(bitPattern: UInt32(bits[o * columns + k]) << 16)
                    sum += Double(rows > 8 ? Float(Float16(w)) : w) * Double(rows > 8 ? Float(Float16(x[r * columns + k])) : x[r * columns + k])
                }
                want[r * width + o] = Float(sum)
            } }
            near(output.floatArray(), want, "BF16 \(rows) rows")
        }
    }

    func testGroupedQuantizedProjectionWithOffsetViewsAndGuards() throws {
        let rt = try runtime(), columns = 256, width = 5, heads = 2, rows = 3
        let source = values(columns * width * heads, 8, 0.004)
        for type: UInt32 in [8, 12] {
            let bytes = Int(GGUF.tensorNBytes(type: type, elements: UInt64(source.count))!)
            var packed = [UInt8](repeating: 0, count: bytes)
            packed.withUnsafeMutableBytes { dst in
                if type == 8 {
                    let half = source.map { Half.bits($0) }
                    half.withUnsafeBytes { Quantize.quantizeF16Q8_0($0.baseAddress!, count: source.count, into: dst.baseAddress!) }
                } else {
                    source.withUnsafeBufferPointer { Quantize.quantizeQ4_K($0.baseAddress!, count: source.count, into: dst.baseAddress!) }
                }
            }
            var dequant = [Float](repeating: 0, count: source.count)
            packed.withUnsafeBytes { src in dequant.withUnsafeMutableBufferPointer { dst in
                if type == 8 { Quantize.dequantQ8_0(src.baseAddress!, count: source.count, into: dst.baseAddress!) }
                else { Quantize.dequantQ4_K(src.baseAddress!, count: source.count, into: dst.baseAddress!) }
            } }
            let slab = [UInt8](repeating: 0xAB, count: 16) + packed + [UInt8](repeating: 0xAB, count: 16)
            let weightSlab = try slab.withUnsafeBytes { try GPUTensor.raw(rt, ptr: $0.baseAddress!, byteLength: $0.count, elementCount: source.count) }
            let weight = weightSlab.subview(byteOffset: 16, byteLength: bytes, count: source.count)
            let x = values(rows * heads * columns, 4, 0.006)
            let inputSlab = try GPUTensor.floats(rt, [31] + x + [37])
            let input = inputSlab.subview(byteOffset: 4, byteLength: x.count * 4, count: x.count)
            let outputSlab = try GPUTensor.floats(rt, Array(repeating: -73, count: rows * heads * width + 2))
            let output = outputSlab.subview(byteOffset: 4, byteLength: rows * heads * width * 4, count: rows * heads * width)
            let graph = GraphContext(rt); try graph.begin()
            try graph.glm53Projection(weight, type: type, input: input, output: output,
                                     columns: columns, width: width, rows: rows, heads: heads)
            try finish(graph)
            var expected = [Float](repeating: 0, count: output.count)
            for r in 0..<rows { for h in 0..<heads { for o in 0..<width {
                var sum: Double = 0
                for k in 0..<columns { sum += Double(dequant[(h * width + o) * columns + k]) * Double(x[(r * heads + h) * columns + k]) }
                expected[(r * heads + h) * width + o] = Float(sum)
            } } }
            near(output.floatArray(), expected, "quantized grouped type \(type)")
            XCTAssertEqual(outputSlab.floatArray().first, -73)
            XCTAssertEqual(outputSlab.floatArray().last, -73)
            XCTAssertEqual(inputSlab.floatArray(), [31] + x + [37])
            XCTAssertEqual(Array(UnsafeBufferPointer(start: weightSlab.buffer.contents().assumingMemoryBound(to: UInt8.self), count: slab.count)), slab)
        }
    }

    func testRoutingUsesBiasedSelectionAndUnbiasedNormalizedSigmoidWeights() throws {
        let rt = try runtime()
        let logits = values(288, 3), bias = values(288, 5, 0.003)
        let input = try GPUTensor.floats(rt, logits), b = try GPUTensor.floats(rt, bias)
        let ids = try GPUTensor.zeros(rt, floatCount: 8), weights = try GPUTensor.zeros(rt, floatCount: 8)
        let graph = GraphContext(rt); try graph.begin()
        try graph.glm53Dispatch("kernel_glm53_route", [1], [input, b, ids, weights], groups: .init(width: 1, height: 1, depth: 1), threads: 32)
        try finish(graph)
        let probs = logits.map { 1 / (1 + exp(-$0)) }
        let expected = (0..<288).sorted {
            let a = probs[$0] + bias[$0], b = probs[$1] + bias[$1]
            return a == b ? $0 < $1 : a > b
        }.prefix(8)
        let ptr = ids.buffer.contents().assumingMemoryBound(to: UInt32.self)
        XCTAssertEqual((0..<8).map { Int(ptr[$0]) }, Array(expected))
        let denominator = expected.reduce(Float(0)) { $0 + probs[$1] }
        near(weights.floatArray(), expected.map { probs[$0] * 2.5 / denominator }, "router")
    }

    func testAbsorbedSparseAttentionIgnoresPaddingAndMatchesCPU() throws {
        let rt = try runtime(), heads = 2, visible = 7
        let query = values(heads * 512, 3, 0.002)
        let cacheBits = values(visible * 512, 4, 0.004).map { Float16($0).bitPattern }
        let selection: [UInt32] = [0, 3, 6, UInt32.max]
        let q = try GPUTensor.floats(rt, query)
        let cache = try cacheBits.withUnsafeBytes { try GPUTensor.raw(rt, ptr: $0.baseAddress!, byteLength: $0.count, elementCount: cacheBits.count) }
        let selected = try selection.withUnsafeBytes { try GPUTensor.raw(rt, ptr: $0.baseAddress!, byteLength: $0.count, elementCount: selection.count) }
        let output = try GPUTensor.zeros(rt, floatCount: heads * 512)
        let graph = GraphContext(rt); try graph.begin()
        try graph.glm53Dispatch("kernel_glm53_attention", [UInt32(visible), 4, UInt32(heads), Float(1.0/16).bitPattern],
            [q, cache, selected, output], groups: .init(width: heads, height: 1, depth: 1), shared: 32)
        try finish(graph)
        let kv = cacheBits.map { Float(Float16(bitPattern: $0)) }
        var expected = [Float](repeating: 0, count: heads * 512)
        for h in 0..<heads {
            let rows = [0, 3, 6]
            let scores = rows.map { row in (0..<512).reduce(Double(0)) { $0 + Double(query[h*512+$1]) * Double(kv[row*512+$1]) } / 16 }
            let m = scores.max()!, p = scores.map { exp($0 - m) }, denominator = p.reduce(0,+)
            for d in 0..<512 {
                expected[h*512+d] = Float(zip(rows,p).reduce(Double(0)) { $0 + Double(kv[$1.0*512+d]) * $1.1 } / denominator)
            }
        }
        near(output.floatArray(), expected, "DSA CPU")
    }

    func testPoolUpdateAcrossPartialGroupsMatchesCPU() throws {
        let rt = try runtime(), n = 7, width = 128
        let keys = values(n * width, 1), gates = values(n * width, 2)
        let norm = values(width, 3, 0.002).map { $0 + 1 }, bias = values(width, 4, 0.001)
        let apeBits = values(4 * width, 5, 0.004).map { UInt16($0.bitPattern >> 16) }
        let nw = try GPUTensor.floats(rt, norm), nb = try GPUTensor.floats(rt, bias)
        let ape = try apeBits.withUnsafeBytes { try GPUTensor.raw(rt, ptr: $0.baseAddress!, byteLength: $0.count, elementCount: apeBits.count) }
        func run(_ chunks: [Int]) throws -> [Float] {
            let cache = try GPUTensor.zerosBytes(rt, byteLength: 2 * width * 2)
            let tailK = try GPUTensor.zeros(rt, floatCount: 4 * width), tailG = try GPUTensor.zeros(rt, floatCount: 4 * width)
            var pos = 0
            for count in chunks {
                let k = try GPUTensor.floats(rt, Array(keys[pos*width..<(pos+count)*width]))
                let g = try GPUTensor.floats(rt, Array(gates[pos*width..<(pos+count)*width]))
                let graph = GraphContext(rt); try graph.begin()
                try graph.glm53PoolUpdate(rawKeys: k, gates: g, norm: nw, bias: nb, ape: ape,
                    cache: cache, tailKeys: tailK, tailGates: tailG,
                    position: pos, rows: count, capacity: n)
                try finish(graph); pos += count
            }
            let p = cache.buffer.contents().assumingMemoryBound(to: UInt16.self)
            return (0..<width).map { Float(Float16(bitPattern: p[$0])) }
        }
        let batch = try run([7]), resumed = try run([3, 4]), decode = try run(Array(repeating: 1, count: 7))
        XCTAssertEqual(batch, resumed); XCTAssertEqual(batch, decode)
        XCTAssertEqual(batch, try run([1, 6])); XCTAssertEqual(batch, try run([2, 5]))
        let means = (0..<4).map { row in keys[row*width..<(row+1)*width].reduce(0,+) / Float(width) }
        let invs = (0..<4).map { row in 1 / sqrt(keys[row*width..<(row+1)*width].reduce(Float(0)) { $0 + ($1-means[row])*($1-means[row]) } / Float(width) + 1e-6) }
        var expected = [Float](repeating: 0, count: width)
        for d in 0..<width {
            let scores = (0..<4).map { gates[$0*width+d] + Float(bitPattern: UInt32(apeBits[$0*width+d]) << 16) }
            let m = scores.max()!, expScores = scores.map { exp($0-m) }, denom = expScores.reduce(0,+)
            for row in 0..<4 { expected[d] += expScores[row]/denom * ((keys[row*width+d]-means[row])*invs[row]*norm[d]+bias[d]) }
            expected[d] = Float(Float16(expected[d]))
        }
        near(batch, expected, "pool CPU", absolute: 0.002, relative: 0.0001)
    }
}
