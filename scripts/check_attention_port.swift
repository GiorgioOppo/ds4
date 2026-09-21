import Foundation
import Metal
import DS4Core
import DS4Metal

// Standalone checks for the attention-output/HC and Q4 prefill pair ports.
// No XCTest, application preferences, GGUF files, or model inference required.
// See attention_port_checks.md. Build/run using check_attention_port.sh.

private struct CheckFailure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

private func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw CheckFailure(message) }
}

private struct RandomValues {
    var seed: UInt64 = 0xD54A7710
    mutating func values(_ count: Int, scale: Float = 0.3) -> [Float] {
        (0..<count).map { _ in
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return (Float(UInt32(truncatingIfNeeded: seed >> 32)) / Float(UInt32.max) * 2 - 1) * scale
        }
    }
}

/// All inputs and outputs are nested views with nonzero offsets. Guard bytes
/// remain part of the owning allocation, so writes outside each view are caught.
private final class GuardedTensor {
    let owner: GPUTensor
    let view: GPUTensor
    let initial: [UInt8]
    private let prefix = 256
    private let suffix = 256
    let name: String

    init(_ runtime: MetalRuntime, name: String, bytes: [UInt8], count: Int) throws {
        self.name = name
        initial = [UInt8](repeating: 0xA5, count: prefix) + bytes + [UInt8](repeating: 0x5A, count: suffix)
        owner = try .bytes(runtime, initial, elementCount: initial.count)
        let outer = owner.subview(byteOffset: 128, byteLength: initial.count - 128, count: initial.count - 128)
        view = outer.subview(byteOffset: prefix - 128, byteLength: bytes.count, count: count)
    }

    convenience init(_ runtime: MetalRuntime, name: String, floats: [Float]) throws {
        try self.init(runtime, name: name, bytes: floats.withUnsafeBytes { Array($0) }, count: floats.count)
    }

    convenience init(_ runtime: MetalRuntime, name: String, outputFloats: Int) throws {
        try self.init(runtime, name: name,
                      floats: [Float](repeating: Float(bitPattern: 0x7FC01234), count: outputFloats))
    }

    func allBytes() -> [UInt8] {
        Array(UnsafeRawBufferPointer(start: owner.buffer.contents(), count: owner.byteLength))
    }

    func verifyUnchanged() throws {
        try require(allBytes() == initial, "input modified: \(name)")
    }

    func verifyGuards() throws {
        let bytes = allBytes()
        try require(bytes.prefix(prefix) == initial.prefix(prefix), "prefix guard changed: \(name)")
        try require(bytes.suffix(suffix) == initial.suffix(suffix), "suffix guard changed: \(name)")
    }

    func reset() {
        initial.withUnsafeBytes { _ = memcpy(owner.buffer.contents(), $0.baseAddress!, $0.count) }
    }
}

private func quantized(_ values: [Float], q4: Bool, rows: Int, columns: Int) throws -> [UInt8] {
    let type: UInt32 = q4 ? 12 : 8
    let bytes = QuantEncode.rowSize(type: type, columns: columns) * rows
    var packed = [UInt8](repeating: 0, count: bytes)
    let written = values.withUnsafeBufferPointer { source in
        packed.withUnsafeMutableBytes { destination in
            QuantEncode.quantizeChunk(type: type, src: source.baseAddress!, dst: destination.baseAddress!,
                                      start: 0, rows: rows, columns: columns, imatrix: nil)
        }
    }
    try require(written == bytes, "quantization byte count")
    return packed
}

private func finish(_ context: GraphContext) throws {
    context.commit()
    if let error = context.lastError { throw error }
}

private func exact(_ actual: GPUTensor, _ reference: GPUTensor, _ label: String) throws {
    let a = actual.floatArray(), b = reference.floatArray()
    try require(a.count == b.count, "\(label): count")
    for i in a.indices {
        try require(a[i].isFinite && b[i].isFinite, "\(label)[\(i)]: nonfinite/unwritten output")
        try require(a[i].bitPattern == b[i].bitPattern,
                    "\(label)[\(i)] not bit-exact: \(a[i]) vs \(b[i]), bits \(a[i].bitPattern)/\(b[i].bitPattern)")
    }
}

private func near(_ actual: [Float], _ reference: [Float], _ label: String,
                  absolute: Float = 0.0001, relative: Float = 0.0001) throws {
    try require(actual.count == reference.count, "\(label): count")
    var maximum: Float = 0, scaled: Float = 0, squared: Double = 0, bitDifferences = 0
    for i in actual.indices {
        let a = actual[i], b = reference[i]
        try require(a.isFinite && b.isFinite, "\(label)[\(i)]: nonfinite/unwritten output")
        let delta = abs(a - b), bound = absolute + relative * abs(b)
        if a.bitPattern != b.bitPattern { bitDifferences += 1 }
        maximum = max(maximum, delta); scaled = max(scaled, delta / bound)
        squared += Double(delta) * Double(delta)
        try require(delta <= bound, "\(label)[\(i)] tolerance exceeded: \(a) vs \(b), delta=\(delta), bound=\(bound)")
    }
    print(String(format: "  parity %@ max_abs=%.8g max_bound_fraction=%.5g rmse=%.8g bit_differences=%d/%d",
                 label, maximum, scaled, sqrt(squared / Double(max(1, actual.count))), bitDifferences, actual.count))
}

private struct BenchmarkOptions {
    var enabled = false
    var pairs = 8
    var repeats = 8
}

private func checkEligibility() throws {
    for (device, supported) in [("Apple M1 Max", true), ("Apple M2 Ultra", true),
                                 ("Apple M3", true), ("Apple M4 Pro", true),
                                 ("Apple M5", false), ("Apple M10", false),
                                 ("Unknown GPU", false)] {
        for nTok in [0, 1, 31, 32, 33, 64, 128, 256, 257] {
            let expected = supported && [32, 64, 128, 256].contains(nTok)
            let actual = GraphContext.q4PrefillPairEligible(deviceName: device,
                inDim: 4096, qOutDim: 1024, kvOutDim: 512, nTok: nTok)
            try require(actual == expected, "eligibility contract: \(device), \(nTok)")
        }
    }
    for (input, q, kv) in [(2048, 1024, 512), (4096, 512, 512), (4096, 1024, 256)] {
        try require(!GraphContext.q4PrefillPairEligible(deviceName: "Apple M1 Max",
            inDim: input, qOutDim: q, kvOutDim: kv, nTok: 32), "wrong geometry accepted")
    }
    print("PASS CPU eligibility: M1–M4, unsupported devices, tile boundaries, wrong geometry")
}

private func median(_ values: [Double]) -> Double {
    let sorted = values.sorted(), middle = sorted.count / 2
    return sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
}

/// Measures resident-kernel batches including CPU encode/submit/wait overhead.
/// Allocations, quantization, validation and printing stay outside timed regions.
/// AB/BA order alternates; no end-to-end inference or GPU-only time is implied.
private func benchmark(_ runtime: MetalRuntime, label: String, options: BenchmarkOptions,
                       reference: (GraphContext) throws -> Void,
                       candidate: (GraphContext) throws -> Void) throws {
    guard options.enabled else { return }
    func sample(_ encode: (GraphContext) throws -> Void) throws -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        let context = GraphContext(runtime); try context.begin()
        for _ in 0..<options.repeats { try encode(context) }
        try finish(context)
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000 / Double(options.repeats)
    }
    for _ in 0..<3 { _ = try sample(reference); _ = try sample(candidate) }
    var a: [Double] = [], b: [Double] = []
    for index in 0..<options.pairs {
        let av: Double, bv: Double
        if index.isMultiple(of: 2) { av = try sample(reference); bv = try sample(candidate) }
        else { bv = try sample(candidate); av = try sample(reference) }
        a.append(av); b.append(bv)
        print(String(format: "  sample %@ pair=%d order=%@ reference_ms=%.6f candidate_ms=%.6f",
                     label, index, index.isMultiple(of: 2) ? "AB" : "BA", av, bv))
    }
    let pairedRatios = zip(a, b).map { $0 / $1 }
    print(String(format: "BENCH %@ reference_median_ms=%.6f candidate_median_ms=%.6f paired_speedup=%.4fx pairs=%d repeats=%d metric=resident_batch_wall",
                 label, median(a), median(b), median(pairedRatios), options.pairs, options.repeats))
}

private func checkDecode(_ runtime: MetalRuntime, options: BenchmarkOptions) throws {
    for q4 in [false, true] {
        for (inDim, outDim) in [(512, 66), (4096, 4096)] {
            let label = "outputHC/\(q4 ? "Q4_K" : "Q8_0")/\(inDim)x\(outDim)"
            var random = RandomValues(seed: UInt64(inDim * 17 + outDim))
            // The old Q4 matvec can read inactive SIMDgroup rows in its final
            // group. Pad weights through NSG=8 while checking output tails.
            let paddedRows = (outDim + 15) / 16 * 16
            let w = try GuardedTensor(runtime, name: label + "/weight",
                bytes: quantized(random.values(paddedRows * inDim, scale: 0.05), q4: q4,
                                 rows: paddedRows, columns: inDim), count: paddedRows * inDim)
            let x = try GuardedTensor(runtime, name: label + "/x", floats: random.values(inDim))
            let residual = try GuardedTensor(runtime, name: label + "/residual", floats: random.values(outDim * 4))
            var splitValues = random.values(24, scale: 0.2)
            for i in 0..<4 { splitValues[4 + i] += 0.7; splitValues[8 + i * 4 + i] += 0.5 }
            let split = try GuardedTensor(runtime, name: label + "/split", floats: splitValues)
            let refBlock = try GuardedTensor(runtime, name: label + "/refBlock", outputFloats: outDim)
            let ref = try GuardedTensor(runtime, name: label + "/reference", outputFloats: outDim * 4)
            let gotBlock = try GuardedTensor(runtime, name: label + "/block", outputFloats: outDim)
            let got = try GuardedTensor(runtime, name: label + "/candidate", outputFloats: outDim * 4)
            func reference(_ c: GraphContext) throws {
                if q4 { try c.matmulQ4_K(weight: w.view, x: x.view, out: refBlock.view, inDim: inDim, outDim: outDim) }
                else { try c.matmulQ8_0(weight: w.view, x: x.view, out: refBlock.view, inDim: inDim, outDim: outDim) }
                try c.hcExpand4(blockOut: refBlock.view, residual: residual.view, post: split.view, comb: split.view,
                                blockAdd: nil, out: ref.view, nEmbd: outDim, nTokens: 1,
                                postByteOffset: 16, combByteOffset: 32)
            }
            func candidate(_ c: GraphContext) throws {
                let used = try c.attentionOutputHCFused(weight: w.view, x: x.view, blockOut: gotBlock.view,
                    residual: residual.view, split: split.view, out: got.view, inDim: inDim, outDim: outDim,
                    q4: q4, enabled: true)
                try require(used, "\(label): supported fusion unexpectedly rejected")
            }
            let c = GraphContext(runtime); try c.begin()
            try reference(c)
            for _ in 0..<3 { try candidate(c) }
            try finish(c)
            try exact(gotBlock.view, refBlock.view, label + "/projection")
            try exact(got.view, ref.view, label + "/HC")

            // Same queue, independent contexts, reused outputs and scratch.
            let first = GraphContext(runtime); try first.begin(); try candidate(first); first.commitAsync()
            let second = GraphContext(runtime); try second.begin(); try candidate(second); try finish(second)
            first.waitCompleted()
            try exact(got.view, ref.view, label + "/queued-repeat")

            // An encoded but uncommitted graph must be disposable without
            // submitting work. GraphContext.deinit closes the encoder safely.
            got.reset(); gotBlock.reset()
            do {
                let abandoned = GraphContext(runtime); try abandoned.begin()
                try candidate(abandoned)
            }
            try got.verifyUnchanged(); try gotBlock.verifyUnchanged()

            // A command buffer retains its resources after local tensor/view
            // wrappers leave scope, up to the actual submission/completion.
            if inDim == 512 {
                let retained = GraphContext(runtime); try retained.begin()
                try autoreleasepool {
                    func duplicate(_ tensor: GPUTensor) throws -> GPUTensor {
                        try .raw(runtime, ptr: tensor.buffer.contents() + tensor.byteOffset,
                                 byteLength: tensor.byteLength, elementCount: tensor.count)
                    }
                    let localWeight = try duplicate(w.view), localX = try duplicate(x.view)
                    let localResidual = try duplicate(residual.view), localSplit = try duplicate(split.view)
                    let used = try retained.attentionOutputHCFused(weight: localWeight, x: localX,
                        blockOut: gotBlock.view, residual: localResidual, split: localSplit,
                        out: got.view, inDim: inDim, outDim: outDim, q4: q4, enabled: true)
                    try require(used, "resource-lifetime fusion rejected")
                }
                try finish(retained)
                try exact(got.view, ref.view, label + "/resource-lifetime")
            }

            for rejection in 0..<5 {
                got.reset(); gotBlock.reset()
                let c = GraphContext(runtime); try c.begin()
                let used = try c.attentionOutputHCFused(weight: w.view, x: x.view,
                    blockOut: rejection == 4 ? got.view : gotBlock.view,
                    residual: residual.view, split: split.view,
                    out: rejection == 3 ? residual.view : got.view,
                    inDim: inDim, outDim: rejection == 2 ? outDim - 1 : outDim,
                    q4: q4, nHC: rejection == 1 ? 3 : 4, enabled: rejection != 0)
                try require(!used, "\(label): rejection \(rejection) unexpectedly dispatched")
                try finish(c)
                try got.verifyUnchanged(); try gotBlock.verifyUnchanged(); try residual.verifyUnchanged()
            }
            try benchmark(runtime, label: label, options: options, reference: reference, candidate: candidate)
            for input in [w, x, residual, split] { try input.verifyUnchanged() }
            for output in [refBlock, ref, gotBlock, got] { try output.verifyGuards() }
            print("PASS \(label): bit-exact projection+HC, tails, offsets, guards, immutability, reuse, fallback")
        }
    }
}

/// Scalar CPU model of the existing GEMM's Q4 operand conversion, including
/// half(d / 16) for high nibbles. Preserving that boundary is intentional even
/// for subnormal scales: the optimization must retain current model numerics.
private func dequantizeGEMMRow(_ weight: [UInt8], row: Int, columns: Int) -> [Float] {
    let rowBytes = columns / 256 * 144
    var values = [Float](repeating: 0, count: columns)
    for block in 0..<(columns / 256) {
        let base = row * rowBytes + block * 144
        let d = Float(Float16(bitPattern: UInt16(weight[base]) | UInt16(weight[base + 1]) << 8))
        let minimum = Float(Float16(bitPattern: UInt16(weight[base + 2]) | UInt16(weight[base + 3]) << 8))
        let scales = Array(weight[(base + 4)..<(base + 16)])
        for group in 0..<8 {
            let scale: UInt8, minScale: UInt8
            if group < 4 { scale = scales[group] & 63; minScale = scales[group + 4] & 63 }
            else {
                scale = (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4)
                minScale = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4)
            }
            let high = !group.isMultiple(of: 2)
            let dl = (high ? Float(Float16(d / 16)) : d) * Float(scale)
            let ml = minimum * Float(minScale)
            for k in 0..<32 {
                let packed = weight[base + 16 + group / 2 * 32 + k]
                let masked = high ? packed & 0xF0 : packed & 0x0F
                values[block * 256 + group * 32 + k] = Float(Float16(dl * Float(masked) - ml))
            }
        }
    }
    return values
}

/// Independent sampled CPU oracle: decoded F16 weights and activations with a
/// Double dot product. Neither GPU projection is the sole numerical authority.
private func sampledPrefillOracle(weight: [UInt8], activation: [Float], output: GPUTensor,
                                  inDim: Int, outDim: Int, nTok: Int, label: String,
                                  absolute: Float = 0.0001) throws {
    let rows = [0, nTok / 2, nTok - 1]
    let columns = [0, 1, outDim / 2, outDim - 2, outDim - 1]
    let gpu = output.floatArray()
    var actual: [Float] = [], expected: [Float] = []
    for column in columns {
        let dequantized = dequantizeGEMMRow(weight, row: column, columns: inDim)
        for row in rows {
            var sum: Double = 0
            for k in 0..<inDim {
                sum += Double(Float16(dequantized[k])) * Double(Float16(activation[row * inDim + k]))
            }
            actual.append(gpu[row * outDim + column]); expected.append(Float(sum))
        }
    }
    try near(actual, expected, label + "/CPU-F16-oracle", absolute: absolute, relative: 0.0001)
}

private func verifyHalfStaging(_ tensor: GPUTensor, activation: [Float]) throws {
    let halves = (tensor.buffer.contents() + tensor.byteOffset).assumingMemoryBound(to: UInt16.self)
    for index in activation.indices {
        try require(halves[index] == Float16(activation[index]).bitPattern,
                    "RHS F16 staging mismatch at \(index)")
    }
}

private func checkPrefill(_ runtime: MetalRuntime, options: BenchmarkOptions) throws {
    let inDim = 4096, qDim = 1024, kvDim = 512
    var random = RandomValues(seed: 0xB850AA)
    let qBytes = try quantized(random.values(inDim * qDim, scale: 0.05), q4: true, rows: qDim, columns: inDim)
    let kvBytes = try quantized(random.values(inDim * kvDim, scale: 0.05), q4: true, rows: kvDim, columns: inDim)
    let qWeight = try GuardedTensor(runtime, name: "prefill/qWeight", bytes: qBytes, count: inDim * qDim)
    let kvWeight = try GuardedTensor(runtime, name: "prefill/kvWeight", bytes: kvBytes, count: inDim * kvDim)
    for nTok in [31, 32, 33, 64, 128, 256] {
        let label = "prefillQ4/nTok\(nTok)"
        let activation = random.values(nTok * inDim)
        let x = try GuardedTensor(runtime, name: label + "/x", floats: activation)
        let rhs = try GuardedTensor(runtime, name: label + "/rhs",
                                   bytes: [UInt8](repeating: 0xA7, count: nTok * inDim * 2), count: nTok * inDim)
        let refQ = try GuardedTensor(runtime, name: label + "/refQ", outputFloats: nTok * qDim)
        let refKV = try GuardedTensor(runtime, name: label + "/refKV", outputFloats: nTok * kvDim)
        let gotQ = try GuardedTensor(runtime, name: label + "/q", outputFloats: nTok * qDim)
        let gotKV = try GuardedTensor(runtime, name: label + "/kv", outputFloats: nTok * kvDim)
        let eligible = GraphContext.q4PrefillPairEligible(deviceName: runtime.device.name,
            inDim: inDim, qOutDim: qDim, kvOutDim: kvDim, nTok: nTok)
        func reference(_ c: GraphContext) throws {
            try c.encodeMMDenseQ4K(weight: qWeight.view, act: x.view, actBase: 0, out: refQ.view,
                                   inDim: inDim, outDim: qDim, nTok: nTok)
            try c.encodeMMDenseQ4K(weight: kvWeight.view, act: x.view, actBase: 0, out: refKV.view,
                                   inDim: inDim, outDim: kvDim, nTok: nTok)
        }
        func candidate(_ c: GraphContext) throws {
            let used = try c.encodeQ4PrefillPair(weightQA: qWeight.view, weightKV: kvWeight.view,
                act: x.view, qOut: gotQ.view, kvOut: gotKV.view, rhsF16: rhs.view,
                inDim: inDim, qOutDim: qDim, kvOutDim: kvDim, nTok: nTok, enabled: true)
            try require(used == eligible, "\(label): selected path disagrees with eligibility")
            if !used {
                try c.encodeMMDenseQ4K(weight: qWeight.view, act: x.view, actBase: 0, out: gotQ.view,
                                       inDim: inDim, outDim: qDim, nTok: nTok)
                try c.encodeMMDenseQ4K(weight: kvWeight.view, act: x.view, actBase: 0, out: gotKV.view,
                                       inDim: inDim, outDim: kvDim, nTok: nTok)
            }
        }
        // Rejections must leave outputs AND shared RHS untouched.
        for rejection in 0..<4 {
            let c = GraphContext(runtime); try c.begin()
            let used = try c.encodeQ4PrefillPair(weightQA: qWeight.view, weightKV: kvWeight.view,
                act: x.view, qOut: gotQ.view, kvOut: rejection == 2 ? gotQ.view : gotKV.view,
                rhsF16: rejection == 3 ? x.view : rhs.view, inDim: rejection == 1 ? 2048 : inDim,
                qOutDim: qDim, kvOutDim: kvDim, nTok: nTok, enabled: rejection != 0)
            try require(!used, "\(label): rejection \(rejection) unexpectedly dispatched")
            try finish(c)
            try gotQ.verifyUnchanged(); try gotKV.verifyUnchanged(); try rhs.verifyUnchanged(); try x.verifyUnchanged()
        }
        if !eligible {
            let c = GraphContext(runtime); try c.begin()
            let used = try c.encodeQ4PrefillPair(weightQA: qWeight.view, weightKV: kvWeight.view,
                act: x.view, qOut: gotQ.view, kvOut: gotKV.view, rhsF16: rhs.view,
                inDim: inDim, qOutDim: qDim, kvOutDim: kvDim, nTok: nTok, enabled: true)
            try require(!used, "\(label): unsupported path accepted")
            try finish(c); try gotQ.verifyUnchanged(); try gotKV.verifyUnchanged(); try rhs.verifyUnchanged()
        }
        let c = GraphContext(runtime); try c.begin(); try reference(c)
        for _ in 0..<3 { try candidate(c) }
        try finish(c)
        if eligible {
            try verifyHalfStaging(rhs.view, activation: activation)
            try near(gotQ.view.floatArray(), refQ.view.floatArray(), label + "/Q")
            try near(gotKV.view.floatArray(), refKV.view.floatArray(), label + "/KV")
            try sampledPrefillOracle(weight: qBytes, activation: activation, output: gotQ.view,
                                     inDim: inDim, outDim: qDim, nTok: nTok, label: label + "/Q")
            try sampledPrefillOracle(weight: kvBytes, activation: activation, output: gotKV.view,
                                     inDim: inDim, outDim: kvDim, nTok: nTok, label: label + "/KV")
        } else {
            try exact(gotQ.view, refQ.view, label + "/fallbackQ")
            try exact(gotKV.view, refKV.view, label + "/fallbackKV")
        }
        let first = GraphContext(runtime); try first.begin(); try candidate(first); first.commitAsync()
        let second = GraphContext(runtime); try second.begin(); try candidate(second); try finish(second)
        first.waitCompleted()
        if eligible {
            try near(gotQ.view.floatArray(), refQ.view.floatArray(), label + "/queueQ")
            try near(gotKV.view.floatArray(), refKV.view.floatArray(), label + "/queueKV")
            try benchmark(runtime, label: label, options: options, reference: reference, candidate: candidate)
        }
        for input in [qWeight, kvWeight, x] { try input.verifyUnchanged() }
        for output in [rhs, refQ, refKV, gotQ, gotKV] { try output.verifyGuards() }
        print("PASS \(label): \(eligible ? "fused" : "fallback (device/shape ineligible)"), offsets, guards, immutable inputs, scratch reuse, no-op rejection")
    }
    try checkPrefillSubnormal(runtime)
}

private func checkPrefillSubnormal(_ runtime: MetalRuntime) throws {
    let inDim = 4096, qDim = 1024, kvDim = 512, nTok = 32
    guard GraphContext.q4PrefillPairEligible(deviceName: runtime.device.name,
        inDim: inDim, qOutDim: qDim, kvOutDim: kvDim, nTok: nTok) else {
        print("SKIP prefillQ4/subnormal: device is ineligible for paired kernel")
        return
    }
    // d is the smallest positive F16. The old GEMM's high-nibble half(d / 16)
    // underflow must be preserved, as well as its nonzero low-nibble outputs.
    var block = [UInt8](repeating: 0, count: 144)
    block[0] = 1
    for i in 4..<16 { block[i] = 1 }
    for i in 16..<144 { block[i] = 0xFF }
    let qBytes = Array(repeating: block, count: inDim / 256 * qDim).flatMap { $0 }
    let kvBytes = Array(repeating: block, count: inDim / 256 * kvDim).flatMap { $0 }
    let activation = [Float](repeating: 0.25, count: inDim * nTok)
    let qWeight = try GuardedTensor(runtime, name: "tiny/qWeight", bytes: qBytes, count: inDim * qDim)
    let kvWeight = try GuardedTensor(runtime, name: "tiny/kvWeight", bytes: kvBytes, count: inDim * kvDim)
    let x = try GuardedTensor(runtime, name: "tiny/x", floats: activation)
    let q = try GuardedTensor(runtime, name: "tiny/q", outputFloats: nTok * qDim)
    let kv = try GuardedTensor(runtime, name: "tiny/kv", outputFloats: nTok * kvDim)
    let refQ = try GuardedTensor(runtime, name: "tiny/refQ", outputFloats: nTok * qDim)
    let refKV = try GuardedTensor(runtime, name: "tiny/refKV", outputFloats: nTok * kvDim)
    let rhs = try GuardedTensor(runtime, name: "tiny/rhs",
                               bytes: [UInt8](repeating: 0xA7, count: nTok * inDim * 2), count: nTok * inDim)
    let c = GraphContext(runtime); try c.begin()
    let used = try c.encodeQ4PrefillPair(weightQA: qWeight.view, weightKV: kvWeight.view,
        act: x.view, qOut: q.view, kvOut: kv.view, rhsF16: rhs.view,
        inDim: inDim, qOutDim: qDim, kvOutDim: kvDim, nTok: nTok, enabled: true)
    try require(used, "subnormal paired kernel rejected")
    try c.encodeMMDenseQ4K(weight: qWeight.view, act: x.view, actBase: 0, out: refQ.view,
                           inDim: inDim, outDim: qDim, nTok: nTok)
    try c.encodeMMDenseQ4K(weight: kvWeight.view, act: x.view, actBase: 0, out: refKV.view,
                           inDim: inDim, outDim: kvDim, nTok: nTok)
    try finish(c)
    try exact(q.view, refQ.view, "subnormal/Q-baseline")
    try exact(kv.view, refKV.view, "subnormal/KV-baseline")
    try verifyHalfStaging(rhs.view, activation: activation)
    try sampledPrefillOracle(weight: qBytes, activation: activation, output: q.view,
        inDim: inDim, outDim: qDim, nTok: nTok, label: "subnormal/Q", absolute: 0.00000001)
    try sampledPrefillOracle(weight: kvBytes, activation: activation, output: kv.view,
        inDim: inDim, outDim: kvDim, nTok: nTok, label: "subnormal/KV", absolute: 0.00000001)
    for input in [qWeight, kvWeight, x] { try input.verifyUnchanged() }
    for output in [rhs, q, kv, refQ, refKV] { try output.verifyGuards() }
    print("PASS prefillQ4/subnormal: independent CPU F16 oracle, RHS bits, guards, immutable inputs")
}

@main
private struct AttentionPortChecks {
    static func main() {
        do {
            var options = BenchmarkOptions(), suite = "all", cpuOnly = false
            var arguments = Array(CommandLine.arguments.dropFirst())
            while !arguments.isEmpty {
                let flag = arguments.removeFirst()
                switch flag {
                case "--cpu-only": cpuOnly = true
                case "--benchmark": options.enabled = true
                case "--suite", "--pairs", "--repeats":
                    guard !arguments.isEmpty else { throw CheckFailure("missing value for \(flag)") }
                    let value = arguments.removeFirst()
                    if flag == "--suite" { suite = value }
                    else {
                        guard let number = Int(value), (1...100).contains(number) else { throw CheckFailure("\(flag) must be 1...100") }
                        if flag == "--pairs" { options.pairs = number } else { options.repeats = number }
                    }
                default: throw CheckFailure("unknown option \(flag)")
                }
            }
            try require(["all", "decode", "prefill"].contains(suite), "suite must be all/decode/prefill")
            try checkEligibility()
            if cpuOnly { return }
            let runtime = try MetalRuntime()
            print("Attention port checks; device=\(runtime.device.name); suite=\(suite); q8_nsg=\(ProcessInfo.processInfo.environment["DS4_Q8_NSG"] ?? "4"); q4_nsg=\(ProcessInfo.processInfo.environment["DS4_DENSE_Q4_NSG"] ?? "4")")
            print("Synthetic quantized weights, no GGUF/model inference. Timings measure resident batch wall time including encode/submit/wait; NOT GPU-only or end-to-end throughput.")
            if suite != "prefill" { try checkDecode(runtime, options: options) }
            if suite != "decode" { try checkPrefill(runtime, options: options) }
            print("ALL SELECTED CHECKS PASSED")
        } catch {
            FileHandle.standardError.write(Data("FAIL: \(error)\n".utf8))
            exit(1)
        }
    }
}
