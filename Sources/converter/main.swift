import Foundation
import DeepSeekKit

// Native Swift port of `Reference/inference/convert.py`, extended to
// transcode every non-native dtype into a type Apple Silicon supports
// natively (BF16 by default).
//
// Apple Silicon GPUs handle these natively in MSL:
//   F32, F16, BF16 (Metal 3+ / macOS 14+), and integer types.
// They do NOT have native types or arithmetic for:
//   FP8-E4M3, FP4-E2M1, E8M0 (MX scale).
// Inference paths on those would need per-element dequant in shader.
// Cheaper to pay the cost once at convert time and ship native-dtype
// weights — at the price of bigger files (FP8 → BF16 doubles, FP4 →
// BF16 quadruples).
//
// Usage:
//   swift run -c release converter \
//       --hf-ckpt-path /path/V4-Flash-HF \
//       --save-path /path/V4-Flash-converted \
//       --n-experts <N> \
//       [--model-parallel 1] \
//       [--target-dtype bf16|f16|keep]   # default: bf16
//       [--shard-size-gb 5]              # default: 5 GB per shard

func usage() -> Never {
    FileHandle.standardError.write(Data("""
    usage: converter --hf-ckpt-path <dir> --save-path <dir> --n-experts <N>
                     [--model-parallel 1] [--target-dtype bf16|f16|keep]
                     [--shard-size-gb 5]

    """.utf8))
    exit(2)
}

enum TargetDType: String {
    case bf16, f16, keep
    var safetensorsTag: String {
        switch self {
        case .bf16: return "BF16"
        case .f16:  return "F16"
        case .keep: return ""
        }
    }
    var bytesPerElement: Int { 2 }   // both bf16 and f16 are 2 bytes
}

// ---------- Argument parsing ----------

var hfPath: String?
var savePath: String?
var nExperts: Int = 0
var mp: Int = 1
var targetDType: TargetDType = .bf16
var shardSizeGB: Double = 5.0

var args = Array(CommandLine.arguments.dropFirst())
while !args.isEmpty {
    let a = args.removeFirst()
    switch a {
    case "--hf-ckpt-path":
        guard !args.isEmpty else { usage() }
        hfPath = args.removeFirst()
    case "--save-path":
        guard !args.isEmpty else { usage() }
        savePath = args.removeFirst()
    case "--n-experts":
        guard !args.isEmpty, let n = Int(args.removeFirst()) else { usage() }
        nExperts = n
    case "--model-parallel":
        guard !args.isEmpty, let n = Int(args.removeFirst()) else { usage() }
        mp = n
    case "--target-dtype":
        guard !args.isEmpty, let t = TargetDType(rawValue: args.removeFirst()) else { usage() }
        targetDType = t
    case "--shard-size-gb":
        guard !args.isEmpty, let v = Double(args.removeFirst()) else { usage() }
        shardSizeGB = v
    default:
        usage()
    }
}

guard let hf = hfPath, let save = savePath, nExperts > 0 else { usage() }
precondition(nExperts % mp == 0, "n_experts must be divisible by model_parallel")
precondition(mp == 1, "model_parallel > 1 not supported by this Swift port (single-rank)")
let shardSizeBytes = Int(shardSizeGB * 1_000_000_000)

let hfDir = URL(fileURLWithPath: hf)
let saveDir = URL(fileURLWithPath: save)
try FileManager.default.createDirectory(at: saveDir, withIntermediateDirectories: true)

// ---------- Rename mapping (port of convert.py:55-79) ----------

let leafMapping: [String: String] = [
    "embed_tokens": "embed",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
    "q_proj": "wq",
    "q_a_proj": "wq_a",
    "q_a_layernorm": "q_norm",
    "q_b_proj": "wq_b",
    "kv_a_proj_with_mqa": "wkv_a",
    "kv_a_layernorm": "kv_norm",
    "kv_b_proj": "wkv_b",
    "o_proj": "wo",
    "gate_proj": "w1",
    "down_proj": "w2",
    "up_proj": "w3",
    "lm_head": "head",
    // Identity / passthrough names that already match.
    "embed": "embed",
    "wq_b": "wq_b",
    "wo_a": "wo_a",
    "wo_b": "wo_b",
    "head": "head",
    "attn_sink": "attn_sink",
    "weights_proj": "weights_proj",
]

func renameKey(_ name: String) -> String {
    var n = name
    if n.hasPrefix("model.") { n = String(n.dropFirst("model.".count)) }
    n = n.replacingOccurrences(of: "self_attn", with: "attn")
    n = n.replacingOccurrences(of: "mlp", with: "ffn")
    n = n.replacingOccurrences(of: "weight_scale_inv", with: "scale")
    n = n.replacingOccurrences(of: "e_score_correction_bias", with: "bias")

    let parts = n.split(separator: ".").map(String.init)
    let leaf: String
    if parts.contains(where: { ["hc", "attn_sink", "tie2eid", "ape"].contains($0) ||
                                $0.hasPrefix("hc_") }) {
        leaf = parts.last ?? n
    } else {
        leaf = parts.count >= 2 ? parts[parts.count - 2] : (parts.last ?? n)
    }
    if let mapped = leafMapping[leaf] {
        n = n.replacingOccurrences(of: leaf, with: mapped)
    }
    return n
}

func shouldSkip(_ name: String) -> Bool {
    // convert.py:105 — skip MTP-tied embed/head; they alias the main ones.
    if name.hasPrefix("mtp.") {
        if name.contains("emb") { return true }
        if name.hasSuffix("head.weight") { return true }
    }
    return false
}

// ---------- Native-dtype conversion helpers ----------

@inline(__always)
func floatToBF16(_ f: Float) -> UInt16 {
    let bits = f.bitPattern
    let rounded = bits &+ ((bits >> 16) & 1) &+ 0x7FFF
    return UInt16(truncatingIfNeeded: rounded >> 16)
}

@inline(__always)
func floatToF16(_ f: Float) -> UInt16 {
    // IEEE 754 single → half with round-to-nearest-even, saturating to
    // ±inf on overflow. Subnormals are flushed via the standard formula.
    let bits = f.bitPattern
    let sign = (bits >> 31) & 1
    let exp = (bits >> 23) & 0xFF
    let mant = bits & 0x7FFFFF
    if exp == 0 { return UInt16(truncatingIfNeeded: sign << 15) }
    if exp == 0xFF {
        let m: UInt32 = mant != 0 ? 0x200 : 0   // NaN if mantissa, else inf
        return UInt16(truncatingIfNeeded: (sign << 15) | (0x1F << 10) | m)
    }
    let unbiased = Int(exp) - 127
    if unbiased > 15 { return UInt16(truncatingIfNeeded: (sign << 15) | (0x1F << 10)) }
    if unbiased < -14 {
        // Subnormal half.
        let shift = -14 - unbiased + 13
        if shift > 24 { return UInt16(truncatingIfNeeded: sign << 15) }
        let full = (mant | 0x800000) >> (shift - 1)
        let halfMant = (full + 1) >> 1
        return UInt16(truncatingIfNeeded: (sign << 15) | halfMant)
    }
    let halfExp = UInt32(unbiased + 15)
    let halfMant = (mant + 0x1000) >> 13
    if halfMant >= 0x400 {
        if halfExp + 1 >= 0x1F {
            return UInt16(truncatingIfNeeded: (sign << 15) | (0x1F << 10))
        }
        return UInt16(truncatingIfNeeded: (sign << 15) | ((halfExp + 1) << 10))
    }
    return UInt16(truncatingIfNeeded: (sign << 15) | (halfExp << 10) | halfMant)
}

@inline(__always)
func packNative(_ f: Float, _ target: TargetDType) -> UInt16 {
    switch target {
    case .bf16: return floatToBF16(f)
    case .f16:  return floatToF16(f)
    case .keep: fatalError("packNative requires bf16 or f16")
    }
}

@inline(__always)
func deqE4M3(_ b: UInt8) -> Float {
    let sign = (UInt32(b) >> 7) & 1
    let exp = (UInt32(b) >> 3) & 0xF
    let mant = UInt32(b) & 0x7
    if exp == 0 && mant == 0 { return sign == 1 ? -0.0 : 0.0 }
    if exp == 0xF && mant == 0x7 { return .nan }
    if exp == 0 {
        let v = Float(mant) * 0x1p-9
        return sign == 1 ? -v : v
    }
    let bp = (sign << 31) | ((exp + 120) << 23) | (mant << 20)
    return Float(bitPattern: bp)
}

@inline(__always)
func deqE2M1(_ nibble: UInt8) -> Float {
    let mag = nibble & 7
    let v: Float
    switch mag {
    case 0: v = 0.0
    case 1: v = 0.5
    case 2: v = 1.0
    case 3: v = 1.5
    case 4: v = 2.0
    case 5: v = 3.0
    case 6: v = 4.0
    default: v = 6.0
    }
    return (nibble & 8) != 0 ? -v : v
}

@inline(__always)
func deqE8M0(_ b: UInt8) -> Float {
    if b == 0xFF { return .nan }
    return Float(bitPattern: UInt32(b) << 23)
}

// ---------- FP8 → native fusion ----------
//
// Weight: [out, in] FP8-E4M3. Scale: [out/128, in/128] E8M0.
//   fused[i, j] = deqE4M3(W[i,j]) * deqE8M0(S[i/128, j/128]).
// Output: [out, in] in `target` (BF16 or F16). Streamed row-by-row so
// memory stays bounded by inDim regardless of total tensor size.
func fuseFP8ToNative(weightURL: URL, weightOffset: Int,
                     scaleURL: URL, scaleOffset: Int,
                     outDim: Int, inDim: Int,
                     target: TargetDType) throws -> Data {
    precondition(outDim % 128 == 0 && inDim % 128 == 0,
                 "FP8 weight dims must be 128-aligned")
    let blocksOut = outDim / 128
    let blocksIn = inDim / 128

    guard let sf = FileHandle(forReadingAtPath: scaleURL.path) else {
        throw NSError(domain: "fuseFP8", code: 1)
    }
    defer { try? sf.close() }
    try sf.seek(toOffset: UInt64(scaleOffset))
    guard let scaleBytes = try sf.read(upToCount: blocksOut * blocksIn) else {
        throw NSError(domain: "fuseFP8", code: 2)
    }

    guard let wf = FileHandle(forReadingAtPath: weightURL.path) else {
        throw NSError(domain: "fuseFP8", code: 3)
    }
    defer { try? wf.close() }
    try wf.seek(toOffset: UInt64(weightOffset))

    var out = Data(count: outDim * inDim * 2)
    out.withUnsafeMutableBytes { outRaw in
        let outPtr = outRaw.bindMemory(to: UInt16.self).baseAddress!
        scaleBytes.withUnsafeBytes { scaleRaw in
            let scalePtr = scaleRaw.bindMemory(to: UInt8.self).baseAddress!
            for i in 0..<outDim {
                guard let row = try? wf.read(upToCount: inDim), row.count == inDim else {
                    fatalError("short read on FP8 weight row \(i)")
                }
                let bo = i / 128
                row.withUnsafeBytes { rowRaw in
                    let rowPtr = rowRaw.bindMemory(to: UInt8.self).baseAddress!
                    for j in 0..<inDim {
                        let s = deqE8M0(scalePtr[bo * blocksIn + (j / 128)])
                        let w = deqE4M3(rowPtr[j])
                        outPtr[i * inDim + j] = packNative(w * s, target)
                    }
                }
            }
        }
    }
    return out
}

// ---------- FP4 → native fusion ----------
//
// Weight: [out, in/2] packed FP4-E2M1 (low nibble = element 2*i,
// high nibble = element 2*i+1). Scale: [out, in/32] E8M0.
//   fused[i, j] = deqE2M1(W_nibble(i, j)) * deqE8M0(S[i, j/32]).
// Output: [out, in] in `target` — 4× the input weight bytes.
func fuseFP4ToNative(weightURL: URL, weightOffset: Int,
                     scaleURL: URL, scaleOffset: Int,
                     outDim: Int, inDim: Int,
                     target: TargetDType) throws -> Data {
    precondition(inDim % 32 == 0, "FP4 weight inDim must be 32-aligned")
    precondition(inDim % 2 == 0, "FP4 weight inDim must be even (packed pairs)")
    let blocksIn = inDim / 32

    guard let sf = FileHandle(forReadingAtPath: scaleURL.path) else {
        throw NSError(domain: "fuseFP4", code: 1)
    }
    defer { try? sf.close() }
    try sf.seek(toOffset: UInt64(scaleOffset))
    guard let scaleBytes = try sf.read(upToCount: outDim * blocksIn) else {
        throw NSError(domain: "fuseFP4", code: 2)
    }

    guard let wf = FileHandle(forReadingAtPath: weightURL.path) else {
        throw NSError(domain: "fuseFP4", code: 3)
    }
    defer { try? wf.close() }
    try wf.seek(toOffset: UInt64(weightOffset))

    let packedRowBytes = inDim / 2
    var out = Data(count: outDim * inDim * 2)
    out.withUnsafeMutableBytes { outRaw in
        let outPtr = outRaw.bindMemory(to: UInt16.self).baseAddress!
        scaleBytes.withUnsafeBytes { scaleRaw in
            let scalePtr = scaleRaw.bindMemory(to: UInt8.self).baseAddress!
            for i in 0..<outDim {
                guard let row = try? wf.read(upToCount: packedRowBytes),
                      row.count == packedRowBytes else {
                    fatalError("short read on FP4 weight row \(i)")
                }
                row.withUnsafeBytes { rowRaw in
                    let rowPtr = rowRaw.bindMemory(to: UInt8.self).baseAddress!
                    for halfIdx in 0..<packedRowBytes {
                        let byte = rowPtr[halfIdx]
                        let jLow = 2 * halfIdx
                        let jHigh = jLow + 1
                        let sLow = deqE8M0(scalePtr[i * blocksIn + (jLow / 32)])
                        let sHigh = deqE8M0(scalePtr[i * blocksIn + (jHigh / 32)])
                        let vLow = deqE2M1(byte & 0xF) * sLow
                        let vHigh = deqE2M1(byte >> 4) * sHigh
                        outPtr[i * inDim + jLow]  = packNative(vLow, target)
                        outPtr[i * inDim + jHigh] = packNative(vHigh, target)
                    }
                }
            }
        }
    }
    return out
}

@inline(__always)
func isFP8DType(_ s: String) -> Bool {
    let u = s.uppercased()
    return u == "F8_E4M3" || u == "F8E4M3" || u == "FLOAT8_E4M3FN"
}

@inline(__always)
func isFP4DType(_ s: String) -> Bool {
    let u = s.uppercased()
    return u == "F4_E2M1" || u == "F4E2M1" || u == "FLOAT4_E2M1FN_X2"
}

@inline(__always)
func isE8M0DType(_ s: String) -> Bool {
    let u = s.uppercased()
    return u == "F8_E8M0" || u == "F8E8M0" || u == "FLOAT8_E8M0FNU"
}

// ---------- Walk + collect ----------

let fm = FileManager.default
let inputs = try fm.contentsOfDirectory(at: hfDir, includingPropertiesForKeys: nil)
    .filter { $0.pathExtension == "safetensors" }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }

guard !inputs.isEmpty else {
    FileHandle.standardError.write(Data("no .safetensors files in \(hf)\n".utf8))
    exit(1)
}

print("Converter: \(inputs.count) input shard(s)")
print("  target dtype:    \(targetDType.rawValue) " +
      (targetDType == .keep ? "(non-native FP8/FP4 preserved)"
                              : "(FP8/FP4+scale fused into native)"))
print("  shard size cap:  \(String(format: "%.1f", shardSizeGB)) GB")
print("  sharding:        layer-aligned (top-level + one bucket per layer)")
if targetDType != .keep {
    print("  note:            FP8 → 2× size, FP4 → 4× size; expect ~3-4× the " +
          "input directory's footprint on disk.")
}
print("Indexing input …")

// Mapping: new_name → (sourceURL, byte_offset, byte_count, dtype, shape)
struct PendingTensor {
    let url: URL
    let offset: Int
    let byteCount: Int
    let dtype: String
    let shape: [Int]
}

var plan: [String: PendingTensor] = [:]
var woAScales: [String: PendingTensor] = [:]   // keyed by the fused weight's new name

for inputURL in inputs {
    let stf = try SafeTensorsFile(url: inputURL)
    for (origName, entry) in stf.entries {
        if shouldSkip(origName) { continue }
        let newName = renameKey(origName)
        // Compute absolute byte range in the input file.
        // SafeTensorsFile keeps the data section start internally; we need
        // to recompute it here by re-reading the header length.
        let absOffset = try absoluteOffset(of: entry, in: inputURL)
        let pt = PendingTensor(url: inputURL, offset: absOffset,
                                byteCount: entry.dataOffsets[1] - entry.dataOffsets[0],
                                dtype: entry.dtype, shape: entry.shape)
        if newName.hasSuffix(".wo_a.scale") {
            // Buffer the scale; we'll fuse with the matching .weight below.
            woAScales[newName.replacingOccurrences(of: ".scale", with: ".weight")] = pt
        } else {
            plan[newName] = pt
        }
    }
}

print("Collected \(plan.count) tensors plus \(woAScales.count) wo_a scales.")

// ---------- Build per-tensor write entries ----------
//
// For each tensor in the input:
//   - If it's a non-native dtype (FP8 or FP4) AND has a companion .scale,
//     fuse them into a single tensor of `targetDType` (BF16 or F16).
//   - If it's an E8M0 scale tensor, drop it: it's been consumed by its
//     parent weight's fusion above.
//   - Otherwise pass through unchanged (BF16/F16/F32/I32/I8 are all
//     native or trivially native-compatible).
//
// When `targetDType == .keep`, fall back to the original convert.py
// behaviour: only fuse wo_a (which the reference inference path expects
// in BF16) and relabel I8 expert weights as F4_E2M1.

struct WriteEntry {
    let name: String
    let dtype: String
    let shape: [Int]
    let byteCount: Int
    let source: SafeTensorsWriter.Source
}

func scaleNameFor(_ weightName: String) -> String {
    return weightName.replacingOccurrences(of: ".weight", with: ".scale")
}

var writeEntries: [WriteEntry] = []
var scalesConsumed = Set<String>()    // .scale entries that have been folded in

for newName in plan.keys.sorted() {
    let pt = plan[newName]!

    // Skip standalone scale tensors when the parent weight is going to
    // consume them (or already has).
    if isE8M0DType(pt.dtype) {
        if scalesConsumed.contains(newName) { continue }
        // If the parent weight isn't in the plan (rare), pass the scale
        // through as-is so nothing is silently dropped.
        let parent = newName.replacingOccurrences(of: ".scale", with: ".weight")
        if plan[parent] != nil { continue }
    }

    // Native-dtype transcoding: FP8/FP4 weight + scale → BF16/F16.
    if targetDType != .keep {
        let scaleName = scaleNameFor(newName)
        if isFP8DType(pt.dtype), let sc = plan[scaleName] {
            let outDim = pt.shape[0]
            let inDim = pt.shape[1]
            let n = outDim * inDim * targetDType.bytesPerElement
            scalesConsumed.insert(scaleName)
            let target = targetDType
            writeEntries.append(WriteEntry(
                name: newName, dtype: target.safetensorsTag, shape: pt.shape, byteCount: n,
                source: .compute(byteCount: n) {
                    try fuseFP8ToNative(weightURL: pt.url, weightOffset: pt.offset,
                                         scaleURL: sc.url, scaleOffset: sc.offset,
                                         outDim: outDim, inDim: inDim, target: target)
                }))
            continue
        }
        if isFP4DType(pt.dtype) || (newName.contains(".experts.") &&
                                     (pt.dtype.uppercased() == "I8" ||
                                      pt.dtype.uppercased() == "U8")),
           let sc = plan[scaleName] {
            // FP4 weights ship as packed [out, in/2]. The logical inDim is
            // double the stored last-dim.
            let outDim = pt.shape[0]
            let inDim = pt.shape[1] * 2
            let n = outDim * inDim * targetDType.bytesPerElement
            scalesConsumed.insert(scaleName)
            let target = targetDType
            let logicalShape = [outDim, inDim]
            writeEntries.append(WriteEntry(
                name: newName, dtype: target.safetensorsTag, shape: logicalShape, byteCount: n,
                source: .compute(byteCount: n) {
                    try fuseFP4ToNative(weightURL: pt.url, weightOffset: pt.offset,
                                         scaleURL: sc.url, scaleOffset: sc.offset,
                                         outDim: outDim, inDim: inDim, target: target)
                }))
            continue
        }
    } else {
        // Legacy convert.py-style behaviour: only fuse wo_a, relabel experts.
        if newName.hasSuffix(".wo_a.weight"),
           let sc = plan[scaleNameFor(newName)] {
            let outDim = pt.shape[0]
            let inDim = pt.shape[1]
            let n = outDim * inDim * 2
            scalesConsumed.insert(scaleNameFor(newName))
            writeEntries.append(WriteEntry(
                name: newName, dtype: "BF16", shape: pt.shape, byteCount: n,
                source: .compute(byteCount: n) {
                    try fuseFP8ToNative(weightURL: pt.url, weightOffset: pt.offset,
                                         scaleURL: sc.url, scaleOffset: sc.offset,
                                         outDim: outDim, inDim: inDim, target: .bf16)
                }))
            continue
        }
        var dtype = pt.dtype
        if newName.contains(".experts.") && newName.hasSuffix(".weight") {
            if pt.dtype.uppercased() == "I8" || pt.dtype.uppercased() == "U8" {
                dtype = "F4_E2M1"
            }
        }
        writeEntries.append(WriteEntry(
            name: newName, dtype: dtype, shape: pt.shape, byteCount: pt.byteCount,
            source: .file(url: pt.url, offset: pt.offset, byteCount: pt.byteCount)))
        continue
    }

    // Default pass-through (native dtype already, or unknown name).
    writeEntries.append(WriteEntry(
        name: newName, dtype: pt.dtype, shape: pt.shape, byteCount: pt.byteCount,
        source: .file(url: pt.url, offset: pt.offset, byteCount: pt.byteCount)))
}

// ---------- Group tensors by depth, then pack ----------
//
// Strategy: each transformer layer goes into its own shard whenever
// possible. If a layer is bigger than the shard cap, split it across
// consecutive shards (the next layer always starts a fresh shard).
// Top-level tensors (embed, head, norm, hc_head_*) go into shard 0.
//
// Why: the forward pass touches layers in strict depth order. Aligning
// shards to layer boundaries means that during one forward pass we read
// shard 0 once, then 1, then 2, ..., never going backwards. The OS page
// cache can prefetch the next shard while we're still computing on the
// current one, and evict already-consumed shards under memory pressure
// without disturbing the layer being actively read.

func depthKey(of name: String) -> Int {
    // Returns -1 for top-level tensors, otherwise the layer index.
    if name.hasPrefix("layers.") {
        let parts = name.split(separator: ".")
        if parts.count >= 2, let i = Int(parts[1]) { return i }
    }
    if name.hasPrefix("mtp.") {
        let parts = name.split(separator: ".")
        if parts.count >= 2, let i = Int(parts[1]) {
            return 100_000 + i   // sort MTP after main layers
        }
    }
    return -1
}

// Bucket entries by depth.
var bucketsByDepth: [Int: [WriteEntry]] = [:]
for e in writeEntries { bucketsByDepth[depthKey(of: e.name), default: []].append(e) }

// Sanity log: report what depth structure we discovered just from the
// tensor names. Useful to cross-check against config.json's `n_layers`
// and `n_mtp_layers` if you have them.
let layerDepths = bucketsByDepth.keys.filter { $0 >= 0 && $0 < 100_000 }.sorted()
let mtpDepths = bucketsByDepth.keys.filter { $0 >= 100_000 }.sorted().map { $0 - 100_000 }
let topLevelTensors = bucketsByDepth[-1]?.count ?? 0
print("Discovered structure (from tensor names, no config.json read):")
print("  top-level tensors:  \(topLevelTensors)")
if let lo = layerDepths.first, let hi = layerDepths.last {
    print("  layers:             \(layerDepths.count) (indices \(lo)…\(hi))")
}
if !mtpDepths.isEmpty {
    print("  mtp blocks:         \(mtpDepths.count)")
}

struct Shard {
    var entries: [WriteEntry]
    var totalBytes: Int
}

var shards: [Shard] = []
let depths = bucketsByDepth.keys.sorted()
for depth in depths {
    let bucket = bucketsByDepth[depth]!
    // Within a depth bucket, sort tensors by name for determinism.
    let sorted = bucket.sorted { $0.name < $1.name }
    var current = Shard(entries: [], totalBytes: 0)
    for e in sorted {
        if !current.entries.isEmpty && current.totalBytes + e.byteCount > shardSizeBytes {
            shards.append(current)
            current = Shard(entries: [], totalBytes: 0)
        }
        current.entries.append(e)
        current.totalBytes += e.byteCount
    }
    if !current.entries.isEmpty { shards.append(current) }
}

print("Packing \(writeEntries.count) tensors into \(shards.count) shard(s) " +
      "(layer-aligned; max \(String(format: "%.1f", shardSizeGB)) GB/shard)")

// ---------- Write each shard + index.json ----------

let total = shards.count
var weightMap: [String: String] = [:]      // tensor name → shard filename
let totalBytes = shards.reduce(0) { $0 + $1.totalBytes }

// Resume detection: if the output dir already contains shards from a
// previous (interrupted) run, skip the ones that are present and the
// right size. Sharding is deterministic given the same input + flags,
// so file `model-(i+1)-of-total.safetensors` produced now must match
// any file with the same name from a prior run.
//
// A shard is considered "complete" iff:
//   - its filename matches model-(i+1)-of-(total).safetensors exactly
//     (so a partial write under a different `total` is skipped)
//   - its size on disk is at least `shard.totalBytes` (the header adds a
//     few KB of JSON, so >= totalBytes implies full write)
//
// We also pre-populate `weightMap` for the skipped shards so the final
// index.json comes out correct.

func expectedFilename(i: Int, total: Int) -> String {
    return String(format: "model-%05d-of-%05d.safetensors", i + 1, total)
}

var resumeFromShard = 0
for (i, shard) in shards.enumerated() {
    let fileName = expectedFilename(i: i, total: total)
    let url = saveDir.appendingPathComponent(fileName)
    guard let attrs = try? fm.attributesOfItem(atPath: url.path),
          let size = attrs[.size] as? Int,
          size >= shard.totalBytes else { break }
    // Mark as already-written, populate the index.
    for e in shard.entries { weightMap[e.name] = fileName }
    resumeFromShard = i + 1
}
if resumeFromShard > 0 {
    let skippedBytes = shards[..<resumeFromShard].reduce(0) { $0 + $1.totalBytes }
    print("Resume: \(resumeFromShard)/\(total) shard(s) already on disk " +
          "(\(String(format: "%.1f", Double(skippedBytes) / 1_000_000_000)) GB skipped).")
}

for (i, shard) in shards.enumerated() {
    let fileName = expectedFilename(i: i, total: total)
    if i < resumeFromShard { continue }
    let outURL = saveDir.appendingPathComponent(fileName)
    let w = SafeTensorsWriter()
    for e in shard.entries {
        w.add(name: e.name, dtype: e.dtype, shape: e.shape, source: e.source)
        weightMap[e.name] = fileName
    }
    print("  [\(i + 1)/\(total)] \(fileName) — " +
          "\(shard.entries.count) tensors, " +
          "\(String(format: "%.2f", Double(shard.totalBytes) / 1_000_000_000)) GB")
    try w.write(to: outURL)
}

// HuggingFace-style index.json, useful for tooling and reproducibility.
let indexObj: [String: Any] = [
    "metadata": ["total_size": totalBytes] as [String: Any],
    "weight_map": weightMap,
]
let indexData = try JSONSerialization.data(withJSONObject: indexObj,
                                            options: [.sortedKeys, .prettyPrinted])
let indexURL = saveDir.appendingPathComponent("model.safetensors.index.json")
try indexData.write(to: indexURL)
print("Wrote \(weightMap.count) tensors across \(total) shard(s); " +
      "\(String(format: "%.1f", Double(totalBytes) / 1_000_000_000)) GB total.")

// ---------- Copy tokenizer ----------

for f in ["tokenizer.json", "tokenizer_config.json"] {
    let src = hfDir.appendingPathComponent(f)
    if fm.fileExists(atPath: src.path) {
        let dst = saveDir.appendingPathComponent(f)
        if fm.fileExists(atPath: dst.path) { try? fm.removeItem(at: dst) }
        try fm.copyItem(at: src, to: dst)
    }
}

print("Done.")
print("Run:  swift run -c release deepseek \(saveDir.path) \"<prompt>\" --mode chat")

// ---------- Helpers ----------

/// Re-reads the header of `url` to compute the absolute byte position of
/// `entry.dataOffsets[0]` within the file. SafeTensorsFile already
/// computed this internally but doesn't expose it; this function repeats
/// the small header read to get it deterministically.
func absoluteOffset(of entry: SafeTensorsFile.Entry, in url: URL) throws -> Int {
    guard let fh = FileHandle(forReadingAtPath: url.path) else {
        throw NSError(domain: "absoluteOffset", code: 1)
    }
    defer { try? fh.close() }
    guard let lenData = try fh.read(upToCount: 8), lenData.count == 8 else {
        throw NSError(domain: "absoluteOffset", code: 2)
    }
    let headerLen = lenData.withUnsafeBytes { $0.load(as: UInt64.self) }
    return 8 + Int(headerLen) + entry.dataOffsets[0]
}
