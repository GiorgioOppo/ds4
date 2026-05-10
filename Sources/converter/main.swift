import Foundation
import DeepSeekKit

// 1:1 Swift port of `Reference/inference/convert.py`.
//
// Walks every `.safetensors` file in `--hf-ckpt-path`, applies the V4
// rename mapping (self_attn → attn, mlp → ffn, weight_scale_inv → scale,
// e_score_correction_bias → bias, plus the leaf-key remapping table),
// fuses `wo_a` weight + scale into a single BF16 tensor, optionally
// re-encodes FP4 expert weights as FP8 (deferred), and writes one or
// more output safetensors shards to `--save-path`.
//
// Usage:
//   swift run -c release converter \
//       --hf-ckpt-path /path/V4-Flash-HF \
//       --save-path /path/V4-Flash-converted \
//       --n-experts <N> \
//       [--model-parallel 1] \
//       [--expert-dtype fp4|fp8]   # default: relabel only (fp4 view)

func usage() -> Never {
    FileHandle.standardError.write(Data("""
    usage: converter --hf-ckpt-path <dir> --save-path <dir> --n-experts <N>
                     [--model-parallel 1] [--expert-dtype fp4|fp8]

    """.utf8))
    exit(2)
}

// ---------- Argument parsing ----------

var hfPath: String?
var savePath: String?
var nExperts: Int = 0
var mp: Int = 1
var expertDtype: String? = nil

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
    case "--expert-dtype":
        guard !args.isEmpty else { usage() }
        let s = args.removeFirst()
        guard ["fp4", "fp8"].contains(s) else { usage() }
        expertDtype = s
    default:
        usage()
    }
}

guard let hf = hfPath, let save = savePath, nExperts > 0 else { usage() }
precondition(nExperts % mp == 0, "n_experts must be divisible by model_parallel")
precondition(mp == 1, "model_parallel > 1 not supported by this Swift port (single-rank)")

if expertDtype == "fp8" {
    FileHandle.standardError.write(Data("""
    --expert-dtype fp8 (lossless FP4→FP8 re-encode) is not implemented yet.
    Falling back to relabel-only (FP4 dtype tag, identical bytes).

    """.utf8))
}

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

// ---------- BF16 conversion helpers ----------

@inline(__always)
func floatToBF16(_ f: Float) -> UInt16 {
    var bits = f.bitPattern
    // round-to-nearest-even
    let rounded = bits &+ ((bits >> 16) & 1) &+ 0x7FFF
    return UInt16(truncatingIfNeeded: rounded >> 16)
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
func deqE8M0(_ b: UInt8) -> Float {
    if b == 0xFF { return .nan }
    return Float(bitPattern: UInt32(b) << 23)
}

// ---------- wo_a fusion ----------
//
// Given an FP8-E4M3 weight `[out, in]` and an E8M0 scale `[out/128, in/128]`,
// produce a BF16 fused tensor `[out, in]` where
//   fused[i, j] = deqE4M3(weight[i, j]) * deqE8M0(scale[i / 128, j / 128]).
// Returns the bytes of the BF16 buffer.
func fuseWoA(weightURL: URL, weightOffset: Int, scaleURL: URL, scaleOffset: Int,
             outDim: Int, inDim: Int) throws -> Data {
    precondition(outDim % 128 == 0 && inDim % 128 == 0, "wo_a dims must be 128-aligned")
    let blocksOut = outDim / 128
    let blocksIn = inDim / 128

    // Read scale: [blocksOut, blocksIn] of u8.
    guard let sf = FileHandle(forReadingAtPath: scaleURL.path) else {
        throw NSError(domain: "fuseWoA", code: 1)
    }
    defer { try? sf.close() }
    try sf.seek(toOffset: UInt64(scaleOffset))
    guard let scaleBytes = try sf.read(upToCount: blocksOut * blocksIn) else {
        throw NSError(domain: "fuseWoA", code: 2)
    }

    // Read weight: [outDim, inDim] of u8 (FP8). Stream via a chunked read.
    guard let wf = FileHandle(forReadingAtPath: weightURL.path) else {
        throw NSError(domain: "fuseWoA", code: 3)
    }
    defer { try? wf.close() }
    try wf.seek(toOffset: UInt64(weightOffset))

    var out = Data(count: outDim * inDim * 2)        // BF16 bytes

    // Process row by row.
    let rowBytes = inDim
    out.withUnsafeMutableBytes { outRaw in
        let outPtr = outRaw.bindMemory(to: UInt16.self).baseAddress!
        scaleBytes.withUnsafeBytes { scaleRaw in
            let scalePtr = scaleRaw.bindMemory(to: UInt8.self).baseAddress!
            for i in 0..<outDim {
                guard let row = try? wf.read(upToCount: rowBytes), row.count == rowBytes else {
                    fatalError("short read on wo_a weight row \(i)")
                }
                let bo = i / 128
                row.withUnsafeBytes { rowRaw in
                    let rowPtr = rowRaw.bindMemory(to: UInt8.self).baseAddress!
                    for j in 0..<inDim {
                        let bi = j / 128
                        let s = deqE8M0(scalePtr[bo * blocksIn + bi])
                        let w = deqE4M3(rowPtr[j])
                        outPtr[i * inDim + j] = floatToBF16(w * s)
                    }
                }
            }
        }
    }
    return out
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

print("Indexing \(inputs.count) input shard(s) …")

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

// ---------- Write ----------

let outURL = saveDir.appendingPathComponent("model0-mp1.safetensors")
let writer = SafeTensorsWriter()

// Sort for deterministic output.
for newName in plan.keys.sorted() {
    let pt = plan[newName]!
    if newName.hasSuffix(".wo_a.weight"), let sc = woAScales[newName] {
        // Fuse FP8 + scale → BF16 single tensor.
        let outDim = pt.shape[0]
        let inDim = pt.shape[1]
        // Lazy compute: the data is produced when the writer streams it out.
        let n = outDim * inDim * 2
        writer.add(name: newName, dtype: "BF16", shape: pt.shape,
                   source: .compute(byteCount: n) {
            try fuseWoA(weightURL: pt.url, weightOffset: pt.offset,
                        scaleURL: sc.url, scaleOffset: sc.offset,
                        outDim: outDim, inDim: inDim)
        })
        continue
    }

    // Expert weight relabeling: convert.py reinterprets I8 → F4_E2M1
    // (`view(torch.float4_e2m1fn_x2)`). Bytes unchanged, only the dtype tag.
    var dtype = pt.dtype
    if newName.contains(".experts.") && newName.hasSuffix(".weight") {
        if pt.dtype.uppercased() == "I8" || pt.dtype.uppercased() == "U8" {
            dtype = "F4_E2M1"
        }
    }
    writer.add(name: newName, dtype: dtype, shape: pt.shape,
               source: .file(url: pt.url, offset: pt.offset, byteCount: pt.byteCount))
}

print("Writing \(outURL.path) …")
try writer.write(to: outURL)
print("Wrote \(plan.count) tensors.")

// ---------- Copy tokenizer ----------

for f in ["tokenizer.json", "tokenizer_config.json"] {
    let src = hfDir.appendingPathComponent(f)
    if fm.fileExists(atPath: src.path) {
        let dst = saveDir.appendingPathComponent(f)
        if fm.fileExists(atPath: dst.path) { try? fm.removeItem(at: dst) }
        try fm.copyItem(at: src, to: dst)
    }
}

print("Done. Run:")
print("  swift run -c release deepseek \(saveDir.path) \"<prompt>\" --mode chat")

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
