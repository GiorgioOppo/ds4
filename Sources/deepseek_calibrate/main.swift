import Foundation
import DeepSeekKit

// Calibration runner for the Llama-family path (TODO §1 follow-up).
// Loads a GGUF model, walks a calibration corpus, accumulates
// per-layer activation stats (ActivationObserver) and optionally
// the per-layer Hessian (HessianObserver), and writes the results
// to disk so a follow-up quantizer (gptqQuantizeBF16ToInt8,
// quantizeBF16ToInt8Calibrated with .awq / .smoothQuant) can ingest
// them.
//
// CLI: deepseek_calibrate <path-to.gguf> <corpus.txt> <out-dir>
//                          [--collect-hessian]            (heavy; required for GPTQ)
//                          [--max-tokens-per-batch N]     (default 1024)
//                          [--load-strategy mmap|preload|streaming]
//                          [--weight-dtype f32|bf16]
//                          [--max-seq-len N]
//
// Output:
//   <out-dir>/stats.json
//     { "layers": [
//         { "name": "blk.0.attn_q", "inDim": 4096,
//           "observedTokens": 12345,
//           "perChannelAbsMax": [...], "perChannelMean": [...] }, ... ] }
//   <out-dir>/hessians/<layer>.f64    (when --collect-hessian is set)
//     raw little-endian Double[inDim * inDim], symmetric.

let args = CommandLine.arguments
let stderr = FileHandle.standardError

func usage() -> Never {
    stderr.write(Data("""
    usage: deepseek_calibrate <path-to.gguf> <corpus.txt> <out-dir> [options]
    See header in Sources/deepseek_calibrate/main.swift for the full flag list.

    """.utf8))
    exit(2)
}

guard args.count >= 4 else { usage() }
let ggufPath = args[1]
let corpusPath = args[2]
let outDir = args[3]

var collectHessian = false
var maxTokensPerBatch = 1024
var loadStrategy: LoadStrategy = .mmap
var weightDtype: DType = .f32
var maxSeqLenOverride: Int? = nil

var i = 4
while i < args.count {
    switch args[i] {
    case "--collect-hessian":
        collectHessian = true; i += 1
    case "--max-tokens-per-batch":
        guard i + 1 < args.count, let v = Int(args[i + 1]), v > 0 else { usage() }
        maxTokensPerBatch = v; i += 2
    case "--load-strategy":
        guard i + 1 < args.count else { usage() }
        switch args[i + 1] {
        case "mmap":      loadStrategy = .mmap
        case "preload":   loadStrategy = .preload
        case "streaming": loadStrategy = .streaming
        default: usage()
        }
        i += 2
    case "--weight-dtype":
        guard i + 1 < args.count else { usage() }
        switch args[i + 1] {
        case "f32":  weightDtype = .f32
        case "bf16": weightDtype = .bf16
        default: usage()
        }
        i += 2
    case "--max-seq-len":
        guard i + 1 < args.count, let v = Int(args[i + 1]), v > 0 else { usage() }
        maxSeqLenOverride = v; i += 2
    default:
        usage()
    }
}

// ---------- Load GGUF ----------
stderr.write(Data("Opening GGUF (\(loadStrategy.rawValue))…\n".utf8))
let gguf: GGUFFile
do {
    gguf = try GGUFFile(url: URL(fileURLWithPath: ggufPath),
                          strategy: loadStrategy)
} catch {
    stderr.write(Data("GGUF open failed: \(error)\n".utf8))
    exit(1)
}

// Streaming model would mean re-dequantizing weights every forward;
// fine for normal inference but a calibration sweep is many forwards
// so we materialize once instead.
guard loadStrategy != .streaming else {
    stderr.write(Data(
        "Refusing to calibrate with --load-strategy streaming: per-forward "
        + "redequant would dominate the sweep cost. Use mmap or preload.\n".utf8))
    exit(1)
}

stderr.write(Data("Building LlamaModel…\n".utf8))
let model: LlamaModel
do {
    model = try LlamaModel.fromGGUF(
        gguf,
        maxSeqLenOverride: maxSeqLenOverride,
        weightDtype: weightDtype)
} catch {
    stderr.write(Data("LlamaModel build failed: \(error)\n".utf8))
    exit(1)
}
stderr.write(Data(
    "Model: \(model.config.nLayers) layers × \(model.config.nHeads) heads, "
    + "vocab=\(model.config.vocabSize)\n".utf8))

stderr.write(Data("Building tokenizer…\n".utf8))
let loaded: LoadedTokenizer
do {
    loaded = try TokenizerLoader.loadFromGGUF(gguf)
} catch {
    stderr.write(Data("Tokenizer build failed: \(error)\n".utf8))
    exit(1)
}
let tokenizer = loaded.tokenizer

// ---------- Read corpus ----------
let corpusContents: String
do {
    corpusContents = try String(
        contentsOf: URL(fileURLWithPath: corpusPath),
        encoding: .utf8)
} catch {
    stderr.write(Data("Failed to read corpus: \(error)\n".utf8))
    exit(1)
}
let samples = corpusContents
    .split(separator: "\n", omittingEmptySubsequences: true)
    .map { String($0) }
    .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
stderr.write(Data("Corpus: \(samples.count) samples\n".utf8))

// ---------- Run calibration ----------
let actObs = ActivationObserver()
let hessObs: HessianObserver? = collectHessian ? HessianObserver() : nil
let runner = LlamaCalibrationRunner(
    model: model, tokenizer: tokenizer,
    activation: actObs, hessian: hessObs)
runner.maxTokensPerBatch = maxTokensPerBatch

let started = Date()
for (idx, sample) in samples.enumerated() {
    runner.observe(sample)
    if (idx + 1) % 10 == 0 || idx == samples.count - 1 {
        let dt = Date().timeIntervalSince(started)
        stderr.write(Data(
            "[\(idx + 1)/\(samples.count)] " + String(format: "%.1f", dt)
            + "s elapsed\n".utf8))
    }
}
stderr.write(Data("Calibration done in "
    + String(format: "%.1f", Date().timeIntervalSince(started))
    + "s\n".utf8))

// ---------- Write outputs ----------
let outURL = URL(fileURLWithPath: outDir)
do {
    try FileManager.default.createDirectory(
        at: outURL, withIntermediateDirectories: true)
} catch {
    stderr.write(Data("Cannot create out-dir: \(error)\n".utf8))
    exit(1)
}

// Layer name set: union of what's been observed. We sort so the
// JSON output has a deterministic order.
var layerNames = Set<String>()
for L in 0..<model.config.nLayers {
    for suffix in ["attn_q", "attn_k", "attn_v", "ffn_gate", "ffn_up"] {
        layerNames.insert("blk.\(L).\(suffix)")
    }
}
let orderedNames = layerNames.sorted()

// stats.json
struct LayerStats: Encodable {
    let name: String
    let inDim: Int
    let observedTokens: Int
    let perChannelAbsMax: [Float]
    let perChannelMean: [Float]
}
struct CalibrationStatsFile: Encodable {
    let model: String
    let nLayers: Int
    let hessianCollected: Bool
    let layers: [LayerStats]
}

var layerStats: [LayerStats] = []
for name in orderedNames {
    guard let s = actObs.finalize(for: name) else { continue }
    layerStats.append(LayerStats(
        name: name,
        inDim: s.perChannelAbsMax.count,
        observedTokens: s.observedTokens,
        perChannelAbsMax: s.perChannelAbsMax,
        perChannelMean: s.perChannelMean ?? []))
}
let statsFile = CalibrationStatsFile(
    model: ggufPath,
    nLayers: model.config.nLayers,
    hessianCollected: collectHessian,
    layers: layerStats)
let statsJSONURL = outURL.appendingPathComponent("stats.json")
do {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let data = try encoder.encode(statsFile)
    try data.write(to: statsJSONURL)
    stderr.write(Data("Wrote \(statsJSONURL.path) "
        + "(\(layerStats.count) layers)\n".utf8))
} catch {
    stderr.write(Data("stats.json write failed: \(error)\n".utf8))
    exit(1)
}

// hessians/<layer>.f64
if let hessObs = hessObs {
    let hessDir = outURL.appendingPathComponent("hessians")
    try? FileManager.default.createDirectory(
        at: hessDir, withIntermediateDirectories: true)
    for name in orderedNames {
        guard let (H, inDim, _) = hessObs.finalize(for: name) else { continue }
        let path = hessDir.appendingPathComponent("\(name).f64")
        do {
            try H.withUnsafeBufferPointer { buf in
                let data = Data(bytes: buf.baseAddress!,
                                 count: buf.count * MemoryLayout<Double>.size)
                try data.write(to: path)
            }
            stderr.write(Data("  hess \(name) (\(inDim)×\(inDim) doubles)\n".utf8))
        } catch {
            stderr.write(Data("hessian write failed for \(name): \(error)\n".utf8))
        }
        // Release immediately so we don't hold every layer's Hessian
        // in memory at once — for inDim=11k that's ~970 MB per layer.
        hessObs.releaseLayer(name)
    }
}
stderr.write(Data("Done.\n".utf8))
