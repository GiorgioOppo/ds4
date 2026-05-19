import Foundation
import DeepSeekKit

// Calibration runner shared by the Llama-family (GGUF) and the
// DeepSeek-V4 (safetensors) paths (TODO §1 follow-up). Loads a
// model, walks a calibration corpus, accumulates per-layer
// activation stats (`ActivationObserver`) and optionally the
// per-layer Hessian (`HessianObserver`), and writes the results
// to disk in the format the converter ingests via
// `--calib-stats`.
//
// CLI:
//   deepseek_calibrate <model-path> <corpus.txt> <out-dir> [options]
//
// `--architecture llama` (default): `<model-path>` is a .gguf file.
//                                   Uses `LlamaCalibrationRunner`.
// `--architecture v4`:              `<model-path>` is a directory
//                                   from the V4 converter (BF16).
//                                   Uses `V4CalibrationRunner`.
//
// Options:
//   --collect-hessian            heavy, required for GPTQ
//   --max-tokens-per-batch N     default 1024
//   --max-seq-len N              optional KV cache cap (llama only)
//   --weight-dtype f32|bf16      llama only; default f32
//   --load-strategy mmap|preload llama only; default mmap
//
// Output:
//   <out-dir>/stats.json           per-layer perChannelAbsMax/Mean
//   <out-dir>/hessians/<name>.f64  raw [inDim*inDim] Doubles (with --collect-hessian)

enum Architecture: String { case llama, v4 }

let args = CommandLine.arguments
let stderr = FileHandle.standardError

func usage() -> Never {
    stderr.write(Data("""
    usage: deepseek_calibrate <model-path> <corpus.txt> <out-dir> [options]
    See header in Sources/deepseek_calibrate/main.swift for the full flag list.

    """.utf8))
    exit(2)
}

guard args.count >= 4 else { usage() }
let modelPath = args[1]
let corpusPath = args[2]
let outDir = args[3]

var architecture: Architecture = .llama
var collectHessian = false
var maxTokensPerBatch = 1024
var loadStrategy: LoadStrategy = .mmap
var weightDtype: DType = .f32
var maxSeqLenOverride: Int? = nil

var i = 4
while i < args.count {
    switch args[i] {
    case "--architecture":
        guard i + 1 < args.count,
              let a = Architecture(rawValue: args[i + 1])
        else { usage() }
        architecture = a; i += 2
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

guard loadStrategy != .streaming else {
    stderr.write(Data(
        "Refusing to calibrate with --load-strategy streaming: per-forward "
        + "redequant would dominate the sweep cost. Use mmap or preload.\n".utf8))
    exit(1)
}

// ---------- Read corpus (shared) ----------
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

let actObs = ActivationObserver()
let hessObs: HessianObserver? = collectHessian ? HessianObserver() : nil

// nLayers and orderedNames depend on the architecture; we fill
// them in inside each branch and use them after for the write-out.
var nLayers = 0
var orderedNames: [String] = []
var modelLabel = modelPath
let started = Date()

switch architecture {

case .llama:
    // ---------- Load GGUF + Llama ----------
    stderr.write(Data("Opening GGUF (\(loadStrategy.rawValue))…\n".utf8))
    let gguf: GGUFFile
    do {
        gguf = try GGUFFile(url: URL(fileURLWithPath: modelPath),
                              strategy: loadStrategy)
    } catch {
        stderr.write(Data("GGUF open failed: \(error)\n".utf8))
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

    let loaded = try TokenizerLoader.loadFromGGUF(gguf)
    let tokenizer = loaded.tokenizer

    let runner = LlamaCalibrationRunner(
        model: model, tokenizer: tokenizer,
        activation: actObs, hessian: hessObs)
    runner.maxTokensPerBatch = maxTokensPerBatch

    for (idx, sample) in samples.enumerated() {
        runner.observe(sample)
        if (idx + 1) % 10 == 0 || idx == samples.count - 1 {
            stderr.write(Data(
                "[\(idx + 1)/\(samples.count)] "
                + String(format: "%.1f", Date().timeIntervalSince(started))
                + "s elapsed\n".utf8))
        }
    }
    nLayers = model.config.nLayers
    var names = Set<String>()
    for L in 0..<nLayers {
        for s in ["attn_q", "attn_k", "attn_v", "ffn_gate", "ffn_up"] {
            names.insert("blk.\(L).\(s)")
        }
    }
    orderedNames = names.sorted()

case .v4:
    // ---------- Load V4 converted model ----------
    let modelDir = URL(fileURLWithPath: modelPath)
    let configURL = modelDir.appendingPathComponent("config.json")
    stderr.write(Data("Reading config.json…\n".utf8))
    let config: ModelConfig
    do {
        config = try ModelConfig.load(from: configURL)
    } catch {
        stderr.write(Data("ModelConfig.load failed: \(error)\n".utf8))
        exit(1)
    }
    stderr.write(Data(
        "Model: \(config.nLayers) layers, dim=\(config.dim), "
        + "nRoutedExperts=\(config.nRoutedExperts)\n".utf8))

    stderr.write(Data("Loading tokenizer…\n".utf8))
    let loaded: LoadedTokenizer
    do {
        loaded = try TokenizerLoader.load(tokenizerDir: modelDir)
    } catch {
        stderr.write(Data("Tokenizer load failed: \(error)\n".utf8))
        exit(1)
    }
    let tokenizer = loaded.tokenizer

    stderr.write(Data("Loading Transformer weights "
        + "(\(loadStrategy.rawValue))…\n".utf8))
    let model: Transformer
    do {
        model = try Transformer.load(
            config: config, from: modelDir,
            strategyOverride: loadStrategy.rawValue)
    } catch {
        stderr.write(Data("Transformer.load failed: \(error)\n".utf8))
        exit(1)
    }

    let runner = V4CalibrationRunner(
        model: model, tokenizer: tokenizer,
        activation: actObs, hessian: hessObs)
    runner.maxTokensPerBatch = maxTokensPerBatch

    for (idx, sample) in samples.enumerated() {
        runner.observe(sample)
        if (idx + 1) % 5 == 0 || idx == samples.count - 1 {
            stderr.write(Data(
                "[\(idx + 1)/\(samples.count)] "
                + String(format: "%.1f", Date().timeIntervalSince(started))
                + "s elapsed\n".utf8))
        }
    }
    nLayers = config.nLayers
    orderedNames = runner.tagNames()
    modelLabel = modelDir.path
}

stderr.write(Data("Calibration done in "
    + String(format: "%.1f", Date().timeIntervalSince(started))
    + "s\n".utf8))

// ---------- Write outputs (shared) ----------
let outURL = URL(fileURLWithPath: outDir)
do {
    try FileManager.default.createDirectory(
        at: outURL, withIntermediateDirectories: true)
} catch {
    stderr.write(Data("Cannot create out-dir: \(error)\n".utf8))
    exit(1)
}

var layerStats: [CalibrationStatsFile.LayerStats] = []
for name in orderedNames {
    guard let s = actObs.finalize(for: name) else { continue }
    layerStats.append(CalibrationStatsFile.LayerStats(
        name: name,
        inDim: s.perChannelAbsMax.count,
        observedTokens: s.observedTokens,
        perChannelAbsMax: s.perChannelAbsMax,
        perChannelMean: s.perChannelMean ?? []))
}
let statsFile = CalibrationStatsFile(
    model: modelLabel,
    nLayers: nLayers,
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
        hessObs.releaseLayer(name)
    }
}
stderr.write(Data("Done.\n".utf8))
