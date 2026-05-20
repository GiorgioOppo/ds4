import Foundation
import DeepSeekKit

// Minimal CLI for running a GGUF Llama-family checkpoint end-to-end
// (TODO §10.2 / T2 plumbing). Keeps the V4-specific `deepseek` CLI
// untouched — that one carries a lot of MLA / streaming / converter
// machinery that doesn't apply here. This file is the smallest
// possible "load, prefill, decode" loop on top of:
//
//   - `GGUFFile` (mmap + dequant kernels)
//   - `LlamaModel.fromGGUF(...)` (factory)
//   - `TokenizerLoader.loadFromGGUF(...)` (BPE reconstruction)
//   - `Sampler.sample(...)` (the same sampler the V4 path uses)
//
// CLI: deepseek_gguf <path-to.gguf> "<prompt>"
//                    [--max-tokens N]                (default 64)
//                    [--temperature T]               (default 1.0)
//                    [--top-k K] [--top-p P]
//                    [--min-p P] [--tfs Z] [--typical P]
//                    [--repetition-penalty R]
//                    [--frequency-penalty F] [--presence-penalty P]
//                    [--mirostat TAU] [--mirostat-eta ETA]
//                    [--mirostat-v1] [--mirostat-m N]
//                    [--logit-bias '{"<id>": <bias>}']
//                    [--dry-multiplier M] [--dry-base B]
//                                                  [--dry-allowed-length N]
//                    [--json-schema PATH]
//                    [--max-seq-len N]               (cap KV cache)
//                    [--no-chat-template]            (skip the GGUF chat template)
//                    [--weight-dtype f32|bf16]       (dequant target; default f32)
//                    [--load-strategy mmap|preload|streaming]  (default mmap)
//                    [--use-map-shared]              (MAP_SHARED instead of MAP_PRIVATE)
//                    [--warmup]                      (POSIX_MADV_WILLNEED on the mmap)

let args = CommandLine.arguments

func usage() -> Never {
    FileHandle.standardError.write(Data("""
    usage: deepseek_gguf <path-to.gguf> "<prompt>" [options]
    See header in Sources/deepseek_gguf/main.swift for full flag list.

    """.utf8))
    exit(2)
}

guard args.count >= 3 else { usage() }

let ggufPath = args[1]
let prompt = args[2]
var i = 3

var maxTokens = 64
var temperature: Float = 1.0
var topK = 0
var topP: Float = 1.0
var minP: Float = 0.0
var tfsZ: Float = 1.0
var typicalP: Float = 1.0
var repetitionPenalty: Float = 1.0
var frequencyPenalty: Float = 0.0
var presencePenalty: Float = 0.0
var mirostatTau: Float = 0.0
var mirostatEta: Float = 0.1
var mirostatV1 = false
var mirostatM = 100
var logitBiasArg: String? = nil
var dryMultiplier: Float = 0.0
var dryBase: Float = 1.75
var dryAllowedLength = 2
var jsonSchemaPath: String? = nil
var maxSeqLenOverride: Int? = nil
var noChatTemplate = false
// `--weight-dtype f32|bf16` — controls the dequant target for
// quantized GGUF weights. f32 is safer numerically; bf16 halves
// the resident memory of the weights at no measurable accuracy
// loss on Q8_0/Q4_0/Q4_K. Q5_K/Q6_K silently fall back to f32.
var weightDtype: DType = .f32
// Mirror of the safetensors load-strategy machinery (TODO §10.2
// follow-up). `.streaming` swaps in `LlamaStreamingModel`, which
// dequantizes weights lazily per forward and emits per-layer
// `POSIX_MADV_DONTNEED` hints so the OS can evict the source mmap
// under pressure.
var loadStrategy: LoadStrategy = .mmap
var useMapShared = false
var warmup = false

while i < args.count {
    switch args[i] {
    case "--max-tokens":
        guard i + 1 < args.count, let n = Int(args[i + 1]), n > 0 else { usage() }
        maxTokens = n; i += 2
    case "--temperature":
        guard i + 1 < args.count, let t = Float(args[i + 1]) else { usage() }
        temperature = t; i += 2
    case "--top-k":
        guard i + 1 < args.count, let n = Int(args[i + 1]), n >= 0 else { usage() }
        topK = n; i += 2
    case "--top-p":
        guard i + 1 < args.count, let v = Float(args[i + 1]) else { usage() }
        topP = v; i += 2
    case "--min-p":
        guard i + 1 < args.count, let v = Float(args[i + 1]) else { usage() }
        minP = v; i += 2
    case "--tfs":
        guard i + 1 < args.count, let v = Float(args[i + 1]) else { usage() }
        tfsZ = v; i += 2
    case "--typical":
        guard i + 1 < args.count, let v = Float(args[i + 1]) else { usage() }
        typicalP = v; i += 2
    case "--repetition-penalty":
        guard i + 1 < args.count, let v = Float(args[i + 1]) else { usage() }
        repetitionPenalty = v; i += 2
    case "--frequency-penalty":
        guard i + 1 < args.count, let v = Float(args[i + 1]) else { usage() }
        frequencyPenalty = v; i += 2
    case "--presence-penalty":
        guard i + 1 < args.count, let v = Float(args[i + 1]) else { usage() }
        presencePenalty = v; i += 2
    case "--mirostat":
        guard i + 1 < args.count, let v = Float(args[i + 1]) else { usage() }
        mirostatTau = v; i += 2
    case "--mirostat-eta":
        guard i + 1 < args.count, let v = Float(args[i + 1]) else { usage() }
        mirostatEta = v; i += 2
    case "--mirostat-v1":
        mirostatV1 = true; i += 1
    case "--mirostat-m":
        guard i + 1 < args.count, let v = Int(args[i + 1]), v > 0 else { usage() }
        mirostatM = v; i += 2
    case "--logit-bias":
        guard i + 1 < args.count else { usage() }
        logitBiasArg = args[i + 1]; i += 2
    case "--dry-multiplier":
        guard i + 1 < args.count, let v = Float(args[i + 1]), v >= 0 else { usage() }
        dryMultiplier = v; i += 2
    case "--dry-base":
        guard i + 1 < args.count, let v = Float(args[i + 1]), v > 0 else { usage() }
        dryBase = v; i += 2
    case "--dry-allowed-length":
        guard i + 1 < args.count, let v = Int(args[i + 1]), v > 0 else { usage() }
        dryAllowedLength = v; i += 2
    case "--json-schema":
        guard i + 1 < args.count else { usage() }
        jsonSchemaPath = args[i + 1]; i += 2
    case "--max-seq-len":
        guard i + 1 < args.count, let n = Int(args[i + 1]), n > 0 else { usage() }
        maxSeqLenOverride = n; i += 2
    case "--no-chat-template":
        noChatTemplate = true; i += 1
    case "--weight-dtype":
        guard i + 1 < args.count else { usage() }
        switch args[i + 1] {
        case "f32":  weightDtype = .f32
        case "bf16": weightDtype = .bf16
        default: usage()
        }
        i += 2
    case "--load-strategy":
        guard i + 1 < args.count else { usage() }
        switch args[i + 1] {
        case "mmap":      loadStrategy = .mmap
        case "preload":   loadStrategy = .preload
        case "streaming": loadStrategy = .streaming
        default: usage()
        }
        i += 2
    case "--use-map-shared":
        useMapShared = true; i += 1
    case "--warmup":
        warmup = true; i += 1
    default:
        usage()
    }
}

let stderr = FileHandle.standardError

// ---------- Load GGUF ----------
let ggufURL = URL(fileURLWithPath: ggufPath)
let gguf: GGUFFile
do {
    gguf = try GGUFFile(url: ggufURL,
                          strategy: loadStrategy,
                          useMapShared: useMapShared,
                          warmup: warmup)
} catch {
    stderr.write(Data("Failed to open GGUF: \(error)\n".utf8))
    exit(1)
}

// ---------- Build model ----------
stderr.write(Data(
    "Loading model weights (strategy=\(loadStrategy.rawValue))…\n".utf8))
let model: any LlamaForwardModel
do {
    switch loadStrategy {
    case .streaming:
        model = try LlamaStreamingModel(
            gguf: gguf,
            maxSeqLenOverride: maxSeqLenOverride,
            weightDtype: weightDtype)
    case .mmap, .preload:
        model = try LlamaModel.fromGGUF(
            gguf,
            maxSeqLenOverride: maxSeqLenOverride,
            weightDtype: weightDtype)
    }
} catch {
    stderr.write(Data("Failed to build LlamaModel: \(error)\n".utf8))
    exit(1)
}
stderr.write(Data(
    ("Loaded: \(model.config.nLayers) layers × \(model.config.nHeads) heads "
     + "× \(model.config.headDim) head_dim, vocab=\(model.config.vocabSize), "
     + "ctx=\(model.config.maxSeqLen)\n").utf8))

// ---------- Build tokenizer ----------
let loaded: LoadedTokenizer
do {
    loaded = try TokenizerLoader.loadFromGGUF(gguf)
} catch {
    stderr.write(Data("Failed to load GGUF tokenizer: \(error)\n".utf8))
    exit(1)
}
let tokenizer = loaded.tokenizer

// ---------- Render the prompt ----------
// By default, ask the GGUF's embedded chat template to wrap the
// user's prompt. `--no-chat-template` skips that and tokenizes the
// raw text directly, useful for testing or for completion-style
// (non-chat) base models.
let promptText: String
if noChatTemplate {
    promptText = prompt
} else {
    let msgs = [Message(role: .user, content: prompt)]
    do {
        promptText = try loaded.chatTemplate.render(messages: msgs,
                                                      options: .init())
    } catch {
        stderr.write(Data(
            ("Chat template render failed (use --no-chat-template to "
             + "skip): \(error)\n").utf8))
        exit(1)
    }
}

let promptIds = tokenizer.encode(promptText)
guard !promptIds.isEmpty else {
    stderr.write(Data("Tokenizer produced 0 tokens for the prompt.\n".utf8))
    exit(1)
}
stderr.write(Data("Prompt: \(promptIds.count) tokens\n".utf8))

// ---------- Sampler config ----------
var parsedLogitBias: [Int32: Float] = [:]
if let raw = logitBiasArg,
   let data = raw.data(using: .utf8),
   let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
{
    for (k, v) in obj {
        guard let id = Int32(k) else { continue }
        let f: Float?
        if let n = v as? NSNumber { f = n.floatValue }
        else if let s = v as? String { f = Float(s) }
        else { f = nil }
        if let f { parsedLogitBias[id] = f }
    }
}

var samplingOpts = SamplingOptions(
    temperature: temperature,
    topK: topK, topP: topP,
    minP: minP, tailFree: tfsZ, typical: typicalP,
    repetitionPenalty: repetitionPenalty,
    frequencyPenalty: frequencyPenalty, presencePenalty: presencePenalty,
    mirostatTau: mirostatTau, mirostatEta: mirostatEta,
    mirostatMu: 2.0 * mirostatTau,
    logitBias: parsedLogitBias,
    dryMultiplier: dryMultiplier, dryBase: dryBase,
    dryAllowedLength: dryAllowedLength,
    mirostatV1: mirostatV1, mirostatM: mirostatM)

if let path = jsonSchemaPath {
    do {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let mask = try SchemaCompiler.compile(
            jsonData: data,
            tokenizer: tokenizer,
            vocabSize: model.config.vocabSize)
        samplingOpts.schemaMask = mask
    } catch {
        stderr.write(Data(
            "JSON schema compile failed: \(error)\n".utf8))
        exit(1)
    }
}

// ---------- Prefill ----------
stderr.write(Data("Prefilling…\n".utf8))
let prefillStart = Date()
var logits = model.forward(inputIds: [promptIds], startPos: 0)
let prefillElapsed = Date().timeIntervalSince(prefillStart)
stderr.write(Data(
    ("Prefill: \(promptIds.count) tokens in \(String(format: "%.2f", prefillElapsed))s "
     + "(\(Int(Double(promptIds.count) / max(prefillElapsed, 1e-9))) tok/s)\n").utf8))

// ---------- Decode loop ----------
let stopTokens: Set<Int> = tokenizer.stopTokenIds
var generatedIds: [Int] = []
var startPos = promptIds.count

let stdout = FileHandle.standardOutput
let decodeStart = Date()

for _ in 0..<maxTokens {
    let nextId = Sampler.sample(logits,
                                  history: promptIds + generatedIds,
                                  options: &samplingOpts)
    if stopTokens.contains(nextId) { break }
    generatedIds.append(nextId)
    let piece = tokenizer.decode([nextId])
    stdout.write(Data(piece.utf8))
    // Step one token forward.
    logits = model.forward(inputIds: [[nextId]], startPos: startPos)
    startPos += 1
}
stdout.write(Data("\n".utf8))

let decodeElapsed = Date().timeIntervalSince(decodeStart)
let tps = generatedIds.isEmpty
    ? 0.0
    : Double(generatedIds.count) / max(decodeElapsed, 1e-9)
stderr.write(Data(
    "Generated \(generatedIds.count) tokens in "
    + "\(String(format: "%.2f", decodeElapsed))s "
    + "(\(String(format: "%.1f", tps)) tok/s)\n".utf8))
