import Foundation
import DeepSeekKit

// CLI: deepseek <model-dir> "<prompt>"
//                   [--max-tokens N]
//                   [--temperature T]
//                   [--mode raw|chat]
//                   [--load-strategy auto|preload|mmap]
//                   [--force-load]
//
// `<model-dir>` should contain config.json (optional) and tokenizer.json.
// safetensors weights are loaded if present, otherwise the model is
// initialised with random f32 weights so the smoke flow still runs.
//
// `--force-load` bypasses the conservative RAM-safety refusals
// (shard > 70% of available, total > 25× available). Use only if you
// know you can tolerate aggressive paging — on small-RAM Macs the
// system can lock up under sustained mmap thrash.

func usage() -> Never {
    FileHandle.standardError.write(Data("""
    usage: deepseek <model-dir> "<prompt>" \
        [--max-tokens N] [--temperature T] [--mode raw|chat] \
        [--load-strategy auto|preload|mmap] [--force-load]

    """.utf8))
    exit(2)
}

let args = CommandLine.arguments
guard args.count >= 3 else { usage() }

let modelDir = URL(fileURLWithPath: args[1])
let prompt = args[2]
var maxTokens = 32
var temperature: Float = 1.0
var mode = "chat"
var loadStrategy: String? = nil
var forceLoad = false

var i = 3
while i < args.count {
    switch args[i] {
    case "--max-tokens":
        guard i + 1 < args.count, let n = Int(args[i + 1]) else { usage() }
        maxTokens = n; i += 2
    case "--temperature":
        guard i + 1 < args.count, let t = Float(args[i + 1]) else { usage() }
        temperature = t; i += 2
    case "--mode":
        guard i + 1 < args.count, ["raw", "chat"].contains(args[i + 1]) else { usage() }
        mode = args[i + 1]; i += 2
    case "--load-strategy":
        guard i + 1 < args.count,
              ["auto", "preload", "mmap", "streaming"].contains(args[i + 1]) else { usage() }
        loadStrategy = args[i + 1]; i += 2
    case "--force-load":
        forceLoad = true; i += 1
    default: usage()
    }
}

// ---------- Config ----------
let configURL = modelDir.appendingPathComponent("config.json")
let config: ModelConfig
if FileManager.default.fileExists(atPath: configURL.path) {
    do {
        config = try ModelConfig.load(from: configURL)
    } catch {
        FileHandle.standardError.write(Data("failed to parse config.json: \(error)\n".utf8))
        exit(1)
    }
} else {
    FileHandle.standardError.write(Data("""
    config.json not found at \(configURL.path) — using ModelArgs defaults.
    Note: defaults are toy-sized (n_layers=7, n_routed_experts=8) and not
    suitable for real inference.

    """.utf8))
    config = ModelConfig()
}

// ---------- Tokenizer ----------
let tokenizerURL = modelDir.appendingPathComponent("tokenizer.json")
let tokenizer: Tokenizer
do {
    tokenizer = try TokenizerLoader.load(from: tokenizerURL)
} catch {
    FileHandle.standardError.write(Data("\(error.localizedDescription)\n".utf8))
    exit(1)
}

// ---------- Model ----------
print("Loading model …", terminator: "")
fflush(stdout)
let model: Transformer
do {
    model = try Transformer.load(config: config, from: modelDir,
                                  strategyOverride: loadStrategy,
                                  forceLoad: forceLoad)
} catch {
    FileHandle.standardError.write(Data("model load failed: \(error.localizedDescription)\n".utf8))
    exit(1)
}
print(" ready.")
MemoryLogger.snapshot("model-ready", force: true)

// ---------- Prompt formatting ----------
let promptText: String
switch mode {
case "raw":
    promptText = prompt
case "chat":
    let msg = Message(role: .user, content: prompt)
    promptText = EncodingDSV4.encodeMessages([msg], mode: .chat)
default: usage()
}

let promptIds = tokenizer.encode(promptText)
print("Prompt tokens: \(promptIds.count)")
if promptIds.isEmpty {
    FileHandle.standardError.write(Data("""
    tokenizer produced 0 tokens for the prompt. Check tokenizer.json's
    pre_tokenizer regex — see `jq '.pre_tokenizer' \(tokenizerURL.path)`.

    """.utf8))
    exit(1)
}

// ---------- Generation loop ----------
// Prefill: feed the entire prompt at start_pos = 0. Then decode one token
// at a time, feeding the previous token at start_pos = (prompt_len + i).
// Each decode call updates the sliding-window KV cache + compressor state
// and produces logits for one new token.

print("---")
fflush(stdout)

let eosId: Int = tokenizer.eosId ?? -1
var generatedIds: [Int] = []

// Prefill.
var logits = model.forward(inputIds: [promptIds], startPos: 0)
MemoryLogger.snapshot("prefill-complete", force: true)
var samplingOpts = SamplingOptions(temperature: temperature,
                                    topK: 0, topP: 1.0,
                                    repetitionPenalty: 1.0)

for step in 0..<maxTokens {
    let nextId = Sampler.sample(logits, history: generatedIds, options: &samplingOpts)
    if nextId == eosId { break }
    generatedIds.append(nextId)
    MemoryLogger.snapshot("decode:token-\(String(format: "%03d", step))",
                          force: true)

    // Stream the new token to stdout immediately. In chat mode, we buffer
    // the whole output and parse <think>...</think> at the end so the
    // reasoning block doesn't leak before its closing tag.
    if mode == "raw" {
        print(tokenizer.decode([nextId]), terminator: "")
        fflush(stdout)
    }

    // Stop after sampling the requested count without doing one more
    // unnecessary forward.
    if step == maxTokens - 1 { break }

    // Decode step: feed only the just-sampled token at startPos = promptLen + step.
    let startPos = promptIds.count + step
    logits = model.forward(inputIds: [[nextId]], startPos: startPos)
}

print("")    // newline after streaming
if mode == "chat" {
    let combined = tokenizer.decode(generatedIds)
    let msg = EncodingDSV4.parseCompletion(combined, mode: .chat)
    if let r = msg.reasoningContent, !r.isEmpty {
        print("[reasoning]\n\(r)\n[/reasoning]")
    }
    print(msg.content)
}
