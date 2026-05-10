import Foundation
import DeepSeekKit

// CLI: deepseek <model-dir> "<prompt>"
//                   [--max-tokens N]
//                   [--temperature T]
//                   [--mode raw|chat]
//
// `<model-dir>` should contain config.json (optional) and tokenizer.json.
// safetensors weights are loaded if present, otherwise the model is
// initialised with random f32 weights so the smoke flow still runs.

func usage() -> Never {
    FileHandle.standardError.write(Data("""
    usage: deepseek <model-dir> "<prompt>" \
        [--max-tokens N] [--temperature T] [--mode raw|chat]

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
    model = try Transformer.load(config: config, from: modelDir)
} catch {
    FileHandle.standardError.write(Data("model load failed: \(error.localizedDescription)\n".utf8))
    exit(1)
}
print(" ready.")

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

// ---------- Generation loop ----------
// Prefill: feed the entire prompt at start_pos = 0. Then decode token by
// token. Single-token decode requires a working MLA decode path which is
// not yet implemented (it traps with a precondition). For now we run the
// prefill, take the last-position logits, sample once, and stop. That
// exercises the full pipeline end-to-end without hitting the decode trap.
//
// Flag `--max-tokens 1` is the safe default; larger values currently
// cause the second forward call to fail.

let logits = model.forward(inputIds: [promptIds], startPos: 0)

// Sampling: argmax for temperature == 0, else sample with temperature.
let sampledId: Int
if temperature == 0 {
    sampledId = Sampler.argmax(logits)
} else {
    Sampler.applyTemperature(logits, temperature)
    sampledId = Sampler.argmax(logits)   // greedy after temperature scaling — proper
                                           // multinomial sampling is a future step
}

let sampled = tokenizer.decode([sampledId])
print("---")
if mode == "chat" {
    let msg = EncodingDSV4.parseCompletion(sampled, mode: .chat)
    print(msg.content)
} else {
    print(sampled)
}

if maxTokens > 1 {
    FileHandle.standardError.write(Data("""

    Note: only one token is generated end-to-end at this stage.
    Multi-token decode requires the sliding-window MLA decode path which
    is still a fatalError in Sources/DeepSeekKit/Layers/Attention.swift.

    """.utf8))
}
