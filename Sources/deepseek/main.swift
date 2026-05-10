import Foundation
import DeepSeekKit

// CLI: deepseek <model-dir> "<prompt>" [--max-tokens N] [--temperature T]
//
// <model-dir> must contain:
//   - config.json
//   - tokenizer.json
//   - one or more *.safetensors shards (with `model.safetensors.index.json`
//     if sharded)
//
// This is a thin wrapper around DeepSeekKit. The model assembly step (loading
// every weight, building DecoderLayers) is intentionally not here yet — see
// `assembleModel` below, which is a TODO until the safetensors weight names
// for V4-Pro are confirmed.

func usage() -> Never {
    FileHandle.standardError.write(Data("""
    usage: deepseek <model-dir> "<prompt>" [--max-tokens N] [--temperature T]
    """.utf8))
    exit(2)
}

let args = CommandLine.arguments
guard args.count >= 3 else { usage() }

let modelDir = URL(fileURLWithPath: args[1])
let prompt = args[2]
var maxTokens = 128
var temperature: Float = 0.0
var i = 3
while i < args.count {
    switch args[i] {
    case "--max-tokens":
        guard i + 1 < args.count, let n = Int(args[i+1]) else { usage() }
        maxTokens = n; i += 2
    case "--temperature":
        guard i + 1 < args.count, let t = Float(args[i+1]) else { usage() }
        temperature = t; i += 2
    default:
        usage()
    }
}

let configURL = modelDir.appendingPathComponent("config.json")
let tokenizerURL = modelDir.appendingPathComponent("tokenizer.json")

let config: ModelConfig
do {
    config = try ModelConfig.load(from: configURL)
} catch {
    FileHandle.standardError.write(Data("failed to load config.json: \(error)\n".utf8))
    exit(1)
}

let tokenizer: Tokenizer
do {
    tokenizer = try TokenizerLoader.load(from: tokenizerURL)
} catch {
    FileHandle.standardError.write(Data("""
    Tokenizer is not implemented yet. See Sources/DeepSeekKit/Tokenizer.swift.
    Once tokenizer.json is parsed, this CLI will run end-to-end.
    \(error)

    """.utf8))
    exit(1)
}

func assembleModel(config: ModelConfig, weightsDir: URL) throws -> DeepSeekV4 {
    // Wire each layer from safetensors shards. This is the largest piece
    // of integration work remaining — see README "Roadmap".
    throw NSError(domain: "Assembly", code: -1, userInfo: [NSLocalizedDescriptionKey:
        "Model assembly not implemented — needs the V4-Pro weight name map."])
}

let model: DeepSeekV4
do {
    model = try assembleModel(config: config, weightsDir: modelDir)
} catch {
    FileHandle.standardError.write(Data("\(error.localizedDescription)\n".utf8))
    exit(1)
}

let gen = Generator(model: model, tokenizer: tokenizer)
gen.generate(prompt: prompt,
             options: GenerationOptions(maxNewTokens: maxTokens, temperature: temperature)) { piece in
    print(piece, terminator: "")
    fflush(stdout)
}
print("")
