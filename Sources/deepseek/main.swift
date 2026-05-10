import Foundation
import DeepSeekKit

// CLI: deepseek <model-dir> "<prompt>" [--max-tokens N] [--temperature T]
//
// <model-dir> must contain:
//   - config.json
//   - tokenizer.json + tokenizer_config.json
//   - one or more *.safetensors shards (after running convert.py upstream
//     to produce model0-mp1.safetensors)
//
// This is a thin wrapper over DeepSeekKit. The model assembly step is
// `assembleModel` below — left as a TODO until the safetensors weight
// names in the V4 checkpoint are confirmed.

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
var temperature: Float = 1.0
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
    FileHandle.standardError.write(Data("\(error.localizedDescription)\n".utf8))
    exit(1)
}

func assembleModel(config: ModelConfig, weightsDir: URL) throws -> Transformer {
    // Wire each layer from safetensors shards. Largest piece of integration
    // work remaining — needs the V4 weight name map (see README "Roadmap").
    throw NSError(domain: "Assembly", code: -1, userInfo: [NSLocalizedDescriptionKey:
        "Model assembly not implemented — needs V4 safetensors weight name map."])
}

let model: Transformer
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
