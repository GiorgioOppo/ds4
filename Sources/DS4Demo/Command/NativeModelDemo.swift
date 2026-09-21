import DS4Core
import DS4Metal
import Foundation

func runNativeModelDemo(model: GGUFModel, architecture: ModelArchitectureID,
                        contextSize: Int, arguments: [String]) throws {
    let env = ProcessInfo.processInfo.environment
    let budget = arguments.count > 2 ? Int(arguments[2]) ?? 4 : Int(env["DS4_MAX_NEW"] ?? "") ?? 4
    var prompt = arguments.count > 3 ? arguments[3] : env["DS4_PROMPT"] ?? "Ciao! Presentati in una frase."
    if prompt.hasPrefix("@") { prompt = try String(contentsOfFile: String(prompt.dropFirst()), encoding: .utf8) }
    let tokenizer = try TokenizerFactory.make(for: model)
    let decoder: any SwiftModelDecoder
    let rendered: String
    let reasoning: ThinkMode = env["DS4_THINK"] == "1" ? .high : .none
    switch architecture {
    case .bonsai2:
        decoder = try BonsaiModel(model: model, contextSize: contextSize)
        rendered = try QwenChatRenderer.render(turns: [.user(prompt)], architecture: architecture, reasoning: reasoning)
    case .qwen38FlashNext:
        decoder = try Qwen38Model(model: model, contextSize: contextSize)
        rendered = try QwenChatRenderer.render(turns: [.user(prompt)], architecture: architecture, reasoning: reasoning)
    case .glm53Flash:
        decoder = try GLM53Model(model: model, contextSize: contextSize)
        rendered = try GLM52ChatRenderer.render(turns: [.user(prompt)], reasoning: reasoning)
    case .deepSeekV41:
        decoder = try DeepSeek41Model(model: model, contextSize: contextSize)
        rendered = try DeepSeek41ChatRenderer.render(turns: [.user(prompt)], reasoning: reasoning)
    default: throw ModelArchitectureError.unsupportedArchitecture(architecture)
    }
    let tokens = tokenizer.tokenizeRenderedChat(rendered).map(Int.init)
    guard tokens.count < contextSize, budget > 0 else {
        throw SwiftModelDecoderError.invalidInput("Il prompt deve lasciare spazio alla generazione e maxNew deve essere positivo.")
    }
    let start = Date()
    var logits = try decoder.evaluate(tokens: tokens, cancelled: { false })
    log(String(format: "DS4Demo %@: prefill %d token in %.3fs", architecture.rawValue, tokens.count, Date().timeIntervalSince(start)))
    let decode = Date()
    var generated = 0
    for index in 0..<min(budget, contextSize - tokens.count) {
        let token = Int32(Sampler.argmax(logits))
        if let qwen = tokenizer as? QwenTokenizer, qwen.stopTokens.contains(token) { break }
        if let glm = tokenizer as? GLM52Tokenizer, glm.isStopToken(token, reasoning: reasoning) { break }
        if let deep = tokenizer as? DeepSeekV4Tokenizer, token == deep.eosId { break }
        FileHandle.standardOutput.write(Data(tokenizer.tokenText(token)))
        generated += 1
        if index + 1 < min(budget, contextSize - tokens.count) {
            logits = try decoder.evaluate(tokens: [Int(token)], cancelled: { false })
        }
    }
    FileHandle.standardOutput.write(Data("\n".utf8))
    log(String(format: "DS4Demo: decode %d token in %.3fs", generated, Date().timeIntervalSince(decode)))
}
