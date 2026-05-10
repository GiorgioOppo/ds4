import Foundation

public struct GenerationOptions {
    public var maxNewTokens: Int = 128
    public var temperature: Float = 0.0   // 0 = greedy

    public init(maxNewTokens: Int = 128, temperature: Float = 0.0) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
    }
}

public final class Generator {
    public let model: DeepSeekV4
    public let tokenizer: Tokenizer

    public init(model: DeepSeekV4, tokenizer: Tokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    public func generate(prompt: String,
                         options: GenerationOptions = GenerationOptions(),
                         onToken: (String) -> Void) {
        let promptIds = tokenizer.encode(prompt)
        let cache = CacheBank(numLayers: model.config.numHiddenLayers,
                              maxTokens: model.config.maxPositionEmbeddings,
                              kvHeads: model.config.numKeyValueHeads,
                              headDim: model.config.headDim)

        // Prefill: feed prompt tokens one at a time. Real prefill should batch
        // them through a single forward pass; the loop below is correct but
        // O(promptLen * decode-cost) — acceptable for the scaffold.
        var lastLogits: Tensor? = nil
        for id in promptIds {
            lastLogits = model.step(tokenId: id, cache: cache)
        }

        var nextId: Int
        for _ in 0..<options.maxNewTokens {
            let logits = lastLogits!
            Sampler.applyTemperature(logits, options.temperature)
            nextId = Sampler.argmax(logits)
            if nextId == (model.config.eosTokenId ?? -1) { break }
            onToken(tokenizer.decode([nextId]))
            lastLogits = model.step(tokenId: nextId, cache: cache)
        }
    }
}
