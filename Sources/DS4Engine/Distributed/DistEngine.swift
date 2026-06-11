import Foundation
import DS4Core
import DS4Metal

/// Node-level engine for distributed inference. Builds the same streaming decoder
/// as `InferenceService` but exposes the low-level slice ops (embed / forwardSlice
/// / head) the coordinator and workers drive directly, plus tokenization and
/// sampling for the coordinator. Stateless w.r.t. conversation: the coordinator
/// sequences positions explicitly across the cluster.
///
/// Memory note: layer weights are no-copy mmap loaded **lazily** per layer, so a
/// worker that only calls `forwardSlice` over its range never faults in the other
/// layers' weights — that is where the per-node memory saving comes from.
public final class DistEngine: @unchecked Sendable {
    public let modelName: String
    public let nLayers: Int
    public let contextSize: Int
    private let rt: MetalRuntime
    private let model: GGUFModel
    private let tok: Tokenizer
    private let decoder: StreamingDecoder

    public init(modelPath: String, contextSize: Int, expertCacheSlots: Int? = nil) throws {
        self.rt = try MetalRuntime()
        self.model = try GGUFModel(path: modelPath, metalMapping: true, prefetchCPU: false)
        self.tok = try Tokenizer(model: model)
        var dims = DSV4Shape.dims
        let mq = GGUFWeights.detectMoEQuant(model)
        dims.gateQuant = mq.gate; dims.upQuant = mq.up; dims.downQuant = mq.down; dims.routerF16 = mq.routerF16
        self.contextSize = contextSize
        self.nLayers = DSV4Shape.nLayer
        self.modelName = (modelPath as NSString).lastPathComponent
        let rope = RopeParams(nCtxOrig: 4096, freqBase: 10000, freqScale: 1, extFactor: 0,
                              attnFactor: 1, betaFast: 32, betaSlow: 1)
        self.decoder = try StreamingDecoder.fromGGUFExpertCachedMapped(
            rt: rt, model: model, dims: dims, rope: rope, nLayers: DSV4Shape.nLayer,
            maxKeys: contextSize, cacheSlots: expertCacheSlots)
    }

    /// HC state width crossing the wire (nHC * nEmbd floats).
    public var hcStateCount: Int { decoder.hcStateCount }

    // MARK: Slice ops (delegate to the decoder)

    public func embed(token: Int, pos: Int) throws -> [Float] {
        try decoder.embed(token: token, pos: pos)
    }

    public func forwardSlice(hc: [Float], pos: Int, nKeys: Int, start: Int, end: Int) throws -> [Float] {
        try decoder.forwardSlice(hc: hc, pos: pos, nKeys: nKeys, start: start, end: end)
    }

    public func head(hc: [Float]) throws -> [Float] {
        try decoder.head(hc: hc)
    }

    // MARK: Coordinator helpers

    /// Tokenize a rendered chat / prompt string into token ids.
    public func tokenize(_ text: String) -> [Int] {
        tok.tokenizeRenderedChat(text).map { Int($0) }
    }

    /// Build the token ids for a single-turn chat prompt (BOS + system + user + assistant open).
    public func chatPromptIds(system: String?, prompt: String) -> [Int] {
        tok.encodeChatPrompt(system: system, prompt: prompt, think: .none).map { Int($0) }
    }

    public func tokenText(_ id: Int) -> String {
        String(bytes: tok.tokenText(Int32(id)), encoding: .utf8) ?? ""
    }

    public var eosId: Int { Int(tok.eosId) }

    /// Sample the next token id from a logits vector.
    public func sample(_ logits: [Float], params: SamplingParams, rng: inout UInt64) -> Int {
        Sampler.sample(logits, temperature: params.temperature, topK: params.topK,
                       topP: params.topP, minP: params.minP, rng: &rng)
    }
}
