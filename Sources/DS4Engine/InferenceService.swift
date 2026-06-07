import Foundation
import DS4Core
import DS4Metal

// DS4Engine: the GUI's inference service, now backed by the PURE-SWIFT engine
// (DS4Core tokenizer/GGUF + DS4Metal StreamingDecoder) instead of the C ds4.
// Same API surface ChatStore already consumes (InferenceService actor, send ->
// event stream, modelInfo, resetConversation) so the SwiftUI app is unchanged.
//
// Generation uses StreamingDecoder (per-layer load/compute/evict) so the 164GB
// model fits in 16GB. NOTE: the naive streaming path reloads every layer per
// token (slow); the expert-cache fast path is a follow-up. Multi-turn context
// reuse is also a follow-up — each send renders system + current user turn.

public enum ChatRole: Sendable { case system, user, assistant }

public enum DS4ThinkMode: Sendable {
    case none, high
    var core: ThinkMode { self == .high ? .high : .none }
}

public struct SamplingParams: Sendable {
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var minP: Float
    public var seed: UInt64
    public init(temperature: Float = 0.6, topK: Int = 0, topP: Float = 0.95, minP: Float = 0.05, seed: UInt64 = 0xD54) {
        self.temperature = temperature; self.topK = topK; self.topP = topP; self.minP = minP; self.seed = seed
    }
}

public struct StreamingOptions: Sendable {
    public var enabled: Bool
    public var cacheSpec: String?
    public init(enabled: Bool = true, cacheSpec: String? = nil) { self.enabled = enabled; self.cacheSpec = cacheSpec }
}

public struct ModelInfo: Sendable {
    public let name: String
    public let layers: Int
    public let nEmbd: Int
    public let nVocab: Int
    public let contextSize: Int
    public let routedQuantBits: Int
    public let kvCacheBytes: UInt64
}

public enum GenEvent: Sendable {
    case reasoning(String)
    case text(String)
}

public actor InferenceService {
    private let rt: MetalRuntime
    private let model: GGUFModel
    private let tok: Tokenizer
    private let decoder: StreamingDecoder
    private let dims: DSV4Dims
    private let contextSize: Int
    private let modelName: String
    private var systemPrompt: String?

    public init(modelPath: String, metalSourceDir: String? = nil, contextSize: Int,
                systemPrompt: String?, streaming: StreamingOptions,
                minimumRAMMode: Bool, perLayerStreaming: Bool) throws {
        // Kernels are embedded in the binary — no metal/ folder needed.
        self.rt = try MetalRuntime()
        self.model = try GGUFModel(path: modelPath, metalMapping: true, prefetchCPU: false)
        self.tok = try Tokenizer(model: model)
        // Configure the MoE/router quant scheme from the GGUF (Q4_K+Q8 vs IQ2_XXS/Q2_K+F16).
        var configuredDims = DSV4Shape.dims
        let mq = GGUFWeights.detectMoEQuant(model)
        configuredDims.gateQuant = mq.gate; configuredDims.upQuant = mq.up
        configuredDims.downQuant = mq.down; configuredDims.routerF16 = mq.routerF16
        self.dims = configuredDims
        self.contextSize = contextSize
        self.systemPrompt = systemPrompt
        self.modelName = (modelPath as NSString).lastPathComponent
        let rope = RopeParams(nCtxOrig: 4096, freqBase: 10000, freqScale: 1, extFactor: 0,
                              attnFactor: 1, betaFast: 32, betaSlow: 1)
        // Mapped-experts streaming: experts are no-copy mmap views over the full
        // expert tensors (all 256/layer); the GPU reads only the selected rows and
        // the OS page cache caches them across tokens — no per-token re-gather.
        self.decoder = try StreamingDecoder.fromGGUFMappedExperts(rt: rt, model: model, dims: dims, rope: rope,
                                                                  nLayers: DSV4Shape.nLayer, maxKeys: contextSize)
    }

    public func modelInfo() -> ModelInfo {
        // Raw KV cache footprint: nLayer x contextSize x headDim x F32.
        let kv = UInt64(DSV4Shape.nLayer) * UInt64(contextSize) * UInt64(dims.headDim) * 4
        return ModelInfo(name: modelName, layers: DSV4Shape.nLayer, nEmbd: dims.nEmbd,
                         nVocab: dims.vocab, contextSize: contextSize, routedQuantBits: 4, kvCacheBytes: kv)
    }

    public func resetConversation(systemPrompt: String?) {
        self.systemPrompt = systemPrompt
    }

    public func send(userText: String, thinkMode: DS4ThinkMode, sampling: SamplingParams,
                     maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.runGeneration(userText: userText, think: thinkMode,
                                                 sampling: sampling, maxTokens: maxTokens,
                                                 continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func runGeneration(userText: String, think: DS4ThinkMode, sampling: SamplingParams,
                               maxTokens: Int, continuation: AsyncThrowingStream<GenEvent, Error>.Continuation) throws {
        let prompt = tok.encodeChatPrompt(system: systemPrompt, prompt: userText, think: think.core)
        precondition(prompt.count < contextSize, "prompt exceeds context")

        // Prefill: feed each prompt token, accumulating the KV cache.
        var pos = 0
        var lastLogits: [Float] = []
        for t in prompt {
            try Task.checkCancellation()
            lastLogits = try decoder.forward(token: Int(t), pos: pos, nKeys: pos + 1)
            pos += 1
        }

        var rng = sampling.seed
        var inReasoning = think == .high       // prompt ends with <think> when enabled
        var pending: [UInt8] = []

        func flush(_ asReasoning: Bool) {
            guard !pending.isEmpty, let s = String(bytes: pending, encoding: .utf8) else { return }
            pending.removeAll(keepingCapacity: true)
            continuation.yield(asReasoning ? .reasoning(s) : .text(s))
        }

        var produced = 0
        while produced < maxTokens && pos < contextSize {
            try Task.checkCancellation()
            let next = Sampler.sample(lastLogits, temperature: sampling.temperature, topK: sampling.topK,
                                      topP: sampling.topP, minP: sampling.minP, rng: &rng)
            if Int32(next) == tok.eosId { break }
            if inReasoning && Int32(next) == tok.thinkEndId {
                flush(true)
                inReasoning = false
            } else {
                pending.append(contentsOf: tok.tokenText(Int32(next)))
                flush(inReasoning)
            }
            produced += 1
            lastLogits = try decoder.forward(token: next, pos: pos, nKeys: pos + 1)
            pos += 1
        }
        flush(inReasoning)
    }
}
