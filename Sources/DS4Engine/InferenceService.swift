import Foundation
import DS4Core
import DS4Metal

// DS4Engine: the GUI's inference service, backed by the PURE-SWIFT engine
// (DS4Core tokenizer/GGUF + DS4Metal StreamingDecoder).
//
// Generation uses StreamingDecoder (per-layer load/compute/evict) so the 164GB
// model fits in 16GB. The service keeps a multi-turn conversation (`turns`) and
// re-renders the whole transcript each generation (no cross-turn KV reuse yet),
// which is what tool calling needs: a tool result is a new turn that continues
// the same conversation. Tool calls are rendered/parsed via DS4Core.ChatRenderer
// / ToolCallParser using the model's own markup tokens (ToolMarkup.discover).

public enum ChatRole: Sendable { case system, user, assistant, tool }

public enum InferenceError: Error, CustomStringConvertible {
    case contextExceeded(prompt: Int, context: Int)
    public var description: String {
        switch self {
        case let .contextExceeded(p, c):
            return "la conversazione (\(p) token) supera il contesto (\(c)). Inizia una nuova chat o aumenta il contesto."
        }
    }
}

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
    case toolCall([ToolCall])   // the model requested one or more tools; generation paused
    case progress(String)       // prefill/decode status (e.g. "prefill 3/11" or "12 tok · 1.4 tok/s")
}

public actor InferenceService {
    private let rt: MetalRuntime
    private let model: GGUFModel
    private let tok: Tokenizer
    private let decoder: StreamingDecoder
    private let dims: DSV4Dims
    private let contextSize: Int
    private let modelName: String
    private let markup: ToolMarkup

    // Conversation state (rendered in full on each generation).
    private var turns: [ChatTurn] = []
    private var tools: [ToolSpec] = []

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
        // Optional active-experts override (DS4_ACTIVE_EXPERTS=2..6): fewer experts
        // per token = less expert I/O, lower quality. Honored by the streaming path.
        if let s = ProcessInfo.processInfo.environment["DS4_ACTIVE_EXPERTS"], let kk = Int(s) {
            configuredDims.activeExperts = max(1, min(kk, configuredDims.k))
        }
        self.dims = configuredDims
        self.contextSize = contextSize
        self.modelName = (modelPath as NSString).lastPathComponent
        self.markup = ToolMarkup.discover(in: tok)
        if let systemPrompt, !systemPrompt.isEmpty { self.turns = [.system(systemPrompt)] }
        let rope = RopeParams(nCtxOrig: 4096, freqBase: 10000, freqScale: 1, extFactor: 0,
                              attnFactor: 1, betaFast: 32, betaSlow: 1)
        // Fast 16GB path (C --ssd-streaming model): non-routed weights are no-copy mmap
        // (resident via page cache, evictable), only the 6 selected experts gathered/token.
        self.decoder = try StreamingDecoder.fromGGUFExpertCachedMapped(rt: rt, model: model, dims: dims, rope: rope,
                                                                       nLayers: DSV4Shape.nLayer, maxKeys: contextSize)
    }

    public func modelInfo() -> ModelInfo {
        // Raw KV cache footprint: nLayer x contextSize x headDim x F32.
        let kv = UInt64(DSV4Shape.nLayer) * UInt64(contextSize) * UInt64(dims.headDim) * 4
        return ModelInfo(name: modelName, layers: DSV4Shape.nLayer, nEmbd: dims.nEmbd,
                         nVocab: dims.vocab, contextSize: contextSize, routedQuantBits: 4, kvCacheBytes: kv)
    }

    /// The raw Jinja chat template embedded in the GGUF (for inspection), if any.
    public func chatTemplate() -> String? { model.string("tokenizer.chat_template") }

    /// Per-phase decode timing (route/attn vs expert gather I/O vs experts compute…).
    public func resetDecodeProfile() { decoder.resetProfile() }
    public func decodeProfileReport() -> String { decoder.profile.report() }

    public func resetConversation(systemPrompt: String?) {
        if let systemPrompt, !systemPrompt.isEmpty { turns = [.system(systemPrompt)] }
        else { turns = [] }
    }

    /// Declare the tools available to the model for subsequent turns.
    public func setTools(_ tools: [ToolSpec]) { self.tools = tools }

    /// Append a user message and generate the assistant reply.
    public func send(userText: String, thinkMode: DS4ThinkMode, sampling: SamplingParams,
                     maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        turns.append(.user(userText))
        return run(think: thinkMode, sampling: sampling, maxTokens: maxTokens)
    }

    /// Append tool results and continue the assistant turn (the model now sees the
    /// outputs of the tools it called and produces its final answer — or more calls).
    public func provideToolResults(_ outputs: [ToolOutput], thinkMode: DS4ThinkMode,
                                   sampling: SamplingParams, maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        for o in outputs { turns.append(.toolResult(callId: o.callId, name: o.name, content: o.content)) }
        return run(think: thinkMode, sampling: sampling, maxTokens: maxTokens)
    }

    private func run(think: DS4ThinkMode, sampling: SamplingParams, maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.runGeneration(think: think, sampling: sampling, maxTokens: maxTokens,
                                                 continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func runGeneration(think: DS4ThinkMode, sampling: SamplingParams, maxTokens: Int,
                               continuation: AsyncThrowingStream<GenEvent, Error>.Continuation) throws {
        let rendered = ChatRenderer.render(turns: turns, tools: tools, think: think.core, markup: markup)
        let prompt = tok.tokenizeRenderedChat(rendered)
        guard prompt.count < contextSize else {
            throw InferenceError.contextExceeded(prompt: prompt.count, context: contextSize)
        }

        // Prefill: LAYER-MAJOR — load each layer's weights once and apply to all
        // prompt tokens at once (amortizes the dominant weight I/O). Returns the
        // last token's logits; populates the KV cache for positions 0..N-1.
        continuation.yield(.progress("prefill \(prompt.count) token…"))
        var lastLogits = try decoder.prefill(tokens: prompt.map { Int($0) })
        var pos = prompt.count

        var rng = sampling.seed
        var inReasoning = think == .high       // prompt ends with <think> when enabled
        var inTool = false
        var pending: [UInt8] = []
        var visible = ""                       // accumulated visible answer (for history)
        var toolBytes: [UInt8] = []            // raw bytes of the tool-call block (after callsBegin)
        let callsBeginId = tok.tokenId(markup.callsBegin)

        func flush(_ asReasoning: Bool) {
            guard !pending.isEmpty, let s = String(bytes: pending, encoding: .utf8) else { return }
            pending.removeAll(keepingCapacity: true)
            if asReasoning { continuation.yield(.reasoning(s)) }
            else { visible += s; continuation.yield(.text(s)) }
        }

        var produced = 0
        let genStart = Date()
        while produced < maxTokens && pos < contextSize {
            try Task.checkCancellation()
            let next = Sampler.sample(lastLogits, temperature: sampling.temperature, topK: sampling.topK,
                                      topP: sampling.topP, minP: sampling.minP, rng: &rng)
            if Int32(next) == tok.eosId { break }
            if !inTool, let cb = callsBeginId, Int32(next) == cb {
                flush(inReasoning)             // emit any buffered text/reasoning, then enter the tool block
                inTool = true
            } else if inTool {
                toolBytes.append(contentsOf: tok.tokenText(Int32(next)))
            } else if inReasoning && Int32(next) == tok.thinkEndId {
                flush(true)
                inReasoning = false
            } else {
                pending.append(contentsOf: tok.tokenText(Int32(next)))
                flush(inReasoning)
            }
            produced += 1
            lastLogits = try decoder.forward(token: next, pos: pos, nKeys: pos + 1)
            pos += 1
            let elapsed = Date().timeIntervalSince(genStart)
            continuation.yield(.progress(String(format: "%d token · %.2f tok/s", produced,
                                                 elapsed > 0 ? Double(produced) / elapsed : 0)))
        }
        flush(inReasoning)

        // Finalize the assistant turn: extract any tool calls and record history.
        var calls: [ToolCall] = []
        var finalVisible = visible
        if inTool {
            let block = markup.callsBegin + (String(bytes: toolBytes, encoding: .utf8) ?? "")
            calls = ToolCallParser.parse(block, markup: markup).calls
        } else {
            // Fallback: the markup may have streamed as plain text (model without a
            // single-token markup). Split it back out of the visible answer.
            let (c, v) = ToolCallParser.parse(visible, markup: markup)
            if !c.isEmpty { calls = c; finalVisible = v }
        }
        turns.append(.assistant(text: finalVisible, toolCalls: calls))
        if !calls.isEmpty { continuation.yield(.toolCall(calls)) }
        continuation.yield(.progress(""))   // clear status when done
    }
}
