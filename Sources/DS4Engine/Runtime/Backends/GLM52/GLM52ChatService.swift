import DS4Core
import DS4Metal
import Foundation

/// GLM 5.2 chat service for the GUI: mirrors the chat subset of the DeepSeek
/// `InferenceService` API (send / sendWithHistory / modelInfo / warmup /
/// quiesceForTeardown) over the GLM resident engine, so `ChatStore` can host
/// either backend.
///
/// Feature parity with the DeepSeek service so far: sampled decoding
/// (temperature, top-K, repetition penalty via `GLM52Sampler`), stoppable
/// generation (the stream's termination cancels the producer), incremental
/// KV between turns (only the rendered conversation's NEW suffix is
/// prefilled when it extends what the engine already holds — otherwise a
/// context reset and a full layer-major batched prefill). Still deliberate
/// v1 limits: no tool calls and no reasoning stream.
public actor GLM52ChatService {
    public let service: GLM52InferenceService
    private let contextSize: Int
    private var systemPrompt: String?
    private var transcript: [ChatTurn] = []
    /// Tokens the engine's caches currently hold, in order.
    private var primedTokens: [Int32] = []

    public init(modelPath: String,
                contextSize: Int,
                systemPrompt: String?) throws {
        let environment = ProcessInfo.processInfo.environment
        var options = GLM52ResidentModelOptions()
        options.cacheCapacity = max(256, contextSize)
        // RAM-adaptive residency (the biggest tok/s lever on streaming);
        // DS4_GLM_RESIDENT_LAYERS overrides.
        options.residentLayerCount = environment["DS4_GLM_RESIDENT_LAYERS"]
            .flatMap(Int.init)
            ?? GLM52ResidentModelOptions.adaptiveResidentLayerCount()
        options.activeExperts = environment["DS4_GLM_ACTIVE_EXPERTS"]
            .flatMap(Int.init)
        if let slots = environment["DS4_GLM_EXPERT_SLOTS"].flatMap(Int.init) {
            options.expertSlotCount = slots
        }
        service = try GLM52InferenceService(modelPath: modelPath,
                                            options: options)
        self.contextSize = max(256, contextSize)
        self.systemPrompt = systemPrompt
    }

    // MARK: - InferenceService-mirroring chat surface

    public func modelInfo() -> ModelInfo {
        let shape = service.engine.configuration.shape
        return ModelInfo(
            name: "GLM 5.2",
            layers: service.engine.loadedLayerCount,
            nEmbd: Int(shape.nEmbd),
            nVocab: Int(shape.nVocab),
            contextSize: contextSize,
            routedQuantBits: 2,
            kvCacheBytes: UInt64(95_232) * UInt64(contextSize),
            architecture: GLM52BackendDefinition.supportedArchitecture,
            displayName: "GLM 5.2 (glm-dsa)",
            quantizationSummary: "routed IQ2_XXS",
            capabilities: GLM52BackendDefinition.runtimeCapabilities)
    }

    public func warmup() -> Bool { true }

    public func quiesceForTeardown() async {}

    public func setAgent(systemPrompt: String?) {
        self.systemPrompt = systemPrompt
    }

    public func committedTokens() -> Int { service.engine.position }

    /// Append a user message to the running transcript and generate.
    public func send(userText: String, thinkMode: DS4ThinkMode,
                     sampling: SamplingParams,
                     maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        transcript.append(.user(userText))
        return generate(turns: renderableTurns(), sampling: sampling,
                        maxTokens: maxTokens)
    }

    /// Restore a persisted conversation and generate: the provided history
    /// replaces the running transcript. The incremental-KV prefix match
    /// still applies, so a reopened chat whose rendering extends the live
    /// caches does not pay a full re-prefill.
    public func sendWithHistory(_ history: [ChatTurn], userText: String,
                                systemPrompt: String?,
                                thinkMode: DS4ThinkMode,
                                sampling: SamplingParams,
                                maxTokens: Int)
        -> AsyncThrowingStream<GenEvent, Error> {
        if let systemPrompt { self.systemPrompt = systemPrompt }
        transcript = history + [.user(userText)]
        return generate(turns: renderableTurns(), sampling: sampling,
                        maxTokens: maxTokens)
    }

    // MARK: - Generation

    private func renderableTurns() -> [ChatTurn] {
        (systemPrompt.map { [ChatTurn.system($0)] } ?? []) + transcript
    }

    private func noteGeneration(fedTokens: [Int32], assistant: String) {
        primedTokens = fedTokens
        transcript.append(.assistant(text: assistant, toolCalls: []))
    }

    private func generate(turns: [ChatTurn],
                          sampling: SamplingParams,
                          maxTokens: Int)
        -> AsyncThrowingStream<GenEvent, Error> {
        let service = self.service
        let contextSize = self.contextSize
        let primed = self.primedTokens
        return AsyncThrowingStream { continuation in
            // Detached: the engine blocks on GPU/SSD waits and must not
            // occupy the cooperative pool (same discipline as the DeepSeek
            // actor's serial GCD executor).
            let producer = Task.detached(priority: .userInitiated) {
                do {
                    let rendered = try GLM52ChatRenderer.render(turns: turns)
                    let tokens = service.tokenizer
                        .tokenizeRenderedChat(rendered)
                    guard tokens.count + 1 < contextSize else {
                        throw GGUFError.cannotOpen(
                            "conversazione (\(tokens.count) token) oltre il "
                            + "contesto GLM configurato (\(contextSize))")
                    }

                    // Incremental KV: prefill only the suffix when the
                    // rendered conversation extends what the caches hold.
                    let common = zip(primed, tokens)
                        .prefix { $0 == $1 }.count
                    let incremental = common == primed.count
                        && common == service.engine.position
                        && common < tokens.count
                    let suffix: [Int32]
                    if incremental {
                        suffix = Array(tokens[common...])
                        continuation.yield(.progress(
                            "prefill incrementale \(suffix.count) token "
                            + "(+\(common) in cache)"))
                    } else {
                        service.engine.resetContext()
                        suffix = tokens
                        continuation.yield(.progress(
                            "prefill \(tokens.count) token (layer-major)"))
                    }
                    let prefillStart = Date()
                    var logits = try service.engine.prefill(suffix)
                    let prefillSeconds = max(
                        Date().timeIntervalSince(prefillStart), 0.001)
                    continuation.yield(.progress(String(
                        format: "prefill %d tok in %.1fs · %.2f tok/s",
                        suffix.count, prefillSeconds,
                        Double(suffix.count) / prefillSeconds)))

                    var fed = tokens
                    var produced = 0
                    var reply: [UInt8] = []
                    let decodeStart = Date()
                    let budget = min(maxTokens,
                                     contextSize - tokens.count - 1)
                    while produced < budget, !Task.isCancelled {
                        guard let token = GLM52Sampler.sample(
                            logits: logits,
                            temperature: sampling.temperature,
                            topK: sampling.topK,
                            repetitionPenalty: sampling.repetitionPenalty,
                            recentTokens: fed.suffix(
                                GLM52Sampler.penaltyWindow)) else { break }
                        produced += 1
                        if service.tokenizer.isStopToken(token,
                                                         reasoning: .none) {
                            break
                        }
                        let bytes = service.tokenizer.tokenText(token)
                        reply.append(contentsOf: bytes)
                        if let piece = String(bytes: bytes,
                                              encoding: .utf8) {
                            continuation.yield(.text(piece))
                        }
                        let elapsed = max(
                            Date().timeIntervalSince(decodeStart), 0.001)
                        continuation.yield(.progress(String(
                            format: "%d tok · %.2f tok/s",
                            produced, Double(produced) / elapsed)))
                        if produced == budget || Task.isCancelled { break }
                        logits = try service.engine.forwardNext(token)
                        fed.append(token)
                    }
                    let text = String(decoding: reply, as: UTF8.self)
                    await self.noteGeneration(fedTokens: fed,
                                              assistant: text)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in producer.cancel() }
        }
    }
}
