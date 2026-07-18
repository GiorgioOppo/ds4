import DS4Core
import DS4Metal
import Foundation

/// GLM 5.2 chat service for the GUI: mirrors the chat subset of the DeepSeek
/// `InferenceService` API (send / sendWithHistory / modelInfo / warmup /
/// quiesceForTeardown) over the GLM resident engine, so `ChatStore` can host
/// either backend. Deliberate v1 limits, stated where the GUI can see them:
/// greedy decoding (SamplingParams accepted, ignored), no tool calls, no
/// reasoning stream, and every generation re-prefills the rendered
/// conversation after a context reset (batched layer-major prefill — the
/// incremental KV suffix reuse of the DeepSeek path comes later).
public actor GLM52ChatService {
    public let service: GLM52InferenceService
    private let contextSize: Int
    private var systemPrompt: String?
    private var transcript: [ChatTurn] = []

    public init(modelPath: String,
                contextSize: Int,
                systemPrompt: String?) throws {
        let environment = ProcessInfo.processInfo.environment
        var options = GLM52ResidentModelOptions()
        options.cacheCapacity = max(256, contextSize)
        // Streaming defaults sized for consumer RAM: three resident dense
        // layers unless the user overrides.
        options.residentLayerCount = environment["DS4_GLM_RESIDENT_LAYERS"]
            .flatMap(Int.init) ?? 3
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
            quantizationSummary: "routed IQ2_XXS · greedy",
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
        return generate(turns: renderableTurns(), maxTokens: maxTokens)
    }

    /// Restore a persisted conversation and generate: the provided history
    /// replaces the running transcript.
    public func sendWithHistory(_ history: [ChatTurn], userText: String,
                                systemPrompt: String?,
                                thinkMode: DS4ThinkMode,
                                sampling: SamplingParams,
                                maxTokens: Int)
        -> AsyncThrowingStream<GenEvent, Error> {
        if let systemPrompt { self.systemPrompt = systemPrompt }
        transcript = history + [.user(userText)]
        return generate(turns: renderableTurns(), maxTokens: maxTokens)
    }

    // MARK: - Generation

    private func renderableTurns() -> [ChatTurn] {
        (systemPrompt.map { [ChatTurn.system($0)] } ?? []) + transcript
    }

    private func appendAssistant(_ text: String) {
        transcript.append(.assistant(text: text, toolCalls: []))
    }

    private func generate(turns: [ChatTurn],
                          maxTokens: Int)
        -> AsyncThrowingStream<GenEvent, Error> {
        let service = self.service
        let contextSize = self.contextSize
        return AsyncThrowingStream { continuation in
            // Detached: the engine blocks on GPU/SSD waits and must not
            // occupy the cooperative pool (same discipline as the DeepSeek
            // actor's serial GCD executor).
            Task.detached(priority: .userInitiated) {
                do {
                    let rendered = try GLM52ChatRenderer.render(turns: turns)
                    let tokens = service.tokenizer
                        .tokenizeRenderedChat(rendered)
                    guard tokens.count + 1 < contextSize else {
                        throw GGUFError.cannotOpen(
                            "conversazione (\(tokens.count) token) oltre il "
                            + "contesto GLM configurato (\(contextSize))")
                    }
                    continuation.yield(.progress(
                        "prefill \(tokens.count) token (layer-major)"))
                    service.engine.resetContext()
                    var logits = try service.engine.prefill(tokens)
                    var produced = 0
                    var reply: [UInt8] = []
                    let budget = min(maxTokens,
                                     contextSize - tokens.count - 1)
                    while produced < budget {
                        guard let token = GLM52GreedyDecoding.argmax(logits)
                        else { break }
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
                        if produced % 8 == 0 {
                            continuation.yield(.progress(
                                "\(produced) token generati"))
                        }
                        if produced == budget { break }
                        logits = try service.engine.forwardNext(token)
                    }
                    let text = String(decoding: reply, as: UTF8.self)
                    await self.appendAssistant(text)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
