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

    /// `residentLayers`/`activeExperts` are the GUI settings (nil = env,
    /// then RAM-adaptive / full top-8).
    public init(modelPath: String,
                contextSize: Int,
                systemPrompt: String?,
                residentLayers: Int? = nil,
                activeExperts: Int? = nil) throws {
        let environment = ProcessInfo.processInfo.environment
        var options = GLM52ResidentModelOptions()
        options.cacheCapacity = max(256, contextSize)
        // RAM-adaptive residency (the biggest tok/s lever on streaming);
        // GUI setting first, DS4_GLM_RESIDENT_LAYERS next.
        options.residentLayerCount = residentLayers
            ?? environment["DS4_GLM_RESIDENT_LAYERS"].flatMap(Int.init)
            ?? GLM52ResidentModelOptions.adaptiveResidentLayerCount()
        options.activeExperts = activeExperts
            ?? environment["DS4_GLM_ACTIVE_EXPERTS"].flatMap(Int.init)
        if let slots = environment["DS4_GLM_EXPERT_SLOTS"].flatMap(Int.init) {
            options.expertSlotCount = slots
        }
        if let slots = environment["DS4_GLM_STREAM_SLOTS"]
            .flatMap(Int.init) {
            options.streamSlotCount = slots
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

    public struct BenchmarkNumbers: Sendable {
        public let prefillTps: Double
        public let genTps: Double
        public let report: String
    }

    /// Synthetic MEASUREMENT benchmark — the DeepSeek analog sweeps prefill
    /// knobs, but GLM v1 has none worth tuning in place, so this reports
    /// prefill/decode tok/s plus the per-phase streaming profile. Resets
    /// the engine context before and after (the next chat re-prefills).
    public func benchmark(contextTokens: Int, genTokens: Int) async throws
        -> BenchmarkNumbers {
        let service = self.service
        let limit = max(8, min(contextTokens, contextSize - genTokens - 1))
        primedTokens = []
        return try await Task.detached(priority: .userInitiated) {
            var tokens = service.tokenizer.tokenizeRenderedChat(
                "benchmark sintetico DwarfStar — misura di prefill e decode ")
            if tokens.isEmpty { tokens = [1] }
            while tokens.count < limit { tokens += tokens }
            tokens = Array(tokens.prefix(limit))
            service.engine.resetContext()
            service.engine.resetStreamingStats()
            let prefillStart = Date()
            var logits = try service.engine.prefill(tokens)
            let prefillSeconds = max(
                Date().timeIntervalSince(prefillStart), 0.001)
            let prefillReport = service.engine.streamingReport()
            service.engine.resetStreamingStats()
            let decodeStart = Date()
            var produced = 0
            for _ in 0..<genTokens {
                guard let token = GLM52GreedyDecoding.argmax(logits) else {
                    break
                }
                logits = try service.engine.forwardNext(token)
                produced += 1
            }
            let decodeSeconds = max(
                Date().timeIntervalSince(decodeStart), 0.001)
            let decodeReport = service.engine.streamingReport()
            service.engine.resetContext()
            return BenchmarkNumbers(
                prefillTps: Double(tokens.count) / prefillSeconds,
                genTps: Double(produced) / decodeSeconds,
                report: "prefill: " + prefillReport
                    + "\ndecode: " + decodeReport)
        }.value
    }

    public func setAgent(systemPrompt: String?) {
        self.systemPrompt = systemPrompt
    }

    /// Tools declared to the model (rendered into the native GLM XML tool
    /// prompt). The compact form is a DeepSeek-only optimization: accepted
    /// and ignored here.
    public func setTools(_ tools: [ToolSpec]) { self.tools = tools }
    public func setCompactTools(_ on: Bool) {}
    private var tools: [ToolSpec] = []

    public func committedTokens() -> Int { service.engine.position }

    /// Append a user message to the running transcript and generate.
    public func send(userText: String, thinkMode: DS4ThinkMode,
                     sampling: SamplingParams,
                     maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        transcript.append(.user(userText))
        return generate(turns: renderableTurns(), sampling: sampling,
                        maxTokens: maxTokens, thinkMode: thinkMode)
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
                        maxTokens: maxTokens, thinkMode: thinkMode)
    }

    // MARK: - Generation

    private func renderableTurns() -> [ChatTurn] {
        (systemPrompt.map { [ChatTurn.system($0)] } ?? []) + transcript
    }

    private func noteGeneration(fedTokens: [Int32], assistant: String) {
        primedTokens = fedTokens
        transcript.append(.assistant(text: assistant, toolCalls: []))
    }

    /// Splits the raw token stream into reasoning/text events on the GLM
    /// `<think>` markers, holding back a short tail so a marker split across
    /// token boundaries is never emitted as text.
    private struct StreamSplitter {
        var inThink = false
        var pending = ""
        mutating func feed(_ piece: String) -> [GenEvent] {
            pending += piece
            var events: [GenEvent] = []
            while true {
                let marker = inThink
                    ? GLM52ConversationProtocol.thinkClose
                    : GLM52ConversationProtocol.thinkOpen
                guard let range = pending.range(of: marker) else { break }
                let chunk = String(pending[..<range.lowerBound])
                if !chunk.isEmpty {
                    events.append(inThink ? .reasoning(chunk) : .text(chunk))
                }
                pending = String(pending[range.upperBound...])
                inThink.toggle()
            }
            let hold = 12
            if pending.count > hold {
                let cut = pending.index(pending.endIndex, offsetBy: -hold)
                let emit = String(pending[..<cut])
                pending = String(pending[cut...])
                if !emit.isEmpty {
                    events.append(inThink ? .reasoning(emit) : .text(emit))
                }
            }
            return events
        }
        mutating func flush() -> [GenEvent] {
            guard !pending.isEmpty else { return [] }
            let event: GenEvent = inThink
                ? .reasoning(pending) : .text(pending)
            pending = ""
            return [event]
        }
    }

    private func generate(turns: [ChatTurn],
                          sampling: SamplingParams,
                          maxTokens: Int,
                          thinkMode: DS4ThinkMode)
        -> AsyncThrowingStream<GenEvent, Error> {
        let service = self.service
        let contextSize = self.contextSize
        let primed = self.primedTokens
        let declaredTools = self.tools
        return AsyncThrowingStream { continuation in
            // Detached: the engine blocks on GPU/SSD waits and must not
            // occupy the cooperative pool (same discipline as the DeepSeek
            // actor's serial GCD executor).
            let producer = Task.detached(priority: .userInitiated) {
                do {
                    let rendered = try GLM52ChatRenderer.render(
                        turns: turns, tools: declaredTools,
                        reasoning: thinkMode.core)
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
                    var splitter = StreamSplitter()
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
                        if service.tokenizer.isStopToken(
                            token, reasoning: thinkMode.core) {
                            break
                        }
                        let bytes = service.tokenizer.tokenText(token)
                        reply.append(contentsOf: bytes)
                        if let piece = String(bytes: bytes,
                                              encoding: .utf8) {
                            for event in splitter.feed(piece) {
                                continuation.yield(event)
                            }
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
                    for event in splitter.flush() {
                        continuation.yield(event)
                    }
                    // Native GLM XML tool calls: parsed from the full reply;
                    // the GUI receives them as one .toolCall event (the
                    // automatic tool loop stays DeepSeek-only in this v1).
                    let raw = String(decoding: reply, as: UTF8.self)
                    let parsed = GLM52ToolCodec.parse(raw,
                                                      tools: declaredTools)
                    if !parsed.calls.isEmpty {
                        continuation.yield(.toolCall(parsed.calls))
                    }
                    await self.noteGeneration(fedTokens: fed,
                                              assistant: parsed.visibleText)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in producer.cancel() }
        }
    }
}
