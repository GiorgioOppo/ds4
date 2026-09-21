import DS4Core
import DS4Metal
import Foundation

/// Chat and OpenAI-compatible API adapter for the native Swift model drivers.
/// The dedicated serial executor owns every decoder operation, including reset,
/// warmup and cancellation cleanup; Metal/SSD waits never occupy the main actor.
public actor SwiftModelChatService: ChatBackend {
    nonisolated let engineQueue = DispatchSerialQueue(label: "ds4.native-model", qos: .userInitiated)
    public nonisolated var unownedExecutor: UnownedSerialExecutor { engineQueue.asUnownedSerialExecutor() }
    private let model: GGUFModel?
    private let decoder: any SwiftModelDecoder
    private let tokenizer: any TokenizerProtocol
    private nonisolated let info: ModelInfo
    private nonisolated let generation = NativeGenerationGate()
    private var systemPrompt: String?
    private var transcript: [ChatTurn] = []
    private var tools: [ToolSpec] = []
    private var compactTools = false
    private var primed: [Int] = []

    public init(modelPath: String, contextSize: Int, systemPrompt: String?) throws {
        let model = try GGUFModel(path: modelPath, metalMapping: true, prefetchCPU: false)
        let descriptor = try ModelInspector.inspect(model)
        let tokenizer = try TokenizerFactory.make(for: model)
        let decoder: any SwiftModelDecoder
        switch descriptor.architecture {
        case .bonsai2: decoder = try BonsaiModel(model: model, contextSize: contextSize)
        case .qwen38FlashNext: decoder = try Qwen38Model(model: model, contextSize: contextSize)
        case .deepSeekV41: decoder = try DeepSeek41Model(model: model, contextSize: contextSize)
        case .glm53Flash: decoder = try GLM53Model(model: model, contextSize: contextSize)
        default: throw ModelArchitectureError.unsupportedArchitecture(descriptor.architecture)
        }
        guard decoder.vocabularySize == tokenizer.nVocab else {
            throw SwiftModelDecoderError.invalidInput("Il vocabolario non corrisponde al decoder.")
        }
        self.model = model; self.decoder = decoder; self.tokenizer = tokenizer
        self.systemPrompt = systemPrompt
        let types = Set(model.tensors.filter { !$0.name.contains("engram_embd") && !$0.name.contains("ngram") }.map(\.typeName)).sorted()
        info = ModelInfo(name: descriptor.displayName, layers: descriptor.layerCount ?? 0,
                         nEmbd: descriptor.embeddingLength ?? 0, nVocab: tokenizer.nVocab,
                         contextSize: decoder.contextCapacity, routedQuantBits: 0, kvCacheBytes: 0,
                         architecture: descriptor.architecture, displayName: descriptor.displayName,
                         quantizationSummary: types.joined(separator: ", "),
                         capabilities: [.generation, .reasoning, .tools])
    }

    // Injection boundary used by state-machine tests, without model weights.
    init(decoder: sending any SwiftModelDecoder, tokenizer: sending any TokenizerProtocol,
         info: ModelInfo) {
        model = nil; self.decoder = decoder; self.tokenizer = tokenizer; self.info = info
    }
    public nonisolated func modelInfo() -> ModelInfo { info }
    public func committedTokens() -> Int { primed.count }
    public func setAgent(_ agent: AgentProfile, tools: [ToolSpec]) { systemPrompt = agent.systemPrompt; self.tools = tools }
    public func setTools(_ tools: [ToolSpec]) { self.tools = tools }
    public func setCompactTools(_ on: Bool) { compactTools = on }

    public func warmup() async -> Bool {
        guard let ticket = try? generation.begin() else { return false }
        defer { try? decoder.reset(); primed = []; generation.finish(ticket) }
        do {
            try decoder.reset()
            guard let token = tokenizer.tokenize("ciao").first else { return false }
            _ = try decoder.evaluate(tokens: [Int(token)], cancelled: { [generation] in
                Task.isCancelled || generation.isCancelled(ticket)
            })
            return true
        } catch { DS4Log.info("native", "warmup fallito: \(error)"); return false }
    }
    public nonisolated func quiesceForTeardown() async {
        // This must run outside the decoder's actor: a synchronous Metal/SSD
        // pass occupies that actor until it observes the cancellation flag.
        generation.retireAndCancel()
        await drainAndReset()
    }
    private func drainAndReset() {
        try? decoder.reset(); primed = []
    }

    public nonisolated func send(userText: String, thinkMode: DS4ThinkMode, sampling: SamplingParams,
                     maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        generate(request: .user(userText),
                 thinkMode: thinkMode, sampling: sampling, maxTokens: maxTokens)
    }
    public nonisolated func sendWithHistory(_ history: [ChatTurn], userText: String, systemPrompt: String?,
                                thinkMode: DS4ThinkMode, sampling: SamplingParams,
                                maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        generate(request: .history(history, userText, systemPrompt),
                 thinkMode: thinkMode, sampling: sampling, maxTokens: maxTokens)
    }
    public nonisolated func provideToolResults(_ outputs: [ToolOutput], thinkMode: DS4ThinkMode,
                                   sampling: SamplingParams, maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        generate(request: .results(outputs), thinkMode: thinkMode, sampling: sampling, maxTokens: maxTokens)
    }
    public nonisolated func complete(turns: [ChatTurn], tools: [ToolSpec], thinkMode: DS4ThinkMode,
                         sampling: SamplingParams, maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        generate(request: .complete(turns, tools), thinkMode: thinkMode,
                 sampling: sampling, maxTokens: maxTokens)
    }

    private func render(_ turns: [ChatTurn], tools: [ToolSpec], reasoning: ThinkMode) throws -> String {
        try ToolHistoryValidator.validate(turns)
        switch info.architecture {
        case .bonsai2, .qwen38FlashNext:
            return try QwenChatRenderer.render(turns: turns, tools: tools, architecture: info.architecture, reasoning: reasoning)
        case .glm53Flash: return try GLM52ChatRenderer.render(turns: turns, tools: tools, reasoning: reasoning)
        case .deepSeekV41: return try DeepSeek41ChatRenderer.render(turns: turns, tools: tools, reasoning: reasoning, compactTools: compactTools)
        default: throw ModelArchitectureError.unsupportedArchitecture(info.architecture)
        }
    }
    private func isStop(_ token: Int32, reasoning: ThinkMode) -> Bool {
        if let tok = tokenizer as? QwenTokenizer { return tok.stopTokens.contains(token) }
        if let tok = tokenizer as? GLM52Tokenizer { return tok.isStopToken(token, reasoning: reasoning) }
        if let tok = tokenizer as? DeepSeekV4Tokenizer { return token == tok.eosId }
        return false
    }
    private func parse(_ text: String, tools: [ToolSpec]) throws -> (calls: [ToolCall], visibleText: String) {
        let result: (calls: [ToolCall], visibleText: String)
        switch info.architecture {
        case .bonsai2, .qwen38FlashNext:
            let parsed = try QwenToolCodec.parseStrict(text, tools: tools)
            result = (parsed.calls, parsed.visibleText)
        case .glm53Flash:
            let parsed = try GLM52ToolCodec.parseStrict(text, tools: tools)
            result = (parsed.calls, parsed.visibleText)
        default: result = try ToolCallParser.parseStrict(text, markup: .dsv4)
        }
        let allowed = Set(tools.map(\.name))
        guard result.calls.allSatisfy({ allowed.contains($0.name) }) else {
            throw SwiftModelDecoderError.invalidInput("Il modello ha richiesto un tool non dichiarato.")
        }
        // IDs must remain unique across all rounds of one transcript.
        let prefix = UUID().uuidString
        return (result.calls.enumerated().map { index, call in
            ToolCall(id: "call_\(prefix)_\(index)", name: call.name, argumentsJSON: call.argumentsJSON)
        }, result.visibleText)
    }
    private nonisolated func generate(request: NativeChatRequest,
                          thinkMode: DS4ThinkMode, sampling: SamplingParams,
                          maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        let (stream, continuation) = AsyncThrowingStream<GenEvent, Error>.makeStream()
        let ticket: UUID
        do { ticket = try generation.begin() }
        catch {
            continuation.finish(throwing: error); return stream
        }
        let task = Task {
            await self.run(request: request, ticket: ticket, thinkMode: thinkMode,
                     sampling: sampling, maxTokens: maxTokens, continuation: continuation)
        }
        continuation.onTermination = { [generation] _ in
            generation.cancel(ticket)
            task.cancel()
        }
        return stream
    }
    private func run(request: NativeChatRequest, ticket: UUID,
                     thinkMode: DS4ThinkMode, sampling: SamplingParams, maxTokens: Int,
                     continuation: AsyncThrowingStream<GenEvent, Error>.Continuation) {
        defer { generation.finish(ticket) }
        let cancelled: @Sendable () -> Bool = { [generation] in
            Task.isCancelled || generation.isCancelled(ticket)
        }
        func checkCancellation() throws {
            if cancelled() { throw CancellationError() }
        }
        do {
            try checkCancellation()
            guard maxTokens > 0 else { generation.finish(ticket); continuation.finish(); return }
            let history: [ChatTurn], system: String?, tools: [ToolSpec]
            switch request {
            case .user(let text):
                history = transcript + [.user(text)]; system = systemPrompt; tools = self.tools
            case .history(let turns, let text, let prompt):
                history = turns + [.user(text)]; system = prompt ?? systemPrompt; tools = self.tools
            case .results(let outputs):
                history = transcript + outputs.map { .toolResult(callId: $0.callId, name: $0.name, content: $0.content) }
                system = systemPrompt; tools = self.tools
            case .complete(let turns, let declaredTools):
                history = turns; system = nil; tools = declaredTools
            }
            let turns = (system.map { [ChatTurn.system($0)] } ?? []) + history
            let rendered = try render(turns, tools: tools, reasoning: thinkMode.core)
            let tokens = tokenizer.tokenizeRenderedChat(rendered).map(Int.init)
            guard !tokens.isEmpty, tokens.count < decoder.contextCapacity else {
                throw InferenceError.contextExceeded(prompt: tokens.count, context: decoder.contextCapacity)
            }
            let common = zip(primed, tokens).prefix { $0 == $1 }.count
            let canAppend = !primed.isEmpty && common == primed.count && common == decoder.position && common < tokens.count
            if !canAppend { try decoder.reset(); primed = [] }
            let suffix = Array(tokens.dropFirst(canAppend ? common : 0))
            continuation.yield(.progress("Prefill \(suffix.count) token" + (canAppend ? " (+\(common) in cache)" : "")))
            let start = Date()
            var logits = try decoder.evaluate(tokens: suffix, cancelled: cancelled)
            let prefillSeconds = max(Date().timeIntervalSince(start), 0.001)
            let summary = String(format: "prefill %d tok in %.1fs · %.2f tok/s", suffix.count, prefillSeconds, Double(suffix.count) / prefillSeconds)
            continuation.yield(.progress(summary))
            var fed = tokens, visibleRaw = "", rng = sampling.seed
            var splitter = GLM52ChatService.StreamSplitter(startsInThink: thinkMode.core.enabled)
            var toolStream = NativeToolTextStream(marker: info.architecture == .deepSeekV41 ? ToolMarkup.dsv4.callsOpen : "<tool_call>")
            func emit(_ events: [GenEvent]) {
                for event in events {
                    if case .text(let text) = event {
                        visibleRaw += text
                        for output in toolStream.feed(text) { continuation.yield(output) }
                    } else { continuation.yield(event) }
                }
            }
            let budget = min(maxTokens, decoder.contextCapacity - tokens.count)
            let decodeStart = Date()
            for index in 0..<budget {
                try checkCancellation()
                guard logits.count == decoder.vocabularySize, logits.contains(where: \.isFinite) else {
                    throw SwiftModelDecoderError.invalidState
                }
                let next = Sampler.sample(logits, temperature: sampling.temperature,
                                          topK: sampling.topK, topP: sampling.topP, minP: sampling.minP,
                                          repetitionPenalty: sampling.repetitionPenalty,
                                          recent: fed.suffix(max(0, sampling.repeatLastN)), rng: &rng)
                let token = Int32(next)
                if isStop(token, reasoning: thinkMode.core) { break }
                emit(splitter.feed(tokenizer.tokenText(token)))
                let elapsed = max(Date().timeIntervalSince(decodeStart), 0.001)
                continuation.yield(.progress(summary + String(format: " · decode %d tok · %.2f tok/s", index + 1, Double(index + 1) / elapsed)))
                if index + 1 < budget {
                    logits = try decoder.evaluate(tokens: [next], cancelled: cancelled)
                    fed.append(next)
                }
            }
            try checkCancellation()
            emit(splitter.flush())
            for event in toolStream.flush() { continuation.yield(event) }
            let parsed = try parse(visibleRaw, tools: tools)
            if !parsed.calls.isEmpty { continuation.yield(.toolCall(parsed.calls)) }
            primed = fed
            transcript = history + [.assistant(text: parsed.visibleText, toolCalls: parsed.calls)]
            self.systemPrompt = system; self.tools = tools
            generation.finish(ticket)
            continuation.finish()
        } catch {
            try? decoder.reset(); primed = []
            generation.finish(ticket)
            continuation.finish(throwing: error)
        }
    }
}

private enum NativeChatRequest: Sendable {
    case user(String)
    case history([ChatTurn], String, String?)
    case results([ToolOutput])
    case complete([ChatTurn], [ToolSpec])
}

/// Only admission/cancellation crosses executors. Decoder, tokenizer and
/// conversation state remain owned exclusively by SwiftModelChatService.
private final class NativeGenerationGate: @unchecked Sendable {
    private let lock = NSLock()
    private var active: UUID?
    private var cancelled = false
    private var retired = false

    func begin() throws -> UUID {
        lock.lock(); defer { lock.unlock() }
        guard !retired else { throw SwiftModelDecoderError.invalidState }
        guard active == nil else {
            throw SwiftModelDecoderError.invalidInput("Il modello sta già generando una risposta.")
        }
        let ticket = UUID(); active = ticket; cancelled = false
        return ticket
    }
    func cancel(_ ticket: UUID) {
        lock.lock(); defer { lock.unlock() }
        if active == ticket { cancelled = true }
    }
    func isCancelled(_ ticket: UUID) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return retired || active != ticket || cancelled
    }
    func finish(_ ticket: UUID) {
        lock.lock(); defer { lock.unlock() }
        if active == ticket { active = nil; cancelled = false }
    }
    func retireAndCancel() {
        lock.lock(); defer { lock.unlock() }
        retired = true; cancelled = true
    }
}

/// Hold only a possible delimiter prefix, so ordinary text streams immediately.
struct NativeToolTextStream {
    let marker: String
    private var pending = ""
    private var inTool = false
    init(marker: String) { self.marker = marker }
    mutating func feed(_ text: String) -> [GenEvent] {
        if inTool { return text.isEmpty ? [] : [.toolStream(text)] }
        pending += text
        if let range = pending.range(of: marker) {
            var events: [GenEvent] = []
            let visible = String(pending[..<range.lowerBound])
            if !visible.isEmpty { events.append(.text(visible)) }
            events.append(.toolStream(String(pending[range.lowerBound...])))
            pending = ""; inTool = true; return events
        }
        var hold = min(pending.count, marker.count - 1)
        while hold > 0 && pending.suffix(hold) != marker.prefix(hold) { hold -= 1 }
        let split = pending.index(pending.endIndex, offsetBy: -hold)
        let visible = String(pending[..<split]); pending = String(pending[split...])
        return visible.isEmpty ? [] : [.text(visible)]
    }
    mutating func flush() -> [GenEvent] {
        defer { pending = "" }
        return pending.isEmpty ? [] : [inTool ? .toolStream(pending) : .text(pending)]
    }
}
