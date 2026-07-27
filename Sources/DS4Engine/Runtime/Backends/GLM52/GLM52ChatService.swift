import DS4Core
import DS4Metal
import Foundation

/// GLM 5.2 chat service for the GUI: mirrors the chat subset of the DeepSeek
/// `InferenceService` API (send / sendWithHistory / modelInfo / warmup /
/// quiesceForTeardown) over the GLM resident engine, so `ChatStore` can host
/// either backend.
///
/// Feature parity with the DeepSeek service so far: sampled decoding
/// (temperature, top-K, top-P, min-P, repetition penalty and a seedable
/// C-compatible RNG via `GLM52Sampler`, which now delegates to the shared
/// DS4Core `Sampler`), stoppable
/// generation (the stream's termination cancels the producer), incremental
/// KV between turns (only the rendered conversation's NEW suffix is
/// prefilled when it extends what the engine already holds — otherwise a
/// context reset and a full layer-major batched prefill), the reasoning
/// stream (`<think>` split on special-token IDs) and native XML tool calls
/// with observation turns, and the disk-KV store (prefix-keyed checkpoints
/// with token budget and eviction, `GLM52DiskKVStore`). Not mirrored (by
/// design): decode profiles, sub-agents, distributed execution.
public actor GLM52ChatService {
    public let service: GLM52InferenceService
    private let contextSize: Int
    private var systemPrompt: String?
    private var transcript: [ChatTurn] = []
    /// Tokens the engine's caches currently hold, in order.
    private var primedTokens: [Int32] = []
    /// The engine is SINGLE-DRIVER: producer, warmup e benchmark girano
    /// detached (bloccano su GPU/SSD) e quindi FUORI dall'actor — ogni
    /// nuovo passo attende qui il completamento del precedente. Senza
    /// questa catena, un warmup lanciato durante una generazione (visto in
    /// campo: il toggle think ri-applica l'agente) interleava le prefetch
    /// di due passi nello stesso streamer.
    private var lastEnginePass: Task<Void, Never>?

    /// Disk-KV store (nil = disabled): per-model scoped directory of
    /// prefix-keyed checkpoints with token budget and eviction, like the
    /// DeepSeek `DiskKVStore` — reopening ANY recent chat restores its
    /// longest stored prefix instead of re-prefilling from zero.
    private let kvStore: GLM52DiskKVStore?

    /// `residentLayers`/`activeExperts` are the GUI settings (nil = env,
    /// then RAM-adaptive / full top-8). `diskKVBudgetTokens` caps the
    /// checkpoint store (0 = byte safety cap only).
    public init(modelPath: String,
                contextSize: Int,
                systemPrompt: String?,
                residentLayers: Int? = nil,
                activeExperts: Int? = nil,
                diskKVDirectory: String? = nil,
                diskKVBudgetTokens: Int = 0) throws {
        if let diskKVDirectory {
            let file = (modelPath as NSString).lastPathComponent
            let attributes = try? FileManager.default
                .attributesOfItem(atPath: modelPath)
            let size = (attributes?[.size] as? UInt64) ?? 0
            let scoped = URL(fileURLWithPath: diskKVDirectory)
                .appendingPathComponent("\(file)-\(size)", isDirectory: true)
            let store = try? GLM52DiskKVStore(
                directory: scoped, budgetTokens: diskKVBudgetTokens)
            // Pre-store upgrade: the old single live checkpoint becomes a
            // regular entry, so the caches it holds survive the migration.
            store?.adoptLegacyCheckpoint(at: scoped.appendingPathComponent(
                GLM52DiskKVStore.legacyFileName))
            kvStore = store
        } else {
            kvStore = nil
        }
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

    /// REAL warmup: one full 1-token pass through the streamed stack — file
    /// handles, residual pipeline compilations, arena and staging slots all
    /// get touched, so the first user token stops paying the one-time
    /// costs. Context and telemetry are reset right after.
    public func warmup() async -> Bool {
        let service = self.service
        primedTokens = []
        let prior = lastEnginePass
        let pass = Task.detached(priority: .userInitiated) {
            await prior?.value
            let probe = service.tokenizer
                .tokenizeRenderedChat("ciao").first ?? 1
            // Il reset di ripristino DEVE correre anche su errore: un
            // prefill fallito a metà lascia righe appese in alcuni layer
            // (position 0, rows 1) e avvelenerebbe il primo prefill vero.
            defer {
                service.engine.resetContext()
                service.engine.resetStreamingStats()
            }
            do {
                service.engine.resetContext()
                _ = try service.engine.prefill([probe])
                return true
            } catch {
                DS4Log.info("glm", "warmup fallito: \(error)")
                return false
            }
        }
        lastEnginePass = Task { _ = await pass.value }
        return await pass.value
    }

    public func quiesceForTeardown() async {
        service.engine.saveUsageProfile()
    }

    public struct BenchmarkNumbers: Sendable {
        public let contextTokens: Int
        public let prefillTps: Double
        public let genTps: Double
        /// p99 of the per-token decode speed — the reached steady state,
        /// insensitive to the cold first token (same metric DeepSeek charts).
        public let genTpsP99: Double
        public let kvBytes: UInt64
        public let report: String
    }

    /// KV bytes per cached token (compact MLA rows + indexer keys), the same
    /// constant `modelInfo` reports.
    private static let kvBytesPerToken: UInt64 = 95_232

    /// Synthetic MEASUREMENT benchmark — the DeepSeek analog sweeps prefill
    /// knobs, but GLM v1 has none worth tuning in place, so this reports
    /// prefill/decode tok/s plus the per-phase streaming profile. Resets
    /// the engine context before and after (the next chat re-prefills).
    public func benchmark(contextTokens: Int, genTokens: Int) async throws
        -> BenchmarkNumbers {
        let service = self.service
        let limit = max(8, min(contextTokens, contextSize - genTokens - 1))
        primedTokens = []
        let prior = lastEnginePass
        let pass = Task.detached(priority: .userInitiated) {
            await prior?.value
            return try Self.benchmarkPass(
                service: service, limit: limit, genTokens: genTokens)
        }
        lastEnginePass = Task { _ = try? await pass.value }
        // Stop del pannello → cancel del task staccato: i loop del pass
        // controllano la cancellazione tra un token e l'altro.
        return try await withTaskCancellationHandler {
            try await pass.value
        } onCancel: {
            pass.cancel()
        }
    }

    private static func benchmarkPass(service: GLM52InferenceService,
                                      limit: Int, genTokens: Int)
        throws -> BenchmarkNumbers {
            var tokens = service.tokenizer.tokenizeRenderedChat(
                "benchmark sintetico DwarfStar — misura di prefill e decode ")
            if tokens.isEmpty { tokens = [1] }
            while tokens.count < limit { tokens += tokens }
            tokens = Array(tokens.prefix(limit))
            // Come nel warmup: cache coerenti anche quando il run fallisce.
            defer { service.engine.resetContext() }
            service.engine.resetContext()
            service.engine.resetStreamingStats()
            let prefillStart = Date()
            var logits = try service.engine.prefill(tokens)
            let prefillSeconds = max(
                Date().timeIntervalSince(prefillStart), 0.001)
            let prefillReport = service.engine.profileReport(
                title: "Profilo prefill")
                + "\n" + service.engine.streamingReport()
            service.engine.resetStreamingStats()
            let decodeStart = Date()
            var produced = 0
            var tokenSpeeds: [Double] = []
            tokenSpeeds.reserveCapacity(genTokens)
            for _ in 0..<genTokens {
                try Task.checkCancellation()
                guard let token = GLM52GreedyDecoding.argmax(logits) else {
                    break
                }
                let tokenStart = Date()
                logits = try service.engine.forwardNext(token)
                let tokenSeconds = Date().timeIntervalSince(tokenStart)
                if tokenSeconds > 0 { tokenSpeeds.append(1.0 / tokenSeconds) }
                produced += 1
            }
            let decodeSeconds = max(
                Date().timeIntervalSince(decodeStart), 0.001)
            let decodeReport = service.engine.profileReport(
                title: "Profilo decode")
                + "\n" + service.engine.streamingReport()
            service.engine.resetContext()
            var p99 = 0.0
            if !tokenSpeeds.isEmpty {
                let sorted = tokenSpeeds.sorted()
                p99 = sorted[min(sorted.count - 1,
                                 Int(Double(sorted.count - 1) * 0.99))]
            }
            return BenchmarkNumbers(
                contextTokens: tokens.count,
                prefillTps: Double(tokens.count) / prefillSeconds,
                genTps: Double(produced) / decodeSeconds,
                genTpsP99: p99,
                kvBytes: Self.kvBytesPerToken * UInt64(tokens.count),
                report: "prefill: " + prefillReport
                    + "\ndecode: " + decodeReport)
    }

    /// Teacher-forced Top-1/2/3 next-token accuracy — the GLM counterpart of
    /// the DeepSeek `accuracyBenchmark`, sharing its pure sampling plan,
    /// accumulator and result types so the Benchmark panel renders both
    /// backends identically. Each sampled piece runs an independent
    /// layer-major `forwardBatch` (weights read once per piece), which
    /// returns the logits of EVERY position: scoring happens at prefill
    /// speed, not at the much slower per-token decode speed.
    public func accuracyBenchmark(
        text: String,
        minContextTokens: Int,
        maxContextTokens: Int,
        maxTokensPerPiece: Int,
        pieceCount: Int,
        seed: UInt64,
        bucketSize: Int,
        onObservation: @escaping @Sendable (InferenceService.AccuracyObservation) -> Void = { _ in }
    ) async throws -> InferenceService.AccuracyResult {
        let service = self.service
        let contextSize = self.contextSize
        primedTokens = []
        let prior = lastEnginePass
        let pass = Task.detached(priority: .userInitiated) {
            await prior?.value
            return try Self.accuracyPass(
                service: service, contextSize: contextSize, text: text,
                minContextTokens: minContextTokens,
                maxContextTokens: maxContextTokens,
                maxTokensPerPiece: maxTokensPerPiece,
                pieceCount: pieceCount, seed: seed, bucketSize: bucketSize,
                onObservation: onObservation)
        }
        lastEnginePass = Task { _ = try? await pass.value }
        return try await withTaskCancellationHandler {
            try await pass.value
        } onCancel: {
            pass.cancel()
        }
    }

    private static func accuracyPass(
        service: GLM52InferenceService,
        contextSize: Int,
        text: String,
        minContextTokens: Int,
        maxContextTokens: Int,
        maxTokensPerPiece: Int,
        pieceCount: Int,
        seed: UInt64,
        bucketSize: Int,
        onObservation: @Sendable (InferenceService.AccuracyObservation) -> Void
    ) throws -> InferenceService.AccuracyResult {
        // Plain-text tokenization: the corpus is data, not chat markup.
        let source = service.tokenizer.tokenize(text).map(Int.init)
        guard source.count >= 2 else {
            throw InferenceService.AccuracyBenchmarkError
                .insufficientTokens(actual: source.count)
        }
        guard contextSize >= 4 else {
            throw InferenceService.AccuracyBenchmarkError
                .contextTooSmall(actual: contextSize)
        }
        // Every GLM prompt starts with `[gMASK]<sop>`; the evaluation feeds
        // the same two-token prefix so scored positions see the trained
        // distribution. The plan reserves one leading token itself (the
        // DeepSeek BOS), so only the second prefix token is subtracted here.
        let prefix: [Int32] = [service.tokenizer.special.mask,
                               service.tokenizer.special.startOfPrompt]
        let plan = InferenceService.makeAccuracySamplingPlan(
            sourceTokenCount: source.count,
            contextSize: contextSize - (prefix.count - 1),
            minContextTokens: minContextTokens,
            maxContextTokens: maxContextTokens,
            maxTokensPerPiece: maxTokensPerPiece,
            pieceCount: pieceCount,
            seed: seed
        )
        try Task.checkCancellation()

        var accumulator = InferenceService.AccuracyAccumulator(bucketSize: bucketSize)
        var pieceResults: [InferenceService.AccuracyPieceResult] = []
        pieceResults.reserveCapacity(plan.pieces.count)
        // Cache coerenti anche su errore/cancellazione a metà piece.
        defer { service.engine.resetContext() }
        let started = Date()

        for piece in plan.pieces {
            try Task.checkCancellation()
            let beforeEvaluated = accumulator.evaluatedTokens
            let beforeTop1 = accumulator.top1CorrectTokens
            let beforeTop2 = accumulator.top2CorrectTokens
            let beforeTop3 = accumulator.top3CorrectTokens

            let sourceInputEnd = piece.targetStartTokenIndex
                + piece.evaluatedTokens - 1
            var inputs = prefix
            inputs.append(contentsOf: source[
                piece.sourceStartTokenIndex..<sourceInputEnd
            ].map(Int32.init))
            service.engine.resetContext()
            let rows = try service.engine.forwardBatch(inputs)
            // rows[j] is the distribution after inputs[0...j]: the prediction
            // for source index targetStart+k sits at row
            // prefix.count + contextTokens + k - 1 (see the DeepSeek
            // position derivation; here BOS is the two-token prefix).
            let base = prefix.count + piece.contextTokens - 1
            for k in 0..<piece.evaluatedTokens {
                try Task.checkCancellation()
                let sourceIndex = piece.targetStartTokenIndex + k
                let candidates = topCandidates(rows[base + k], count: 3)
                let observation = accumulator.append(
                    sourceTokenIndex: sourceIndex,
                    expectedTokenId: source[sourceIndex],
                    predictedTokenIds: candidates,
                    pieceIndex: piece.index
                )
                onObservation(observation)
            }

            let evaluated = accumulator.evaluatedTokens - beforeEvaluated
            let top1Correct = accumulator.top1CorrectTokens - beforeTop1
            let top2Correct = accumulator.top2CorrectTokens - beforeTop2
            let top3Correct = accumulator.top3CorrectTokens - beforeTop3
            let denominator = Double(max(1, evaluated))
            pieceResults.append(InferenceService.AccuracyPieceResult(
                index: piece.index,
                sourceStartTokenIndex: piece.sourceStartTokenIndex,
                targetStartTokenIndex: piece.targetStartTokenIndex,
                contextTokens: piece.contextTokens,
                evaluatedTokens: evaluated,
                top1CorrectTokens: top1Correct,
                top2CorrectTokens: top2Correct,
                top3CorrectTokens: top3Correct,
                top1Accuracy: evaluated > 0 ? Double(top1Correct) / denominator : 0,
                top2Accuracy: evaluated > 0 ? Double(top2Correct) / denominator : 0,
                top3Accuracy: evaluated > 0 ? Double(top3Correct) / denominator : 0,
                truncated: piece.truncated || evaluated < piece.evaluatedTokens
            ))
        }

        let duration = Date().timeIntervalSince(started)
        return accumulator.result(
            originalTokens: source.count,
            contextTokens: plan.effectiveMaxContextTokens,
            duration: duration,
            truncated: plan.truncated,
            pieces: pieceResults,
            requestedPieceCount: plan.requestedPieceCount,
            seed: plan.seed,
            effectiveMinContextTokens: plan.effectiveMinContextTokens,
            effectiveMaxContextTokens: plan.effectiveMaxContextTokens
        )
    }

    /// Top-`count` token ids by logit, one partial-selection pass over the
    /// vocabulary. Strict `>` keeps the LOWEST id on ties — the same rule as
    /// the greedy argmax kernels. Internal for the unit test.
    static func topCandidates(_ logits: [Float], count: Int) -> [Int] {
        var best: [(id: Int, value: Float)] = []
        best.reserveCapacity(count)
        for (id, value) in logits.enumerated() {
            if best.count < count {
                best.append((id, value))
                best.sort { $0.value > $1.value }
            } else if value > best[count - 1].value {
                best[count - 1] = (id, value)
                var i = count - 1
                while i > 0 && best[i].value > best[i - 1].value {
                    best.swapAt(i, i - 1)
                    i -= 1
                }
            }
        }
        return best.map(\.id)
    }

    public func setAgent(systemPrompt: String?) {
        self.systemPrompt = systemPrompt
    }

    /// DS-shaped agent application (the `ChatBackend` contract): system
    /// prompt + declared tools from the profile, like the DeepSeek service.
    public func setAgent(_ agent: AgentProfile, tools: [ToolSpec]) {
        systemPrompt = agent.systemPrompt.isEmpty
            ? nil : agent.systemPrompt
        self.tools = tools
    }

    /// Tools declared to the model (rendered into the native GLM XML tool
    /// prompt). The compact form is a DeepSeek-only optimization: accepted
    /// and ignored here.
    public func setTools(_ tools: [ToolSpec]) { self.tools = tools }
    public func setCompactTools(_ on: Bool) {}
    private var tools: [ToolSpec] = []

    public func committedTokens() -> Int { service.engine.position }

    /// Hit/miss dell'arena esperti per la riga di tuning della GUI (snapshot
    /// lock-protetto sul motore, sicuro anche a generazione in corso).
    public func expertArenaCounters() -> (hits: Int, misses: Int)? {
        service.engine.expertArenaCounters()
    }

    /// Esito dell'auto-tune dei knob di caricamento (report + vincitori
    /// come nome-env → valore; "" = default del motore).
    public struct AutoTuneOutcome: Sendable {
        public let report: String
        public let winners: [String: String]
    }

    /// Auto-tune SENZA servizio vivo: il chiamante scarica il modello prima
    /// (un solo motore alla volta — ogni misura costruisce e getta il suo)
    /// e lo ricarica coi vincitori dopo. Wrapper GUI-friendly di
    /// GLM52ResidentModel.autoTune: knob ESATTI di caricamento, un gradino
    /// alla volta, ~1 min a configurazione.
    public static func autoTune(modelPath: String, genTokens: Int = 12)
        throws -> AutoTuneOutcome {
        let runtime = try MetalRuntime()
        let model = try GGUFModel(path: modelPath, metalMapping: false,
                                  prefetchCPU: false)
        let tokenizer = try GLM52Tokenizer(model: model)
        let tokens = try tokenizer.encodeChatPrompt(
            prompt: "Auto-tune: misura sintetica dei knob di caricamento.")
        let outcome = try GLM52ResidentModel.autoTune(
            runtime: runtime, path: modelPath, prompt: tokens,
            genTokens: genTokens)
        return AutoTuneOutcome(report: outcome.report,
                               winners: outcome.winners)
    }

    /// Append a user message to the running transcript and generate.
    public func send(userText: String, thinkMode: DS4ThinkMode,
                     sampling: SamplingParams,
                     maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        transcript.append(.user(userText))
        return generate(turns: renderableTurns(), sampling: sampling,
                        maxTokens: maxTokens, thinkMode: thinkMode)
    }

    /// Stateless completion for the local HTTP server (OpenAI semantics:
    /// each request carries the whole conversation). The request replaces
    /// the running transcript; the incremental-KV prefix match still
    /// applies, so consecutive requests of the same client conversation
    /// prefill only the new suffix.
    public func complete(turns: [ChatTurn], tools: [ToolSpec],
                         thinkMode: DS4ThinkMode, sampling: SamplingParams,
                         maxTokens: Int)
        -> AsyncThrowingStream<GenEvent, Error> {
        self.tools = tools
        systemPrompt = nil
        transcript = turns
        return generate(turns: turns, sampling: sampling,
                        maxTokens: maxTokens, thinkMode: thinkMode)
    }

    /// Feed tool outputs back (native GLM observation turns) and stream the
    /// assistant continuation — the DeepSeek tool loop's GLM counterpart.
    /// v1 honesty: the re-render after a tool round rarely prefix-matches
    /// the live caches (thinking is not re-rendered), so the continuation
    /// usually pays a fresh layer-major prefill.
    public func provideToolResults(_ outputs: [ToolOutput],
                                   thinkMode: DS4ThinkMode,
                                   sampling: SamplingParams,
                                   maxTokens: Int)
        -> AsyncThrowingStream<GenEvent, Error> {
        transcript.append(contentsOf: outputs.map {
            .toolResult(callId: $0.callId, name: $0.name,
                        content: $0.content)
        })
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

    private func noteGeneration(fedTokens: [Int32], assistant: String,
                                calls: [ToolCall]) {
        primedTokens = fedTokens
        // The tool calls stay ON the assistant turn: the tool-loop re-render
        // needs them between the assistant text and the observations.
        transcript.append(.assistant(text: assistant, toolCalls: calls))
    }

    /// Splits the raw token stream into reasoning/text events on the GLM
    /// `<think>` markers, holding back a short tail so a marker split across
    /// token boundaries is never emitted as text.
    private struct StreamSplitter {
        var inThink = false
        var pending = ""

        /// Col think ATTIVO il prompt renderizzato termina con `<think>`:
        /// la generazione comincia già DENTRO il blocco e contiene solo il
        /// `</think>` di chiusura — lo splitter deve partire in modalità
        /// reasoning, o l'intera catena passa come testo e la chiusura
        /// finisce letterale nella UI.
        init(startsInThink: Bool) {
            inThink = startsInThink
        }
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
        let kvStore = self.kvStore
        let (stream, continuation) =
            AsyncThrowingStream<GenEvent, Error>.makeStream()
        let prior = lastEnginePass
        // Detached: the engine blocks on GPU/SSD waits and must not
        // occupy the cooperative pool (same discipline as the DeepSeek
        // actor's serial GCD executor). The await on `prior` keeps the
        // engine single-driver across producer, warmup e benchmark.
        let producer = Task.detached(priority: .userInitiated) {
            await prior?.value
            do {
                    // FINESTRA SCORREVOLE sull'overflow: quando la
                    // conversazione supera il contesto, i turni più vecchi
                    // (mai il system) vengono scartati dal RENDER — il
                    // transcript completo resta persistito a monte — finché
                    // resta almeno una riserva di generazione.
                    var activeTurns = turns
                    var rendered = try GLM52ChatRenderer.render(
                        turns: activeTurns, tools: declaredTools,
                        reasoning: thinkMode.core)
                    var tokens = service.tokenizer
                        .tokenizeRenderedChat(rendered)
                    let reserve = min(256, max(32, maxTokens))
                    var droppedTurns = 0
                    while tokens.count + reserve >= contextSize {
                        guard let dropIndex = activeTurns.firstIndex(
                                  where: { turn in
                                      if case .system = turn { return false }
                                      return true
                                  }),
                              dropIndex < activeTurns.count - 1 else {
                            throw GGUFError.cannotOpen(
                                "conversazione (\(tokens.count) token) "
                                + "oltre il contesto GLM (\(contextSize)) "
                                + "anche dopo la finestra scorrevole")
                        }
                        activeTurns.remove(at: dropIndex)
                        droppedTurns += 1
                        rendered = try GLM52ChatRenderer.render(
                            turns: activeTurns, tools: declaredTools,
                            reasoning: thinkMode.core)
                        tokens = service.tokenizer
                            .tokenizeRenderedChat(rendered)
                    }
                    if droppedTurns > 0 {
                        continuation.yield(.progress(
                            "contesto pieno: finestra scorrevole, scartati "
                            + "\(droppedTurns) turni più vecchi dal render"))
                    }

                    // Incremental KV: prefill only the suffix when the
                    // rendered conversation extends what the caches hold.
                    let common = zip(primed, tokens)
                        .prefix { $0 == $1 }.count
                    // primed vuoto NON è incrementale: uno stato ignoto
                    // (warmup fallito, generazione interrotta) deve sempre
                    // passare dal reset, che riporta le cache a zero righe.
                    let incremental = !primed.isEmpty
                        && common == primed.count
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
                        // Disk KV: lo store cerca tra TUTTI i checkpoint il
                        // prefisso stretto più lungo della nuova
                        // conversazione — si prefilla solo il resto.
                        var restored = 0
                        if let kvStore,
                           let hit = kvStore.findLongestPrefix(of: tokens) {
                            if (try? service.engine.restoreKVCheckpoint(
                                    from: hit.url)) != nil {
                                restored = hit.tokens.count
                                kvStore.bumpHit(hit.name)
                                continuation.yield(.progress(
                                    "KV da disco: \(restored) token "
                                    + "ripristinati"))
                            } else {
                                // Un restore parziale lascia righe appese:
                                // torna a zero righe prima del prefill.
                                service.engine.resetContext()
                            }
                        }
                        suffix = Array(tokens[restored...])
                        continuation.yield(.progress(
                            "prefill \(suffix.count) token (layer-major)"))
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
                    var splitter = StreamSplitter(
                        startsInThink: thinkMode.core == .high)
                    let decodeStart = Date()
                    let budget = min(maxTokens,
                                     contextSize - tokens.count - 1)
                    // Seedable, C-compatible xorshift RNG (shared with DeepSeek):
                    // same seed → same stream, so GLM sampling is reproducible.
                    var samplerRNG = sampling.seed
                    while produced < budget, !Task.isCancelled {
                        guard let token = GLM52Sampler.sample(
                            logits: logits,
                            temperature: sampling.temperature,
                            topK: sampling.topK,
                            topP: sampling.topP,
                            minP: sampling.minP,
                            repetitionPenalty: sampling.repetitionPenalty,
                            recentTokens: fed.suffix(
                                GLM52Sampler.penaltyWindow),
                            rng: &samplerRNG) else { break }
                        produced += 1
                        if service.tokenizer.isStopToken(
                            token, reasoning: thinkMode.core) {
                            break
                        }
                        let bytes = service.tokenizer.tokenText(token)
                        reply.append(contentsOf: bytes)
                        let specials = service.tokenizer.special
                        if token == specials.thinkOpen
                            || token == specials.thinkClose {
                            // I marcatori think sono TOKEN SPECIALI
                            // dedicati: la transizione scatta sull'ID —
                            // robusta anche se la resa testuale cambiasse —
                            // svuotando prima il testo trattenuto nel modo
                            // corrente.
                            for event in splitter.flush() {
                                continuation.yield(event)
                            }
                            splitter.inThink = token == specials.thinkOpen
                        } else if let piece = String(bytes: bytes,
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
                    // Native GLM XML tool calls: parsed from the full reply.
                    // Col think attivo il grezzo inizia col ragionamento
                    // (senza marcatore di apertura — sta nel prompt): via
                    // il prefisso fino al `</think>`, così transcript e
                    // parser vedono solo la risposta visibile.
                    var raw = String(decoding: reply, as: UTF8.self)
                    if thinkMode.core == .high,
                       let close = raw.range(
                           of: GLM52ConversationProtocol.thinkClose) {
                        raw = String(raw[close.upperBound...])
                    }
                    let parsed = GLM52ToolCodec.parse(raw,
                                                      tools: declaredTools)
                    if !parsed.calls.isEmpty {
                        continuation.yield(.toolCall(parsed.calls))
                    }
                    await self.noteGeneration(fedTokens: fed,
                                              assistant: parsed.visibleText,
                                              calls: parsed.calls)
                    // Checkpoint disk-KV a fine generazione (best-effort,
                    // ~96 KB/token): dedup sui token esatti e intervallo
                    // minimo dall'ultimo prefisso salvato; lo store evicta
                    // sotto budget prima di pubblicare.
                    if let kvStore, kvStore.shouldStore(tokens: fed) {
                        kvStore.store(tokens: fed) { url in
                            try service.engine.saveKVCheckpoint(
                                to: url, tokens: fed)
                        }
                    }
                    service.engine.saveUsageProfile()
                    continuation.finish()
                } catch {
                    // Un prefill abortito (Stop, disconnessione, errore
                    // Metal) lascia righe parziali e prefetch dello
                    // streamer non consumati: il reset drena e azzera —
                    // il prossimo turno ricostruisce da disk-KV/zero via
                    // mismatch di posizione.
                    service.engine.resetContext()
                    continuation.finish(throwing: error)
                }
        }
        lastEnginePass = Task { _ = await producer.value }
        continuation.onTermination = { _ in producer.cancel() }
        return stream
    }
}
