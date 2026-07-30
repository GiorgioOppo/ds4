import DS4Core
import DS4Metal
import Foundation

/// Laguna S 2.1 chat service for the GUI: mirrors the chat subset of the
/// DeepSeek `InferenceService` API (send / sendWithHistory / modelInfo /
/// warmup / quiesceForTeardown) over the Laguna resident engine, so
/// `ChatStore` can host the backend behind the common `ChatBackend`
/// contract — the same integration shape as `GLM52ChatService`.
///
/// Feature parity with the GLM service so far: sampled decoding
/// (temperature, top-K, top-P, min-P, repetition penalty, seedable
/// C-compatible RNG via `LagunaSampler`), stoppable generation, incremental
/// KV between turns (only the rendered conversation's NEW suffix is
/// prefilled when it extends what the engine already holds), the reasoning
/// stream (`<think>` split on special-token IDs) and native `<tool_call>`
/// turns via `LagunaToolCodec`, plus prefix-indexed native disk-KV
/// checkpoints. Not mirrored: auto-tune — the streaming engine profile
/// lives in the demo CLI and benchmark panel.
public actor LagunaChatService {
    public let service: LagunaInferenceService
    private let contextSize: Int
    private var systemPrompt: String?
    private var transcript: [ChatTurn] = []
    /// Tokens the engine's caches currently hold, in order.
    private var primedTokens: [Int32] = []
    /// The engine is SINGLE-DRIVER (same discipline as GLM): producer and
    /// warmup run detached — they block on GPU/SSD — and every new pass
    /// awaits the previous one here.
    private var lastEnginePass: Task<Void, Never>?
    /// Completed prefixes persisted beside the DeepSeek/GLM stores. The
    /// active decode cache stays in Metal; this store is for cold restore.
    private let kvStore: LagunaDiskKVStore?

    public init(modelPath: String,
                contextSize: Int,
                systemPrompt: String?,
                initialKVCapacity: Int? = nil,
                diskKVDirectory: String? = nil,
                diskKVBudgetTokens: Int = 0) throws {
        if let diskKVDirectory {
            let file = (modelPath as NSString).lastPathComponent
            let attributes = try? FileManager.default
                .attributesOfItem(atPath: modelPath)
            let size = (attributes?[.size] as? UInt64) ?? 0
            let scoped = URL(fileURLWithPath: diskKVDirectory)
                .appendingPathComponent(
                    "laguna-\(file)-\(size)", isDirectory: true)
            kvStore = try? LagunaDiskKVStore(
                directory: scoped, budgetTokens: diskKVBudgetTokens)
        } else {
            kvStore = nil
        }
        var options = LagunaInferenceService.defaultOptions(
            modelPath: modelPath, contextSize: contextSize)
        if let initialKVCapacity {
            options.initialFullCacheCapacity = initialKVCapacity
        }
        service = try LagunaInferenceService(modelPath: modelPath,
                                             options: options)
        self.contextSize = max(256, contextSize)
        self.systemPrompt = systemPrompt
    }

    // MARK: - InferenceService-mirroring chat surface

    /// F16 ring KV bytes at `tokens` cached positions: 2 planes ×
    /// keyValueProjectionWidth halves per row; sliding-window layers hold a
    /// fixed 512-row ring regardless of the context, only full-attention
    /// layers scale with it.
    static func kvCacheBytes(shape: LagunaShape, tokens: Int) -> UInt64 {
        let rowBytes = Int(shape.keyValueProjectionWidth) * 2 * 2
        var perTokenBytes = 0
        var fixedBytes = 0
        for layer in 0..<Int(shape.nLayer) {
            if shape.isSlidingWindowLayer(layer) {
                fixedBytes += min(Int(shape.nSWA), tokens) * rowBytes
            } else {
                perTokenBytes += rowBytes
            }
        }
        return UInt64(perTokenBytes) * UInt64(tokens) + UInt64(fixedBytes)
    }

    public func modelInfo() -> ModelInfo {
        let shape = service.engine.configuration.shape
        let streaming = service.engine.isExpertStreaming
        let cacheSlots = service.engine.expertCacheSlots
        var quantization = "routed K-quant"
        if streaming {
            quantization += cacheSlots > 0
                ? " · streaming SSD (cache \(cacheSlots) slot)"
                : " · streaming SSD"
        } else {
            quantization += " residente"
        }
        return ModelInfo(
            name: "Laguna S 2.1",
            layers: service.engine.loadedLayerCount,
            nEmbd: Int(shape.nEmbd),
            nVocab: Int(shape.nVocab),
            contextSize: contextSize,
            routedQuantBits: 2,
            // Laguna allocates the full-attention KV lazily. Report what is
            // actually resident at load, not the theoretical 32k maximum.
            kvCacheBytes: UInt64(service.engine.allocatedKVCacheBytes),
            architecture: LagunaBackendDefinition.supportedArchitecture,
            displayName: "Laguna S 2.1",
            quantizationSummary: quantization,
            capabilities: LagunaBackendDefinition.runtimeCapabilities)
    }

    /// REAL warmup: one 1-token pass through the full stack (file handles,
    /// pipeline compilations, staging/cache slots), so the first user token
    /// stops paying the one-time costs. Context is reset right after; a
    /// failed prefill must also reset, or partial rows poison the first
    /// real prefill.
    public func warmup() async -> Bool {
        let service = self.service
        primedTokens = []
        let prior = lastEnginePass
        let pass = Task.detached(priority: .userInitiated) {
            await prior?.value
            let probe = service.tokenizer
                .tokenizeRenderedChat("ciao").first ?? 1
            defer {
                service.engine.resetContext()
                service.engine.resetProfile()
            }
            do {
                service.engine.resetContext()
                _ = try service.engine.prefill([probe])
                return true
            } catch {
                DS4Log.info("laguna", "warmup fallito: \(error)")
                return false
            }
        }
        lastEnginePass = Task { _ = await pass.value }
        return await pass.value
    }

    public func quiesceForTeardown() async {
        // No persisted usage profile in the Laguna v1 — the await still
        // serializes teardown behind any in-flight engine pass.
        await lastEnginePass?.value
    }

    // MARK: - Benchmark (GLM-shaped surfaces for the Benchmark panel)

    public struct BenchmarkNumbers: Sendable {
        public let contextTokens: Int
        public let prefillTps: Double
        public let genTps: Double
        /// Throughput equivalent to p99 per-token decode latency: the slow
        /// generation tail (same metric as DeepSeek/GLM).
        public let genTpsP99: Double
        public let kvBytes: UInt64
        public let report: String
    }

    /// Synthetic MEASUREMENT benchmark: prefill (layer-major) + greedy
    /// decode tok/s plus the per-phase profile. Resets the engine context
    /// before and after (the next chat re-prefills).
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
        return try await withTaskCancellationHandler {
            try await pass.value
        } onCancel: {
            pass.cancel()
        }
    }

    private static func benchmarkPass(service: LagunaInferenceService,
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
        service.engine.resetProfile()
        let prefillStart = Date()
        var logits = try service.engine.prefill(tokens)
        let prefillSeconds = max(
            Date().timeIntervalSince(prefillStart), 0.001)
        let prefillReport = service.engine.profileReport(
            title: "Profilo prefill")
        service.engine.resetProfile()
        let decodeStart = Date()
        var produced = 0
        var tokenSpeeds: [Double] = []
        tokenSpeeds.reserveCapacity(genTokens)
        for _ in 0..<genTokens {
            try Task.checkCancellation()
            var best = logits[0]
            var bestIndex: Int32 = 0
            for index in 1..<logits.count where logits[index] > best {
                best = logits[index]
                bestIndex = Int32(index)
            }
            let tokenStart = Date()
            logits = try service.engine.forwardNext(bestIndex)
            let tokenSeconds = Date().timeIntervalSince(tokenStart)
            if tokenSeconds > 0 { tokenSpeeds.append(1.0 / tokenSeconds) }
            produced += 1
        }
        let decodeSeconds = max(
            Date().timeIntervalSince(decodeStart), 0.001)
        let decodeReport = service.engine.profileReport(
            title: "Profilo decode")
        var p99 = 0.0
        if !tokenSpeeds.isEmpty {
            let sorted = tokenSpeeds.sorted()
            let latencyRank = Int(ceil(0.99 * Double(sorted.count)))
            p99 = sorted[max(0, sorted.count - latencyRank)]
        }
        return BenchmarkNumbers(
            contextTokens: tokens.count,
            prefillTps: Double(tokens.count) / prefillSeconds,
            genTps: Double(produced) / decodeSeconds,
            genTpsP99: p99,
            kvBytes: kvCacheBytes(
                shape: service.engine.configuration.shape,
                tokens: tokens.count),
            report: "prefill: " + prefillReport
                + "\ndecode: " + decodeReport)
    }

    /// Teacher-forced Top-1/2/3 next-token accuracy — shares the DeepSeek
    /// sampling plan, accumulator and result types so the Benchmark panel
    /// renders every backend identically. Each sampled piece runs an
    /// independent layer-major `forwardBatch` (weights read once per
    /// piece): scoring happens at prefill speed.
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
        service: LagunaInferenceService,
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
        // Every Laguna prompt opens with the BOS marker; the evaluation
        // feeds the same one-token prefix so scored positions see the
        // trained distribution. The plan already reserves one leading
        // token itself.
        let prefix: [Int32] = [service.tokenizer.special.beginOfSequence]
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

        var accumulator = InferenceService.AccuracyAccumulator(
            bucketSize: bucketSize)
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
            // rows[j] is the distribution after inputs[0...j]: the
            // prediction for source index targetStart+k sits at row
            // prefix.count + contextTokens + k - 1 (same derivation as
            // DeepSeek/GLM; here BOS is the one-token prefix).
            let base = prefix.count + piece.contextTokens - 1
            for k in 0..<piece.evaluatedTokens {
                try Task.checkCancellation()
                let sourceIndex = piece.targetStartTokenIndex + k
                let candidates = GLM52ChatService.topCandidates(
                    rows[base + k], count: 3)
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
                top1Accuracy: evaluated > 0
                    ? Double(top1Correct) / denominator : 0,
                top2Accuracy: evaluated > 0
                    ? Double(top2Correct) / denominator : 0,
                top3Accuracy: evaluated > 0
                    ? Double(top3Correct) / denominator : 0,
                truncated: piece.truncated
                    || evaluated < piece.evaluatedTokens
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

    public func setAgent(systemPrompt: String?) {
        self.systemPrompt = systemPrompt
    }

    /// DS-shaped agent application (the `ChatBackend` contract): system
    /// prompt + declared tools from the profile.
    public func setAgent(_ agent: AgentProfile, tools: [ToolSpec]) {
        systemPrompt = agent.systemPrompt.isEmpty
            ? nil : agent.systemPrompt
        self.tools = tools
    }

    /// Tools declared to the model (rendered into the native Laguna
    /// `<tools>` system section). The compact form is a DeepSeek-only
    /// optimization: accepted and ignored here.
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

    /// Stateless completion for the local HTTP server (OpenAI semantics).
    /// The request replaces the running transcript; the incremental-KV
    /// prefix match still applies.
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

    /// Feed tool outputs back (native `<tool_response>` turns) and stream
    /// the assistant continuation. v1 honesty: like GLM, the re-render
    /// after a tool round rarely prefix-matches the live caches, so the
    /// continuation usually pays a fresh prefill.
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
    /// replaces the running transcript.
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
        // The tool calls stay ON the assistant turn: the tool-loop
        // re-render needs them between the assistant text and the
        // observations.
        transcript.append(.assistant(text: assistant, toolCalls: calls))
    }

    /// Splits the raw token stream into reasoning/text events on the Laguna
    /// `<think>` markers, holding back a short tail so a marker split
    /// across token boundaries is never emitted as text. Same shape as the
    /// GLM splitter; transitions normally fire on the dedicated special
    /// token IDs in the decode loop.
    private struct StreamSplitter {
        var inThink = false
        var pending = ""

        /// With reasoning active the rendered prompt ends inside `<think>`:
        /// generation starts in reasoning mode and only carries the closing
        /// marker.
        init(startsInThink: Bool) {
            inThink = startsInThink
        }
        mutating func feed(_ piece: String) -> [GenEvent] {
            pending += piece
            var events: [GenEvent] = []
            while true {
                let marker = inThink
                    ? LagunaConversationProtocol.thinkClose
                    : LagunaConversationProtocol.thinkOpen
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
        // Detached: the engine blocks on GPU/SSD waits and must not occupy
        // the cooperative pool. The await on `prior` keeps the engine
        // single-driver across producer and warmup.
        let producer = Task.detached(priority: .userInitiated) {
            await prior?.value
            do {
                // FINESTRA SCORREVOLE sull'overflow: quando la conversazione
                // supera il contesto, i turni più vecchi (mai il system)
                // vengono scartati dal RENDER — il transcript completo resta
                // persistito a monte — finché resta una riserva di
                // generazione.
                var activeTurns = turns
                var rendered = try LagunaChatRenderer.render(
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
                            "conversazione (\(tokens.count) token) oltre "
                            + "il contesto Laguna (\(contextSize)) anche "
                            + "dopo la finestra scorrevole")
                    }
                    activeTurns.remove(at: dropIndex)
                    droppedTurns += 1
                    rendered = try LagunaChatRenderer.render(
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

                // Incremental KV: prefill only the suffix when the rendered
                // conversation extends what the caches hold. primed vuoto
                // NON è incrementale: uno stato ignoto passa dal reset.
                let common = zip(primed, tokens)
                    .prefix { $0 == $1 }.count
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
                    // Drop a formerly-grown full KV before opening another
                    // conversation, then restore the longest exact prefix
                    // available on disk. The strict-prefix lookup guarantees
                    // at least one suffix token remains to produce logits.
                    service.engine.resetContext(releaseExcessKV: true)
                    var restored = 0
                    if let kvStore,
                       let hit = kvStore.findLongestPrefix(of: tokens) {
                        if (try? service.engine.restoreKVCheckpoint(
                                from: hit.url)) != nil {
                            restored = hit.tokens.count
                            kvStore.bumpHit(hit.name)
                            continuation.yield(.progress(
                                "KV da disco: \(restored) token ripristinati"))
                        } else {
                            service.engine.resetContext(
                                releaseExcessKV: true)
                        }
                    }
                    suffix = Array(tokens[restored...])
                    continuation.yield(.progress(
                        "prefill \(suffix.count) token"
                        + (restored > 0
                           ? " (+\(restored) da disco)" : "")))
                }
                // Prefill layer-major (engine.prefill): il progresso vivo
                // arriva per LAYER attraversato — con lo streaming SSD il
                // prompt sono decine di secondi che altrimenti sembrano un
                // freeze — e lo Stop interrompe anche il prefill (il motore
                // controlla la cancellazione a ogni layer). Throttle come
                // DeepSeek: ogni yield è un hop sul MainActor.
                let prefillStart = Date()
                let suffixCount = suffix.count
                nonisolated(unsafe) var lastPrefillProgress = prefillStart
                let logits: [Float]
                do {
                    logits = try service.engine.prefill(suffix) {
                        layersDone, layersTotal in
                        let now = Date()
                        guard layersDone < layersTotal,
                              now.timeIntervalSince(lastPrefillProgress)
                                  >= 0.25 else { return }
                        lastPrefillProgress = now
                        let dt = now.timeIntervalSince(prefillStart)
                        continuation.yield(.progress(String(
                            format: "prefill %d token · layer %d/%d · %.0fs…",
                            suffixCount, layersDone, layersTotal, dt)))
                    }
                } catch is CancellationError {
                    // Stop a metà prefill: righe parziali in cache — il
                    // reset azzera, e il mismatch di posizione al prossimo
                    // turno forza il re-prefill da zero.
                    service.engine.resetContext()
                    continuation.finish()
                    return
                }
                let prefillSeconds = max(
                    Date().timeIntervalSince(prefillStart), 0.001)
                // Riepilogo del prefill: resta nello status finché il primo
                // progress del decode non lo sostituisce — e il decode lo
                // ripropone in coda alla sua riga, come il backend DeepSeek,
                // così il tempo del prompt resta leggibile in GUI.
                let prefillSummary = String(
                    format: "prefill %d tok in %.1fs · %.2f tok/s",
                    suffix.count, prefillSeconds,
                    Double(suffix.count) / prefillSeconds)
                continuation.yield(.progress(prefillSummary))

                var fed = tokens
                var produced = 0
                var reply: [UInt8] = []
                var splitter = StreamSplitter(
                    startsInThink: thinkMode.core == .high)
                let decodeStart = Date()
                let budget = min(maxTokens,
                                 contextSize - tokens.count - 1)
                // Seedable, C-compatible xorshift RNG (shared Sampler):
                // same seed → same stream.
                var samplerRNG = sampling.seed
                var nextToken: Int32?
                if budget > 0 {
                    nextToken = LagunaSampler.sample(
                        logits: logits,
                        temperature: sampling.temperature,
                        topK: sampling.topK,
                        topP: sampling.topP,
                        minP: sampling.minP,
                        repetitionPenalty: sampling.repetitionPenalty,
                        recentTokens: fed.suffix(
                            LagunaSampler.penaltyWindow),
                        rng: &samplerRNG)
                }
                while produced < budget, !Task.isCancelled {
                    guard let token = nextToken else { break }
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
                        // I marcatori think sono TOKEN SPECIALI dedicati:
                        // la transizione scatta sull'ID, svuotando prima il
                        // testo trattenuto nel modo corrente.
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
                        format: "%d tok · %.2f tok/s — %@",
                        produced, Double(produced) / elapsed,
                        prefillSummary)))
                    if produced == budget || Task.isCancelled { break }
                    fed.append(token)
                    nextToken = try service.engine.forwardNextSampled(
                        token,
                        temperature: sampling.temperature,
                        topK: sampling.topK,
                        topP: sampling.topP,
                        minP: sampling.minP,
                        repetitionPenalty: sampling.repetitionPenalty,
                        recentTokens: fed.suffix(
                            LagunaSampler.penaltyWindow),
                        rng: &samplerRNG)
                }
                for event in splitter.flush() {
                    continuation.yield(event)
                }
                // Native Laguna tool calls: parsed from the full reply.
                // Col think attivo il grezzo inizia col ragionamento (senza
                // marcatore di apertura — sta nel prompt): via il prefisso
                // fino al `</think>`, così transcript e parser vedono solo
                // la risposta visibile.
                var raw = String(decoding: reply, as: UTF8.self)
                if thinkMode.core == .high,
                   let close = raw.range(
                       of: LagunaConversationProtocol.thinkClose) {
                    raw = String(raw[close.upperBound...])
                }
                let parsed = LagunaToolCodec.parse(raw, tools: declaredTools)
                if !parsed.calls.isEmpty {
                    continuation.yield(.toolCall(parsed.calls))
                }
                await self.noteGeneration(fedTokens: fed,
                                          assistant: parsed.visibleText,
                                          calls: parsed.calls)
                // Best-effort native LKV1 checkpoint. The writer streams
                // Metal planes directly to F_NOCACHE storage, so saving a
                // long context does not create a second model-sized Data.
                if let kvStore, kvStore.shouldStore(tokens: fed) {
                    kvStore.store(tokens: fed) { url in
                        try service.engine.saveKVCheckpoint(
                            to: url, tokens: fed)
                    }
                }
                continuation.finish()
            } catch {
                // Un prefill abortito (Stop, errore Metal) lascia righe
                // parziali: il reset azzera — il prossimo turno ricostruisce
                // da zero via mismatch di posizione.
                service.engine.resetContext(releaseExcessKV: true)
                continuation.finish(throwing: error)
            }
        }
        lastEnginePass = Task { _ = await producer.value }
        continuation.onTermination = { _ in producer.cancel() }
        return stream
    }
}
