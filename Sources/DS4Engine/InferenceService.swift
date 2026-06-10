import Foundation
import DS4Core
import DS4Metal

// DS4Engine: the GUI's inference service, backed by the PURE-SWIFT engine
// (DS4Core tokenizer/GGUF + DS4Metal StreamingDecoder).
//
// Generation uses StreamingDecoder (per-layer load/compute/evict) so the 164GB
// model fits in 16GB. The conversation is APPEND-ONLY with KV-cache reuse: the
// service tracks the exact token ids already in the KV (`committedIds`) and each
// turn prefills ONLY the new suffix (the new user turn, or the tool result +
// assistant open), reusing the KV of all prior turns. No full re-render — so the
// generated reasoning/tool-call tokens stay verbatim in the KV and the next turn
// is a clean token-level extension. Tool calls are parsed via DS4Core.ToolCallParser;
// the tool/declaration format follows the model's chat_template (ChatRenderer).

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

    // Append-only conversation state: `committedIds` are the exact token ids already
    // in the KV cache. Each turn prefills ONLY the new suffix and appends here, so
    // the KV is reused across turns (no full re-prefill). `needsClose` is true when
    // the committed KV ends with an open assistant turn (its <eos> not yet in the KV).
    private var committedIds: [Int] = []
    private var needsClose = false
    private var systemPrompt: String?
    private var tools: [ToolSpec] = []
    // Compact tool declaration (just name(params)) to save prefill tokens.
    // Defaults to true (local inference); the GUI toggle is the single source of
    // truth and pushes its value via setCompactTools (no env override).
    private var compactTools = true
    /// Set when a generation was interrupted (cancel/error) mid-stream: the GPU
    /// KV cache and the recurrent NSA-compressor state may then be inconsistent
    /// with `committedIds`. The next generation rebuilds the KV from the exact
    /// committed ids (slow once, but correct) before continuing.
    private var kvDirty = false

    /// The pure-Swift engine always runs the SSD-streaming path (no-copy mmap
    /// non-routed weights + per-token expert gather); there are no resident/
    /// per-layer variants to configure, hence no streaming options here.
    /// `expertCacheSlots` enables the per-layer expert slot-cache (0/nil = off);
    /// the persisted usage stats pre-warm it with the hottest experts.
    public init(modelPath: String, contextSize: Int, systemPrompt: String?,
                expertCacheSlots: Int? = nil) throws {
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
        self.systemPrompt = (systemPrompt?.isEmpty == false) ? systemPrompt : nil
        let rope = RopeParams(nCtxOrig: 4096, freqBase: 10000, freqScale: 1, extFactor: 0,
                              attnFactor: 1, betaFast: 32, betaSlow: 1)
        // Fast 16GB path (C --ssd-streaming model): non-routed weights are no-copy mmap
        // (resident via page cache, evictable), only the 6 selected experts gathered/token.
        self.decoder = try StreamingDecoder.fromGGUFExpertCachedMapped(rt: rt, model: model, dims: dims, rope: rope,
                                                                       nLayers: DSV4Shape.nLayer, maxKeys: contextSize,
                                                                       cacheSlots: expertCacheSlots)
        // Load the persisted usage stats ("usage imatrix") BEFORE any generation,
        // so the slot-cache warms with the historically hottest experts. The
        // profile is PER-AGENT: different roles route to different experts.
        if let data = try? Data(contentsOf: Self.usageURL(modelName: modelName, agentId: "generale")) {
            decoder.usage?.load(data)
        }
    }

    // MARK: - Agents (roles) + per-agent expert usage

    /// The active agent's id — keys the persisted usage profile.
    private var agentId = "generale"

    /// Switch the conversation to `agent`: persists the outgoing agent's usage
    /// profile, swaps in the new agent's one, drops the slot-cache pools (they
    /// re-warm lazily with the NEW profile), declares the agent's tools and
    /// starts a fresh conversation with its role (system prompt).
    public func setAgent(_ agent: AgentProfile, tools: [ToolSpec]) {
        saveExpertUsage()
        agentId = agent.id
        decoder.usage?.replace(with: try? Data(contentsOf: Self.usageURL(modelName: modelName, agentId: agentId)))
        decoder.slotCache?.invalidate()
        self.tools = tools
        resetConversation(systemPrompt: agent.systemPrompt.isEmpty ? nil : agent.systemPrompt)
    }

    // MARK: - Expert usage ("usage imatrix") persistence + tuning info

    /// Per-(model, agent) usage-profile file. Nonisolated so the actor's init
    /// can resolve it before isolation is established.
    nonisolated static func usageURL(modelName: String, agentId: String) -> URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DwarfStar", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("expert-usage-\(modelName)-\(agentId).json")
    }

    /// Persist the routing-frequency stats (called automatically after each
    /// generation; cheap — at most nLayer×nExpert small ints).
    public func saveExpertUsage() {
        guard let data = decoder.usage?.serialize() else { return }
        try? data.write(to: Self.usageURL(modelName: modelName, agentId: agentId))
    }

    public func resetExpertUsage() {
        decoder.usage?.reset()
        try? FileManager.default.removeItem(at: Self.usageURL(modelName: modelName, agentId: agentId))
    }

    public struct TuningInfo: Sendable {
        public let agentId: String          // whose usage profile is active
        public let cacheSlots: Int          // 0 = cache off
        public let cacheHits: Int
        public let cacheMisses: Int
        public let totalRoutes: Int
        /// Per-layer summary: "L7 · top8 = 43% · expert 17,4,99,…".
        public let layerSummaries: [String]
    }

    public func tuningInfo() -> TuningInfo {
        let usage = decoder.usage
        var summaries: [String] = []
        if let usage, usage.totalRoutes > 0 {
            for il in 0..<DSV4Shape.nLayer {
                let conc = usage.concentration(layer: il, n: 8)
                let top = usage.top(layer: il, n: 8).map(String.init).joined(separator: ",")
                guard conc > 0 else { continue }
                summaries.append(String(format: "L%-3d top8 = %2.0f%%  ·  expert %@", il, conc * 100, top))
            }
        }
        return TuningInfo(agentId: agentId,
                          cacheSlots: decoder.slotCache?.slotsPerLayer ?? 0,
                          cacheHits: decoder.slotCache?.hits ?? 0,
                          cacheMisses: decoder.slotCache?.misses ?? 0,
                          totalRoutes: usage?.totalRoutes ?? 0,
                          layerSummaries: summaries)
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
        self.systemPrompt = (systemPrompt?.isEmpty == false) ? systemPrompt : nil
        committedIds = []
        needsClose = false
        kvDirty = false   // next generation starts at pos 0 and resets the compressor
    }

    /// Declare the tools available to the model. Tools are baked into the first
    /// prompt of a conversation, so a change takes effect on the next new chat.
    public func setTools(_ tools: [ToolSpec]) { self.tools = tools }

    /// Use the compact (name-list) tool declaration to save prefill tokens.
    public func setCompactTools(_ on: Bool) { compactTools = on }

    private func assistantOpen(_ think: DS4ThinkMode) -> String {
        "<｜Assistant｜>" + (think == .high ? "<think>" : "</think>")
    }

    /// The prefix that opens a new user/tool turn: BOS + system (+ tools) the first
    /// time, otherwise the <eos> that closes the previous (still-open) assistant turn.
    private func openingPrefix() -> String {
        if committedIds.isEmpty {
            let sys = ChatRenderer.systemBlock(turns: systemPrompt.map { [.system($0)] } ?? [],
                                               tools: tools, markup: markup, compact: compactTools)
            return "<｜begin▁of▁sentence｜>" + sys
        }
        return needsClose ? "<｜end▁of▁sentence｜>" : ""
    }

    /// Append a user message and generate the assistant reply (prefills only the
    /// new suffix, reusing the KV cache of the prior turns).
    public func send(userText: String, thinkMode: DS4ThinkMode, sampling: SamplingParams,
                     maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        let suffix = openingPrefix() + "<｜User｜>" + userText + assistantOpen(thinkMode)
        return run(suffix: suffix, think: thinkMode, sampling: sampling, maxTokens: maxTokens)
    }

    /// Append tool results (inside a user turn) and continue the assistant turn.
    public func provideToolResults(_ outputs: [ToolOutput], thinkMode: DS4ThinkMode,
                                   sampling: SamplingParams, maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        var suffix = openingPrefix() + "<｜User｜>"
        for o in outputs { suffix += "<tool_result>" + o.content + "</tool_result>" }
        suffix += assistantOpen(thinkMode)
        return run(suffix: suffix, think: thinkMode, sampling: sampling, maxTokens: maxTokens)
    }

    private func run(suffix: String, think: DS4ThinkMode, sampling: SamplingParams,
                     maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.generate(suffix: suffix, think: think, sampling: sampling,
                                            maxTokens: maxTokens, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func generate(suffix: String, think: DS4ThinkMode, sampling: SamplingParams, maxTokens: Int,
                          continuation: AsyncThrowingStream<GenEvent, Error>.Continuation) throws {
        let suffixIds = tok.tokenizeRenderedChat(suffix).map { Int($0) }
        let startPos = committedIds.count
        guard startPos + suffixIds.count < contextSize else {
            throw InferenceError.contextExceeded(prompt: startPos + suffixIds.count, context: contextSize)
        }
        guard !suffixIds.isEmpty else { continuation.yield(.progress("")); return }

        // Dirty-until-clean: any throw below (user cancel, error) leaves the GPU
        // KV/compressor state possibly out of sync with committedIds; the flag makes
        // the NEXT generation rebuild before continuing.
        let needsRebuild = kvDirty
        kvDirty = true
        if needsRebuild && !committedIds.isEmpty {
            // Recover from an interrupted generation: replay the exact committed ids
            // from position 0 (resets the recurrent compressor) — slow once, correct.
            continuation.yield(.progress("ripristino KV (\(committedIds.count) token)…"))
            _ = try decoder.prefill(tokens: committedIds, startPos: 0)
        }

        // Prefill ONLY the new suffix; positions 0..startPos-1 are reused from the KV.
        continuation.yield(.progress(startPos == 0 ? "prefill \(suffixIds.count) token…"
                                                   : "prefill +\(suffixIds.count) token (riuso KV)…"))
        var lastLogits = try decoder.prefill(tokens: suffixIds, startPos: startPos)
        committedIds.append(contentsOf: suffixIds)
        // The committed KV now ends with an open assistant turn; mark it immediately
        // so a mid-decode interruption still closes the turn on the next suffix.
        needsClose = true
        var pos = committedIds.count

        var rng = sampling.seed
        var inReasoning = think == .high       // suffix ends with <think> when enabled
        var inTool = false
        var pending: [UInt8] = []
        var visible = ""
        var toolBytes: [UInt8] = []
        let dsmlId = tok.dsmlId

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
            if Int32(next) == tok.eosId { break }   // eos closes the turn; not forwarded (next suffix re-adds it)
            if !inTool, Int32(next) == dsmlId {
                flush(inReasoning)
                inTool = true
                toolBytes.append(contentsOf: tok.tokenText(Int32(next)))
            } else if inTool {
                toolBytes.append(contentsOf: tok.tokenText(Int32(next)))
            } else if Int32(next) == tok.thinkStartId {
                // The model opened a reasoning block on its own (even with think
                // off): route it to the reasoning stream, don't show the tag.
                flush(inReasoning)
                inReasoning = true
            } else if Int32(next) == tok.thinkEndId {
                // Close reasoning (also when we weren't in it: suppress a stray
                // literal "</think>" instead of showing it as text).
                flush(inReasoning)
                inReasoning = false
            } else {
                pending.append(contentsOf: tok.tokenText(Int32(next)))
                flush(inReasoning)
            }
            produced += 1
            lastLogits = try decoder.forward(token: next, pos: pos, nKeys: pos + 1)
            committedIds.append(next)           // the generated token is now in the KV
            pos += 1
            let elapsed = Date().timeIntervalSince(genStart)
            continuation.yield(.progress(String(format: "%d token · %.2f tok/s", produced,
                                                 elapsed > 0 ? Double(produced) / elapsed : 0)))
        }
        flush(inReasoning)

        // Extract any tool calls from the generated output.
        var calls: [ToolCall] = []
        if inTool {
            let block = String(bytes: toolBytes, encoding: .utf8) ?? ""
            calls = ToolCallParser.parse(visible + block, markup: markup).calls
            if calls.isEmpty {
                // The model opened a DSML block we could not parse: surface it
                // instead of dropping it silently (the user can see what it tried).
                continuation.yield(.text("\n[chiamata tool non interpretabile]\n" + block))
            }
        } else {
            let c = ToolCallParser.parse(visible, markup: markup).calls
            if !c.isEmpty { calls = c }
        }
        if !calls.isEmpty { continuation.yield(.toolCall(calls)) }
        kvDirty = false                         // clean completion: KV matches committedIds
        saveExpertUsage()                       // persist the usage imatrix (cheap)
        continuation.yield(.progress(""))
    }
}
