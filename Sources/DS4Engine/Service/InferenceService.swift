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
    /// Repetition penalty (llama.cpp `penalty_repeat`): >1 discourages re-emitting
    /// the last `repeatLastN` tokens — breaks the repeat-loop collapse on
    /// quantized models. 1.0 = off — the DEFAULT: the C original (ds4.c
    /// sample_top_p_min_p) has no penalty, so engine/server/demo stay faithful
    /// to it; the chat GUI opts in explicitly with its own user-set value.
    public var repetitionPenalty: Float
    public var repeatLastN: Int
    public init(temperature: Float = 0.6, topK: Int = 0, topP: Float = 0.95, minP: Float = 0.05,
                seed: UInt64 = 0xD54, repetitionPenalty: Float = 1.0, repeatLastN: Int = 64) {
        self.temperature = temperature; self.topK = topK; self.topP = topP; self.minP = minP
        self.seed = seed; self.repetitionPenalty = repetitionPenalty; self.repeatLastN = repeatLastN
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
    case toolStream(String)     // raw tool-call markup, streamed live during generation
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
    /// Disk KV cache (ds4_kvstore model): nil = off. Checkpoints completed
    /// generations and restores matching prefixes on cold starts.
    private var diskKV: DiskKVStore?
    private var lastDiskStoreCount = 0
    /// Content-keyed KV cache for sub-agents: one entry per file/project context
    /// (key = the content prefix tokens), built lazily on first use and reused.
    private var subKV: DiskKVStore?

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
        // Mixed-precision GGUFs (some routed layers upcast, e.g. to Q4_K): those
        // layers decode per-layer and bypass the single-class expert slot-cache,
        // reading experts via the mmap gather. Uniform models report 0 (no-op).
        let mixed = GGUFWeights.mixedPrecisionLayerCount(model, nLayers: DSV4Shape.nLayer)
        if mixed > 0 {
            FileHandle.standardError.write(Data(
                "ds4: GGUF a precisione mista: \(mixed)/\(DSV4Shape.nLayer) layer routed fuori dalla classe \(mq.gate)/\(mq.up)/\(mq.down) — decodificati per-layer, bypassano la cache esperti\n".utf8))
        }
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
        // Sub-agent KV cache (separate directory from the chat disk-KV; content-keyed).
        let subBits: UInt8 = configuredDims.gateQuant == .iq2_xxs ? 2 : 4
        self.subKV = try? DiskKVStore(directory: Self.subAgentKVDir(modelName: modelName),
                                      budgetMB: 8192, quantBits: subBits, contextSize: contextSize)
    }

    /// Directory holding the per-file / per-project sub-agent KV caches.
    nonisolated static func subAgentKVDir(modelName: String) -> URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DwarfStar/subagent-kv", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
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
        return ModelInfo(name: modelName, layers: DSV4Shape.nLayer, nEmbd: dims.nEmbd,
                         nVocab: dims.vocab, contextSize: contextSize, routedQuantBits: 4,
                         kvCacheBytes: estimatedKVCacheBytes())
    }

    /// Worst-case (full-context) KV-cache RAM, matching what the decoder actually
    /// allocates — so the figure reflects the raw-KV ring. Without the ring the raw
    /// cache is nLayer × ctx × headDim × F32 (dominant); with it the raw cache is a
    /// constant nSWA-row window and only the NSA-compressed rows (ctx/ratio) + the
    /// indexer scale with the context. A static ctx×headDim formula would keep
    /// reporting the huge number even with the ring on.
    private func estimatedKVCacheBytes() -> UInt64 {
        let ringOn = getenv("DS4_RAW_RING").map { String(cString: $0) == "1" } ?? false
        let headDim = UInt64(dims.headDim)
        let ctx = UInt64(contextSize)
        let rawRows = ringOn ? UInt64(min(dims.nSWA, contextSize)) : ctx
        var bytes: UInt64 = 0
        for il in 0..<DSV4Shape.nLayer {
            bytes += rawRows * headDim * 4                                  // raw cache (every layer)
            let ratio = DSV4Shape.compressRatio(layer: il)
            guard ratio > 0 else { continue }
            bytes += (ctx / UInt64(ratio)) * headDim * 4                   // NSA compressor cache
            if ratio == 4 {
                bytes += (ctx / 4) * UInt64(dims.nIndexerHeadDim) * 4       // indexer compressor (ratio-4 only)
            }
        }
        return bytes
    }

    /// The raw Jinja chat template embedded in the GGUF (for inspection), if any.
    public func chatTemplate() -> String? { model.string("tokenizer.chat_template") }

    /// Tokens currently committed to the KV (used to warn before the context fills).
    public func committedTokens() -> Int { committedIds.count }

    public struct BenchPoint: Sendable {
        public let contextTokens: Int
        public let prefillTps: Double
        public let genTps: Double
        public let kvBytes: UInt64
        public init(contextTokens: Int, prefillTps: Double, genTps: Double, kvBytes: UInt64) {
            self.contextTokens = contextTokens; self.prefillTps = prefillTps
            self.genTps = genTps; self.kvBytes = kvBytes
        }
    }

    /// Native benchmark (replaces the removed `ds4-bench` binary): prefill a
    /// synthetic prompt of `contextTokens` tokens and decode `genTokens` from it,
    /// returning prefill/generation throughput at that context frontier. Resets
    /// the conversation; `contextTokens` is clamped to fit the loaded context.
    public func benchmark(contextTokens: Int, genTokens: Int) throws -> BenchPoint {
        resetConversation(systemPrompt: nil)
        let ctx = max(8, min(contextTokens, contextSize - genTokens - 4))
        // Synthetic prompt: BOS + a tiled filler tokenization (output quality is
        // irrelevant for timing; the work — attention, MoE gather — is the same).
        var ids: [Int] = [Int(tok.bosId)]
        let filler = tok.tokenizeRenderedChat("The quick brown fox jumps over the lazy dog. ").map { Int($0) }
        let pad = filler.isEmpty ? [Int(tok.eosId)] : filler
        var i = 0
        while ids.count < ctx { ids.append(pad[i % pad.count]); i += 1 }
        ids = Array(ids.prefix(ctx))

        let t0 = Date()
        var lastLogits = try decoder.prefill(tokens: ids, startPos: 0)
        let prefillDt = Date().timeIntervalSince(t0)

        var pos = ids.count
        var rng: UInt64 = 0xD54
        var produced = 0
        let g0 = Date()
        while produced < genTokens {
            try Task.checkCancellation()
            let next = Sampler.sample(lastLogits, temperature: 0.6, topK: 0, topP: 0.95, minP: 0.05, rng: &rng)
            lastLogits = try decoder.forward(token: next, pos: pos, nKeys: pos + 1)
            pos += 1; produced += 1
        }
        let genDt = Date().timeIntervalSince(g0)
        kvDirty = true   // synthetic KV state — force a rebuild on the next real turn
        let kv = UInt64(DSV4Shape.nLayer) * UInt64(ctx) * UInt64(dims.headDim) * 4
        return BenchPoint(contextTokens: ctx,
                          prefillTps: prefillDt > 0 ? Double(ctx) / prefillDt : 0,
                          genTps: genDt > 0 && produced > 0 ? Double(produced) / genDt : 0,
                          kvBytes: kv)
    }

    // MARK: - Sub-agents (isolated context, returns only the answer)

    /// The outcome of a sub-agent run. `answer` is fed back to the main agent as a
    /// tool result (so the main KV commits only the question + this answer); `steps`
    /// is a display-only trace of the sub-agent's internal tool rounds.
    public struct SubAgentRun: Sendable {
        public let target: String
        public let question: String
        public let answer: String
        public let steps: [String]
        public init(target: String, question: String, answer: String, steps: [String]) {
            self.target = target; self.question = question; self.answer = answer; self.steps = steps
        }
    }

    private struct SubContext { let system: String; let content: String; let tools: [ToolSpec]; let label: String; let toolNames: [String] }

    /// Resolve a sub-agent target ("project"/"" or a project file path), an optional
    /// ROLE to assume (`agentId`), and the MINIMAL tool set into the sub-agent's
    /// system prompt, the content block that seeds the KV, and the declared tools.
    /// Granted tools = explicit `requested` (∩ grantable), else the role's tools,
    /// else a read-only default. Works WITHOUT an imported project: that is not an
    /// error — the sub-agent then runs on the task alone (no project content/tools).
    private func subContext(for target: String, agent agentId: String, toolNames requested: [String]) -> SubContext {
        let t = target.trimmingCharacters(in: .whitespacesAndNewlines)
        let isProject = t.isEmpty || t.lowercased() == "project" || t == "."
        let role = agentId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil
            : AgentRegistry.shared.all().first { $0.id == agentId.trimmingCharacters(in: .whitespacesAndNewlines) }
        let projectInfo = ProjectCache.shared.info()
        let fileText = (projectInfo != nil && !isProject) ? ProjectCache.shared.fullText(of: t) : nil

        // Granted = explicit ∩ grantable, else role's tools ∩ grantable, else default.
        var granted = requested.filter { ToolRegistry.subAgentGrantable.contains($0) }
        if granted.isEmpty, let role { granted = role.toolNames.filter { ToolRegistry.subAgentGrantable.contains($0) } }
        if granted.isEmpty {
            granted = projectInfo == nil ? []
                : (isProject ? ["project_list", "project_read", "project_search"]
                             : ["project_read", "project_search"])
        }
        // Without an imported project, project-scoped tools can't do anything → drop them.
        if projectInfo == nil { granted = granted.filter { !ToolRegistry.projectScoped.contains($0) } }
        var seen = Set<String>(); granted = granted.filter { seen.insert($0).inserted }   // stable de-dup
        let specs = ToolRegistry.specs(enabled: Set(granted))
        let toolLine = granted.isEmpty ? "Non hai tool: rispondi con le tue conoscenze."
                                       : "Tool a disposizione (usa SOLO questi): " + granted.joined(separator: ", ") + "."
        let rolePrefix = (role.map { $0.systemPrompt.isEmpty ? "" : $0.systemPrompt + "\n\n" }) ?? ""
        let roleLabel = role.map { " · \($0.name)" } ?? ""

        if let info = projectInfo, isProject {
            let map = ProjectCache.shared.fileList().prefix(200).joined(separator: "\n")
            let content = "Progetto «\(info.name)» — \(info.fileCount) file.\nMappa (parziale):\n\(map)\n\n"
            let sys = rolePrefix + "Sei un sub-agent autonomo che lavora SOLO sul progetto importato. \(toolLine) Concludi con una risposta sintetica: cosa hai trovato/fatto, con file:riga."
            return SubContext(system: sys, content: content, tools: specs, label: "progetto:\(info.name)\(roleLabel)", toolNames: granted)
        }
        if let text = fileText {
            let content = "Contenuto del file «\(t)» (già in contesto):\n```\n\(text)\n```\n\n"
            let sys = rolePrefix + "Sei un sub-agent focalizzato sul file «\(t)», già in contesto. \(toolLine) Se modifichi, agisci SOLO su questo file (find esatto e unico, indentazione inclusa). Concludi con una risposta sintetica."
            return SubContext(system: sys, content: content, tools: specs, label: "file:\(t)\(roleLabel)", toolNames: granted)
        }
        // No project imported (or the file isn't in it): a plain sub-agent that
        // answers the task directly — NOT an error (a chat may have no project).
        let note = projectInfo == nil ? "" : "Nota: «\(t)» non è nel progetto importato. "
        let sys = rolePrefix + "Sei un sub-agent. \(note)\(toolLine) Esegui il compito e concludi con una risposta sintetica."
        return SubContext(system: sys, content: "", tools: specs, label: "task\(roleLabel)", toolNames: granted)
    }

    /// Run an isolated sub-agent on `target` with `question`. The MAIN conversation
    /// KV is snapshotted and restored around the run, so the caller's context only
    /// ever sees the question (the tool call) and this answer (the tool result) —
    /// the sub-agent's internal tool rounds happen in a separate, discarded context.
    /// The target's content prefix is cached (content-keyed) and reused next time.
    public func runSubAgent(target: String, question: String, agent: String = "", tools: [String] = [],
                            maxTokens: Int = 1024, maxRounds: Int = .max) async throws -> SubAgentRun {
        let ctx = subContext(for: target, agent: agent, toolNames: tools)
        // A dirty main KV must be rebuilt before snapshotting so the restore is exact.
        if kvDirty, !committedIds.isEmpty { _ = try decoder.prefill(tokens: committedIds, startPos: 0); kvDirty = false }

        // Snapshot the MAIN context and restore it however the sub-agent ends.
        let savedIds = committedIds, savedClose = needsClose, savedDirty = kvDirty, savedDisk = lastDiskStoreCount
        let mainSnap: KVSnapshot? = savedIds.isEmpty ? nil : decoder.exportKV(nKeys: savedIds.count)
        defer {
            committedIds = savedIds; needsClose = savedClose; lastDiskStoreCount = savedDisk
            if let mainSnap {
                do { try decoder.importKV(mainSnap); kvDirty = savedDirty }
                catch { kvDirty = true }   // next main turn rebuilds from committedIds
            } else { kvDirty = true }
        }

        var steps: [String] = []
        let sampling = SamplingParams()

        // 1. Build or restore the content-keyed KV prefix (lazy cache).
        let prefixText = "<｜begin▁of▁sentence｜>"
            + ChatRenderer.systemBlock(turns: [.system(ctx.system)], tools: ctx.tools, markup: markup, compact: true)
            + "<｜User｜>" + ctx.content
        let prefixIds = tok.tokenizeRenderedChat(prefixText).map { Int($0) }
        guard prefixIds.count < contextSize - 32 else {
            return SubAgentRun(target: ctx.label, question: question,
                               answer: "Il contenuto di «\(ctx.label)» eccede il contesto del sub-agent.", steps: steps)
        }
        var pos = 0
        if let snap = subKV?.snapshot(forTokens: prefixIds, modelName: modelName) {
            try decoder.importKV(snap); pos = prefixIds.count
            steps.append("KV «\(ctx.label)» riusata (\(pos) token)")
        } else {
            _ = try decoder.prefill(tokens: prefixIds, startPos: 0); pos = prefixIds.count
            subKV?.store(tokens: prefixIds, modelName: modelName,
                         snapshot: decoder.exportKV(nKeys: pos), reason: .cold)
            steps.append("KV «\(ctx.label)» creata (\(pos) token)")
        }
        steps.append("tool: " + ctx.toolNames.joined(separator: ", "))

        // 2. Sub-agent tool loop: question → answer/tool-calls → results → … (bounded).
        var recent: [Int] = []
        var suffix = question + assistantOpen(.none)
        var answer = ""
        var round = 0
        while true {
            let suffixIds = tok.tokenizeRenderedChat(suffix).map { Int($0) }
            guard pos + suffixIds.count < contextSize else { steps.append("contesto sub-agent esaurito"); break }
            var lastLogits = try decoder.prefill(tokens: suffixIds, startPos: pos)
            pos += suffixIds.count
            let turn = try decodeSubTurn(lastLogits: &lastLogits, pos: &pos, recent: &recent,
                                         sampling: sampling, maxTokens: maxTokens)
            answer = turn.visible
            guard !turn.calls.isEmpty, round < maxRounds else { break }
            round += 1
            var results = ""
            for c in turn.calls {
                let out = ToolRegistry.execute(c)
                    ?? ToolOutput(callId: c.id, name: c.name, content: #"{"error":"tool non disponibile nel sub-agent"}"#)
                steps.append("\(c.name) \(c.argumentsJSON) → " + String(out.content.prefix(160)))
                results += "<tool_result>" + out.content + "</tool_result>"
            }
            suffix = "<｜end▁of▁sentence｜><｜User｜>" + results + assistantOpen(.none)
        }
        let final = answer.trimmingCharacters(in: .whitespacesAndNewlines)
        steps.append("risposta: \(final.count) caratteri")
        return SubAgentRun(target: ctx.label, question: question,
                           answer: final.isEmpty ? "(nessuna risposta)" : final, steps: steps)
    }

    /// Decode one assistant turn in the sub-agent context: returns the visible
    /// answer text and any tool calls. Reasoning (<think>…</think>) is discarded —
    /// only the answer and tool calls matter for the sub-agent.
    private func decodeSubTurn(lastLogits: inout [Float], pos: inout Int, recent: inout [Int],
                               sampling: SamplingParams, maxTokens: Int) throws
        -> (visible: String, calls: [ToolCall]) {
        var rng = sampling.seed &+ UInt64(pos)
        var inTool = false, inReasoning = false
        var visibleBytes: [UInt8] = []
        var toolBytes: [UInt8] = []
        let dsmlId = tok.dsmlId
        var produced = 0
        while produced < maxTokens && pos < contextSize {
            try Task.checkCancellation()
            let lo = max(0, recent.count - sampling.repeatLastN)
            let next = Sampler.sample(lastLogits, temperature: sampling.temperature, topK: sampling.topK,
                                      topP: sampling.topP, minP: sampling.minP,
                                      repetitionPenalty: sampling.repetitionPenalty,
                                      recent: recent[lo...], rng: &rng)
            if Int32(next) == tok.eosId { break }
            if !inTool, Int32(next) == dsmlId {
                inTool = true; toolBytes.append(contentsOf: tok.tokenText(Int32(next)))
            } else if inTool {
                toolBytes.append(contentsOf: tok.tokenText(Int32(next)))
            } else if Int32(next) == tok.thinkStartId {
                inReasoning = true
            } else if Int32(next) == tok.thinkEndId {
                inReasoning = false
            } else if !inReasoning {
                visibleBytes.append(contentsOf: tok.tokenText(Int32(next)))
            }
            produced += 1
            recent.append(next)
            lastLogits = try decoder.forward(token: next, pos: pos, nKeys: pos + 1)
            pos += 1
        }
        let visibleRaw = String(bytes: visibleBytes, encoding: .utf8) ?? ""
        let toolText = String(bytes: toolBytes, encoding: .utf8) ?? ""
        let parsed = ToolCallParser.parse(inTool ? visibleRaw + toolText : visibleRaw, markup: markup)
        return (parsed.visibleText.trimmingCharacters(in: .whitespacesAndNewlines), parsed.calls)
    }

    /// Per-phase decode timing (route/attn vs expert gather I/O vs experts compute…).
    public func resetDecodeProfile() { decoder.resetProfile() }
    public func decodeProfileReport() -> String { decoder.profile.report() }

    public func resetConversation(systemPrompt: String?) {
        self.systemPrompt = (systemPrompt?.isEmpty == false) ? systemPrompt : nil
        committedIds = []
        needsClose = false
        kvDirty = false   // next generation starts at pos 0 and resets the compressor
        lastDiskStoreCount = 0
    }

    /// Enable/disable the disk KV cache. `dir` nil turns it off. Takes effect on
    /// the next generation; existing checkpoints in `dir` become restorable.
    public func setDiskKV(directory: URL?, budgetMB: Int) {
        guard let directory else { diskKV = nil; return }
        let bits: UInt8 = dims.gateQuant == .iq2_xxs ? 2 : 4
        diskKV = try? DiskKVStore(directory: directory, budgetMB: budgetMB,
                                  quantBits: bits, contextSize: contextSize)
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

    /// Re-prime a REOPENED conversation whose KV the engine no longer holds (the
    /// GUI restored a persisted chat), then generate the reply. The prior `history`
    /// turns + the new user turn are rendered as one prompt: on a cold KV the disk
    /// cache restores the longest matching prefix (so this is NOT a full re-prefill
    /// when the chat was checkpointed), and only the remainder is prefilled. After
    /// this call the KV holds the whole conversation, so the next turns reuse it
    /// incrementally via `send`. `tools` (set by the agent) are preserved.
    public func sendWithHistory(_ history: [ChatTurn], userText: String, systemPrompt: String?,
                                thinkMode: DS4ThinkMode, sampling: SamplingParams,
                                maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        resetConversation(systemPrompt: systemPrompt)
        var turns: [ChatTurn] = self.systemPrompt.map { [.system($0)] } ?? []
        turns.append(contentsOf: history)
        turns.append(.user(userText))
        let suffix = ChatRenderer.render(turns: turns, tools: tools, think: thinkMode.core,
                                         markup: markup, compactTools: compactTools)
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

    /// Stateless completion for the local HTTP server: reset the KV and render the
    /// FULL message list as a fresh prompt, then generate. Mirrors OpenAI semantics
    /// where each request carries the whole conversation (no server-side history).
    public func complete(turns: [ChatTurn], tools: [ToolSpec], thinkMode: DS4ThinkMode,
                         sampling: SamplingParams, maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        resetConversation(systemPrompt: nil)
        self.tools = tools
        let suffix = ChatRenderer.render(turns: turns, tools: tools, think: thinkMode.core,
                                         markup: markup, compactTools: compactTools)
        return run(suffix: suffix, think: thinkMode, sampling: sampling, maxTokens: maxTokens)
    }

    private func run(suffix: String, think: DS4ThinkMode, sampling: SamplingParams,
                     maxTokens: Int) -> AsyncThrowingStream<GenEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try self.generate(suffix: suffix, think: think, sampling: sampling,
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
        var suffixIds = tok.tokenizeRenderedChat(suffix).map { Int($0) }

        // Disk KV (ds4_kvstore model): on a COLD start, restore the longest stored
        // checkpoint that is an exact token prefix of this prompt and prefill only
        // the remainder. Covers both a fresh chat (system/agent prefix) and the
        // stateless HTTP server (each request re-sends the whole transcript).
        if committedIds.isEmpty, !kvDirty, let store = diskKV,
           let hit = store.findLongestPrefix(of: suffixIds, modelName: modelName) {
            continuation.yield(.progress("ripristino KV da disco (\(hit.tokens.count) token)…"))
            do {
                try decoder.importKV(hit.snapshot)
                committedIds = hit.tokens
                suffixIds.removeFirst(hit.tokens.count)
                lastDiskStoreCount = hit.tokens.count
            } catch {
                committedIds = []          // fall back to a cold prefill
            }
        }

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
        var toolEmitted = 0
        let dsmlId = tok.dsmlId
        let lt = UInt8(ascii: "<")

        func flush(_ asReasoning: Bool) {
            guard !pending.isEmpty, let s = String(bytes: pending, encoding: .utf8) else { return }
            pending.removeAll(keepingCapacity: true)
            if asReasoning { continuation.yield(.reasoning(s)) }
            else { visible += s; continuation.yield(.text(s)) }
        }

        // Stream the not-yet-emitted suffix of the tool block as raw markup, so the
        // user watches the tool call being generated. Holds back partial UTF-8.
        func streamTool() {
            guard toolEmitted < toolBytes.count,
                  let s = String(bytes: toolBytes[toolEmitted...], encoding: .utf8) else { return }
            toolEmitted = toolBytes.count
            continuation.yield(.toolStream(s))
        }

        // Flush pending text but keep a trailing '<' buffered: it may begin the
        // tool-call opener "<｜DSML｜…" (the '<' and ｜DSML｜ are separate tokens) and
        // must not be streamed as a stray bubble before we know what follows.
        func flushHoldingOpener(_ asReasoning: Bool) {
            guard pending.last == lt else { flush(asReasoning); return }
            pending.removeLast()
            flush(asReasoning)        // emit everything before the '<'
            pending.append(lt)        // re-buffer the '<' for the next round
        }

        var produced = 0
        let genStart = Date()
        while produced < maxTokens && pos < contextSize {
            try Task.checkCancellation()
            // Penalize the recently produced tokens to break repeat-loop collapse.
            let lo = max(0, committedIds.count - sampling.repeatLastN)
            let next = Sampler.sample(lastLogits, temperature: sampling.temperature, topK: sampling.topK,
                                      topP: sampling.topP, minP: sampling.minP,
                                      repetitionPenalty: sampling.repetitionPenalty,
                                      recent: committedIds[lo...], rng: &rng)
            if Int32(next) == tok.eosId { break }   // eos closes the turn; not forwarded (next suffix re-adds it)
            if !inTool, Int32(next) == dsmlId {
                // A held opener '<' belongs to the tool block, not the visible text:
                // move it into toolBytes (so the parser sees "<｜DSML｜…") without ever
                // streaming it as a stray bubble.
                if pending.last == lt { pending.removeLast(); toolBytes.append(lt) }
                flush(inReasoning)                               // flush any remaining real text
                inTool = true
                toolBytes.append(contentsOf: tok.tokenText(Int32(next)))
                streamTool()                                     // begin streaming the raw markup
            } else if inTool {
                toolBytes.append(contentsOf: tok.tokenText(Int32(next)))
                streamTool()
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
                flushHoldingOpener(inReasoning)
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
        // Disk KV checkpoint (interval-gated: each entry is tens of MB).
        if let store = diskKV,
           committedIds.count - lastDiskStoreCount >= store.options.storeIntervalTokens {
            continuation.yield(.progress("salvataggio KV su disco…"))
            // First checkpoint of a conversation = "cold" (anchor: the shared
            // system/agent prefix, 2× protected in eviction); later = "continued"
            // (superseded under pressure by longer checkpoints of the same chat).
            let reason: KVCFile.Reason = lastDiskStoreCount == 0 ? .cold : .continued
            store.store(tokens: committedIds, modelName: modelName,
                        snapshot: decoder.exportKV(nKeys: committedIds.count), reason: reason)
            lastDiskStoreCount = committedIds.count   // gate even on dedup/failure
        }
        continuation.yield(.progress(""))
    }
}
