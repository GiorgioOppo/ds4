import Foundation
import DS4Core
import DS4Metal

/// Node-level engine for distributed inference. Builds the same streaming decoder
/// as `InferenceService` but exposes the low-level slice ops (embed / forwardSlice
/// / head) the coordinator and workers drive directly, plus tokenization and
/// sampling for the coordinator. Stateless w.r.t. conversation: the coordinator
/// sequences positions explicitly across the cluster.
///
/// Memory note: layer weights are no-copy mmap loaded **lazily** per layer, so a
/// worker that only calls `forwardSlice` over its range never faults in the other
/// layers' weights — that is where the per-node memory saving comes from.
public final class DistEngine: @unchecked Sendable {
    public let modelName: String
    public let nLayers: Int
    public let nExperts: Int
    public let headDim: Int
    public let contextSize: Int
    private let rt: MetalRuntime
    private let model: GGUFModel
    private let tok: Tokenizer
    private let decoder: StreamingDecoder
    private let markup: ToolMarkup
    /// Checkpoint header quant tag (same rule as InferenceService.setDiskKV).
    private let kvQuantBits: UInt8
    /// Worker-side disk KV for this shard (enabled by the coordinator's ASSIGN).
    private var diskKV: DiskKVStore?

    /// `kvLayers` restricts KV/compressor allocation to a layer range (a worker
    /// allocates only its slice's caches; a pure coordinator can pass 0..<0).
    /// nil = full model. `onLoadLog` marks each init PHASE (breadcrumbs for the
    /// worker log: LoadProgress only covers the decoder factory, so a stall in
    /// Metal init or mmap would otherwise be silent and undiagnosable).
    public init(modelPath: String, contextSize: Int, expertCacheSlots: Int? = nil,
                kvLayers: Range<Int>? = nil,
                onLoadLog: (@Sendable (String) -> Void)? = nil) throws {
        onLoadLog?("init Metal runtime (compilazione kernel)…")
        self.rt = try MetalRuntime()
        onLoadLog?("mmap gguf…")
        self.model = try GGUFModel(path: modelPath, metalMapping: true, prefetchCPU: false)
        let backend = try RuntimeBackendFactory.prepare(model: model)
        onLoadLog?("backend \(backend.backend.rawValue) · \(backend.descriptor.architecture.rawValue)")
        onLoadLog?("tokenizer…")
        self.tok = try Tokenizer(model: model)
        // Same load-time validation as InferenceService: metadata like the C
        // config_validate_model and the closed expert-quant set. Geometry is
        // instance-owned: a Flash and a Pro engine may coexist in one process
        // without sharing static dimensions, compression ratios or RoPE.
        let config = try ModelConfig(model: model)
        let geometry = DSV4RuntimeGeometry(configuration: config)
        var dims = geometry.dims
        let mq = GGUFWeights.detectMoEQuant(model)
        dims.gateQuant = mq.gate; dims.upQuant = mq.up; dims.downQuant = mq.down; dims.routerF16 = mq.routerF16
        if let kvLayers {
            guard kvLayers.lowerBound >= 0,
                  kvLayers.upperBound <= geometry.nLayers else {
                throw GGUFWeights.LoadError.message(
                    "distributed KV slice \(kvLayers) outside model layers 0..<\(geometry.nLayers)")
            }
        }
        guard contextSize > 0 else {
            throw GGUFWeights.LoadError.message("distributed context size must be positive")
        }
        self.contextSize = contextSize
        self.nLayers = geometry.nLayers
        self.nExperts = dims.nExperts
        self.headDim = dims.headDim
        self.modelName = (modelPath as NSString).lastPathComponent
        let rope = geometry.ropeParams(layer: 0)
        self.markup = ToolMarkup.discover(in: tok)
        self.kvQuantBits = dims.gateQuant == .iq2_xxs ? 2 : 4
        onLoadLog?("geometria \(config.shape.name): \(geometry.nLayers) layer · \(dims.nExperts) esperti")
        onLoadLog?("costruzione decoder (pesi, cache, KV)…")
        self.decoder = try StreamingDecoder.fromGGUFExpertCachedMapped(
            rt: rt, model: model, dims: dims, rope: rope, nLayers: geometry.nLayers,
            maxKeys: contextSize, cacheSlots: expertCacheSlots, kvLayers: kvLayers,
            geometry: geometry)
        onLoadLog?("decoder pronto")
    }

    /// Metadata-only geometry inspection used by workers before allocating
    /// Metal/KV for a coordinator-provided slice. The returned values come from
    /// the validated GGUF, never from the local build's legacy Flash constants.
    static func inspectLayout(modelPath: String) throws -> (nLayers: Int, nExperts: Int) {
        let inspected = try GGUFModel(path: modelPath, metalMapping: false, prefetchCPU: false)
        _ = try RuntimeBackendFactory.prepare(model: inspected)
        let geometry = DSV4RuntimeGeometry(configuration: try ModelConfig(model: inspected))
        return (geometry.nLayers, geometry.dims.nExperts)
    }

    // MARK: Usage imatrix (worker slot-cache pre-warm)

    /// Load a usage profile (ExpertUsageStats JSON): the slot cache pre-warms
    /// with the historically hottest experts on first use.
    public func loadUsage(_ data: Data) { decoder.usage?.load(data) }

    /// Serialized usage collected so far (nil when the decoder tracks none).
    public func usageData() -> Data? { decoder.usage?.serialize() }

    // MARK: Verticale (expert parallelism, Fase C)

    /// Instrada la FFN routed sui worker remoti: (layer, id, pesi, attivazione)
    /// → somma parziale già ridotta. nil = percorso locale. Impostare PRIMA di
    /// qualunque decode.
    public func setRemoteExperts(_ f: (@Sendable (Int, [Int32], [Float], [Float]) throws -> [Float])?) {
        decoder.remoteExperts = f
    }

    /// Prefill/forward diretti del decoder (route verticale: il backbone è
    /// QUESTO motore). BLOCCANTI sui round-trip di rete quando remoteExperts
    /// è impostata — chiamare via Task.detached, mai sul cooperative pool.
    public func verticalPrefill(tokens: [Int], startPos: Int) throws -> [Float] {
        try decoder.prefill(tokens: tokens, startPos: startPos)
    }
    public func verticalForward(token: Int, pos: Int) throws -> [Float] {
        try decoder.forward(token: token, pos: pos, nKeys: pos + 1)
    }

    /// Benchmark del motore verticale (backbone locale + esperti remoti):
    /// identico a InferenceService.benchmark — prompt sintetico, prefill, poi
    /// genTokens di decode con velocità per-token. Da chiamare FUORI dal main
    /// actor e fuori dal cooperative pool (il decode blocca sul round-trip di
    /// rete): Task.detached.
    public func benchmarkVertical(contextTokens: Int, genTokens: Int) throws -> InferenceService.BenchPoint {
        let ctx = max(8, min(contextTokens, contextSize - genTokens - 4))
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
        var tokenSpeeds: [Double] = []
        let g0 = Date()
        while produced < genTokens {
            try Task.checkCancellation()
            let next = Sampler.sample(lastLogits, temperature: 0.6, topK: 0, topP: 0.95, minP: 0.05, rng: &rng)
            let t = Date()
            lastLogits = try decoder.forward(token: next, pos: pos, nKeys: pos + 1)
            let dt = Date().timeIntervalSince(t)
            if dt > 0 { tokenSpeeds.append(1.0 / dt) }
            pos += 1; produced += 1
        }
        let genDt = Date().timeIntervalSince(g0)
        var p99 = 0.0
        if !tokenSpeeds.isEmpty {
            let sorted = tokenSpeeds.sorted()
            let latencyRank = Int(ceil(0.99 * Double(sorted.count)))
            p99 = sorted[max(0, sorted.count - latencyRank)]
        }
        let kv = UInt64(nLayers) * UInt64(ctx) * UInt64(headDim) * 4
        return InferenceService.BenchPoint(contextTokens: ctx,
                                           prefillTps: prefillDt > 0 ? Double(ctx) / prefillDt : 0,
                                           genTps: genDt > 0 && produced > 0 ? Double(produced) / genDt : 0,
                                           kvBytes: kv, genTpsP99: p99, genSpeeds: tokenSpeeds)
    }

    // MARK: Worker-side disk KV (shard checkpoints, driven by the coordinator)

    /// Enable/disable shard checkpoints. Budget in TOKENS; nil directory = off.
    public func setDiskKV(directory: URL?, budgetTokens: Int) {
        guard let directory, budgetTokens > 0 else { diskKV = nil; return }
        diskKV = try? DiskKVStore(directory: directory, budgetMB: 0, quantBits: kvQuantBits,
                                  contextSize: contextSize, budgetTokens: budgetTokens)
    }

    /// Stored checkpoint lengths that are strict prefixes of `ids` (for the
    /// coordinator's restore negotiation). Empty when disk KV is off.
    public func storedPrefixLengths(of ids: [Int]) -> [Int] {
        diskKV?.storedPrefixLengths(of: ids, modelName: modelName) ?? []
    }

    /// Restore the shard checkpoint stored for EXACTLY `tokens` (streamed one
    /// layer at a time into the shard's KV buffers). false when absent/corrupt.
    public func restoreKV(tokens: [Int]) -> Bool {
        guard let diskKV else { return false }
        return diskKV.restore(forTokens: tokens, modelName: modelName, into: decoder)
    }

    /// Checkpoint the shard for `tokens`: the KV export happens synchronously
    /// (the state must not move underneath it); the F_NOCACHE write streams in
    /// the background from a uniquely-owned box (layers freed as written).
    public func saveKV(tokens: [Int], cold: Bool) {
        guard let diskKV, !tokens.isEmpty else { return }
        let box = DiskKVStore.SnapshotBox(decoder.exportKV(nKeys: tokens.count))
        let name = modelName
        Task.detached(priority: .utility) {
            diskKV.store(tokens: tokens, modelName: name, box: box,
                         reason: cold ? .cold : .continued)
        }
    }

    public var thinkStartId: Int { Int(tok.thinkStartId) }
    public var thinkEndId: Int { Int(tok.thinkEndId) }
    public var dsmlId: Int { Int(tok.dsmlId) }

    /// Render a FULL conversation to token ids (BOS + system + tools + turns +
    /// assistant open) — the coordinator re-renders the whole chat each turn
    /// (stateless). Tools use the compact declaration like the local chat.
    public func chatPromptIds(turns: [ChatTurn], tools: [ToolSpec] = [], think: Bool) -> [Int] {
        let safeTurns = turns.map { turn -> ChatTurn in
            switch turn {
            case .system(let text):
                return .system(tok.neutralizeSpecialTokenLiterals(in: text))
            case .user(let text):
                return .user(tok.neutralizeSpecialTokenLiterals(in: text))
            case .assistant(let text, let calls):
                let safeCalls = calls.map {
                    ToolCall(id: $0.id,
                             name: tok.neutralizeSpecialTokenLiterals(in: $0.name),
                             argumentsJSON: tok.neutralizeSpecialTokenLiterals(inJSON: $0.argumentsJSON))
                }
                return .assistant(text: tok.neutralizeSpecialTokenLiterals(in: text),
                                  toolCalls: safeCalls)
            case .toolResult(let callId, let name, let content):
                return .toolResult(callId: callId,
                                   name: tok.neutralizeSpecialTokenLiterals(in: name),
                                   content: tok.neutralizeSpecialTokenLiterals(in: content))
            }
        }
        let safeTools = tools.map {
            ToolSpec(name: tok.neutralizeSpecialTokenLiterals(in: $0.name),
                     description: tok.neutralizeSpecialTokenLiterals(in: $0.description),
                     parametersJSON: tok.neutralizeSpecialTokenLiterals(inJSON: $0.parametersJSON))
        }
        let s = ChatRenderer.render(turns: safeTurns, tools: safeTools, think: think ? .high : .none,
                                    markup: markup, compactTools: true)
        return tok.tokenizeRenderedChat(s).map { Int($0) }
    }

    /// Parse DSML tool calls out of generated text (coordinator-side tool loop).
    public func parseToolCalls(_ text: String) -> (calls: [ToolCall], visible: String) {
        do {
            let r = try ToolCallParser.parseStrict(text, markup: markup)
            return (r.calls, r.visibleText)
        } catch {
            // Distributed execution has the same all-or-nothing boundary as the
            // local engine: recovery parsing is display-only and cannot authorize
            // a truncated or structurally malformed call.
            let visible = ToolCallParser.parse(text, markup: markup).visibleText
            return ([], visible)
        }
    }

    /// HC state width crossing the wire (nHC * nEmbd floats).
    public var hcStateCount: Int { decoder.hcStateCount }

    // MARK: Slice ops (delegate to the decoder)

    public func embed(token: Int, pos: Int) throws -> [Float] {
        try decoder.embed(token: token, pos: pos)
    }

    public func forwardSlice(hc: [Float], pos: Int, nKeys: Int, start: Int, end: Int,
                             token: Int = -1) throws -> [Float] {
        try decoder.forwardSlice(hc: hc, pos: pos, nKeys: nKeys, start: start, end: end, token: token)
    }

    /// Chunked prefill: run the slice over `hcs.count` consecutive tokens from `posBase`.
    /// `tokens` (one id per state) feeds the hash-routed layers (0..2).
    public func forwardSliceBatch(hcs: [[Float]], posBase: Int, start: Int, end: Int,
                                  tokens: [Int]? = nil) throws -> [[Float]] {
        try decoder.forwardSliceBatch(hcs: hcs, posBase: posBase, start: start, end: end, tokens: tokens)
    }

    public func head(hc: [Float]) throws -> [Float] {
        try decoder.head(hc: hc)
    }

    // MARK: Coordinator helpers

    /// Tokenize a rendered chat / prompt string into token ids.
    public func tokenize(_ text: String) -> [Int] {
        tok.tokenizeRenderedChat(text).map { Int($0) }
    }

    public func tokenText(_ id: Int) -> String {
        String(bytes: tok.tokenText(Int32(id)), encoding: .utf8) ?? ""
    }

    public var eosId: Int { Int(tok.eosId) }
    public var bosId: Int { Int(tok.bosId) }

    /// Sample the next token id from a logits vector, penalizing `recent` tokens.
    public func sample(_ logits: [Float], params: SamplingParams, recent: ArraySlice<Int> = ArraySlice<Int>(),
                       rng: inout UInt64) -> Int {
        Sampler.sample(logits, temperature: params.temperature, topK: params.topK,
                       topP: params.topP, minP: params.minP,
                       repetitionPenalty: params.repetitionPenalty, recent: recent, rng: &rng)
    }
}
