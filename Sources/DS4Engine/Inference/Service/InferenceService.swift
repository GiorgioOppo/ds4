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

public actor InferenceService: DS4Logging {
    public static let logTag = "engine"

    /// Executor dedicato su coda GCD SERIALE (SE-0392): tutto il lavoro
    /// dell'actor — prefill, decode, warmup, benchmark — gira su un thread
    /// GCD classico, lo stesso contesto della demo CLI (main thread). Il
    /// motore fa fan-out con DispatchQueue.concurrentPerform a ogni token
    /// (gather esperti, fill slot-cache, staging denso): chiamato da un
    /// thread del pool cooperativo di Swift Concurrency può degradare fino
    /// al quasi-seriale (un core al 100% — misurato sul requant Q4: ore
    /// invece di secondi). La coda è seriale, quindi la semantica actor non
    /// cambia; cambia solo il thread su cui gira.
    nonisolated let engineQueue = DispatchSerialQueue(label: "ds4.engine", qos: .userInitiated)
    public nonisolated var unownedExecutor: UnownedSerialExecutor { engineQueue.asUnownedSerialExecutor() }

    let rt: MetalRuntime
    let model: GGUFModel
    let tok: Tokenizer
    let decoder: StreamingDecoder
    let dims: DSV4Dims
    let runtimeGeometry: DSV4RuntimeGeometry
    let contextSize: Int
    let modelName: String
    let markup: ToolMarkup
    /// Architecture-neutral metadata + implemented runtime capabilities. The
    /// decoder stays concrete; this descriptor is for selection, UI and optional
    /// feature gates only.
    let backendDescriptor: RuntimeModelDescriptor

    // Append-only conversation state: `committedIds` are the exact token ids already
    // in the KV cache. Each turn prefills ONLY the new suffix and appends here, so
    // the KV is reused across turns (no full re-prefill). `needsClose` is true when
    // the committed KV ends with an open assistant turn (its <eos> not yet in the KV).
    var committedIds: [Int] = []
    var needsClose = false
    var systemPrompt: String?
    var tools: [ToolSpec] = []
    // Compact tool declaration (just name(params)) to save prefill tokens.
    // Defaults to true (local inference); the GUI toggle is the single source of
    // truth and pushes its value via setCompactTools (no env override).
    var compactTools = true
    /// Set when a generation was interrupted (cancel/error) mid-stream: the GPU
    /// KV cache and the recurrent NSA-compressor state may then be inconsistent
    /// with `committedIds`. The next generation rebuilds the KV from the exact
    /// committed ids (slow once, but correct) before continuing.
    var kvDirty = false
    /// Disk KV cache (ds4_kvstore model): nil = off. Checkpoints completed
    /// generations and restores matching prefixes on cold starts.
    var diskKV: DiskKVStore?
    var lastDiskStoreCount = 0
    /// Background checkpoint writers own large exported snapshots. Track them
    /// explicitly so an in-process model reload can join their RAM/SSD work
    /// instead of contaminating the following benchmark trial.
    var diskKVWriterTasks: [UUID: Task<Void, Never>] = [:]
    /// Content-keyed KV cache for sub-agents: one entry per file/project context
    /// (key = the content prefix tokens), built lazily on first use and reused.
    var subKV: DiskKVStore?

    /// The pure-Swift engine always runs the SSD-streaming path (no-copy mmap
    /// non-routed weights + per-token expert gather); there are no resident/
    /// per-layer variants to configure, hence no streaming options here.
    /// `expertCacheSlots` enables the per-layer expert slot-cache (0/nil = off);
    /// the persisted usage stats pre-warm it with the hottest experts.
    /// Engine revision stamp, printed to stderr at every init so the engine log
    /// always says WHICH build is running ("I rebuilt but nothing changed" is
    /// otherwise undiagnosable). Bump when engine behaviour changes materially.
    /// Feature state shared by the focused extensions below.
    var agentId = "generale"
    var warmedUp = false
    var lastPrefillProfile = "Profilo prefill: nessun prefill registrato."
    /// Session-scoped like upstream `ds4_session`: low-yield evidence and
    /// backoff survive assistant turns instead of charging every reply a fresh
    /// multi-second DSpark probe. Reset only with the conversation/model.
    var dsparkScheduler = DSparkAdaptiveScheduler()

    public static let engineRevision = "2026-08-06 DSpark adaptive scheduler + confidence-first"

    public init(modelPath: String, contextSize: Int, systemPrompt: String?,
                expertCacheSlots: Int? = nil,
                frozenUsageSeed: Data? = nil) throws {
        Self.log("revisione \(Self.engineRevision)")
        // Active DS4_* knobs, in the log of EVERY consumer (GUI included): "does
        // the app even see the env vars?" must be answerable from the log alone.
        let knobs = ["DS4_EXPERT_CACHE_SLOTS", "DS4_EXPERT_CACHE_UNIFORM", "DS4_EXPERT_CACHE_HOT_EVICTION", "DS4_EXPERT_CACHE_REUSE_EVICTION", "DS4_MULTI_QUANT_CACHE", "DS4_EXPERT_PREAD",
                     "DS4_WILLNEED_EXPERTS",
                     "DS4_PREFETCH", "DS4_PREFETCH_EXPERTS", "DS4_EXPERT_LOOKAHEAD", "DS4_ASYNC_FFN", "DS4_ASYNC_ROUTE", "DS4_PREFILL_UNION",
                     "DS4_PREFILL_FFN_BATCH", "DS4_PREFILL_ROUTE_BATCH", "DS4_PREFILL_CHUNK",
                     "DS4_PREFILL_MM", "DS4_POOL_INTERLEAVE", "DS4_Q8_NSG", "DS4_MOE_NSG", "DS4_DENSE_Q4_NSG",
                     "DS4_ACTIVE_EXPERTS", "DS4_RAW_RING", "DS4_RESIDENT_DENSE",
                     "DS4_DENSE_STREAM", "DS4_DENSE_AHEAD", "DS4_DENSE_Q4", "DS4_SHARED_Q4",
                     "DS4_QKV_Q4", "DS4_COMP_Q8", "DS4_LAZY_IDX", "DS4_GPU_INDEXER_TOPK", "DS4_ADAPTIVE_SPLITK", "DS4_DENSE_Q4_KERNEL", "DS4_FUSED_ROUTER_PROBS", "DS4_FUSED_ROUTER_FINALIZE", "DS4_FUSED_COMP_PROJ",
                     "DS4_VECTOR_COPY", "DS4_FLASH_KV_STAGE", "DS4_ROPE_PAIR", "DS4_ROPE_AFFINE",
                     "DS4_MTLIO", "DS4_RESIDENT_COMP", "DS4_FUSED_HC",
                     "DS4_MLOCK", "DS4_PROFILE_ROUTE", "DS4_Q4_CACHE_DIR",
                     "DS4_DSPARK_GGUF", "DS4_DSPARK_SCHEDULER",
                     "DS4_DSPARK_SCHEDULER_WINDOW",
                     "DS4_DSPARK_SCHEDULER_TAIL_MIN_TOKENS"]
        let env = ProcessInfo.processInfo.environment
        let knobLine = knobs.map { "\($0)=\(env[$0] ?? "·")" }.joined(separator: "  ")
        Self.log("knob \(knobLine)")
        Self.log("contextSize=\(contextSize) cacheSlots=\(expertCacheSlots.map(String.init) ?? "env/off")")
        // Inspect and select BEFORE constructing the DeepSeek tokenizer/config.
        // A recognized Qwen GGUF must fail as "backend non ancora implementato",
        // never as a missing deepseek4.* key after an expensive Metal bring-up.
        if let issue = ModelFileDiagnostics.openabilityIssue(path: modelPath) {
            throw GGUFError.cannotOpen(issue)
        }
        let openedModel = try GGUFModel(path: modelPath, metalMapping: true, prefetchCPU: false)
        let selection = try RuntimeBackendFactory.prepare(model: openedModel)
        // This actor IS the DeepSeek hot loop. A .glm52 or .laguna selection
        // must stop HERE with an actionable message, not fall through into
        // the DeepSeek tokenizer/config validators and their misleading
        // refusals — those chat surfaces are their own engines
        // (GLM52ChatService / LagunaChatService), hosted by ChatStore
        // directly. Reaching this guard means a DeepSeek-only surface
        // (benchmark, auto-tune, distributed) was pointed at a non-DeepSeek
        // GGUF.
        guard selection.backend == .deepSeekV4 else {
            throw GGUFError.cannotOpen(
                "backend \(selection.backend.rawValue) riconosciuto e "
                + "abilitato, ma questa superficie è solo DeepSeek: la chat "
                + "GUI serve \(selection.backend.rawValue) col suo motore "
                + "dedicato (ChatStore), mentre benchmark/auto-tune/"
                + "distribuito non hanno ancora quel backend.")
        }
        self.model = openedModel
        self.backendDescriptor = selection.descriptor
        Self.log("backend=\(selection.backend.rawValue) architecture=\(selection.descriptor.architecture.rawValue) model=\(selection.descriptor.displayName)")
        // Kernels are embedded in the binary — no metal/ folder needed.
        self.rt = try MetalRuntime()
        self.tok = try Tokenizer(model: model)
        // Validate the metadata like the C loader (config_validate_model), then
        // bind one immutable instance-owned geometry. Flash and Pro share the
        // same hot decoder implementation but never share static dimensions,
        // compression ratios or per-layer RoPE decisions.
        let config = try ModelConfig(model: model)
        let geometry = DSV4RuntimeGeometry(configuration: config)
        // Configure the MoE/router quant scheme from the GGUF (Q4_K+Q8 vs IQ2_XXS/Q2_K+F16).
        var configuredDims = geometry.dims
        let mq = GGUFWeights.detectMoEQuant(model)
        configuredDims.gateQuant = mq.gate; configuredDims.upQuant = mq.up
        configuredDims.downQuant = mq.down; configuredDims.routerF16 = mq.routerF16
        // Mixed-precision GGUFs (some routed layers upcast, e.g. to Q4_K) always
        // decode with their real per-layer geometry. The opt-in cache gives each
        // size class a byte-budgeted pool; the legacy path keeps bypassing them.
        let mixed = GGUFWeights.mixedPrecisionLayerCount(model, nLayers: geometry.nLayers)
        if mixed > 0 {
            let multiQuantCache = env["DS4_MULTI_QUANT_CACHE"] == "1"
            let cacheMode = multiQuantCache
                ? "served by byte-budgeted per-layer expert pools"
                : "bypassing legacy single-class expert cache"
            Self.log("mixed-precision GGUF: \(mixed)/\(geometry.nLayers) routed layers outside class \(mq.gate)/\(mq.up)/\(mq.down); \(cacheMode)")
        }
        // Optional active-experts override (DS4_ACTIVE_EXPERTS=2..6): fewer experts
        // per token = less expert I/O, lower quality. Honored by the streaming path.
        if let s = ProcessInfo.processInfo.environment["DS4_ACTIVE_EXPERTS"], let kk = Int(s) {
            configuredDims.activeExperts = max(1, min(kk, configuredDims.k))
        }
        self.dims = configuredDims
        self.runtimeGeometry = geometry
        self.contextSize = contextSize
        let loadedModelName = (modelPath as NSString).lastPathComponent
        self.modelName = loadedModelName
        self.markup = ToolMarkup.discover(in: tok)
        self.systemPrompt = (systemPrompt?.isEmpty == false) ? systemPrompt : nil
        let rope = geometry.ropeParams(layer: 0)
        // Fast 16GB path (C --ssd-streaming model): non-routed weights are no-copy mmap
        // (resident via page cache, evictable), only the 6 selected experts gathered/token.
        self.decoder = try StreamingDecoder.fromGGUFExpertCachedMapped(rt: rt, model: model, dims: dims, rope: rope,
                                                                       nLayers: geometry.nLayers, maxKeys: contextSize,
                                                                       cacheSlots: expertCacheSlots,
                                                                       geometry: geometry)
        if let requested = env["DS4_DSPARK_GGUF"], !requested.isEmpty {
            let supportPath = requested == "1"
                ? DSparkSupportModel.locate(near: modelPath)
                : requested
            if let supportPath,
               FileManager.default.fileExists(atPath: supportPath) {
                do {
                    try decoder.enableDSparkStage0(
                        supportPath: supportPath,
                        mainModelPath: modelPath)
                    let runtime = decoder.dsparkStage0Runtime
                    let residentMiB = (runtime?.residentWeightBytes ?? 0) >> 20
                    let mappedGiB = Double(runtime?.mappedTransformerBytes ?? 0)
                        / Double(1 << 30)
                    let kvMiB = (runtime?.privateKVBytes ?? 0) >> 20
                    Self.log(String(format:
                        "DSpark capture/stage0 + 3 transformer eseguibili: %.2f GiB mmap, %d MiB residenti, KV privato %d MiB",
                        mappedGiB, residentMiB, kvMiB))
                } catch {
                    Self.log("DSpark non attivato; decode target invariato: \(error)")
                }
            } else {
                Self.log("DSpark richiesto ma supporto completo abbinato non trovato; file .part ignorati")
            }
        }
        // Load the persisted usage stats ("usage imatrix") BEFORE any generation,
        // so the slot-cache warms with the historically hottest experts. The
        // profile is PER-AGENT: different roles route to different experts.
        let usageData: Data?
        if let frozenUsageSeed {
            usageData = frozenUsageSeed
        } else {
            usageData = InferenceService.usageDataSeeded(
                modelName: loadedModelName,
                agentId: "generale"
            )
        }
        if let data = usageData {
            decoder.usage?.load(data)
        }
        // Sub-agent KV cache (separate directory from the chat disk-KV; content-
        // keyed). Same 1M-token total budget as the chat store default: prefix
        // snapshots for big files/projects are exactly where reuse pays most.
        let subBits: UInt8 = configuredDims.gateQuant == .iq2_xxs ? 2 : 4
        self.subKV = try? DiskKVStore(directory: Self.subAgentKVDir(modelName: loadedModelName),
                                      budgetMB: 0, quantBits: subBits, contextSize: contextSize,
                                      budgetTokens: 1_000_000)
    }

    /// Metadata-only inspection used by the GUI before loading a backend. It does
    /// not allocate Metal resources and may return a recognized, unavailable
    /// architecture such as Qwen.
    public nonisolated static func inspectModel(path: String) throws -> RuntimeModelDescriptor {
        try RuntimeBackendFactory.inspect(modelPath: path)
    }

    /// Drain model-owned GPU/I/O background work before an in-process reload.
    /// The actor executor serializes this with every benchmark/generation call.
    public func quiesceForTeardown() async {
        decoder.quiesceForTeardown()
        while !diskKVWriterTasks.isEmpty {
            let writers = Array(diskKVWriterTasks.values)
            for writer in writers { await writer.value }
        }
    }

    /// Directory holding the per-file / per-project sub-agent KV caches.
    nonisolated static func subAgentKVDir(modelName: String) -> URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DwarfStar/subagent-kv", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}
