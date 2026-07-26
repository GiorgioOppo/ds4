import SwiftUI
import AppKit
import UniformTypeIdentifiers
import DS4Engine
import DS4Core

/// Immutable identity of the load-time state that actually constructed the
/// visible inference service. Settings remain editable for the next load, so
/// the rigorous tuner must compare against this snapshot rather than assuming
/// that today's controls describe the engine already resident in memory.
struct LoadedEngineSignature: Equatable, Sendable {
    let modelPath: String
    let contextSize: Int
    let tuning: MachineAutoTuneConfiguration
    let fixedEnvironment: [String: String]
}

/// Main-thread view model. Owns the `InferenceService` actor and mirrors its
/// streamed output into observable UI state.
@MainActor
@Observable
final class ChatStore {
    enum Phase: Equatable {
        case needsModel
        case loading
        case ready
        case failed(String)
    }

    // Configuration (editable before loading). Defaults adapt to dev vs bundle.
    // NOTE: the pure-Swift engine always runs the SSD-streaming path (mmap no-copy
    // + per-token expert gather); the old C-engine streaming/RAM-mode toggles were
    // dead and have been removed.
    /// Shared app settings: model path / context / mode are owned by the
    /// Impostazioni screen; this store (like every other controller) proxies them.
    let settings: AppSettings
    var modelPath: String {
        get { settings.modelPath }
        set {
            if settings.modelPath != newValue { inspectedModelDescriptor = nil }
            settings.modelPath = newValue
        }
    }
    var contextSize: Int {
        get { settings.contextSize }
        set { settings.contextSize = newValue }
    }
    var scriptDir = AppEnvironment.resourceDir   // download_model.sh / gguf

    init(settings: AppSettings) {
        self.settings = settings
        AgentRegistry.shared.set(agents)   // didSet doesn't fire for the initial value
        // MCP servers connect asynchronously: when one (dis)connects, bump the
        // observable mirror (so pickers listing MCP tools re-render) and re-push
        // the declared tools to the engine (specs exist only while connected —
        // without this, an agent with mcp_* tools active at launch would never
        // expose them to the model).
        MCPManager.shared.addChangeHandler { [weak self] in
            Task { @MainActor in
                guard let self else { return }
                self.mcpVersion = MCPManager.shared.version
                self.syncTools()
            }
        }
        // Migrazione storica v1 del preset 2026-07. Mantiene i valori di quella
        // release per compatibilità, ma NON descrive il profilo GUI corrente: la
        // migrazione v8 più sotto rende il raw-KV ring obbligatoriamente ON e
        // allinea i nuovi knob persistibili. Dentro init i didSet non scattano,
        // quindi ogni migrazione scrive esplicitamente i propri valori.
        if !UserDefaults.standard.bool(forKey: "DS4FastConfig2026_07") {
            UserDefaults.standard.set(true, forKey: "DS4FastConfig2026_07")
            expertCacheSlots = 16
            UserDefaults.standard.set(16, forKey: "DS4ExpertCacheSlots")
            rawRingEnabled = false
            UserDefaults.standard.set(false, forKey: "DS4RawRing")
            if settings.contextSize > 32768 { settings.contextSize = 8192 }
        }
        // Migrazione UNA TANTUM v2 (2026-07-04): allinea TUTTI i toggle alla
        // configurazione della demo veloce (8 tok/s prefill / 2.5+ decode) —
        // i valori persistiti derivano dagli esperimenti fatti in Settings
        // (es. bundle spento per un test A/B) e restavano incollati per sempre.
        // NB: dentro init i didSet non scattano — persistenza esplicita; i
        // setenv qui sotto leggono già i valori allineati.
        if !UserDefaults.standard.bool(forKey: "DS4DemoAlign2026_07_04") {
            UserDefaults.standard.set(true, forKey: "DS4DemoAlign2026_07_04")
            applyFastDemoDefaults(persistExplicitly: true)
        }
        // Migrazione UNA TANTUM v3 (2026-07-06): allinea alla configurazione
        // MISURATA della sessione di tuning (matrice completa di A/B in demo):
        // slot 16, dense-ahead 2, look-ahead 0 (speculativo neutro, hash layers
        // sempre attivi), niente SHARED_Q4. I valori sperimentali persistiti
        // (slot 12, look-ahead 4) derivavano dai test intermedi.
        if !UserDefaults.standard.bool(forKey: "DS4MeasuredAlign2026_07_06") {
            UserDefaults.standard.set(true, forKey: "DS4MeasuredAlign2026_07_06")
            applyFastDemoDefaults(persistExplicitly: true)
        }
        // Migrazione UNA TANTUM v4 (2026-07-08): matrice A/B in demo su M1 Pro
        // 16 GB — QKV_Q4 promosso (+10%, output coerente) e slot 24 (73% hit,
        // 3.33 tok/s di regime, nessun collasso). MOE_NSG=4 confermato,
        // PREAD_SPLIT e allineamento pread falsificati dal probe DIAG.
        // Dettagli e numeri: docs/VALUTAZIONE-DEMO-PERF.md §8.
        if !UserDefaults.standard.bool(forKey: "DS4MeasuredAlign2026_07_08") {
            UserDefaults.standard.set(true, forKey: "DS4MeasuredAlign2026_07_08")
            applyFastDemoDefaults(persistExplicitly: true)
        }
        // Migrazione UNA TANTUM v5 (2026-07-08, sera): union 256 (+19% prefill
        // misurato a 3.5k token) e SHARED_Q4 (+7% decode, output coerente).
        if !UserDefaults.standard.bool(forKey: "DS4MeasuredAlign2026_07_08b") {
            UserDefaults.standard.set(true, forKey: "DS4MeasuredAlign2026_07_08b")
            applyFastDemoDefaults(persistExplicitly: true)
        }
        // Migrazione UNA TANTUM v6 (2026-07-13): preset finale misurato sul
        // M1 Pro 16 GB con RAM libera. Slot 20 evita pressione/swap, MetalIO
        // e' sicuro grazie al fallback automatico, Q4 completo mantiene una
        // generazione coerente a ~3.2-3.4 tok/s e il prefill arriva a ~4 tok/s.
        if !UserDefaults.standard.bool(forKey: "DS4MeasuredAlign2026_07_13") {
            UserDefaults.standard.set(true, forKey: "DS4MeasuredAlign2026_07_13")
            applyFastDemoDefaults(persistExplicitly: true)
        }
        // Migrazione v7: confronto greedy identico 20→22 slot sul M1 Pro 16 GB:
        // miss/byte SSD -7,8%, gather -6,1%, regime 3,18→3,30 tok/s, senza
        // pressione RAM osservabile. Aggiorna anche installazioni già migrate.
        if !UserDefaults.standard.bool(forKey: "DS4MeasuredSlots22_2026_07_13") {
            UserDefaults.standard.set(true, forKey: "DS4MeasuredSlots22_2026_07_13")
            applyFastDemoDefaults(persistExplicitly: true)
        }
        // Migrazione v8: questa macchina usa obbligatoriamente il ring KV a
        // dimensione costante. Allinea anche la queue depth pread al vincitore
        // A/B e rende persistibile l'occupancy Q4 densa per l'auto-tune GUI.
        if !UserDefaults.standard.bool(forKey: "DS4RigorousAutoTune2026_07_16") {
            UserDefaults.standard.set(true, forKey: "DS4RigorousAutoTune2026_07_16")
            rawRingEnabled = true
            preadSplit = 4
            denseQ4NSG = 4
            UserDefaults.standard.set(true, forKey: "DS4RawRing")
            UserDefaults.standard.set(4, forKey: "DS4PreadSplit")
            UserDefaults.standard.set(4, forKey: "DS4DenseQ4NSG")
        }
        // The mixed-quant cache was promoted after an exact-logit A/B on the
        // 37/43-layer IQ2/Q4 Flash GGUF (2026-07-16): +28.9% decode, -31.1%
        // expert bytes/token, same planned cache RAM. Persist the new default
        // for existing installs only when they have not already made an
        // explicit choice; OFF remains available as the legacy fallback.
        if UserDefaults.standard.object(forKey: "DS4MultiQuantCache") == nil {
            UserDefaults.standard.set(true, forKey: "DS4MultiQuantCache")
        }
        _ = setenv("DS4_RAW_RING", rawRingEnabled ? "1" : "0", 1)   // apply the persisted value at startup
        _ = setenv("DS4_INDEXED_ATTN", indexedAttnEnabled ? "1" : "0", 1) // attention DSA indicizzata (>4k)
        _ = setenv("DS4_MULTI_QUANT_CACHE", multiQuantCacheEnabled ? "1" : "0", 1) // default ON; exact mixed IQ2/Q4 pools
        _ = setenv("DS4_EXPERT_CACHE_UNIFORM", expertCacheUniform ? "1" : "0", 1) // auto-tune: usage-driven vs uniforme
        _ = setenv("DS4_WILLNEED_EXPERTS", willNeedEnabled ? "1" : "0", 1)   // default ON
        _ = setenv("DS4_EXPERT_PREAD", expertPreadEnabled ? "1" : "0", 1)    // default ON <24GB RAM
        _ = setenv("DS4_PREAD_SPLIT", String(preadSplit), 1)                   // NVMe queue depth, auto-tune
        _ = setenv("DS4_DENSE_STREAM", denseStreamEnabled ? "1" : "0", 1)    // default ON <24GB RAM
        _ = setenv("DS4_MLOCK", mlockEnabled ? "1" : "0", 1)                 // default ON (misurato: -38% ms/token)
        _ = setenv("DS4_DENSE_Q4", denseQ4Enabled ? "1" : "0", 1)            // default ON (lossy, disattivabile)
        _ = setenv("DS4_QKV_Q4", qkvQ4Enabled ? "1" : "0", 1)                // default ON (+10% misurato, lossy, disattivabile)
        _ = setenv("DS4_SHARED_Q4", sharedQ4Enabled ? "1" : "0", 1)          // default ON nel preset misurato (+7%, lossy)
        _ = setenv("DS4_EXPERT_LOOKAHEAD", "\(expertLookahead)", 1)          // 0 = solo layer hash (esatto)
        _ = setenv("DS4_DENSE_AHEAD", "\(denseAhead)", 1)                    // 2 = staging un layer avanti (misurato)
        _ = setenv("DS4_ASYNC_FFN", asyncFFNEnabled ? "1" : "0", 1)           // pipeline FFN asincrona (+10% misurato)
        _ = setenv("DS4_Q8_NSG", String(q8NSG), 1)                            // K-split Q8; escluso dal tuner exact
        _ = setenv("DS4_MOE_NSG", String(moeNSG), 1)                          // occupancy kernel MoE/Q4 (auto-tune)
        _ = setenv("DS4_DENSE_Q4_NSG", String(denseQ4NSG), 1)                 // occupancy Q4 densa, auto-tune
        _ = setenv("DS4_EXPERT_BUNDLE", expertBundleEnabled ? "1" : "0", 1)  // opt-in (duplica gli esperti su disco)
        _ = setenv("DS4_MTLIO", metalIOEnabled ? "1" : "0", 1)              // Metal fast resource loading (A/B sperimentale)
        _ = setenv("DS4_MTLIO_MIN_GBS", "4.0", 1)                           // M1 Pro: pread arriva a ~5 GB/s
        _ = setenv("DS4_POOL_INTERLEAVE", "1", 1)                            // un record contiguo per slot/esperto
        _ = setenv("DS4_PREFILL_FFN_BATCH", "1", 1)                          // un command buffer FFN per gruppo
        _ = setenv("DS4_PREFILL_MM", "1", 1)                                 // esperti prefill in GEMM (misurato: 48.9 → 20.7 ms/token su M1 Pro)
        _ = setenv("DS4_GPU_INDEXER_TOPK", "1", 1)                           // evita top-k/readback sulla CPU
        _ = setenv("DS4_DENSE_Q4_KERNEL", "1", 1)                            // matvec Q4 dense senza wrapper MoE k=1
        _ = setenv("DS4_FUSED_ROUTER_PROBS", "1", 1)                         // softplus+sqrt in un dispatch
        _ = setenv("DS4_FUSED_ROUTER_FINALIZE", "1", 1)                      // top-6 + pesi in un dispatch
        _ = setenv("DS4_FUSED_COMP_PROJ", "1", 1)                            // compressore KV+gate, attivazione condivisa
        // La cache del requant Q4 va in Application Support: l'app sandboxed
        // non può scrivere accanto al GGUF scelto col picker (il fallimento
        // sarebbe silenzioso e il requant si ripeterebbe a ogni load).
        _ = setenv("DS4_Q4_CACHE_DIR", Self.q4CacheDirectory.path, 1)
        // Stessa ragione per l'expert-bundle: il sidecar accanto al GGUF resta
        // leggibile quando la sandbox lo consente (riuso di quello della demo,
        // 72 GB non copiati), altrimenti la COSTRUZIONE va qui.
        _ = setenv("DS4_BUNDLE_DIR", Self.bundleDirectory.path, 1)
        // Densi residenti: SOLO automatico dalla RAM (niente toggle in GUI) —
        // su 16 GB nell'app rallenta; il valore persistito di vecchie build
        // viene ripulito così non può restare incollato un ON stantio.
        // (Con DS4_DENSE_STREAM attivo il motore da' comunque precedenza allo stream.)
        UserDefaults.standard.removeObject(forKey: "DS4ResidentDense")
        _ = setenv("DS4_RESIDENT_DENSE", Self.residentDenseAuto ? "1" : "0", 1)
        _ = setenv("DS4_PROFILE_ROUTE", profileRouteEnabled ? "1" : "0", 1)     // diagnostic, default OFF
        // Riparazione una-tantum: la prima versione del benchmark rapido
        // poteva scegliere union=64 su una misura da 128 token troppo corta
        // (rumore) — a scala reale 64 è il valore catastrofico. Riporta al
        // consigliato; il benchmark corretto non lo riproporrà.
        if !UserDefaults.standard.bool(forKey: "DS4BenchUnionFix2026_07_04"),
           prefillUnion < 192 {
            UserDefaults.standard.set(true, forKey: "DS4BenchUnionFix2026_07_04")
            prefillUnion = 192
            UserDefaults.standard.set(192, forKey: "DS4PrefillUnion")
        }
        // Migrazione una-tantum ai knob prefill misurati con le leve 1-8
        // (2026-07-22, testo reale 2.7k: 17.5 → 26+ t/s): chunk 2048
        // (4096 rende di più ma i transienti + slab full-layer stringono i
        // 16 GB con lo stack decode attivo), route batch 128, union 192 + MM
        // (256 + MM misurato PEGGIORE: GEMM esperti diluito). Applicata solo
        // ai valori dei vecchi default, non a scelte esplicite più alte.
        if !UserDefaults.standard.bool(forKey: "DS4PrefillLevers2026_07_22") {
            UserDefaults.standard.set(true, forKey: "DS4PrefillLevers2026_07_22")
            if prefillChunk < 2048 {
                prefillChunk = 2048
                UserDefaults.standard.set(2048, forKey: "DS4PrefillChunk")
            }
            if prefillRouteBatch < 128 {
                prefillRouteBatch = 128
                UserDefaults.standard.set(128, forKey: "DS4PrefillRouteBatch")
            }
            if prefillUnion > 192 {
                prefillUnion = 192
                UserDefaults.standard.set(192, forKey: "DS4PrefillUnion")
            }
        }
        // Migrazione una-tantum: SHARED_Q4 OFF (2026-07-24). Il +7% di decode
        // a contesto corto non ripaga il crollo del prefill misurato col
        // bisect (experts 147 → 368 ms/token, anche dopo aver batchato la
        // shared-FFN Q4). Il preset era già stato corretto, ma il VALORE
        // PERSISTITO vince all'avvio: senza questa migrazione chi lo aveva
        // acceso resta lento per sempre — è il caso che ha prodotto un
        // prefill a ~1 t/s in GUI mentre la CLI ne faceva 32.
        if !UserDefaults.standard.bool(forKey: "DS4SharedQ4Off2026_07_24") {
            UserDefaults.standard.set(true, forKey: "DS4SharedQ4Off2026_07_24")
            if sharedQ4Enabled {
                sharedQ4Enabled = false
                UserDefaults.standard.set(false, forKey: "DS4SharedQ4")
            }
        }
        // Migrazione una-tantum: slot cache ≤ 12 su <24 GB (2026-07-26). A 22
        // slot la cache wired è 6.2 GB su 16: i transienti del prefill batchato
        // (slab full-layer 3.4 GB + flash + GEMM) sfondano la RAM → swap →
        // prefill 1275s invece di 123s (A/B misurato, argmax identico). Il
        // motore ora clampa comunque a runtime, ma allineare il VALORE PERSISTITO
        // evita che la GUI mostri 22 mentre il motore ne usa 11.
        if !UserDefaults.standard.bool(forKey: "DS4CacheSlotsRAM2026_07_26"),
           MemoryInfo.physicalBytes < 24 * 1_073_741_824 {
            UserDefaults.standard.set(true, forKey: "DS4CacheSlotsRAM2026_07_26")
            if expertCacheSlots > 12 {
                expertCacheSlots = 12
                UserDefaults.standard.set(12, forKey: "DS4ExpertCacheSlots")
            }
        }
        // Knob del prefill regolabili a caldo (persistiti dal benchmark in
        // Settings): letti dal motore a ogni chiamata di prefill.
        _ = setenv("DS4_PREFILL_UNION", String(prefillUnion), 1)
        _ = setenv("DS4_PREFILL_CHUNK", String(prefillChunk), 1)
        _ = setenv("DS4_PREFILL_ROUTE_BATCH", String(prefillRouteBatch), 1)
        // Restore the persisted chats (newest first). Always keep at least one so
        // there is an active conversation to write into.
        sessions = ChatSessionStore.loadAll()
        if let first = sessions.first {
            activeSessionId = first.id
        } else {
            let s = ChatSession(agentId: selectedAgentId, systemNote: systemPrompt)
            sessions = [s]
            activeSessionId = s.id
        }
    }
    var systemPrompt = ""
    /// Expert slot-cache slots per layer (0 = off). On Flash IQ2, memory is
    /// ≈ 6.9 MB/slot × 43 layers; Pro has 61 larger expert slabs. Applied on
    /// the NEXT model load.
    /// DEFAULT 22: punto misurato su M1 Pro 16 GB con dense stream + pread +
    /// MLOCK + Q4 completo (A/B greedy 2026-07-13: 70% hit, 3.30 tok/s,
    /// miss/byte SSD -7,8% vs 20 senza collasso). Senza MLOCK i
    /// pool grandi vengono compressi/paginati e conviene 8; senza Q4 il
    /// budget bloccato è più tirato e il punto dolce scende a 12-16.
    var expertCacheSlots: Int = (UserDefaults.standard.object(forKey: "DS4ExpertCacheSlots") as? Int) ?? 22 {
        didSet { UserDefaults.standard.set(expertCacheSlots, forKey: "DS4ExpertCacheSlots") }
    }
    /// GLM 5.2: layer residenti (0 = adattivo alla RAM fisica). MISURATO su
    /// 16 GB: pochi residenti extra sono controproducenti (il sistema li
    /// pagina e ogni commit ne ripaga la residency, ~+750 ms/token con 11
    /// contro 3); l'adattivo resta al floor dense sotto pressione. Si
    /// applica al prossimo caricamento del modello.
    var glmResidentLayers: Int = (UserDefaults.standard.object(forKey: "GLMResidentLayers") as? Int) ?? 0 {
        didSet { UserDefaults.standard.set(glmResidentLayers, forKey: "GLMResidentLayers") }
    }
    /// Hit/miss dell'arena esperti GLM (riga di tuning; refresh esplicito).
    var glmArenaCounters: (hits: Int, misses: Int)?
    /// GLM 5.2: esperti eseguiti per token (0 = tutti gli 8 del router).
    /// Meno esperti = meno I/O, qualità ridotta.
    var glmActiveExperts: Int = (UserDefaults.standard.object(forKey: "GLMActiveExperts") as? Int) ?? 0 {
        didSet { UserDefaults.standard.set(glmActiveExperts, forKey: "GLMActiveExperts") }
    }
    /// GLM 5.2: slot dell'arena esperti (0 = default 24; ~10 MiB l'uno).
    /// Più slot = più riuso keyed nel prefill e più margine speculativo.
    var glmExpertArena: Int = (UserDefaults.standard.object(forKey: "GLMExpertArena") as? Int) ?? 0 {
        didSet { UserDefaults.standard.set(glmExpertArena, forKey: "GLMExpertArena") }
    }
    /// GLM 5.2: slot di staging del layer streamer (0 = default 3; ogni
    /// slot extra ≈ 250 MiB di RAM e un fill SSD in più in volo).
    var glmStreamSlots: Int = (UserDefaults.standard.object(forKey: "GLMStreamSlots") as? Int) ?? 0 {
        didSet { UserDefaults.standard.set(glmStreamSlots, forKey: "GLMStreamSlots") }
    }
    /// GLM 5.2: MetalIO SSD→GPU per i tensori layer (fallback pread
    /// automatico e permanente su qualunque anomalia).
    var glmMetalIOEnabled: Bool = (UserDefaults.standard.object(forKey: "GLMMetalIO") as? Bool) ?? true {
        didSet { UserDefaults.standard.set(glmMetalIOEnabled, forKey: "GLMMetalIO") }
    }
    /// GLM 5.2: staging speculativo degli esperti (misurato su M1 Pro:
    /// paga solo con banda SSD di riserva — default OFF).
    var glmSpeculativeExperts: Bool = (UserDefaults.standard.object(forKey: "GLMSpecExperts") as? Bool) ?? false {
        didSet { UserDefaults.standard.set(glmSpeculativeExperts, forKey: "GLMSpecExperts") }
    }
    /// GLM 5.2: usa i tensori Q4_K del sidecar (lossy, ~2× meno I/O layer).
    /// OFF = layer Q8 dal GGUF; gli esperti unificati restano attivi.
    var glmUseQ4Sidecar: Bool = (UserDefaults.standard.object(forKey: "GLMUseQ4Sidecar") as? Bool) ?? true {
        didSet { UserDefaults.standard.set(glmUseQ4Sidecar, forKey: "GLMUseQ4Sidecar") }
    }
    /// GLM 5.2: fusione dei commit (FFN layer N + trunk N+1 in un command
    /// buffer — ~metà delle attese sincrone). OFF = percorso storico.
    /// Leva 1 del prefill GLM (DS4_GLM_PREFILL_BATCH): route a gruppi con
    /// due commit per gruppo invece di 2-3 per token. Parità bit-esatta nei
    /// test sintetici; OFF finché non validata sul GGUF reale.
    var glmPrefillBatchEnabled: Bool = (UserDefaults.standard.object(forKey: "GLMPrefillBatch") as? Bool) ?? false {
        didSet { UserDefaults.standard.set(glmPrefillBatchEnabled, forKey: "GLMPrefillBatch") }
    }
    var glmFuseEnabled: Bool = (UserDefaults.standard.object(forKey: "GLMFuse") as? Bool) ?? true {
        didSet { UserDefaults.standard.set(glmFuseEnabled, forKey: "GLMFuse") }
    }
    /// GLM 5.2: MoE batched (tutti gli esperti instradati in due dispatch).
    /// OFF = swiglu+down+add per esperto, il percorso di riferimento.
    var glmMoEBatchEnabled: Bool = (UserDefaults.standard.object(forKey: "GLMMoEBatch") as? Bool) ?? true {
        didSet { UserDefaults.standard.set(glmMoEBatchEnabled, forKey: "GLMMoEBatch") }
    }
    /// Fase B del prefill multi-token (DS4_GLM_PREFILL_MOE): pesi esperti
    /// letti una volta per tile di token invece che una volta per token.
    var glmPrefillMoEEnabled: Bool = (UserDefaults.standard.object(forKey: "GLMPrefillMoE") as? Bool) ?? true {
        didSet { UserDefaults.standard.set(glmPrefillMoEEnabled, forKey: "GLMPrefillMoE") }
    }
    /// GLM 5.2: router fuso su GPU (matvec+sigmoid+top-8 nel commit del
    /// trunk; −18% di prefill misurato). OFF = router CPU di riferimento.
    var glmGpuRouterEnabled: Bool = (UserDefaults.standard.object(forKey: "GLMGpuRouter") as? Bool) ?? true {
        didSet { UserDefaults.standard.set(glmGpuRouterEnabled, forKey: "GLMGpuRouter") }
    }
    /// GLM 5.2: mlock dei pesi residenti (head ~0.8 GB + attn/FFN dei layer
    /// residenti; misurato −394 ms/token sul head). OFF = paginabili.
    var glmMlockEnabled: Bool = (UserDefaults.standard.object(forKey: "GLMMlock") as? Bool) ?? true {
        didSet { UserDefaults.standard.set(glmMlockEnabled, forKey: "GLMMlock") }
    }
    /// GLM 5.2: sotto-intervalli pread del fill parallelo nel PREFILL
    /// (0 = default 4; −15% di prefill misurato). Il decode resta seriale.
    var glmReadSplit: Int = (UserDefaults.standard.object(forKey: "GLMReadSplit") as? Int) ?? 0 {
        didSet { UserDefaults.standard.set(glmReadSplit, forKey: "GLMReadSplit") }
    }
    /// GLM 5.2: simdgroup per threadgroup dei kernel matvec (0 = default 4).
    var glmNSG: Int = (UserDefaults.standard.object(forKey: "GLMNSG") as? Int) ?? 0 {
        didSet { UserDefaults.standard.set(glmNSG, forKey: "GLMNSG") }
    }
    /// Layer-aware expert-cache layout for mixed-quant GGUFs. Each routed layer
    /// gets a pool with its real IQ2/Q4 record size, while the allocator keeps
    /// the total byte budget at or below the legacy 22-slot plan. Exact: it only
    /// changes where identical expert bytes are retained. OFF restores the old
    /// single-size-class cache and bypasses off-class routed layers. Applied on
    /// the NEXT model load.
    var multiQuantCacheEnabled: Bool =
        (UserDefaults.standard.object(forKey: "DS4MultiQuantCache") as? Bool) ?? true {
        didSet {
            UserDefaults.standard.set(multiQuantCacheEnabled, forKey: "DS4MultiQuantCache")
            _ = setenv("DS4_MULTI_QUANT_CACHE", multiQuantCacheEnabled ? "1" : "0", 1)
        }
    }
    /// `false` distribuisce il budget cache in base al profilo di routing;
    /// `true` assegna lo stesso numero di slot a ogni layer. Entrambi conservano
    /// gli stessi byte e sono confrontati con usage profile congelato.
    var expertCacheUniform: Bool =
        (UserDefaults.standard.object(forKey: "DS4ExpertCacheUniform") as? Bool) ?? false {
        didSet {
            UserDefaults.standard.set(expertCacheUniform, forKey: "DS4ExpertCacheUniform")
            _ = setenv("DS4_EXPERT_CACHE_UNIFORM", expertCacheUniform ? "1" : "0", 1)
        }
    }
    /// Look-ahead speculativo della slot-cache (DS4_EXPERT_LOOKAHEAD): mentre il
    /// layer i calcola, il pool del layer i+1 viene PREriempito con i top-N
    /// esperti del prior d'uso (I/O reale nella finestra in cui l'SSD è idle;
    /// il demand ha priorità — un prefill in corso cede il passo entro ~2 slab).
    /// I layer hash 0-2 sono SEMPRE prefetchati esatti (selezione nota dal token
    /// id), indipendentemente da questo valore. 0 = solo hash. Si applica al
    /// prossimo caricamento del modello. Speculativo: A/B per macchina.
    var expertLookahead: Int = (UserDefaults.standard.object(forKey: "DS4ExpertLookahead") as? Int) ?? 0 {
        didSet {
            UserDefaults.standard.set(expertLookahead, forKey: "DS4ExpertLookahead")
            _ = setenv("DS4_EXPERT_LOOKAHEAD", "\(expertLookahead)", 1)
        }
    }
    /// Profondita' del ring di staging del dense stream (DS4_DENSE_AHEAD):
    /// 2 = legge un layer piu' avanti, +1,5% misurato su M1 Pro (costo ~150 MB
    /// di staging). Nessun toggle in GUI: e' il default misurato; regolabile
    /// via UserDefaults per esperimenti. Si applica al prossimo load.
    var denseAhead: Int = (UserDefaults.standard.object(forKey: "DS4DenseAhead") as? Int) ?? 2 {
        didSet {
            UserDefaults.standard.set(denseAhead, forKey: "DS4DenseAhead")
            _ = setenv("DS4_DENSE_AHEAD", "\(denseAhead)", 1)
        }
    }
    /// Pipeline asincrona del FFN routed (DS4_ASYNC_FFN): certificata da A/B
    /// (+10%, 335 vs 374 ms/token, output identico token-per-token — la
    /// correttezza e' garantita dalla coda Metal in-order). Default ON;
    /// persistito senza UI: e' il paracadute per debug (=false ripristina il
    /// percorso storico coi wait sincroni). Si applica al prossimo load.
    var asyncFFNEnabled: Bool = (UserDefaults.standard.object(forKey: "DS4AsyncFFN") as? Bool) ?? true {
        didSet {
            UserDefaults.standard.set(asyncFFNEnabled, forKey: "DS4AsyncFFN")
            _ = setenv("DS4_ASYNC_FFN", asyncFFNEnabled ? "1" : "0", 1)
        }
    }

    /// DS4_Q8_NSG: simdgroup per threadgroup nei matvec Q8 (partizione della
    /// riduzione K). Cambia l'occupancy ma puo' cambiare gli ultimi bit Float32:
    /// il manifest exact della GUI NON lo modifica; l'autotuner di processo lo
    /// include solo con --allow-numeric e un gate logits esplicito.
    /// L'ottimo dipende dai core GPU: 4 = riferimento (migliore su M1 Pro);
    /// su GPU più larghe (Max/Ultra) possono vincere 6-8. Il motore lo rilegge
    /// a ogni load per consentire A/B espliciti senza riavviare l'app.
    var q8NSG: Int = (UserDefaults.standard.object(forKey: "DS4Q8NSG") as? Int) ?? 4 {
        didSet {
            UserDefaults.standard.set(q8NSG, forKey: "DS4Q8NSG")
            _ = setenv("DS4_Q8_NSG", String(q8NSG), 1)
        }
    }

    /// DS4_MOE_NSG: simdgroup per threadgroup nei kernel MoE id (FFN routed —
    /// la voce compute più grossa misurata, ~100 ms/token — e matvec densi
    /// Q4). Partizione per RIGHE: numerica identica per costruzione, cambia
    /// solo l'occupancy. Come q8NSG: default 4 (storico), l'ottimo dipende
    /// dai core GPU e l'auto-tune lo esplora con un reload.
    var moeNSG: Int = (UserDefaults.standard.object(forKey: "DS4MoeNSG") as? Int) ?? 4 {
        didSet {
            UserDefaults.standard.set(moeNSG, forKey: "DS4MoeNSG")
            _ = setenv("DS4_MOE_NSG", String(moeNSG), 1)
        }
    }

    // Disk KV cache (ds4_kvstore model): checkpoints completed generations and
    // restores matching prefixes on cold starts. Applied on the NEXT model load.
    // ON by default so conversations are checkpointed and re-prefill is avoided
    // across reloads; the explicit user choice is then persisted.
    var diskKVEnabled: Bool = (UserDefaults.standard.object(forKey: "DS4DiskKV") as? Bool) ?? true {
        didSet { UserDefaults.standard.set(diskKVEnabled, forKey: "DS4DiskKV") }
    }
    /// Disk-KV budget in THOUSANDS of tokens (default 1000 = 1M tokens total
    /// across checkpoints — the live window stays `contextSize`). Tokens, not MB:
    /// per-token checkpoint bytes depend on the model (~22 KB/token on the 61-layer
    /// 2-bit Flash → 1M tokens ≈ 22 GB on disk), so tokens are the stable unit.
    var diskKVBudgetKTok: Int = UserDefaults.standard.object(forKey: "DS4DiskKVBudgetKTok") as? Int ?? 1000 {
        didSet { UserDefaults.standard.set(diskKVBudgetKTok, forKey: "DS4DiskKVBudgetKTok") }
    }
    /// Raw-KV ring buffer (experimental): keep only the nSWA attention window in RAM
    /// instead of the full context, so the KV RAM is constant. Sets the engine env
    /// var; applied on the NEXT model load.
    var rawRingEnabled: Bool = (UserDefaults.standard.object(forKey: "DS4RawRing") as? Bool) ?? true {
        didSet {
            UserDefaults.standard.set(rawRingEnabled, forKey: "DS4RawRing")
            _ = setenv("DS4_RAW_RING", rawRingEnabled ? "1" : "0", 1)
        }
    }
    /// Attention DSA INDICIZZATA (DS4_INDEXED_ATTN): oltre la soglia
    /// dell'indexer (~4k token) attende SOLO sulle 512 righe compresse
    /// selezionate — prefill 8k misurato 686→256s (32.3 t/s, sopra ds4),
    /// decode −5-8% nella fascia 4-8k. Sotto soglia è inerte. Default ON in
    /// GUI (l'attesa dell'utente sui contesti lunghi è il prefill); il
    /// toggle serve all'A/B. Si applica al prossimo caricamento.
    var indexedAttnEnabled: Bool = (UserDefaults.standard.object(forKey: "DS4IndexedAttn") as? Bool) ?? true {
        didSet {
            UserDefaults.standard.set(indexedAttnEnabled, forKey: "DS4IndexedAttn")
            _ = setenv("DS4_INDEXED_ATTN", indexedAttnEnabled ? "1" : "0", 1)
        }
    }
    /// madvise(WILLNEED) sui 6 esperti selezionati prima del gather: anticipa e
    /// batcha il read-ahead a freddo dei soli slab che servono. Solo advisory (non
    /// cambia i numeri). DEFAULT ON; si applica al PROSSIMO caricamento del modello
    /// (l'engine legge l'env alla costruzione del decoder).
    var willNeedEnabled: Bool = (UserDefaults.standard.object(forKey: "DS4WillNeed") as? Bool) ?? true {
        didSet {
            UserDefaults.standard.set(willNeedEnabled, forKey: "DS4WillNeed")
            _ = setenv("DS4_WILLNEED_EXPERTS", willNeedEnabled ? "1" : "0", 1)
        }
    }
    /// Lettura diretta degli esperti (DS4_EXPERT_PREAD): pread + F_NOCACHE dei soli
    /// slab selezionati, direttamente nei buffer di destinazione, SENZA passare
    /// dalla page cache. Su RAM stretta il churn degli esperti (~1 GB/token)
    /// smette così di evictare i pesi densi: misurato su M1 Pro 16 GB vale ~+20%
    /// di tok/s da solo e rende conveniente i densi residenti (insieme: 0.17 →
    /// 0.27 tok/s). Con RAM abbondante conviene invece la page cache (i re-read
    /// degli esperti tiepidi sono gratis) → DEFAULT ON sotto i 24 GB, OFF sopra.
    /// Stessi byte, stesse numeriche. Si applica al prossimo caricamento.
    var expertPreadEnabled: Bool =
        (UserDefaults.standard.object(forKey: "DS4ExpertPread") as? Bool)
        ?? (MemoryInfo.physicalBytes < 24 * 1_073_741_824) {
        didSet {
            UserDefaults.standard.set(expertPreadEnabled, forKey: "DS4ExpertPread")
            _ = setenv("DS4_EXPERT_PREAD", expertPreadEnabled ? "1" : "0", 1)
        }
    }
    /// Numero di range pread concorrenti per slab esperto. Stessi byte e stessa
    /// numerica; cambia soltanto la queue depth NVMe. Il punto migliore dipende
    /// dall'SSD e viene cercato dall'auto-tune record-holder.
    var preadSplit: Int = (UserDefaults.standard.object(forKey: "DS4PreadSplit") as? Int) ?? 4 {
        didSet {
            UserDefaults.standard.set(preadSplit, forKey: "DS4PreadSplit")
            _ = setenv("DS4_PREAD_SPLIT", String(preadSplit), 1)
        }
    }
    /// Streaming double-buffered dei pesi densi (DS4_DENSE_STREAM): invece di
    /// tenere ~6 GB di densi residenti/cachati, ogni layer viene letto con
    /// pread+F_NOCACHE in un ring a 2 slot (~300 MB) UN LAYER IN ANTICIPO, così
    /// l'I/O SSD del layer i+1 si sovrappone al compute del layer i. Byte
    /// identici → numeriche identiche. MISURATO su M1 Pro 16 GB: route/attn
    /// −87%, 0.17 → 0.34 tok/s dalla baseline, primo token 8.7 s invece di 24,
    /// prefill −65%. DEFAULT ON sotto i 24 GB; sopra conviene la residenza
    /// piena (automatica), quindi lì default OFF. Prevale su DS4_RESIDENT_DENSE.
    /// Si applica al prossimo caricamento del modello.
    var denseStreamEnabled: Bool =
        (UserDefaults.standard.object(forKey: "DS4DenseStream") as? Bool)
        ?? (MemoryInfo.physicalBytes < 24 * 1_073_741_824) {
        didSet {
            UserDefaults.standard.set(denseStreamEnabled, forKey: "DS4DenseStream")
            _ = setenv("DS4_DENSE_STREAM", denseStreamEnabled ? "1" : "0", 1)
        }
    }
    /// Requant Q4_K delle tre proiezioni giganti dell'attention (DS4_DENSE_Q4):
    /// q_b / output_a / output_b (Q8, 107 dei ~145 MB/layer) requantizzate al
    /// load e tenute residenti (~1.4 GB, mlockate). MISURATO su M1 Pro 16 GB:
    /// 0.88 → 1.17 tok/s. ATTENZIONE: è l'unica opzione LOSSY — deriva di
    /// logit ~0.01%, output greedy occasionalmente diverso ma coerente
    /// (validato dall'autore sul campo). DEFAULT ON con avviso in GUI; il
    /// toggle resta per chi preferisce la fedeltà piena. Richiede il dense
    /// stream; si applica al prossimo caricamento del modello.
    var denseQ4Enabled: Bool = (UserDefaults.standard.object(forKey: "DS4DenseQ4") as? Bool) ?? true {
        didSet {
            UserDefaults.standard.set(denseQ4Enabled, forKey: "DS4DenseQ4")
            _ = setenv("DS4_DENSE_Q4", denseQ4Enabled ? "1" : "0", 1)
        }
    }
    /// Requant Q4 anche di q_a e kv (DS4_QKV_Q4, richiede il Q4 sopra): gli
    /// ultimi densi Q8 medi nello stream (~0.7 GB/token) diventano residenti
    /// (~0.35 GB). MISURATO su M1 Pro (demo A/B 2026-07-08): decode 2.78 →
    /// 3.06 tok/s (+10%), gather 627 → 606 MB/token, output coerente. LOSSY
    /// come gli altri Q4 — default ON con avviso in GUI (come DENSE_Q4), il
    /// toggle resta per chi preferisce la fedeltà piena; la cache Q4
    /// esistente si estende in modo incrementale (~30 s una tantum).
    var qkvQ4Enabled: Bool = (UserDefaults.standard.object(forKey: "DS4QkvQ4") as? Bool) ?? true {
        didSet {
            UserDefaults.standard.set(qkvQ4Enabled, forKey: "DS4QkvQ4")
            _ = setenv("DS4_QKV_Q4", qkvQ4Enabled ? "1" : "0", 1)
        }
    }

    /// Requant Q4 delle shared-expert FFN (DS4_SHARED_Q4, richiede il Q4
    /// sopra): l'ultima voce grossa rimasta nello stream denso. MISURATO su
    /// M1 Pro (A/B 2026-07-08, sera): 3.13 → 3.36 tok/s (+7%) — era neutro
    /// nel tuning del 07-06, ma allora trio/q_a/kv non erano residenti.
    /// LOSSY (la continuazione greedy cambia restando coerente): default ON nel
    /// preset misurato; il toggle resta disponibile per la massima fedelta'.
    var sharedQ4Enabled: Bool = (UserDefaults.standard.object(forKey: "DS4SharedQ4") as? Bool) ?? true {
        didSet {
            UserDefaults.standard.set(sharedQ4Enabled, forKey: "DS4SharedQ4")
            _ = setenv("DS4_SHARED_Q4", sharedQ4Enabled ? "1" : "0", 1)
        }
    }
    /// Occupancy indipendente dei kernel Q4 densi. È una partizione per righe,
    /// quindi il gate dell'auto-tune deve risultare bit-identico.
    var denseQ4NSG: Int = (UserDefaults.standard.object(forKey: "DS4DenseQ4NSG") as? Int) ?? 4 {
        didSet {
            UserDefaults.standard.set(denseQ4NSG, forKey: "DS4DenseQ4NSG")
            _ = setenv("DS4_DENSE_Q4_NSG", String(denseQ4NSG), 1)
        }
    }
    /// Sidecar expert-bundle (DS4_EXPERT_BUNDLE): gli slab gate/up/down di ogni
    /// esperto riimpacchettati CONTIGUI in <gguf>.expbundle — un miss della
    /// cache diventa un burst sequenziale da ~7 MB invece di 3 letture sparse.
    /// MISURATO: gather 2.7 → 4.8 GB/s (79% del tetto SSD), 2.10 → 2.66 tok/s.
    /// Stessi byte, numeriche identiche. DEFAULT ON: al load il file viene
    /// cercato (accanto al GGUF, poi in Application Support), costruito una
    /// tantum se assente — e SALTATO con log esplicito quando mancano i ~73 GB
    /// liberi che il sidecar duplica su disco. Si applica al prossimo load.
    var expertBundleEnabled: Bool = (UserDefaults.standard.object(forKey: "DS4ExpertBundle") as? Bool) ?? true {
        didSet {
            UserDefaults.standard.set(expertBundleEnabled, forKey: "DS4ExpertBundle")
            _ = setenv("DS4_EXPERT_BUNDLE", expertBundleEnabled ? "1" : "0", 1)
        }
    }
    /// Metal fast resource loading: batch di range dell'expert-bundle caricati
    /// direttamente negli MTLBuffer della slot cache durante il decode. Il
    /// prefill resta sul percorso pread. Default ON nel preset 16 GB misurato;
    /// il fallback a pread è automatico su errore o banda insufficiente, quindi
    /// una sessione sfavorevole non resta bloccata sul backend lento. Prossimo load.
    var metalIOEnabled: Bool = (UserDefaults.standard.object(forKey: "DS4MetalIO") as? Bool) ?? true {
        didSet {
            UserDefaults.standard.set(metalIOEnabled, forKey: "DS4MetalIO")
            _ = setenv("DS4_MTLIO", metalIOEnabled ? "1" : "0", 1)
        }
    }
    /// Unione massima di esperti per gruppo nel prefill (DS4_PREFILL_UNION) e
    /// token per chunk (DS4_PREFILL_CHUNK): gli unici knob del prefill letti a
    /// OGNI chiamata dal motore, quindi regolabili senza ricaricare il modello.
    /// Persistiti; il benchmark in Settings li misura e applica i migliori.
    var prefillUnion: Int = (UserDefaults.standard.object(forKey: "DS4PrefillUnion") as? Int) ?? 192 {
        didSet {
            UserDefaults.standard.set(prefillUnion, forKey: "DS4PrefillUnion")
            _ = setenv("DS4_PREFILL_UNION", String(prefillUnion), 1)
        }
    }
    var prefillChunk: Int = (UserDefaults.standard.object(forKey: "DS4PrefillChunk") as? Int) ?? 2048 {
        didSet {
            UserDefaults.standard.set(prefillChunk, forKey: "DS4PrefillChunk")
            _ = setenv("DS4_PREFILL_CHUNK", String(prefillChunk), 1)
        }
    }

    /// Numero di token route/attention codificati nello stesso command buffer
    /// durante il prefill. Numerica invariata: i dispatch restano seriali e nello
    /// stesso ordine, ma si riducono commit e attese CPU↔GPU. È letto a ogni layer
    /// di prefill, quindi il benchmark può regolarlo senza ricaricare il modello.
    var prefillRouteBatch: Int =
        (UserDefaults.standard.object(forKey: "DS4PrefillRouteBatch") as? Int) ?? 128 {
        didSet {
            UserDefaults.standard.set(prefillRouteBatch, forKey: "DS4PrefillRouteBatch")
            _ = setenv("DS4_PREFILL_ROUTE_BATCH", String(prefillRouteBatch), 1)
        }
    }

    // Benchmark in-app (Settings): prova le combinazioni dei knob a caldo sul
    // motore GIÀ caricato e applica la migliore.
    var benchRunning = false
    var benchStatus: String?
    var benchResults: String = ""
    var benchSucceeded: Bool?
    var benchProgressDone = 0
    var benchProgressTotal = 0
    var autoTuneReportURL: URL?
    var benchTask: Task<Void, Never>?



    /// Esito dell'ultima generazione manuale dell'expert-bundle (bottone Settings).
    var bundleBuildStatus: String?
    /// A bundle build can stream tens of gigabytes from the same GGUF/SSD used by
    /// inference. Keep its task visible and hold the engine-activity lease until
    /// the underlying synchronous builder has really returned.
    var bundleBuildRunning = false
    var bundleBuildTask: Task<Void, Never>?


    /// mlock dei buffer residenti caldi (DS4_MLOCK): pool della cache esperti,
    /// output head residente e staging dello stream (~3.3 GB con i default).
    /// I buffer Metal shared sono memoria anonima che macOS COMPRIME tra un
    /// token e l'altro: MISURATO su M1 Pro 16 GB, bloccarli vale head 235→19 ms
    /// ed experts 748→144 ms (totale −38%, 0.39 → 0.52+ tok/s). Best-effort
    /// (fallimenti ignorati), numeriche identiche. DEFAULT ON; si applica al
    /// prossimo caricamento del modello.
    var mlockEnabled: Bool = (UserDefaults.standard.object(forKey: "DS4MLock") as? Bool) ?? true {
        didSet {
            UserDefaults.standard.set(mlockEnabled, forKey: "DS4MLock")
            _ = setenv("DS4_MLOCK", mlockEnabled ? "1" : "0", 1)
        }
    }
    /// Pesi densi residenti (DS4_RESIDENT_DENSE): copia i ~5 GB di pesi non-esperti
    /// in buffer residenti invece di mapparli no-copy. NON esposto nella GUI e
    /// deciso SOLO dalla RAM (ON ≥24 GB): nella demo CLI a contesto corto su
    /// 16 GB risultava più veloce, ma nell'app reale (SwiftUI, trascrizioni,
    /// server, contesti più lunghi) il budget wired sfora e RALLENTA — misurato.
    /// Con DS4_DENSE_STREAM attivo il motore da' comunque precedenza allo stream.
    /// Il knob env DS4_RESIDENT_DENSE resta per la demo/gli esperimenti.

    /// Profilo decode route/attn (DS4_PROFILE_ROUTE): splitta route/attn in 5 fasi
    /// (comp/q/kv/attn/out), ognuna con commit+wait dedicato. DIAGNOSTICO: i commit
    /// extra RALLENTANO la generazione (gli assoluti si gonfiano; conta il RAPPORTO),
    /// quindi DEFAULT OFF e NON usarlo mentre misuri la velocità. Il report finisce
    /// nel Log motore a fine turno. Si applica al prossimo caricamento del modello
    /// (l'engine legge l'env alla costruzione del decoder).
    var profileRouteEnabled: Bool = (UserDefaults.standard.object(forKey: "DS4ProfileRoute") as? Bool) ?? false {
        didSet {
            UserDefaults.standard.set(profileRouteEnabled, forKey: "DS4ProfileRoute")
            _ = setenv("DS4_PROFILE_ROUTE", profileRouteEnabled ? "1" : "0", 1)
        }
    }
    /// Application Support/DwarfStar/kv-cache (shared by chat and HTTP server).


    // Tuning tab state.
    var tuningInfo: InferenceService.TuningInfo?

    // Agents (roles). Selecting one starts a fresh chat with its role and swaps
    // the per-agent expert-usage profile (the cache re-warms with ITS experts).
    // Editable + persisted; edits apply on the next new chat / agent switch.
    var agents: [AgentProfile] = ChatStore.loadAgents() {
        didSet { AgentRegistry.shared.set(agents) }   // keep the engine-side agents_list tool in sync
    }
    var selectedAgentId: String = UserDefaults.standard.string(forKey: "DS4SelectedAgent") ?? "generale" {
        didSet { UserDefaults.standard.set(selectedAgentId, forKey: "DS4SelectedAgent") }
    }

    // Discovered GGUF files on disk.
    var discoveredModels: [DiscoveredModel] = []

    // Last applied preset explanation, shown in the load screen.
    var presetNote: String?

    // Sampling. Temperature is user-tunable (lower = more focused, less drift —
    // helps on aggressively-quantized models). Persisted across launches.
    var temperature: Double = UserDefaults.standard.object(forKey: "DS4Temperature") as? Double ?? 0.6 {
        didSet { UserDefaults.standard.set(temperature, forKey: "DS4Temperature") }
    }
    /// Repetition penalty (>1 discourages repeats; breaks the repeat-loop collapse
    /// on quantized models). 1.0 = off. Persisted.
    var repetitionPenalty: Double = UserDefaults.standard.object(forKey: "DS4RepPenalty") as? Double ?? 1.1 {
        didSet { UserDefaults.standard.set(repetitionPenalty, forKey: "DS4RepPenalty") }
    }


    // Tools.
    var toolsEnabled = false
    var enabledToolNames: Set<String> = Set(ToolRegistry.builtins.map { $0.spec.name })
    /// Exact allow-list baked into the active conversation prefix. Tool picker
    /// changes affect the next fresh chat; execution must follow what was actually
    /// declared, not a mutable UI selection changed mid-turn.
    var activeConversationToolNames: Set<String> = []
    /// Trusted transitive-capability ceiling captured with the active agent.
    /// Model arguments to subagent_run may only narrow this set.
    var activeConversationDelegatedToolNames: Set<String> = []
    /// Compact tool declaration (just name(params)) — fewer prefill tokens. On by
    /// default for local inference.
    var compactTools = true
    /// Tool calls awaiting a manually-entered result (non-built-in tools).
    var pendingManualCalls: [ToolCall] = []
    /// Drives the manual-results sheet (set when `pendingManualCalls` is filled).
    var awaitingManualResults = false
    /// One slot per tool call in the model-emitted order. Automatic/error results
    /// are filled immediately; manual results fill their recorded positions when
    /// the sheet is submitted. Never concatenate result classes: DSML associates
    /// consecutive `tool_result` blocks positionally with a multi-call batch.
    var pendingOrderedToolOutputs: [ToolOutput?] = []
    var pendingManualOutputIndices: [Int] = []
    /// Epoch which owns the pending manual sheet. A session change/Stop invalidates
    /// it so a late sheet action cannot resume a different conversation.
    var pendingManualEpoch: UInt64?
    /// Tool-loop safety budget. A local quantized model can otherwise repeat an
    /// expensive or mutating call until the context is exhausted.
    var toolRounds = 0
    let maxToolRounds = 16
    let maxToolCallsPerRound = 8
    /// Fingerprints from the immediately previous round: identical consecutive
    /// calls are rejected, while a later verification read after another action
    /// remains possible.
    var previousToolRoundFingerprints: Set<String> = []

    // Live state.
    var phase: Phase = .needsModel
    var info: ModelInfo?
    /// Metadata-only inspection of the selected GGUF. It is available before a
    /// backend loads, so settings can hide architecture-specific controls even
    /// for a recognized backend that is not implemented yet.
    var inspectedModelDescriptor: RuntimeModelDescriptor?
    var modelCapabilities: BackendCapabilities {
        info?.capabilities ?? inspectedModelDescriptor?.capabilities ?? []
    }
    var supportsReasoning: Bool {
        modelCapabilities.contains(.reasoning)
    }
    var messages: [UIMessage] = []
    var input = ""
    /// Text files staged for the next message (folded into the user turn on send).
    var attachments: [ChatAttachment] = []
    /// Transient composer note (e.g. a file that couldn't be decoded as text).
    var attachmentNote: String?
    /// Rough token estimate of the staged attachments (≈4 chars/token); nil if none.
    /// Used to warn before they overflow the context window.
    var attachmentTokenEstimate: Int? {
        guard !attachments.isEmpty else { return nil }
        return attachments.reduce(0) { $0 + $1.content.count } / 4
    }
    var think = false
    var isGenerating = false
    var status = ""          // live prefill/decode progress
    /// Tokens committed to the KV (≈ context used); drives the near-full warning.
    var contextUsed = 0

    // MARK: - Chat sessions (persistent, multiple)

    /// All persisted chats, newest first. The active one's transcript is mirrored
    /// in `messages`; the others live on disk and are loaded on demand.
    var sessions: [ChatSession] = []
    /// Id of the chat currently shown in `messages`.
    var activeSessionId: String = ""
    /// True when the engine's KV already holds the active chat. False right after a
    /// persisted chat is restored: the next send re-primes the engine from the
    /// visible history (the disk-KV cache restores the prefix), after which turns
    /// are incremental again.
    var enginePrimed = true

    var service: InferenceService?
    /// GLM 5.2 chat service — mutually exclusive with `service`. The chat
    /// send path branches on whichever is live; DeepSeek-only surfaces
    /// (decode profiles, sub-agents, distributed) stay disabled while this
    /// is set, while benchmark/correctness/diagnostics have GLM paths.
    var glmService: GLM52ChatService?
    /// Full load signature of `service`, including fixed knobs and context.
    var loadedEngineSignature: LoadedEngineSignature?
    var generation: Task<Void, Never>?
    /// Serialized role/tool setup for the live engine. The Bool is the result of
    /// the post-role-change warmup. Reload and auto-tune await the stable latest
    /// epoch before publishing, persisting, or releasing the current engine.
    var engineSetupTask: Task<Bool, Never>?
    var engineSetupEpoch: UInt64 = 0
    var engineSetupCompletedEpoch: UInt64 = 0
    var engineSetupWarmupSucceeded = true
    /// Monotonic ownership token for all asynchronous chat work. Every fresh user
    /// turn, Stop, new chat, or session activation advances it. Stream/tool tasks
    /// may mutate UI state only while the captured value still matches.
    var conversationEpoch: UInt64 = 0

    /// THE single in-process engine, loaded once in Settings and SHARED by every
    /// feature (Chat, Benchmark, Server). There is never a second full engine:
    /// with the default resident-Q4 + mlock config a second copy doubles wired
    /// memory and OOM-crashes on 16 GB. `InferenceService` is an actor, so
    /// concurrent callers are serialized safely. nil until a model is ready.
    var sharedEngine: InferenceService? { isReady ? service : nil }
    /// Il backend chat attivo dietro il contratto comune `ChatBackend` —
    /// DeepSeek o GLM, uno solo alla volta. I punti che servono capacità
    /// specifiche (profili, disk-KV DeepSeek, sub-agent) continuano a usare
    /// `service`/`glmService` concreti.
    var chatBackend: (any ChatBackend)? {
        if let service { return service }
        return glmService
    }
    /// Shared engine gated for KV-mutating uses (benchmark): a run rewrites the
    /// KV, so it's refused while the chat is mid-generation.
    var benchmarkService: InferenceService? { (isReady && !isGenerating) ? service : nil }
    /// GLM counterpart of `benchmarkService`, same KV-mutating gate.
    var glmBenchmarkService: GLM52ChatService? { (isReady && !isGenerating) ? glmService : nil }
    var loadedModelPath: String? { isReady ? modelPath : nil }

    var isReady: Bool { if case .ready = phase { return true } else { return false } }
    /// Everything togglable in pickers/agents: built-ins + connected MCP tools.
    var availableTools: [ToolSpec] { builtinTools + mcpTools }
    var builtinTools: [ToolSpec] { ToolRegistry.builtins.map(\.spec) }
    /// Reading `mcpVersion` here registers the SwiftUI dependency: MCPManager is
    /// a plain singleton, so views listing MCP tools re-render only because the
    /// change handler (see init) bumps this observable mirror.
    var mcpTools: [ToolSpec] { _ = mcpVersion; return MCPManager.shared.toolSpecs() }
    /// Observable mirror of MCPManager.version (bumped by the change handler).
    var mcpVersion = 0

    var bookmarkRestored = false


    /// Avanzamento del caricamento del modello (0…1) + fase corrente: scritti
    /// dal motore via LoadProgress e riletti qui a ~8 Hz mentre `.loading`.
    var loadFraction: Double = 0
    var loadStage: String = ""

}
