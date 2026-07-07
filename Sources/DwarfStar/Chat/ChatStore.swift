import SwiftUI
import AppKit
import UniformTypeIdentifiers
import DS4Engine
import DS4Core

/// A message as shown in the UI: reasoning and visible answer are kept apart so
/// the chain-of-thought can be collapsed. Assistant messages may carry tool
/// calls; tool results are shown as `.tool` messages.
struct UIMessage: Identifiable {
    let id = UUID()
    let role: ChatRole
    var reasoning: String = ""
    var text: String
    var toolStreamText: String = ""   // raw tool markup shown live while it generates
    var toolCalls: [ToolCall] = []
    /// Names of text files imported with this (user) message — shown as badges; the
    /// full content was folded into the turn actually sent to the model.
    var attachments: [String] = []
    /// Set on a `.tool` message that reports an isolated sub-agent run (question,
    /// answer, and a collapsible trace of its internal steps).
    var subAgent: InferenceService.SubAgentRun?
    /// True while that sub-agent is still executing: the card shows a spinner and
    /// the latest internal step live; flipped off when the final run replaces it.
    var subAgentRunning: Bool = false
}

/// A text file staged in the composer: its full content is folded into the next
/// user turn sent to the model; the transcript shows only the filename + size.
struct ChatAttachment: Identifiable, Equatable {
    let id = UUID()
    let name: String
    let content: String
    var bytes: Int { content.utf8.count }
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
        set { settings.modelPath = newValue }
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
        // Migrazione UNA TANTUM alla configurazione veloce misurata (2026-07,
        // demo A/B sul campo: slot 16 + ring off + bundle = 2.3-2.6 tok/s
        // contro ~1 con slot 6/ring on/contesto 302k). Applica i valori buoni
        // ai default persistiti da vecchi esperimenti; le modifiche manuali
        // FUTURE dell'utente restano sovrane (il flag impedisce di ripeterla).
        // NB: dentro init i didSet non scattano — persistenza esplicita.
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
        _ = setenv("DS4_RAW_RING", rawRingEnabled ? "1" : "0", 1)   // apply the persisted value at startup
        _ = setenv("DS4_WILLNEED_EXPERTS", willNeedEnabled ? "1" : "0", 1)   // default ON
        _ = setenv("DS4_EXPERT_PREAD", expertPreadEnabled ? "1" : "0", 1)    // default ON <24GB RAM
        _ = setenv("DS4_DENSE_STREAM", denseStreamEnabled ? "1" : "0", 1)    // default ON <24GB RAM
        _ = setenv("DS4_MLOCK", mlockEnabled ? "1" : "0", 1)                 // default ON (misurato: -38% ms/token)
        _ = setenv("DS4_DENSE_Q4", denseQ4Enabled ? "1" : "0", 1)            // default ON (lossy, disattivabile)
        _ = setenv("DS4_EXPERT_LOOKAHEAD", "\(expertLookahead)", 1)          // 0 = solo layer hash (esatto)
        _ = setenv("DS4_DENSE_AHEAD", "\(denseAhead)", 1)                    // 2 = staging un layer avanti (misurato)
        _ = setenv("DS4_ASYNC_FFN", asyncFFNEnabled ? "1" : "0", 1)           // pipeline FFN asincrona (+10% misurato)
        _ = setenv("DS4_Q8_NSG", String(q8NSG), 1)                            // partizione K matvec Q8 (auto-tune)
        _ = setenv("DS4_EXPERT_BUNDLE", expertBundleEnabled ? "1" : "0", 1)  // opt-in (duplica gli esperti su disco)
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
        // Knob del prefill regolabili a caldo (persistiti dal benchmark in
        // Settings): letti dal motore a ogni chiamata di prefill.
        _ = setenv("DS4_PREFILL_UNION", String(prefillUnion), 1)
        _ = setenv("DS4_PREFILL_CHUNK", String(prefillChunk), 1)
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
    /// Expert slot-cache slots per layer (0 = off). Memory ≈ 6,9 MB/slot ×
    /// 43 layer on the 2-bit model. Applied on the NEXT model load.
    /// DEFAULT 16: il punto dolce misurato su M1 Pro 16 GB con dense stream +
    /// pread + MLOCK + Q4 (63% hit, ridistribuzione usage-driven attiva,
    /// 1.17 tok/s a regime). Senza MLOCK i pool grandi vengono compressi/
    /// paginati e conviene 8; senza Q4 il budget bloccato è più tirato e il
    /// punto dolce scende a 12.
    var expertCacheSlots: Int = (UserDefaults.standard.object(forKey: "DS4ExpertCacheSlots") as? Int) ?? 16 {
        didSet { UserDefaults.standard.set(expertCacheSlots, forKey: "DS4ExpertCacheSlots") }
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
    /// riduzione K — stesso risultato numerico, cambia solo l'occupancy).
    /// L'ottimo dipende dai core GPU: 4 = riferimento (migliore su M1 Pro);
    /// su GPU più larghe (Max/Ultra) possono vincere 6-8. Il motore lo
    /// rilegge a ogni load del modello: l'auto-tune lo esplora con un reload.
    var q8NSG: Int = (UserDefaults.standard.object(forKey: "DS4Q8NSG") as? Int) ?? 4 {
        didSet {
            UserDefaults.standard.set(q8NSG, forKey: "DS4Q8NSG")
            _ = setenv("DS4_Q8_NSG", String(q8NSG), 1)
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
    var rawRingEnabled: Bool = (UserDefaults.standard.object(forKey: "DS4RawRing") as? Bool) ?? false {
        didSet {
            UserDefaults.standard.set(rawRingEnabled, forKey: "DS4RawRing")
            _ = setenv("DS4_RAW_RING", rawRingEnabled ? "1" : "0", 1)
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
    /// Riporta TUTTI i toggle di performance ai valori della demo veloce
    /// misurata su M1 Pro (prefill ~8 tok/s, decode 2.5+): slot 16, ring off,
    /// willneed+pread+dense stream+mlock+Q4+bundle ON. Usato dalla migrazione
    /// una-tantum in init e dal bottone in Settings ("i toggle persistiti
    /// derivano dai vecchi esperimenti e restano incollati per sempre").
    /// Con `persistExplicitly` scrive anche UserDefaults/env a mano — dentro
    /// init i didSet delle stored property NON scattano.
    func applyFastDemoDefaults(persistExplicitly: Bool = false) {
        expertCacheSlots = 16
        rawRingEnabled = false
        willNeedEnabled = true
        expertPreadEnabled = true
        denseStreamEnabled = true
        mlockEnabled = true
        denseQ4Enabled = true
        expertBundleEnabled = true
        expertLookahead = 0        // speculativo misurato neutro; i layer hash restano sempre attivi
        denseAhead = 2             // staging un layer avanti: +1,5% misurato
        asyncFFNEnabled = true     // pipeline FFN asincrona: +10% misurato, parita' certificata
        if persistExplicitly {
            let d = UserDefaults.standard
            d.set(16, forKey: "DS4ExpertCacheSlots")
            d.set(false, forKey: "DS4RawRing")
            d.set(true, forKey: "DS4WillNeed")
            d.set(true, forKey: "DS4ExpertPread")
            d.set(true, forKey: "DS4DenseStream")
            d.set(true, forKey: "DS4MLock")
            d.set(true, forKey: "DS4DenseQ4")
            d.set(true, forKey: "DS4ExpertBundle")
            d.set(0, forKey: "DS4ExpertLookahead")
            d.set(2, forKey: "DS4DenseAhead")
            d.set(true, forKey: "DS4AsyncFFN")
            _ = setenv("DS4_RAW_RING", "0", 1)
            _ = setenv("DS4_WILLNEED_EXPERTS", "1", 1)
            _ = setenv("DS4_EXPERT_PREAD", "1", 1)
            _ = setenv("DS4_DENSE_STREAM", "1", 1)
            _ = setenv("DS4_MLOCK", "1", 1)
            _ = setenv("DS4_DENSE_Q4", "1", 1)
            _ = setenv("DS4_EXPERT_BUNDLE", "1", 1)
            _ = setenv("DS4_EXPERT_LOOKAHEAD", "0", 1)
            _ = setenv("DS4_DENSE_AHEAD", "2", 1)
            _ = setenv("DS4_ASYNC_FFN", "1", 1)
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
    var prefillChunk: Int = (UserDefaults.standard.object(forKey: "DS4PrefillChunk") as? Int) ?? 512 {
        didSet {
            UserDefaults.standard.set(prefillChunk, forKey: "DS4PrefillChunk")
            _ = setenv("DS4_PREFILL_CHUNK", String(prefillChunk), 1)
        }
    }

    // Benchmark in-app (Settings): prova le combinazioni dei knob a caldo sul
    // motore GIÀ caricato e applica la migliore.
    var benchRunning = false
    var benchStatus: String?
    var benchResults: String = ""

    /// Misura DS4_PREFILL_UNION (64/192/256) e DS4_PREFILL_CHUNK (512/1024) sul
    /// modello caricato con il benchmark sintetico del motore (prefill 512 o
    /// 1024 token + 8 di decode), poi APPLICA e persiste la combinazione col
    /// prefill più veloce. ~5 run, alcuni minuti; la chat resta inutilizzabile
    /// nel frattempo (il motore è un actor seriale). I knob che richiedono un
    /// reload (DS4_PREFILL_MM, FFN/ROUTE_BATCH) non sono coperti.
    /// `quick`: SOLO l'unione esperti su un prefill da 128 token (~2-3 min) —
    /// il divario fra le unioni emerge già a decine di token, mentre il chunk
    /// richiede prefill > 512 token per esistere e resta nella modalità
    /// completa (che usa 512 + 1024 token, ~10 min).
    func runSettingsBenchmark(quick: Bool = false) {
        guard let service else { benchStatus = "Carica prima il modello."; return }
        guard phase == .ready else { benchStatus = "Attendi che il modello sia pronto."; return }
        guard !benchRunning else { return }
        benchRunning = true
        benchResults = ""
        benchStatus = quick ? "Benchmark rapido in corso… (~3 min, non usare la chat)"
                            : "Benchmark in corso… (~10 min, non usare la chat)"
        Task {
            // Nested funcs non ereditano l'isolamento MainActor in Swift 6:
            // annotazione esplicita (chiamata sincrona dal Task, che eredita
            // il MainActor del contesto).
            @MainActor func log(_ s: String) {
                benchResults += s + "\n"
                FileHandle.standardError.write(Data(("DS4 bench: " + s + "\n").utf8))
            }
            do {
                // 1) unione esperti, a chunk fisso 512. Il rapido (256 token)
                //    confronta SOLO 192 e 256: a scala corta il rumore può
                //    premiare 64, che sui prefill reali è il valore
                //    catastrofico (misurato sul campo: ~1.7 GB/token di
                //    riletture contro ~0.6 con 192) — il completo lo include
                //    solo perché a 512 token la misura è affidabile.
                let unionCtx = quick ? 256 : 512
                var bestUnion = prefillUnion
                var bestTps = 0.0
                _ = setenv("DS4_PREFILL_CHUNK", "512", 1)
                // No-op se già caldo: la prima misura union non deve pagare
                // la partenza fredda (pool esperti) e vincere/perdere per quello.
                await service.warmup()
                for union in (quick ? [192, 256] : [64, 192, 256]) {
                    _ = setenv("DS4_PREFILL_UNION", String(union), 1)
                    benchStatus = "Benchmark union=\(union)…"
                    let p = try await service.benchmark(contextTokens: unionCtx, genTokens: quick ? 4 : 8)
                    log(String(format: "union=%d (%d token): prefill %.2f tok/s, decode %.2f tok/s",
                               union, unionCtx, p.prefillTps, p.genTps))
                    if p.prefillTps > bestTps { bestTps = p.prefillTps; bestUnion = union }
                }
                _ = setenv("DS4_PREFILL_UNION", String(bestUnion), 1)
                // 2) chunk (solo modalità completa: sotto i 512 token un secondo
                //    chunk non esiste — con l'unione vincente, 1024 token).
                var bestChunk = prefillChunk
                if !quick {
                    var bestChunkTps = 0.0
                    for chunk in [512, 1024] {
                        _ = setenv("DS4_PREFILL_CHUNK", String(chunk), 1)
                        benchStatus = "Benchmark chunk=\(chunk)…"
                        let p = try await service.benchmark(contextTokens: 1024, genTokens: 8)
                        log(String(format: "union=%d chunk=%d (1024 token): prefill %.2f tok/s, decode %.2f tok/s",
                                   bestUnion, chunk, p.prefillTps, p.genTps))
                        if p.prefillTps > bestChunkTps { bestChunkTps = p.prefillTps; bestChunk = chunk }
                    }
                }
                // Applica e persisti i vincitori (i didSet rifanno i setenv).
                prefillUnion = bestUnion
                prefillChunk = bestChunk
                log("MIGLIORI: union=\(bestUnion) chunk=\(bestChunk) — applicati e salvati.")
                benchStatus = "Fatto: union=\(bestUnion), chunk=\(bestChunk) applicati."
            } catch {
                // Ripristina i valori persistiti dopo un errore/annullamento.
                _ = setenv("DS4_PREFILL_UNION", String(prefillUnion), 1)
                _ = setenv("DS4_PREFILL_CHUNK", String(prefillChunk), 1)
                benchStatus = "Benchmark fallito: \(error)"
            }
            benchRunning = false
        }
    }

    // MARK: Auto-tune per-macchina (M1/M2/…/base/Pro/Max: RAM e SSD diversi)

    /// Nome del chip (es. "Apple M1 Pro") per il referto dell'auto-tune.
    nonisolated static func chipName() -> String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        guard size > 0 else { return "Apple Silicon" }
        var buf = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &buf, &size, nil, 0)
        return String(cString: buf).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Trova i migliori knob di CARICAMENTO per QUESTA macchina — slot della
    /// cache esperti, dense-ahead, async FFN, expert look-ahead — con una
    /// coordinate descent: un reload del modello per candidato (quei knob sono
    /// letti alla creazione del decoder), misurato con un decode breve.
    ///
    /// - Candidati RAM-gated (una base M1 da 16 GB non prova mai 32 slot; una
    ///   Max da 96 GB sì), in ordine crescente di RAM: al primo COLLASSO da
    ///   pressione di memoria i candidati più grossi si saltano.
    /// - Metrica: genTpsP99 (velocità di regime), ma un candidato conta solo
    ///   se STABILE — media/p99 ≥ 0.72. Il collasso da swap ha la firma
    ///   opposta (p99 alto raggiunto presto, media che crolla token dopo
    ///   token): è esattamente ciò che distingue "più slot = più hit" da
    ///   "più slot = spirale di swap" (misurato: 24 slot su 16 GB → 1.55 GB/s).
    /// - Un candidato vince solo con un margine >2% (anti-rumore): a parità
    ///   resta il valore persistito.
    /// I vincitori sono applicati e PERSISTITI (didSet → UserDefaults+setenv);
    /// il referto resta in `benchResults` e nel log motore.
    func runAutoTune() {
        guard service != nil else { benchStatus = "Carica prima il modello."; return }
        guard phase == .ready else { benchStatus = "Attendi che il modello sia pronto."; return }
        guard !benchRunning else { return }
        benchRunning = true
        benchResults = ""
        benchStatus = "Auto-tune in corso… (~15-25 min, non usare la chat)"
        Task {
            @MainActor func log(_ s: String) {
                benchResults += s + "\n"
                FileHandle.standardError.write(Data(("DS4 autotune: " + s + "\n").utf8))
            }
            let ramGB = Int(ProcessInfo.processInfo.physicalMemory >> 30)
            let chip = Self.chipName()
            log("Auto-tune su \(chip), \(ramGB) GB RAM — un reload del modello per configurazione.")

            /// Reload col set di knob corrente e misura: prefill corto (il
            /// prefill ha il suo benchmark) + 28 token di decode.
            @MainActor func measure(_ label: String) async throws -> (score: Double, note: String) {
                benchStatus = "Auto-tune: \(label) — teardown…"
                service = nil                      // libera la RAM wired PRIMA del nuovo load
                // Il teardown di ~GB di buffer wired/mlock non è istantaneo:
                // caricare subito sopra fa girare i primi token del benchmark
                // in piena pressione di memoria (falso collasso, misurato).
                try await Task.sleep(nanoseconds: 4_000_000_000)
                benchStatus = "Auto-tune: \(label) — reload…"
                load()
                while phase == .loading { try await Task.sleep(nanoseconds: 500_000_000) }
                guard phase == .ready, let svc = service else {
                    throw NSError(domain: "DS4AutoTune", code: 1,
                                  userInfo: [NSLocalizedDescriptionKey: "reload fallito (\(label))"])
                }
                try await Task.sleep(nanoseconds: 2_000_000_000)   // assestamento memoria
                // Warmup NON misurato (pool esperti, ring denso, transitorio
                // post-load) — l'equivalente del "REGIME dal token 5" della demo.
                benchStatus = "Auto-tune: \(label) — warmup…"
                _ = try await svc.benchmark(contextTokens: 16, genTokens: 4)
                benchStatus = "Auto-tune: \(label) — misura…"
                let p = try await svc.benchmark(contextTokens: 96, genTokens: 28)
                // Stabilità dal PROFILO temporale, non da media/p99: la spirale
                // di swap ha la CODA più lenta della TESTA (degrado progressivo,
                // misurato con 24 slot su 16 GB); una partenza fredda ha la
                // firma opposta e non deve squalificare la configurazione.
                func median(_ a: [Double]) -> Double {
                    guard !a.isEmpty else { return 0 }
                    return a.sorted()[a.count / 2]
                }
                let speeds = p.genSpeeds.count > 12 ? Array(p.genSpeeds.dropFirst(4)) : p.genSpeeds
                let head = median(Array(speeds.prefix(speeds.count / 2)))
                let tail = median(Array(speeds.suffix(speeds.count / 2)))
                let ratio = head > 0 ? tail / head : 1
                let stable = ratio >= 0.75
                let score = stable ? p.genTpsP99 : 0
                let note = String(format: "%@: %.2f tok/s regime (media %.2f, coda/testa %.0f%%)%@",
                                  label, p.genTpsP99, p.genTps, ratio * 100,
                                  stable ? "" : "  ← COLLASSO PROGRESSIVO (pressione memoria)")
                log(note)
                return (score, note)
            }

            // Snapshot per il ripristino su errore: un reload fallito a metà
            // sweep non deve lasciare persistito il candidato perdente.
            let snapSlots = expertCacheSlots, snapAhead = denseAhead
            let snapAsync = asyncFFNEnabled, snapLook = expertLookahead
            let snapNSG = q8NSG
            do {
                let baseLabel = "baseline slot=\(expertCacheSlots) ahead=\(denseAhead) " +
                                "async=\(asyncFFNEnabled ? 1 : 0) look=\(expertLookahead) q8nsg=\(q8NSG)"
                var best = try await measure(baseLabel).score
                if best == 0 {
                    // Un solo retry: il primo giro paga il transitorio peggiore
                    // (teardown del motore della chat + primo reload).
                    log("baseline instabile — riprovo una volta dopo l'assestamento…")
                    try await Task.sleep(nanoseconds: 8_000_000_000)
                    best = try await measure(baseLabel + " (retry)").score
                }
                guard best > 0 else {
                    throw NSError(domain: "DS4AutoTune", code: 2, userInfo: [
                        NSLocalizedDescriptionKey: "la baseline è instabile anche al retry: " +
                        "liberare RAM (chiudere le altre app) e riprovare"])
                }

                // 1) Slot cache esperti — il knob più importante e più rischioso.
                let slotCands: [Int] = ramGB >= 96 ? [16, 24, 32, 48]
                                     : ramGB >= 48 ? [16, 24, 32]
                                     : ramGB >= 24 ? [12, 16, 24]
                                     : [12, 16, 20]
                for cand in slotCands where cand != expertCacheSlots {
                    let saved = expertCacheSlots
                    expertCacheSlots = cand
                    let r = try await measure("slot=\(cand)")
                    if r.score > best * 1.02 { best = r.score }
                    else {
                        expertCacheSlots = saved
                        if r.score == 0 && cand > saved { break }   // collasso: niente candidati più grossi
                    }
                }

                // 2) Dense-ahead (profondità dello staging ring: 1-3 slot extra).
                for cand in [1, 2, 3] where cand != denseAhead {
                    let saved = denseAhead
                    denseAhead = cand
                    let r = try await measure("ahead=\(cand)")
                    if r.score > best * 1.02 { best = r.score } else { denseAhead = saved }
                }

                // 3) FFN asincrona (misurata +10% su M1 Pro, ma va verificata per macchina).
                do {
                    let saved = asyncFFNEnabled
                    asyncFFNEnabled = !saved
                    let r = try await measure("asyncFFN=\(asyncFFNEnabled ? 1 : 0)")
                    if r.score > best * 1.02 { best = r.score } else { asyncFFNEnabled = saved }
                }

                // 4) Expert look-ahead speculativo (0 = solo layer hash, esatto).
                for cand in [4, 8] where cand != expertLookahead {
                    let saved = expertLookahead
                    expertLookahead = cand
                    let r = try await measure("look=\(cand)")
                    if r.score > best * 1.02 { best = r.score } else { expertLookahead = saved }
                }

                // 5) DS4_Q8_NSG: partizione K dei matvec Q8 — l'unico knob
                //    davvero legato ai CORE GPU (occupancy). Numerica identica
                //    per costruzione; il motore lo rilegge a ogni load.
                for cand in [2, 8] where cand != q8NSG {
                    let saved = q8NSG
                    q8NSG = cand
                    let r = try await measure("q8nsg=\(cand)")
                    if r.score > best * 1.02 { best = r.score } else { q8NSG = saved }
                }

                // Reload finale con i vincitori (le proprietà sono già persistite).
                _ = try await measure("finale slot=\(expertCacheSlots) ahead=\(denseAhead) " +
                                      "async=\(asyncFFNEnabled ? 1 : 0) look=\(expertLookahead) q8nsg=\(q8NSG)")
                let summary = "slot=\(expertCacheSlots) ahead=\(denseAhead) " +
                              "async=\(asyncFFNEnabled ? 1 : 0) look=\(expertLookahead) q8nsg=\(q8NSG)"
                log("MIGLIORI per \(chip)/\(ramGB)GB: \(summary) — applicati e salvati.")
                UserDefaults.standard.set("\(summary) @ \(Date().formatted())",
                                          forKey: "DS4AutoTune-\(chip)-\(ramGB)")
                benchStatus = "Auto-tune completato: \(summary)"
            } catch {
                expertCacheSlots = snapSlots; denseAhead = snapAhead
                asyncFFNEnabled = snapAsync; expertLookahead = snapLook
                q8NSG = snapNSG
                benchStatus = "Auto-tune fallito: \(error.localizedDescription)"
                log("INTERROTTO: \(error.localizedDescription) — knob ripristinati ai valori di partenza; " +
                    "se il modello è scarico, ricaricalo dalle Settings.")
                if phase != .loading && phase != .ready { load() }
            }
            benchRunning = false
        }
    }

    /// Esito dell'ultima generazione manuale dell'expert-bundle (bottone Settings).
    var bundleBuildStatus: String?

    /// Verifica/crea l'expert-bundle ORA, senza aspettare un load del modello.
    /// Gira in background (una build da ~72 GB dura minuti); l'esito compare
    /// accanto al bottone e i dettagli nel Log motore ("DS4 expbundle:").
    func buildExpertBundleNow() {
        guard !modelPath.isEmpty else { bundleBuildStatus = "Nessun modello selezionato."; return }
        guard phase != .loading else { bundleBuildStatus = "Attendi la fine del load in corso."; return }
        bundleBuildStatus = "Verifica/creazione in corso… (dettagli nel Log motore)"
        let path = modelPath
        Task.detached(priority: .userInitiated) {
            let outcome = ExpertBundleTool.ensure(modelPath: path)
            await MainActor.run { self.bundleBuildStatus = outcome }
        }
    }

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
    static var residentDenseAuto: Bool { MemoryInfo.physicalBytes >= 24 * 1_073_741_824 }
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
    static var diskKVDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("DwarfStar/kv-cache", isDirectory: true)
    }

    /// Application Support/DwarfStar/q4-cache: i .q4dense del requant Q4
    /// (~1.4 GB per modello) — scrivibile anche in sandbox.
    static var q4CacheDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("DwarfStar/q4-cache", isDirectory: true)
    }

    /// Dove l'app costruisce l'expert-bundle quando non può scrivere accanto al
    /// GGUF (sandbox). ATTENZIONE: il sidecar duplica la regione esperti (~72 GB
    /// sul Flash 2-bit) — la GUI lo dice esplicitamente nel toggle.
    static var bundleDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("DwarfStar/expert-bundle", isDirectory: true)
    }

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
    var selectedAgent: AgentProfile { agents.first { $0.id == selectedAgentId } ?? agents[0] }

    static func loadAgents() -> [AgentProfile] {
        if let data = UserDefaults.standard.data(forKey: "DS4Agents"),
           var arr = try? JSONDecoder().decode([AgentProfile].self, from: data), !arr.isEmpty {
            // New DEFAULT agents (e.g. "code") must appear even for users with a
            // persisted list: append the missing ones without touching edits.
            for d in AgentProfile.defaults where !arr.contains(where: { $0.id == d.id }) {
                arr.append(d)
            }
            return arr
        }
        return AgentProfile.defaults
    }

    func saveAgents() {
        if let data = try? JSONEncoder().encode(agents) {
            UserDefaults.standard.set(data, forKey: "DS4Agents")
        }
    }

    func isDefaultAgent(_ id: String) -> Bool { AgentProfile.defaults.contains { $0.id == id } }

    func addAgent() {
        let id = "custom-\(UUID().uuidString.prefix(8))"
        agents.append(AgentProfile(id: id, name: "New Agent", icon: "person.fill.questionmark",
                                   systemPrompt: "", toolNames: []))
        saveAgents()
    }

    func deleteAgent(_ id: String) {
        guard !isDefaultAgent(id), agents.count > 1 else { return }
        agents.removeAll { $0.id == id }
        if selectedAgentId == id { selectAgent(agents[0].id) }
        saveAgents()
    }

    func restoreDefaultAgents() {
        agents = AgentProfile.defaults
        if !agents.contains(where: { $0.id == selectedAgentId }) { selectAgent(agents[0].id) }
        saveAgents()
    }

    /// Agents as pretty JSON (for export/sharing between machines).
    func exportAgentsData() -> Data? {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try? enc.encode(agents)
    }

    /// Merge agents from JSON: matching ids are updated, new ones appended.
    /// Returns how many agents were imported (0 = invalid file).
    @discardableResult
    func importAgents(from data: Data) -> Int {
        guard let imported = try? JSONDecoder().decode([AgentProfile].self, from: data),
              !imported.isEmpty else { return 0 }
        for a in imported {
            if let i = agents.firstIndex(where: { $0.id == a.id }) { agents[i] = a }
            else { agents.append(a) }
        }
        saveAgents()
        return imported.count
    }

    /// The agent with the user's extra system prompt appended (if any).
    private func resolvedAgent() -> AgentProfile {
        var agent = selectedAgent
        let extra = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !extra.isEmpty {
            agent.systemPrompt = agent.systemPrompt.isEmpty ? extra : agent.systemPrompt + "\n\n" + extra
        }
        return agent
    }

    /// Apply the agent to the running service: fresh chat with its role + tools,
    /// per-agent usage profile swapped in, slot-cache re-warmed.
    private func applyAgent() {
        guard let service else { return }
        let agent = resolvedAgent()
        toolsEnabled = !agent.toolNames.isEmpty
        enabledToolNames = Set(agent.toolNames)
        let tools = toolsEnabled ? ToolRegistry.autoSpecs(enabled: enabledToolNames) : []
        Task {
            await service.setAgent(agent, tools: tools)
            await service.setCompactTools(compactTools)
            refreshTuningInfo()
            // Se il CAMBIO di agente ha invalidato la slot-cache (profilo
            // usage nuovo), i pool si riscaldano ORA in background invece che
            // dentro il primo messaggio. No-op quando l'agente non è cambiato.
            await service.warmup()
        }
    }

    func selectAgent(_ id: String) {
        selectedAgentId = id
        startNewChat()   // a role switch starts a fresh persisted chat with that role
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
    private var sampling: SamplingParams {
        // topK 40 (default llama.cpp): con topK=0 si campiona sull'INTERO
        // vocabolario DeepSeek (129k token, in gran parte cinesi) e sulla coda
        // rumorosa di un modello 2-bit basta pescare UN token cinese perché il
        // contesto trascini tutta la risposta in cinese — visto in campo a
        // temperature del tutto normali (0.6). Il tetto a 40 taglia quella
        // coda senza togliere varietà; motore/server/demo restano fedeli al C.
        SamplingParams(temperature: Float(temperature), topK: 40,
                       repetitionPenalty: Float(repetitionPenalty))
    }

    // Tools.
    var toolsEnabled = false
    var enabledToolNames: Set<String> = Set(ToolRegistry.builtins.map { $0.spec.name })
    /// Compact tool declaration (just name(params)) — fewer prefill tokens. On by
    /// default for local inference.
    var compactTools = true
    /// Tool calls awaiting a manually-entered result (non-built-in tools).
    var pendingManualCalls: [ToolCall] = []
    /// Drives the manual-results sheet (set when `pendingManualCalls` is filled).
    var awaitingManualResults = false
    private var partialAutoOutputs: [ToolOutput] = []
    /// Tool-loop bound. Illimitato su richiesta: il loop si ferma comunque quando
    /// il contesto è pieno o l'utente preme Stop.
    private var toolRounds = 0
    private var maxToolRounds: Int { .max }

    // Live state.
    var phase: Phase = .needsModel
    var info: ModelInfo?
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
    private var enginePrimed = true

    private var service: InferenceService?
    private var generation: Task<Void, Never>?

    /// THE single in-process engine, loaded once in Settings and SHARED by every
    /// feature (Chat, Benchmark, Server). There is never a second full engine:
    /// with the default resident-Q4 + mlock config a second copy doubles wired
    /// memory and OOM-crashes on 16 GB. `InferenceService` is an actor, so
    /// concurrent callers are serialized safely. nil until a model is ready.
    var sharedEngine: InferenceService? { isReady ? service : nil }
    /// Shared engine gated for KV-mutating uses (benchmark): a run rewrites the
    /// KV, so it's refused while the chat is mid-generation.
    var benchmarkService: InferenceService? { (isReady && !isGenerating) ? service : nil }
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

    private var bookmarkRestored = false

    /// Under the App Sandbox, re-open the last user-picked GGUF via its persisted
    /// security-scoped bookmark (starts access). No-op if none / already restored.
    func restoreModelBookmark() {
        guard !bookmarkRestored else { return }
        bookmarkRestored = true
        if let path = ModelPicker.restoreBookmark() { modelPath = path }
        // Folder grant (sandbox): re-arm access to the model's directory so the
        // sidecar caches next to the GGUF stay readable across launches.
        ModelPicker.restoreFolderBookmark()
    }

    /// Scan the configured directories for GGUF files.
    func scanModels() {
        let gguf = (scriptDir as NSString).appendingPathComponent("gguf")
        discoveredModels = ModelCatalog.scan(directories: [scriptDir, gguf])
    }

    /// Apply the preset recommended for the detected RAM.
    func applyRecommendedPreset() {
        scanModels()
        let preset = HardwarePresets.forRAM(MemoryInfo.physicalBytes)
        contextSize = preset.contextSize

        var note = preset.summary
        if preset.prefersTwoBit {
            if let twoBit = discoveredModels.first(where: { HardwarePresets.isTwoBit($0.name) }) {
                modelPath = twoBit.path
                note += " Selected 2-bit model: \(twoBit.name)."
            } else {
                note += " No 2-bit model found: download it with the Download button (target q2-imatrix) or `./download_model.sh q2-imatrix`."
            }
        }
        presetNote = note
    }

    /// Open the model off the main thread, then flip to `.ready`.
    /// Avanzamento del caricamento del modello (0…1) + fase corrente: scritti
    /// dal motore via LoadProgress e riletti qui a ~8 Hz mentre `.loading`.
    var loadFraction: Double = 0
    var loadStage: String = ""

    func load() {
        guard phase != .loading else { return }
        phase = .loading
        loadFraction = 0
        loadStage = ""
        LoadProgress.shared.reset()
        // Poll del progresso finché il load è in corso (si auto-cancella).
        let poller = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                let s = LoadProgress.shared.snapshot
                self?.loadFraction = s.fraction
                self?.loadStage = s.stage
                try? await Task.sleep(nanoseconds: 120_000_000)
            }
        }
        let path = modelPath, ctx = contextSize
        let cacheSlots = expertCacheSlots
        let kvDir = diskKVEnabled ? Self.diskKVDirectory : nil
        let kvBudgetTokens = diskKVBudgetKTok * 1000
        Task.detached(priority: .userInitiated) {
            defer { poller.cancel() }
            do {
                // Il motore si costruisce su un thread GCD CLASSICO, non sul
                // thread del pool cooperativo di Swift Concurrency di questo
                // Task: il load usa DispatchQueue.concurrentPerform ovunque
                // (riquantizzazione Q4, lettura cache, fill esperti) e chiamato
                // da un thread cooperativo può degradare a esecuzione quasi
                // seriale — un core al 100% e la riquantizzazione in ore. La
                // demo CLI, che carica dal main thread, ha sempre avuto il
                // fan-out pieno: stesso contesto anche qui.
                let svc = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<InferenceService, Error>) in
                    DispatchQueue.global(qos: .userInitiated).async {
                        do {
                            cont.resume(returning: try InferenceService(
                                modelPath: path,
                                contextSize: ctx,
                                systemPrompt: nil,   // set by applyAgent below
                                expertCacheSlots: cacheSlots > 0 ? cacheSlots : nil))
                        } catch {
                            cont.resume(throwing: error)
                        }
                    }
                }
                await svc.setDiskKV(directory: kvDir, budgetTokens: kvBudgetTokens)
                let info = await svc.modelInfo()
                await MainActor.run {
                    self.service = svc
                    self.info = info
                    self.phase = .ready
                    self.activate(self.activeSessionId)   // load the active chat + apply its role
                }
                // Warmup in background A UI GIÀ PRONTA: paga ORA il costo
                // una-tantum della prima generazione (creazione pool esperti
                // + fill top-usage, ~GB da SSD, kernel Metal freddi) invece
                // che sul primo messaggio. Un send immediato si accoda al
                // warmup sull'actor: mai più lento di prima, di norma il
                // primo token passa da ~5-7s a ~1s.
                await svc.warmup()
            } catch {
                await MainActor.run { self.phase = .failed("\(error)") }
            }
        }
    }

    /// Push the current tool selection to the engine (call after toggling tools).
    /// Also re-run when an MCP server (dis)connects: the declared set includes
    /// MCP specs, which exist only while their server is connected.
    func syncTools() {
        guard let service else { return }
        let tools = toolsEnabled ? ToolRegistry.autoSpecs(enabled: enabledToolNames) : []
        let compact = compactTools
        Task { await service.setTools(tools); await service.setCompactTools(compact) }
    }

    private var thinkMode: DS4ThinkMode { think ? .high : .none }

    /// Send the current input (+ any imported text files) and stream the reply,
    /// running the tool loop. Attachments are folded into the turn sent to the
    /// model; the transcript shows just the typed text and the filenames.
    func send() {
        let typed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let service, !isGenerating, !(typed.isEmpty && attachments.isEmpty) else { return }
        let atts = attachments
        let text = Self.composeUserText(typed: typed, attachments: atts)
        input = ""
        attachments = []
        attachmentNote = nil
        // If the engine doesn't hold this (reopened) chat yet, re-feed the prior
        // turns on this first send. Capture them BEFORE appending the new rows.
        let primed = enginePrimed
        let history = primed ? [] : Self.chatTurns(from: messages)
        let sys = primed ? nil : (resolvedAgent().systemPrompt.isEmpty ? nil : resolvedAgent().systemPrompt)
        enginePrimed = true
        messages.append(UIMessage(role: .user, text: typed, attachments: atts.map(\.name)))
        let index = appendAssistant()
        isGenerating = true
        toolRounds = 0                     // fresh user turn resets the tool-loop guard
        persistActiveSession()             // checkpoint the user turn right away

        let mode = thinkMode
        let params = sampling             // capture: `self` is weak inside the Task
        // .userInitiated: the decode runs on the InferenceService actor's executor at
        // THIS task's QoS. A default-priority task lets macOS deprioritize the work —
        // and, worse for the SSD-streaming path, throttle its expert-gather reads — so
        // the app decodes slower than the foreground CLI demo. Match the model-load
        // task's priority (and the CLI's foreground QoS) explicitly.
        generation = Task(priority: .userInitiated) { [weak self] in
            let stream = primed
                ? await service.send(userText: text, thinkMode: mode, sampling: params, maxTokens: 4096)
                : await service.sendWithHistory(history, userText: text, systemPrompt: sys,
                                                thinkMode: mode, sampling: params, maxTokens: 4096)
            await self?.consume(stream, into: index)
            let continued = await self?.handleToolCalls(assistantIndex: index) ?? false
            if !continued { await MainActor.run { self?.finishIfIdle() } }
        }
    }

    /// Submit manually-entered results for the pending (non-built-in) tool calls.
    func submitManualResults(_ contents: [String: String]) {
        guard let service, !pendingManualCalls.isEmpty else { return }
        var outputs = partialAutoOutputs
        for c in pendingManualCalls {
            let content = contents[c.id] ?? ""
            outputs.append(ToolOutput(callId: c.id, name: c.name, content: content))
            messages.append(UIMessage(role: .tool, text: "\(c.name) → \(content)"))
        }
        pendingManualCalls = []
        partialAutoOutputs = []
        awaitingManualResults = false
        continueWithToolOutputs(outputs, service: service)
    }

    /// Abandon the pending manual tool calls without answering them. The calls
    /// stay in the committed KV (the model emitted them), so the next user turn
    /// follows an unanswered call — the model generally copes, but we surface the
    /// abandonment in the transcript so the state is visible.
    func cancelManualResults() {
        if !pendingManualCalls.isEmpty {
            let names = pendingManualCalls.map(\.name).joined(separator: ", ")
            messages.append(UIMessage(role: .tool, text: "✗ no results provided for: \(names)"))
        }
        pendingManualCalls = []
        partialAutoOutputs = []
        awaitingManualResults = false
        finishIfIdle()
    }

    func stop() { generation?.cancel() }

    // MARK: - Text-file attachments

    /// Present an open panel for one or more text files and stage their contents.
    /// Honors the App Sandbox: each pick grants security-scoped access for the
    /// one-shot read (entitlement: files.user-selected.read-write).
    func pickAndAttachFiles() {
        attachmentNote = nil
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.title = "Import Text Files"
        panel.prompt = "Import"
        // Prefer text types; allow any file (.data) so odd extensions can still be
        // picked — non-text content simply fails to decode and is reported.
        panel.allowedContentTypes = [.text, .plainText, .sourceCode, .json, .xml,
                                     .commaSeparatedText, .log, .data]
        guard panel.runModal() == .OK else { return }
        importFiles(panel.urls)
    }

    /// Read each URL as text (UTF-8, then Latin-1) and stage it; collect failures.
    func importFiles(_ urls: [URL]) {
        var failed: [String] = []
        for url in urls {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            guard let text = Self.readText(url) else { failed.append(url.lastPathComponent); continue }
            let name = url.lastPathComponent
            // Re-importing the identical file is a no-op (avoid duplicate context).
            if !attachments.contains(where: { $0.name == name && $0.content == text }) {
                attachments.append(ChatAttachment(name: name, content: text))
            }
        }
        if !failed.isEmpty {
            attachmentNote = "Could not read as text: \(failed.joined(separator: ", "))"
        }
    }

    func removeAttachment(_ id: UUID) { attachments.removeAll { $0.id == id } }

    /// Decode a file as text: UTF-8 first, then Latin-1 (covers most legacy files).
    static func readText(_ url: URL) -> String? {
        if let s = try? String(contentsOf: url, encoding: .utf8) { return s }
        return try? String(contentsOf: url, encoding: .isoLatin1)
    }

    /// Fold staged attachments + the typed message into the text sent to the model.
    /// Each file is delimited so the model can tell content apart from the prompt.
    static func composeUserText(typed: String, attachments: [ChatAttachment]) -> String {
        guard !attachments.isEmpty else { return typed }
        var parts: [String] = attachments.map {
            "--- Attached file: \($0.name) ---\n\($0.content)\n--- end: \($0.name) ---"
        }
        if !typed.isEmpty { parts.append(typed) }
        return parts.joined(separator: "\n\n")
    }

    // MARK: - Tuning tab

    func refreshTuningInfo() {
        guard let service else { tuningInfo = nil; return }
        Task { tuningInfo = await service.tuningInfo() }
    }

    func saveExpertUsage() {
        guard let service else { return }
        Task { await service.saveExpertUsage(); refreshTuningInfo() }
    }

    func resetExpertUsage() {
        guard let service else { return }
        Task { await service.resetExpertUsage(); refreshTuningInfo() }
    }

    // MARK: - Sessions (create / switch / delete / rename / persist)

    func newChat() { startNewChat() }

    /// Make a fresh persisted chat with the current role active. Reuses the current
    /// chat if it's still empty (so flipping the agent before sending anything
    /// doesn't pile up blank chats).
    private func startNewChat() {
        generation?.cancel()
        isGenerating = false
        status = ""
        clearTransientTurnState()
        contextUsed = 0
        enginePrimed = true
        if let i = sessions.firstIndex(where: { $0.id == activeSessionId }),
           messages.isEmpty, sessions[i].messages.isEmpty {
            sessions[i].agentId = selectedAgentId
            sessions[i].systemNote = systemPrompt
            ChatSessionStore.save(sessions[i])
        } else {
            persistActiveSession()
            let session = ChatSession(agentId: selectedAgentId, systemNote: systemPrompt,
                                      modelName: info?.name ?? "")
            sessions.insert(session, at: 0)
            activeSessionId = session.id
            messages.removeAll()
            ChatSessionStore.save(session)
        }
        applyAgent()                      // role + tools + usage profile + resetConversation
    }

    /// Switch to an existing chat: persist the current one, then restore the target.
    func switchSession(_ id: String) {
        guard id != activeSessionId else { return }
        persistActiveSession()
        activate(id)
    }

    func deleteSession(_ id: String) {
        let wasActive = (id == activeSessionId)
        ChatSessionStore.delete(id)
        sessions.removeAll { $0.id == id }
        guard wasActive else { return }
        if let next = sessions.first { activate(next.id) } else { startNewChat() }
    }

    func renameSession(_ id: String, to title: String) {
        guard let i = sessions.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        sessions[i].title = trimmed.isEmpty ? ChatSession.untitled : trimmed
        ChatSessionStore.save(sessions[i])
    }

    /// Restore a session into the live UI and reset the engine to its role (without
    /// persisting the previous one — callers do that first when needed). A non-empty
    /// chat must re-prime on the next send, since the engine no longer holds its KV.
    private func activate(_ id: String) {
        guard let target = sessions.first(where: { $0.id == id }) else { return }
        generation?.cancel()
        isGenerating = false
        status = ""
        clearTransientTurnState()
        activeSessionId = id
        messages = target.messages.map { UIMessage(stored: $0) }
        contextUsed = 0
        systemPrompt = target.systemNote
        if target.agentId != selectedAgentId, agents.contains(where: { $0.id == target.agentId }) {
            selectedAgentId = target.agentId
        }
        enginePrimed = messages.isEmpty
        applyAgent()                      // reset engine to the role; first send re-primes
    }

    /// Snapshot the live transcript into the active session and write it to disk.
    /// Trailing/empty assistant placeholders are dropped so a chat interrupted
    /// mid-generation doesn't reopen with a blank bubble.
    private func persistActiveSession() {
        guard let i = sessions.firstIndex(where: { $0.id == activeSessionId }) else { return }
        let kept = messages.filter {
            !($0.role == .assistant && $0.text.isEmpty && $0.reasoning.isEmpty
              && $0.toolCalls.isEmpty && $0.subAgent == nil)
        }
        sessions[i].messages = kept.map { StoredMessage(from: $0) }
        sessions[i].agentId = selectedAgentId
        sessions[i].systemNote = systemPrompt
        if let name = info?.name { sessions[i].modelName = name }
        sessions[i].updatedAt = Date()
        if sessions[i].title == ChatSession.untitled {
            sessions[i].title = Self.deriveTitle(from: messages)
        }
        ChatSessionStore.save(sessions[i])
    }

    private func clearTransientTurnState() {
        attachments = []
        attachmentNote = nil
        pendingManualCalls = []
        partialAutoOutputs = []
        awaitingManualResults = false
        toolRounds = 0
    }

    /// First non-empty user line, for an auto title.
    private static func deriveTitle(from messages: [UIMessage]) -> String {
        guard let first = messages.first(where: { $0.role == .user && !$0.text.isEmpty }) else {
            return ChatSession.untitled
        }
        let line = first.text.split(separator: "\n").first.map(String.init) ?? first.text
        return String(line.prefix(48))
    }

    // MARK: - Internals

    /// Parse the (target, question, agent, tools) arguments of a `subagent_run`
    /// call. `tools` accepts a JSON array or a comma/space-separated string (some
    /// models quote list arguments).
    private static func subAgentArgs(_ json: String) -> (target: String, question: String, agent: String, tools: [String]) {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return ("", "", "", []) }
        var tools: [String] = []
        if let arr = obj["tools"] as? [Any] { tools = arr.compactMap { $0 as? String } }
        else if let s = obj["tools"] as? String {
            tools = s.split(whereSeparator: { $0 == "," || $0 == " " }).map(String.init)
        }
        return ((obj["target"] as? String) ?? "", (obj["question"] as? String) ?? "",
                (obj["agent"] as? String) ?? "", tools)
    }

    /// Validate a `subagent_run` call before executing it: nil when well-formed,
    /// otherwise an explanatory error the model can act on (fix and retry).
    /// Silent fallbacks here (empty question, ignored unknown role…) would waste
    /// a whole sub-agent run and leave the user staring at a garbage answer.
    private static func subAgentCallProblem(_ argumentsJSON: String,
                                            question: String, agent: String, tools: [String]) -> String? {
        guard let data = argumentsJSON.data(using: .utf8),
              (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] != nil else {
            return #"the arguments are not a JSON object; expected {"target":"<file path or project>","question":"<self-contained task>"}"#
        }
        if question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "missing 'question': pass a self-contained task (the sub-agent does not see this chat)"
        }
        let agentId = agent.trimmingCharacters(in: .whitespacesAndNewlines)
        if !agentId.isEmpty, !AgentRegistry.shared.all().contains(where: { $0.id == agentId }) {
            let ids = AgentRegistry.shared.all().map(\.id).joined(separator: ", ")
            return "unknown agent '\(agentId)'; available agent ids: \(ids) (see agents_list)"
        }
        // MCP tools run app-side against external servers; the sub-agent loop is
        // engine-side and would silently drop them — fail loudly instead so the
        // model retries with built-ins (or the user learns why it can't work).
        let mcpRequested = tools.filter { MCPManager.shared.isMCPTool(named: $0) }
        if !mcpRequested.isEmpty {
            return "MCP tools cannot be granted to sub-agents: \(mcpRequested.joined(separator: ", ")); retry with built-in tools only"
        }
        if !tools.isEmpty, !tools.contains(where: { ToolRegistry.subAgentGrantable.contains($0) }) {
            return "none of the requested tools exist or are grantable to sub-agents; grantable tools: \(ToolRegistry.subAgentGrantable.sorted().joined(separator: ", "))"
        }
        return nil
    }

    private func appendAssistant() -> Int {
        messages.append(UIMessage(role: .assistant, text: ""))
        return messages.count - 1
    }

    private func finishIfIdle() {
        if pendingManualCalls.isEmpty {
            isGenerating = false
            status = ""
            refreshContextUsage()
            persistActiveSession()        // checkpoint the completed turn
            // Il Profilo decode va nel Log motore DOPO OGNI risposta: i contatori
            // sono raccolti comunque, il report costa nulla, e "a quanto genera
            // davvero l'app e dove va il tempo" deve essere leggibile dal log
            // senza attivare niente. (profileRouteEnabled resta il gate della
            // sola scomposizione route/attn, che aggiunge sync GPU.)
            emitDecodeProfile()
        }
    }

    /// Print the last turn's prefill + decode profiles to stderr so they land in
    /// the Log motore (EngineLog captures fd 2), mirroring the demo's DIAG output.
    private func emitDecodeProfile() {
        guard let service else { return }
        Task {
            let prefill = await service.prefillProfileReport()
            let report = await service.decodeProfileReport()
            FileHandle.standardError.write(Data(("\n" + prefill + "\n\n" + report + "\n").utf8))
        }
    }

    /// Refresh the committed-token count (context usage) from the engine.
    private func refreshContextUsage() {
        guard let service else { contextUsed = 0; return }
        Task { contextUsed = await service.committedTokens() }
    }

    /// Drain one generation stream into the assistant message at `index`.
    private func consume(_ stream: AsyncThrowingStream<GenEvent, Error>, into index: Int) async {
        do {
            for try await event in stream {
                guard index < messages.count else { break }
                switch event {
                case .reasoning(let r): messages[index].reasoning += r
                case .text(let t): messages[index].text += t
                case .toolStream(let s): messages[index].toolStreamText += s
                case .toolCall(let calls):
                    messages[index].toolCalls = calls
                    // The block closed: drop the raw live markup; the formatted card
                    // (ToolCallView) takes over.
                    messages[index].toolStreamText = ""
                    // When the model spelled the DSML markup out as plain text (it
                    // streamed into the bubble), strip the parsed block + any leaked
                    // malformed markup from view.
                    if !messages[index].text.isEmpty {
                        let visible = ToolCallParser.parse(messages[index].text, markup: .dsv4).visibleText
                        messages[index].text = ToolCallParser.stripLeakedMarkup(visible, markup: .dsv4)
                    }
                case .progress(let p): status = p
                }
            }
            // The stream ended: the raw live markup was ephemeral feedback — drop it
            // (a parsed call shows as a card). Also scrub any malformed tool markup
            // the model emitted as text (degraded 2-bit output) so the final bubble
            // shows clean prose. A tool block that streamed but never parsed into a
            // call must NOT vanish silently: surface it as an explicit error row so
            // the user sees the model attempted (and botched) a tool call.
            if index < messages.count {
                let unparsed = messages[index].toolStreamText.trimmingCharacters(in: .whitespacesAndNewlines)
                messages[index].toolStreamText = ""
                messages[index].text = ToolCallParser.stripLeakedMarkup(messages[index].text, markup: .dsv4)
                if !unparsed.isEmpty, messages[index].toolCalls.isEmpty {
                    messages.append(UIMessage(role: .tool,
                        text: "✗ malformed tool call (not executed): \(String(unparsed.prefix(300)))"))
                }
            }
        } catch is CancellationError {
            // User-initiated stop: keep the partial text, no error banner.
        } catch {
            let tail = EngineLog.shared.tail()
            if index < messages.count {
                messages[index].text += "\n[errore: \(error)]"
                if !tail.isEmpty { messages[index].text += "\n\n--- log motore ---\n\(tail)" }
            }
        }
    }

    /// Execute the tool calls the assistant emitted: auto-run built-ins, collect
    /// manual ones, and continue the conversation. Returns true if generation
    /// continues (a continuation was spawned or we're awaiting manual input) — in
    /// which case the caller must NOT mark generation finished.
    private func handleToolCalls(assistantIndex index: Int) async -> Bool {
        guard let service, index < messages.count else { return false }
        let calls = messages[index].toolCalls
        guard !calls.isEmpty else { return false }

        toolRounds += 1
        if toolRounds > maxToolRounds {
            messages.append(UIMessage(role: .tool, text: "Too many tool rounds (\(maxToolRounds)); stopped."))
            return false
        }

        var outputs: [ToolOutput] = []
        var manual: [ToolCall] = []
        for c in calls {
            // subagent_run runs ON the engine (it drives the decoder in an isolated
            // context): the main KV only commits this call + the returned answer.
            if c.name == "subagent_run" {
                let (target, question, agent, tools) = Self.subAgentArgs(c.argumentsJSON)
                // A malformed call must fail loudly BEFORE spending a sub-agent run
                // on it: the explanatory error goes back to the model (so it can fix
                // the call) and into the transcript (so the failure is visible).
                if let problem = Self.subAgentCallProblem(c.argumentsJSON, question: question,
                                                          agent: agent, tools: tools) {
                    messages.append(UIMessage(role: .tool, text: "✗ subagent_run not executed: \(problem)"))
                    outputs.append(ToolOutput(callId: c.id, name: c.name,
                                              content: "Error, sub-agent NOT run: \(problem)"))
                    continue
                }
                status = "sub-agent su \(target)…"
                // Show the run in the transcript IMMEDIATELY (a sub-agent can take
                // minutes) and stream its internal steps into the card as they
                // happen; the placeholder is replaced in place when it finishes.
                let placeholder = messages.count
                messages.append(UIMessage(role: .tool, text: "",
                    subAgent: InferenceService.SubAgentRun(
                        target: target.isEmpty ? "project" : target, question: question,
                        answer: "", steps: []),
                    subAgentRunning: true))
                // The steps streamed so far: kept when the run errors out/stops, so
                // the transcript shows how far it got instead of losing the trace.
                func streamedSteps() -> [String] {
                    placeholder < messages.count ? (messages[placeholder].subAgent?.steps ?? []) : []
                }
                let run: InferenceService.SubAgentRun
                do {
                    run = try await service.runSubAgent(
                        target: target, question: question, agent: agent, tools: tools,
                        onStep: { [weak self] step in
                            Task { @MainActor in
                                guard let self, placeholder < self.messages.count,
                                      self.messages[placeholder].subAgentRunning,
                                      let sa = self.messages[placeholder].subAgent else { return }
                                self.messages[placeholder].subAgent = InferenceService.SubAgentRun(
                                    target: sa.target, question: sa.question,
                                    answer: sa.answer, steps: sa.steps + [step])
                            }
                        })
                } catch is CancellationError {
                    run = InferenceService.SubAgentRun(target: target, question: question,
                                                       answer: "(sub-agent stopped)", steps: streamedSteps())
                } catch {
                    run = InferenceService.SubAgentRun(target: target, question: question,
                                                       answer: "Sub-agent error: \(error)", steps: streamedSteps())
                }
                if placeholder < messages.count, messages[placeholder].subAgent != nil {
                    messages[placeholder].subAgent = run
                    messages[placeholder].subAgentRunning = false
                } else {
                    messages.append(UIMessage(role: .tool, text: "", subAgent: run))
                }
                outputs.append(ToolOutput(callId: c.id, name: c.name, content: run.answer))
                continue
            }
            // Built-ins run synchronously; MCP tools go async to their server
            // (a failure — server gone, timeout — comes back as an error output
            // the model can react to).
            if MCPManager.shared.isMCPTool(named: c.name) { status = "MCP: \(c.name)…" }
            if let out = await ToolRegistry.executeAuto(c) {
                outputs.append(out)
                messages.append(UIMessage(role: .tool, text: "\(c.name) → \(out.content)"))
                // Stop pressed during the (long) MCP await: show the result but
                // do NOT spawn a continuation — the user asked this turn to end.
                if Task.isCancelled { return false }
                continue
            }
            manual.append(c)
        }

        if !manual.isEmpty {
            partialAutoOutputs = outputs
            pendingManualCalls = manual
            awaitingManualResults = true
            return true
        }
        continueWithToolOutputs(outputs, service: service)
        return true
    }

    /// Feed tool outputs back and stream the model's continuation (which may emit
    /// further tool calls — the loop repeats, bounded by maxToolRounds).
    private func continueWithToolOutputs(_ outputs: [ToolOutput], service: InferenceService) {
        let index = appendAssistant()
        isGenerating = true
        let mode = thinkMode
        let params = sampling             // capture: `self` is weak inside the Task
        generation = Task(priority: .userInitiated) { [weak self] in   // see send(): keep decode QoS high
            let stream = await service.provideToolResults(outputs, thinkMode: mode,
                                                          sampling: params, maxTokens: 4096)
            await self?.consume(stream, into: index)
            let continued = await self?.handleToolCalls(assistantIndex: index) ?? false
            if !continued { await MainActor.run { self?.finishIfIdle() } }
        }
    }
}

// MARK: - Persistence mapping (UIMessage <-> StoredMessage)

extension ChatRole {
    var persistedString: String {
        switch self {
        case .system: return "system"
        case .user: return "user"
        case .assistant: return "assistant"
        case .tool: return "tool"
        }
    }
    init(persisted: String) {
        switch persisted {
        case "user": self = .user
        case "assistant": self = .assistant
        case "tool": self = .tool
        default: self = .system
        }
    }
}

extension StoredMessage {
    init(from m: UIMessage) {
        self.role = m.role.persistedString
        self.reasoning = m.reasoning
        self.text = m.text
        self.attachments = m.attachments
        self.toolCalls = m.toolCalls.map { StoredToolCall(id: $0.id, name: $0.name, argumentsJSON: $0.argumentsJSON) }
        self.subAgent = m.subAgent.map {
            StoredSubAgent(target: $0.target, question: $0.question, answer: $0.answer, steps: $0.steps)
        }
    }
}

extension UIMessage {
    init(stored s: StoredMessage) {
        self.init(role: ChatRole(persisted: s.role),
                  reasoning: s.reasoning,
                  text: s.text,
                  toolStreamText: "",
                  toolCalls: s.toolCalls.map { ToolCall(id: $0.id, name: $0.name, argumentsJSON: $0.argumentsJSON) },
                  attachments: s.attachments,
                  subAgent: s.subAgent.map {
                      InferenceService.SubAgentRun(target: $0.target, question: $0.question,
                                                   answer: $0.answer, steps: $0.steps)
                  })
    }
}

extension ChatStore {
    /// Rebuild engine turns from the visible transcript to re-prime a reopened chat.
    /// Attachments (one-shot context) are not restored; tool results are re-fed by
    /// their displayed content so the model keeps the thread.
    static func chatTurns(from messages: [UIMessage]) -> [ChatTurn] {
        var turns: [ChatTurn] = []
        for m in messages {
            switch m.role {
            case .user:
                turns.append(.user(m.text))
            case .assistant:
                if m.text.isEmpty && m.toolCalls.isEmpty { continue }
                turns.append(.assistant(text: m.text, toolCalls: m.toolCalls))
            case .tool:
                let content = m.subAgent?.answer ?? m.text
                turns.append(.toolResult(callId: m.toolCalls.first?.id ?? "", name: "", content: content))
            case .system:
                continue
            }
        }
        return turns
    }
}
