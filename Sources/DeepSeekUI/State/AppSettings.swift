import SwiftUI

/// `@AppStorage`-backed user preferences. Only the keys needed by
/// commits 2 land here; sampler + advanced settings get added in
/// later commits.
enum AppSettingsKey {
    static let loadStrategy = "deepseek.loadStrategy"      // "auto" | "preload" | "mmap"
    static let forceLoad    = "deepseek.forceLoad"         // Bool
    static let lastModelDir = "deepseek.lastModelDir"      // path
    static let recentModelDirs = "deepseek.recentModelDirs" // JSON-encoded [String]
    /// Filesystem path to the `converter` CLI binary, used by
    /// ConvertSheet. Empty → ConverterRunner auto-discovers.
    static let converterBinaryPath = "deepseek.converterBinaryPath"

    /// Default `ProjectContextMode` per nuovi progetti (e per i
    /// progetti che non hanno override). Valori:
    /// "pathsOnly" | "indexedContent". Default: "pathsOnly".
    static let projectContextMode = "deepseek.projectContextMode"

    /// Cap globale sul numero di file inclusi nell'albero
    /// dell'inventario per i progetti in modalità `pathsOnly`.
    /// Default: 500. Override per-progetto in
    /// `Project.maxInventoryFiles`.
    static let projectInventoryMaxFiles = "deepseek.projectInventoryMaxFiles"

    /// Preferenza globale per il W8A8 path: quando true (e i pesi
    /// sono int8), `Linear` quantizza l'attivazione a int8 e usa il
    /// kernel `gemm_int8_w8a8_*` invece del W8A16 standard. Letta
    /// dal model loader al momento della costruzione dei Linear —
    /// non ha effetto a runtime sui modelli già caricati. Default
    /// false. Vedi `Sources/DeepSeekKit/Layers/Linear.swift`.
    static let useW8A8Activations = "deepseek.useW8A8Activations"

    /// Pre-fault tutte le pagine dei weight shards al model load.
    /// Riduce il time-to-first-token (no page-in latency staccato
    /// sul primo prefill) ma alloca subito tutto in RAM, quindi
    /// utile solo se il modello sta in memoria. Skip automatico se
    /// model size > RAM × 1.5. Default false. Vedi
    /// `WeightLoader.warmupAllShards`.
    static let warmupOnLoad = "deepseek.warmupOnLoad"

    /// MoE lazy-expert load (streaming strategy only). When true,
    /// `StreamingPool.ensureLayer(K)` only preads the non-expert
    /// tensors of layer K (attention, norms, gate, shared expert);
    /// the per-expert weights are loaded on demand after the gate
    /// runs, via `MoEFFN.ensureExpertsHook` → `ensureExperts`.
    /// On checkpoints with 17×+ memory oversubscription the per-token
    /// I/O drops from ~full-shard to ~core + (topK/nExperts) × experts
    /// — a multi-× reduction on V4-Pro. Default `true`. Set to false
    /// only if you're debugging a regression suspected of the lazy
    /// path. The `DEEPSEEK_LAZY_EXPERT` env var, if present at app
    /// launch, overrides this setting.
    static let lazyExpertLoad = "deepseek.lazyExpertLoad"

    /// When true, the per-token active-expert count is taken from
    /// `activeExpertsPerToken` instead of the engine's built-in
    /// default. Bridged into `ModelConfig.activeExpertsOverride`.
    static let overrideActiveExperts = "deepseek.overrideActiveExperts"
    /// Per-token active-expert count used when `overrideActiveExperts`
    /// is on. Applied on the next model load; clamped engine-side to
    /// [1, 16].
    static let activeExpertsPerToken = "deepseek.activeExpertsPerToken"

    /// EXPERIMENTAL. Rilassa il match strict-prefix della KV cache
    /// in-memory a common-prefix, supportando il caso "user edita
    /// il proprio ultimo messaggio" senza full reset. Rischio basso
    /// ma non-zero — la KV cache fisica oltre il common-prefix
    /// viene sovrascritta dal forward pass senza esplicito rewind.
    /// Default false. Vedi `InferenceService.enableCommonPrefixRewind`.
    static let commonPrefixRewind = "deepseek.commonPrefixRewind"

    /// EXPERIMENTAL. Usa `MAP_SHARED` invece di `MAP_PRIVATE` per
    /// l'mmap dei weight shards in modalità `.mmap`. Su Apple Silicon
    /// (APFS + unified memory) dovrebbe permettere zero-copy
    /// MTLBuffer wrap (ds4 lo usa così). Fallback automatico a
    /// `MAP_PRIVATE` se mmap fallisce. Default false per safety
    /// contro Darwin VM accounting issues su setup esotici.
    static let useMapSharedWeights = "deepseek.useMapSharedWeights"

    /// Cross-restart KV cache persistente. Quando attivo, l'app
    /// salva lo snapshot della KV cache su disco a 4 trigger
    /// (cold/continued/evict/shutdown) tramite `KVCacheLifecycle`,
    /// e al rientro in una conversation prova a ripristinare lo
    /// stato dal file `.kvcache` invece di fare cold prefill.
    /// Default false. Vedi `InferenceService.kvLifecycle`.
    static let crossRestartKVCache = "deepseek.crossRestartKVCache"

    /// Compression del KV cache snapshot sul disco. "f32" (default,
    /// lossless), "f16" (2× compression), "bf16" (2× compression,
    /// range completo F32). Letta solo se `crossRestartKVCache`
    /// attivo.
    static let kvCacheCompression = "deepseek.kvCacheCompression"

    /// Precompute the deterministic tool prefix (BOS + tools block)
    /// of the first chat prompt once at model load — tokens + a
    /// KV-cache snapshot — and reuse it for every new chat's first
    /// turn so the prefill skips re-tokenising and re-prefilling that
    /// block. Persisted under `tool-prefix/` in Application Support,
    /// keyed on the model + tools-block hash. Default ON (a missing
    /// key reads as true). Vedi `InferenceService.precomputeToolPrefix`.
    static let precomputedToolPrefix = "deepseek.precomputedToolPrefix"

    /// Quando attivo, sul primo turn di una conversazione (cold
    /// prefill, KV cache vuota) la UI mostra fra il messaggio
    /// dell'utente e la risposta dell'assistente un blocco
    /// collassabile in grigio con il testo del prompt completo
    /// (system + tools + history + user) che il modello sta per
    /// vedere — emesso in chunk dal tokenizer mentre il prefill è
    /// in corso. Solo i turn cold (no cache hit) lo mostrano; sui
    /// turn incrementali, dove la prefill è solo il delta del
    /// nuovo user message, il trace è omesso per non rumoreggiare.
    /// Default true.
    static let showPrefillTrace = "deepseek.showPrefillTrace"

    /// Local OpenAI-compatible HTTP server (TODO §10.1 / T1). Flag is
    /// off by default — the server only runs when the user explicitly
    /// flips the Settings → Server toggle. Port defaults to 8080,
    /// bind address to `127.0.0.1`.
    static let serverEnabled = "deepseek.server.enabled"
    static let serverPort = "deepseek.server.port"
    static let serverBindAddress = "deepseek.server.bindAddress"

    /// Web-search backend choice (TODO §8 follow-up). Values:
    /// "duckduckgo" (default, no key needed, fragile scraper),
    /// "tavily", "brave", "serper" (each requires the matching
    /// Keychain entry under `KeychainAccount.*APIKey`). When the
    /// selected backend's key is missing, `NativeToolHost` falls
    /// back to the DuckDuckGo provider with a stderr note.
    static let webSearchProvider = "deepseek.webSearch.provider"

    /// TODO §9 sandbox toggle. Opt-in `sandbox-exec` wrapper for
    /// `ShellTool` (DeepSeekIntegrations/Sandbox/Sandbox.swift).
    /// Default false because the bundled profile is strict; turn
    /// it on once you've tuned `<root>/sandbox/default.sb` for
    /// your workflow.
    static let useShellSandbox = "deepseek.shell.useSandbox"

    /// Register the 50 Unix-style tools (ls, head, tail, sort, sed,
    /// awk, find, …) under `DeepSeekTools/Tools/Unix/` so the model
    /// can call them directly instead of falling through to the
    /// .dangerous `shell` tool. Default true. Unset (missing key)
    /// is read as true so a fresh install gets the full toolbox.
    /// Cap on how many user-led turns a remote chat sends back to
    /// the provider. 0 = unlimited (current behaviour). When > 0
    /// the request keeps system messages + the last N user
    /// turns and everything that follows each — older turns are
    /// dropped from the request body, NOT from the on-screen
    /// transcript.
    static let remoteHistoryTurnsCap = "deepseek.remote.historyTurnsCap"
    static let enableUnixTools = "deepseek.tools.unix.enabled"

    /// Register the 30 Xcode / Apple-platform tools (xcodebuild_*,
    /// swift_*, simctl_*, devicectl_*, codesign_*, …) under
    /// `DeepSeekTools/Tools/Xcode/` for macOS / iOS / visionOS app
    /// development. Default true on macOS. Tools that need a binary
    /// missing on the host (jq, devicectl on pre-Xcode-15 setups)
    /// return `not_found` at call time.
    static let enableXcodeTools = "deepseek.tools.xcode.enabled"
}

/// Helper for code paths that need to read defaults without going
/// through SwiftUI property wrappers.
enum AppSettings {
    static var loadStrategy: String {
        UserDefaults.standard.string(forKey: AppSettingsKey.loadStrategy) ?? "auto"
    }

    /// Remote-chat sliding-window cap. 0 means "send the full
    /// history", any positive value caps the request to system
    /// messages + the last N user turns and everything that
    /// follows each. The on-screen transcript stays untouched —
    /// this only affects what's serialised into `messages` for
    /// the next API call.
    static var remoteHistoryTurnsCap: Int {
        let raw = UserDefaults.standard.integer(
            forKey: AppSettingsKey.remoteHistoryTurnsCap)
        return max(0, raw)
    }
    static var forceLoad: Bool {
        UserDefaults.standard.bool(forKey: AppSettingsKey.forceLoad)
    }
    /// `bool(forKey:)` always defaults to `false`; this getter keeps
    /// that semantics for the lazy-expert toggle (default OFF) until
    /// the correctness regression on V4-Pro checkpoints is tracked
    /// down — the lazy path emits degenerate logits ("<|begin_of_
    /// sentence|>" loop) on the user's 148 GB checkpoint while the
    /// legacy full-shard pread produces correct text. Re-enable via
    /// the Loading settings tab once we have a fix.
    static var lazyExpertLoad: Bool {
        UserDefaults.standard.bool(forKey: AppSettingsKey.lazyExpertLoad)
    }

    /// The per-token active-expert override the UI has configured:
    /// `activeExpertsPerToken` when `overrideActiveExperts` is on,
    /// else `nil` (engine uses its built-in default). Bridged into
    /// `ModelConfig.activeExpertsOverride` by `DeepSeekUIApp`.
    static var activeExpertsOverride: Int? {
        guard UserDefaults.standard.bool(
            forKey: AppSettingsKey.overrideActiveExperts) else { return nil }
        let n = UserDefaults.standard.integer(
            forKey: AppSettingsKey.activeExpertsPerToken)
        return n > 0 ? n : nil
    }

    static var lastModelDir: String? {
        UserDefaults.standard.string(forKey: AppSettingsKey.lastModelDir)
    }
    static func setLastModelDir(_ path: String) {
        UserDefaults.standard.set(path, forKey: AppSettingsKey.lastModelDir)
        pushRecentDir(path)
    }

    /// MRU list of model directories, capped at 8. JSON-encoded into
    /// a single UserDefaults key.
    static func recentDirs() -> [String] {
        guard let data = UserDefaults.standard.data(forKey: AppSettingsKey.recentModelDirs),
              let arr = try? JSONDecoder().decode([String].self, from: data) else {
            return []
        }
        return arr
    }

    static func pushRecentDir(_ path: String) {
        var arr = recentDirs().filter { $0 != path }
        arr.insert(path, at: 0)
        if arr.count > 8 { arr = Array(arr.prefix(8)) }
        if let data = try? JSONEncoder().encode(arr) {
            UserDefaults.standard.set(data, forKey: AppSettingsKey.recentModelDirs)
        }
    }

    static func forgetRecentDir(_ path: String) {
        let arr = recentDirs().filter { $0 != path }
        if let data = try? JSONEncoder().encode(arr) {
            UserDefaults.standard.set(data, forKey: AppSettingsKey.recentModelDirs)
        }
    }

    // MARK: - Project context

    /// Default mode per i progetti che non hanno override. Nuovi
    /// progetti partono da qui. Default: `.pathsOnly`.
    static var projectContextMode: ProjectContextMode {
        get {
            let raw = UserDefaults.standard.string(
                forKey: AppSettingsKey.projectContextMode)
            return raw.flatMap(ProjectContextMode.init(rawValue:))
                ?? .pathsOnly
        }
        set {
            UserDefaults.standard.set(
                newValue.rawValue,
                forKey: AppSettingsKey.projectContextMode)
        }
    }

    /// Cap globale (override per-progetto via
    /// `Project.maxInventoryFiles`). 0 → usa default
    /// `ProjectInventoryBuilder.defaultMaxFiles`.
    static var projectInventoryMaxFiles: Int {
        get {
            let v = UserDefaults.standard.integer(
                forKey: AppSettingsKey.projectInventoryMaxFiles)
            return v > 0 ? v : ProjectInventoryBuilder.defaultMaxFiles
        }
        set {
            UserDefaults.standard.set(
                max(0, newValue),
                forKey: AppSettingsKey.projectInventoryMaxFiles)
        }
    }
}
