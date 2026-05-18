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
}

/// Helper for code paths that need to read defaults without going
/// through SwiftUI property wrappers.
enum AppSettings {
    static var loadStrategy: String {
        UserDefaults.standard.string(forKey: AppSettingsKey.loadStrategy) ?? "auto"
    }
    static var forceLoad: Bool {
        UserDefaults.standard.bool(forKey: AppSettingsKey.forceLoad)
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
