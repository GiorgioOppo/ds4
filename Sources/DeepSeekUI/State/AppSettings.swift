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
