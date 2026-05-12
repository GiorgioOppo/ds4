import SwiftUI

/// `@AppStorage`-backed user preferences. Only the keys needed by
/// commits 2 land here; sampler + advanced settings get added in
/// later commits.
enum AppSettingsKey {
    static let loadStrategy = "deepseek.loadStrategy"      // "auto" | "preload" | "mmap"
    static let forceLoad    = "deepseek.forceLoad"         // Bool
    static let lastModelDir = "deepseek.lastModelDir"      // bookmark path
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
    }
}
