import SwiftUI

/// Single source of truth for WHICH model the app uses and HOW (local engine or
/// distributed cluster). Configured once in the Impostazioni tab; every screen
/// (chat, server, benchmark, diagnostics, worker) inherits these values instead
/// of owning its own model picker.
@MainActor
@Observable
final class AppSettings {
    enum EngineMode: String, CaseIterable, Identifiable {
        case local = "Locale"
        case distributed = "Distribuito"
        var id: String { rawValue }
    }

    var modelPath: String = UserDefaults.standard.string(forKey: "DS4ModelPath")
        ?? AppEnvironment.defaultModelPath {
        didSet { UserDefaults.standard.set(modelPath, forKey: "DS4ModelPath") }
    }

    // Default context window. NOTE: 1M is intentional but heavy — the KV caches +
    // scratch scale with it (≈tens of GB at 1M even with the raw-KV ring on; ≈88 GB
    // raw alone with the ring OFF). Lower it in Impostazioni on small-RAM machines.
    var contextSize: Int = UserDefaults.standard.object(forKey: "DS4ContextSize") as? Int ?? 1_000_000 {
        didSet { UserDefaults.standard.set(contextSize, forKey: "DS4ContextSize") }
    }

    var mode: EngineMode = EngineMode(rawValue: UserDefaults.standard.string(forKey: "DS4EngineMode") ?? "")
        ?? .local {
        didSet { UserDefaults.standard.set(mode.rawValue, forKey: "DS4EngineMode") }
    }

    var modelName: String { (modelPath as NSString).lastPathComponent }
}
