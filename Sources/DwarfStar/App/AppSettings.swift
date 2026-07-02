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

    // Default context window is sized to the machine's RAM, same tiers as the
    // "Configura per la tua RAM" preset (4096 on ≤16 GB … up to 32768 on big
    // machines). Provisioning the full 1M everywhere made `maxKeys` 1M (vs the
    // DS4Demo's 4096): the KV caches + scratch scale with it (≈tens of GB at 1M
    // even with the raw-KV ring on; ≈88 GB raw alone with the ring OFF), which
    // starved the expert page cache and made the GUI several times slower than the
    // demo on the same machine. The Stepper still goes up to 1M for anyone who
    // knowingly wants a longer window (and accepts the memory cost).
    var contextSize: Int = UserDefaults.standard.object(forKey: "DS4ContextSize") as? Int
        ?? HardwarePresets.forRAM(MemoryInfo.physicalBytes).contextSize {
        didSet { UserDefaults.standard.set(contextSize, forKey: "DS4ContextSize") }
    }

    var mode: EngineMode = EngineMode(rawValue: UserDefaults.standard.string(forKey: "DS4EngineMode") ?? "")
        ?? .local {
        didSet { UserDefaults.standard.set(mode.rawValue, forKey: "DS4EngineMode") }
    }

    var modelName: String { (modelPath as NSString).lastPathComponent }
}
