import Foundation

/// Resolves default paths so the same code works both when run from SwiftPM
/// during development and when packaged into a .app bundle.
///
/// In dev mode we use ABSOLUTE paths into the upstream ds4 project so the app
/// works regardless of the working directory Xcode runs it from. In a bundle
/// we ship `metal/` (kernel sources, required), and optionally the `ds4*`
/// helper binaries under `bin/`, `download_model.sh`, and `speed-bench/`.
enum AppEnvironment {
    /// Root of the upstream ds4 project on this machine. Hardcoded for the
    /// user's machine; override with the DS4_ROOT environment variable.
    static let projectRoot: String = {
        if let env = ProcessInfo.processInfo.environment["DS4_ROOT"], !env.isEmpty {
            return env
        }
        return "/Users/oppog/Downloads/ds4-main"
    }()

    /// Root of the DS4-gui project itself (where the bundled metal/ kernels live).
    static let guiRoot: String = (projectRoot as NSString).appendingPathComponent("DS4-gui")

    static var isBundled: Bool {
        Bundle.main.bundleURL.pathExtension == "app"
    }

    /// Directory used as the working dir / project root for subprocesses and
    /// for scanning models. The upstream project root in dev; Resources in a bundle.
    static var resourceDir: String {
        if isBundled, let r = Bundle.main.resourceURL?.path { return r }
        return projectRoot
    }

    /// Directory containing the metal/*.metal kernel sources. These now live
    /// inside the DS4-gui project (copied from the upstream engine), so the GUI
    /// no longer depends on ../metal. A bundle ships them under Resources/metal.
    static var metalDir: String {
        if isBundled, let r = Bundle.main.resourceURL?.appendingPathComponent("metal").path,
           FileManager.default.fileExists(atPath: r) {
            return r
        }
        return (guiRoot as NSString).appendingPathComponent("metal")
    }

    /// Directory containing the ds4 helper binaries.
    static var binDir: String {
        if isBundled, let r = Bundle.main.resourceURL?.appendingPathComponent("bin").path,
           FileManager.default.fileExists(atPath: r) {
            return r
        }
        return projectRoot
    }

    /// Default GGUF path. In dev we point at the user's chosen full-precision
    /// Flash GGUF; in a bundle we have no bundled model, so the user selects one.
    static var defaultModelPath: String {
        if isBundled { return "" }
        return (projectRoot as NSString).appendingPathComponent(
            "gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf")
    }

    static func binary(_ name: String) -> String {
        (binDir as NSString).appendingPathComponent(name)
    }
}

/// A recommended starting configuration for a RAM tier. The pure-Swift engine
/// always runs the SSD-streaming path (mmap no-copy + per-token expert gather),
/// so the preset only tunes what is actually configurable: context size + quant.
struct HardwarePreset {
    let contextSize: Int
    let prefersTwoBit: Bool    // recommend the 2-bit quant
    let summary: String
}

enum HardwarePresets {
    static let gib = 1_073_741_824.0

    /// Map detected RAM to a conservative preset. Lower tiers trade quality and
    /// speed for any chance of fitting; 16 GB is below the project's 64 GB floor.
    static func forRAM(_ ramBytes: UInt64) -> HardwarePreset {
        let gb = Double(ramBytes) / gib
        switch gb {
        case ..<24:
            return HardwarePreset(contextSize: 4096, prefersTwoBit: true,
                summary: "≈\(Int(gb.rounded())) GB: sotto il minimo del progetto (64 GB). Quant 2-bit, contesto 4096; i pesi vengono streamati da SSD (lento ma funziona).")
        case ..<80:
            return HardwarePreset(contextSize: 8192, prefersTwoBit: true,
                summary: "≈\(Int(gb.rounded())) GB: quant 2-bit, contesto 8192; gran parte dei pesi resta in page cache.")
        case ..<200:
            return HardwarePreset(contextSize: 32768, prefersTwoBit: true,
                summary: "≈\(Int(gb.rounded())) GB: quant 2-bit interamente in RAM, contesto 32768.")
        default:
            return HardwarePreset(contextSize: 32768, prefersTwoBit: false,
                summary: "≈\(Int(gb.rounded())) GB: anche la quant Q4 entra in RAM, contesto 32768.")
        }
    }

    /// Heuristic: does this GGUF filename look like a 2-bit (IQ2/Q2) quant?
    static func isTwoBit(_ name: String) -> Bool {
        let n = name.lowercased()
        return n.contains("iq2") || n.contains("q2k") || n.contains("-q2") || n.contains("q2-")
    }
}

/// Memory feasibility helper. The engine/OS will hard-fail (prefill error) or be
/// killed (OOM) if the configuration cannot fit; this lets the UI warn first.
enum MemoryInfo {
    static var physicalBytes: UInt64 { ProcessInfo.processInfo.physicalMemory }

    static func fileSize(_ path: String) -> UInt64? {
        guard !path.isEmpty else { return nil }
        let resolved = URL(fileURLWithPath: path).resolvingSymlinksInPath().path
        guard let attr = try? FileManager.default.attributesOfItem(atPath: resolved),
              let n = attr[.size] as? NSNumber else { return nil }
        return n.uint64Value
    }

    static func gib(_ bytes: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }

    /// Returns a warning string when the model is unlikely to fit, else nil.
    /// (The engine always streams from SSD, so the only hard limit is that the
    /// non-routed weights + KV cache must fit in RAM.)
    static func loadWarning(modelPath: String) -> String? {
        let ram = physicalBytes
        guard let size = fileSize(modelPath), size > 0 else { return nil }
        if size > ram * 4 {
            return "Le parti non-routed del modello e la KV cache devono stare in RAM. Con \(gib(ram)) di RAM e un modello da \(gib(size)), il rischio di esaurire la memoria (crash) è alto: usa un contesto ridotto o la quant a 2 bit."
        }
        if size > ram {
            return "Il modello (\(gib(size))) è più grande della RAM (\(gib(ram))): i pesi verranno streamati da SSD a ogni token — funziona, ma molto lentamente (decine di secondi per token)."
        }
        return nil
    }
}
