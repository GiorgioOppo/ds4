import DS4Kit
import Foundation

// Phase 0 smoke test for the Swift -> C engine bridge.
//
// Usage:
//   swift run ds4gui-smoke [MODEL.gguf] [--prompt "..."] [--metal-dir DIR]
//
// Validates that Swift can: link the engine static lib, point Metal at the
// bundled kernel sources, open a real GGUF, and drive one greedy generation.
// With no model present it still proves the build/link wiring and exits cleanly.

func arg(_ name: String) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: name), i + 1 < a.count else { return nil }
    return a[i + 1]
}

// Resolve the engine's working files relative to the parent ds4 project.
let projectRoot = ".."
// Metal kernels are vendored inside DS4-gui (run from the package root).
let metalDir = arg("--metal-dir") ?? "metal"
let prompt = arg("--prompt") ?? "Salutami in una frase."

// Pick a model: explicit positional arg, else common defaults.
let positional = CommandLine.arguments.dropFirst().first { !$0.hasPrefix("--") }
let modelCandidates = [positional, "\(projectRoot)/ds4flash.gguf"].compactMap { $0 }
let fm = FileManager.default
let modelPath = modelCandidates.first { fm.fileExists(atPath: $0) }

print("== ds4gui-smoke ==")
print("metal sources : \(metalDir)  (exists: \(fm.fileExists(atPath: metalDir)))")

guard let modelPath else {
    print("""
    No complete GGUF found. Build/link wiring is OK — the engine library is \
    linked and the bridge compiled.
    Provide a model to run a real generation, e.g.:
      swift run ds4gui-smoke ../ds4flash.gguf
    """)
    exit(0)
}

print("model         : \(modelPath)")
print("loading engine (this maps the GGUF and compiles Metal kernels)...")

do {
    let engine = try DS4Engine(modelPath: modelPath, backend: .metal, metalSourceDir: metalDir)
    print("model name    : \(engine.modelName)")
    print("layers        : \(engine.layerCount)")
    print("vocab         : \(engine.vocabSize)")
    print("routed quant  : \(engine.routedQuantBits)-bit")

    let session = try ChatSession(engine: engine, contextSize: 4096)
    print("context       : \(session.contextSize)")
    print("\n--- prompt ---\n\(prompt)\n--- reply ---")

    let n = try session.generateGreedy(system: nil,
                                        prompt: prompt,
                                        thinkMode: .none,
                                        maxTokens: 128) { text in
        FileHandle.standardOutput.write(Data(text.utf8))
    }
    print("\n--- done (\(n) tokens) ---")
} catch {
    print("ERROR: \(error)")
    exit(1)
}
