import Foundation
import DeepSeekKit

/// Wraps `DeepSeekKit.Transformer` so the UI can drive it from
/// SwiftUI Tasks without fighting actor isolation. `Transformer` and
/// `Tokenizer` are non-Sendable (mutable KV caches, ref types) so we
/// guard them behind a dedicated serial queue and mark the whole
/// class `@unchecked Sendable` — every property access happens on
/// `q`, and the `async` entry points bridge to it.
///
/// This commit lands `loadModel(...)`. The streaming `generate(...)`
/// API arrives in commit 3.
final class InferenceService: @unchecked Sendable {
    private var transformer: Transformer?
    private var tokenizer: Tokenizer?
    private(set) var loadedConfig: ModelConfig?
    private(set) var loadedModelDir: URL?

    private let q = DispatchQueue(label: "deepseek.inference", qos: .userInitiated)

    init() {}

    /// Probe the filesystem + pre-flight, surface the resulting
    /// `LoadPlan` to the UI via `onPlan`, then load. Returns the
    /// (possibly auto-inferred) `ModelConfig` on success.
    /// Errors propagate verbatim — `LoadStrategyError` conforms to
    /// `LocalizedError` so `error.localizedDescription` carries the
    /// rich text the UI prints.
    func loadModel(at url: URL,
                    strategyOverride: String?,
                    forceLoad: Bool,
                    onPlan: @escaping @Sendable (LoadPlan) -> Void
    ) async throws -> ModelConfig {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<ModelConfig, Error>) in
            q.async {
                do {
                    // Pre-flight first so the UI can render the seven
                    // diagnostic fields before the slow mmap/preload
                    // phase begins.
                    let plan = try LoadPlan.decide(modelDir: url,
                                                    override: strategyOverride,
                                                    forceLoad: forceLoad)
                    onPlan(plan)

                    // Tokenizer (cheap, ~30 ms even for 130k-vocab BPE).
                    let tokURL = url.appendingPathComponent("tokenizer.json")
                    let tok = try TokenizerLoader.load(from: tokURL)

                    // Config: prefer on-disk if present, else defaults
                    // (Transformer.load will then call .inferred()
                    // and patch from the actual tensor shapes).
                    let configURL = url.appendingPathComponent("config.json")
                    let cfg: ModelConfig
                    if FileManager.default.fileExists(atPath: configURL.path) {
                        cfg = try ModelConfig.load(from: configURL)
                    } else {
                        cfg = ModelConfig()
                    }

                    let model = try Transformer.load(
                        config: cfg, from: url,
                        strategyOverride: strategyOverride,
                        forceLoad: forceLoad)

                    self.transformer = model
                    self.tokenizer = tok
                    self.loadedConfig = cfg
                    self.loadedModelDir = url
                    cont.resume(returning: cfg)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }
}
