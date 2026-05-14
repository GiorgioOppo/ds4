import Foundation
import DeepSeekKit

/// One event in a streaming generation: an incremental token text,
/// a finalized parsed Message (Reasoning extracted, tool calls
/// decoded), or a status note used for progress logging.
enum GenerationEvent: Sendable {
    case token(String)
    case done(final: Message)
    case status(String)
    /// Emitted right before the prefill forward pass starts. The UI uses
    /// it to swap the bubble into a "prefilling" indicator (no text
    /// streamed yet because no tokens have been sampled).
    case prefillStart(promptTokens: Int)
    /// Emitted once the prefill forward completes. `tokPerMin` is the
    /// throughput of the prefill phase alone (prompt tokens / elapsed).
    case prefillDone(promptTokens: Int, elapsed: TimeInterval, tokPerMin: Double)
    /// Periodic + final throughput sample during the decode loop.
    /// `tokPerMin` is the running rate; emit every ~0.5 s and once at
    /// the end so the UI can show a live ticker.
    case generationProgress(generated: Int, elapsed: TimeInterval, tokPerMin: Double)
}

/// Wraps `DeepSeekKit.Transformer` so the UI can drive it from
/// SwiftUI Tasks without fighting actor isolation. `Transformer` and
/// `Tokenizer` are non-Sendable (mutable KV caches, ref types) so we
/// guard them behind a dedicated serial queue and mark the whole
/// class `@unchecked Sendable` — every property access happens on
/// `q`, and the `async` entry points bridge to it.
final class InferenceService: @unchecked Sendable {
    private var transformer: Transformer?
    private var tokenizer: Tokenizer?
    private(set) var loadedConfig: ModelConfig?
    private(set) var loadedModelDir: URL?

    private let q = DispatchQueue(label: "deepseek.inference", qos: .userInitiated)

    /// Set by `cancelCurrent()`; the generate loop checks this between
    /// tokens and exits early. We can't preempt a mid-token forward
    /// (Metal commands are already in flight), so cancellation always
    /// finishes the current token before stopping.
    private var cancelFlag = false
    private let cancelLock = NSLock()

    init() {}

    func cancelCurrent() {
        cancelLock.lock(); defer { cancelLock.unlock() }
        cancelFlag = true
    }

    private func resetCancelFlag() {
        cancelLock.lock(); defer { cancelLock.unlock() }
        cancelFlag = false
    }

    private func isCancelled() -> Bool {
        cancelLock.lock(); defer { cancelLock.unlock() }
        return cancelFlag
    }

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
                    var cfg: ModelConfig
                    if FileManager.default.fileExists(atPath: configURL.path) {
                        cfg = try ModelConfig.load(from: configURL)
                    } else {
                        cfg = ModelConfig()
                    }

                    // Apply user overrides from
                    //   ~/Library/Application Support/<app>/config-overrides.json
                    // (same file the ModelConfigSettingsTab writes to).
                    // .inferred() will still patch architectural fields from
                    // the checkpoint, so only the truly user-tunable fields
                    // — maxBatchSize and maxSeqLen — actually need merging
                    // here; the rest survive only if not contradicted by the
                    // tensors on disk.
                    if let overrideURL = try? PersistencePaths.conversationsDir()
                        .deletingLastPathComponent()
                        .appendingPathComponent("config-overrides.json"),
                       FileManager.default.fileExists(atPath: overrideURL.path),
                       let data = try? Data(contentsOf: overrideURL),
                       let ov = try? JSONDecoder().decode(ModelConfig.self, from: data) {
                        cfg.maxBatchSize = ov.maxBatchSize
                        cfg.maxSeqLen    = ov.maxSeqLen
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

    /// Drive one generation round: encode the conversation, prefill,
    /// then decode one token at a time emitting `.token(text)` events
    /// until eos or `maxTokens`. Finishes with `.done(final:)`, the
    /// fully-parsed `Message` (reasoning extracted, tool calls
    /// decoded) so the UI can re-render through Markdown / disclosure
    /// once streaming completes.
    ///
    /// `Transformer`'s KV cache is reset between conversations via
    /// `releaseCache()` so two unrelated chats can share the same
    /// loaded weights without cross-talk.
    func generate(history: [Message],
                   mode: ThinkingMode,
                   options: SamplingOptions,
                   maxTokens: Int
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        return AsyncThrowingStream { continuation in
            q.async { [weak self] in
                guard let self else { continuation.finish(); return }
                guard let model = self.transformer,
                      let tok = self.tokenizer else {
                    continuation.finish(throwing: NSError(
                        domain: "InferenceService", code: 1, userInfo: [
                            NSLocalizedDescriptionKey:
                                "No model loaded yet — pick a folder first."
                        ]))
                    return
                }
                self.resetCancelFlag()
                model.releaseCache()

                // 1. Encode prompt.
                let prompt = EncodingDSV4.encodeMessages(history, mode: mode)
                let promptIds = tok.encode(prompt)
                if promptIds.isEmpty {
                    continuation.finish(throwing: NSError(
                        domain: "InferenceService", code: 2, userInfo: [
                            NSLocalizedDescriptionKey:
                                "Tokenizer produced 0 tokens for the prompt."
                        ]))
                    return
                }
                // 2. Prefill. Wrapped in prefillStart / prefillDone so
                //    the UI can render a dedicated indicator while the
                //    forward runs (no tokens are streamed yet — prefill
                //    on a small-RAM Mac can take tens of seconds while
                //    weights page through the streaming slot).
                continuation.yield(.prefillStart(promptTokens: promptIds.count))
                let prefillStart = Date()
                var logits = model.forward(inputIds: [promptIds], startPos: 0)
                let prefillElapsed = Date().timeIntervalSince(prefillStart)
                let prefillTPM = prefillElapsed > 0
                    ? Double(promptIds.count) / prefillElapsed * 60
                    : 0
                continuation.yield(.prefillDone(
                    promptTokens: promptIds.count,
                    elapsed: prefillElapsed,
                    tokPerMin: prefillTPM))

                var opts = options
                // V4-Flash chat: stop on either `<｜end▁of▁sentence｜>`
                // (eosId, end of conversation) or `<|EOT|>` (end of
                // assistant turn). Checking only eosId lets EOT slip
                // through and the model loops on filler tokens.
                let stops = tok.stopTokenIds

                // 3. Decode loop. Emit a generationProgress event roughly
                //    every 500 ms so the UI ticker updates without
                //    flooding the actor mailbox.
                var generated: [Int] = []
                var generatedText = ""
                let decodeStart = Date()
                var lastSample = decodeStart
                for step in 0..<maxTokens {
                    if self.isCancelled() { break }

                    let nextId = Sampler.sample(logits,
                                                  history: generated,
                                                  options: &opts)
                    if stops.contains(nextId) { break }
                    generated.append(nextId)

                    let piece = tok.decode([nextId])
                    generatedText += piece
                    continuation.yield(.token(piece))

                    let now = Date()
                    if now.timeIntervalSince(lastSample) >= 0.5 {
                        let elapsedSoFar = now.timeIntervalSince(decodeStart)
                        let tpm = elapsedSoFar > 0
                            ? Double(generated.count) / elapsedSoFar * 60
                            : 0
                        continuation.yield(.generationProgress(
                            generated: generated.count,
                            elapsed: elapsedSoFar,
                            tokPerMin: tpm))
                        lastSample = now
                    }

                    if step == maxTokens - 1 { break }
                    let startPos = promptIds.count + step
                    logits = model.forward(inputIds: [[nextId]],
                                            startPos: startPos)
                }

                let elapsed = Date().timeIntervalSince(decodeStart)
                let genTPM = elapsed > 0
                    ? Double(generated.count) / elapsed * 60
                    : 0
                continuation.yield(.generationProgress(
                    generated: generated.count,
                    elapsed: elapsed,
                    tokPerMin: genTPM))

                // 4. Finalize: re-parse through EncodingDSV4 so any
                //    `<think>` block is split off into reasoningContent
                //    and tool_calls into structured ToolCall objects.
                let final = EncodingDSV4.parseCompletion(generatedText,
                                                           mode: mode)
                continuation.yield(.done(final: final))
                continuation.finish()
            }
        }
    }
}
