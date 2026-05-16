import Foundation
import DeepSeekKit

/// One event in a streaming generation: an incremental token text,
/// a finalized parsed Message (Reasoning extracted, tool calls
/// decoded), or a status note used for progress logging.
enum GenerationEvent: Sendable {
    /// One sampled token. `text` is what the tokenizer decoded for
    /// this single id; `id` is the raw sample, persisted by
    /// ChatStore so a crash mid-generation can be resumed bit-
    /// identically (decoding-then-re-encoding the partial text
    /// isn't round-trip-safe in BPE).
    case token(text: String, id: Int32)
    /// Stream completed. `final` is the parsed assistant Message;
    /// `promptTokens` is the full BPE prompt the model saw (prefix +
    /// delta) and `generatedTokens` are every sampled id including
    /// the trailing eos when present. Together they let `ChatStore`
    /// append to `Conversation.encodedTokens` without re-tokenizing
    /// anything.
    case done(final: Message,
               promptTokens: [Int32],
               generatedTokens: [Int32])
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
    private var _tokenizer: Tokenizer?
    private(set) var loadedConfig: ModelConfig?
    private(set) var loadedModelDir: URL?

    /// Live mirror of what's currently sitting in the model's KV
    /// cache. When the next `generateForConversation` call hands us a
    /// `promptTokens` that begins with `tokens` and runs against the
    /// same `conversationID` / `mode`, we can skip `releaseCache()`
    /// and prefill *only* the trailing delta — turning the per-turn
    /// O(history) BPE+forward into O(|new user turn|) forwards.
    /// Reset whenever we have to call `releaseCache()` (different
    /// conversation, mode change, or model unload).
    private struct CacheImage {
        let conversationID: UUID
        var tokens: [Int32]
        var mode: ThinkingMode
    }
    private var cacheImage: CacheImage?

    /// Snapshot of the active tokenizer for read-only use outside the
    /// generation path (e.g. the document import flow). Reads the
    /// stored reference on the serial queue so it never races a load.
    /// Returns nil until a model has been loaded.
    func currentTokenizer() -> Tokenizer? {
        q.sync { _tokenizer }
    }

    /// Snapshot of the active model directory. The document library
    /// uses it as a fingerprint to detect "different model selected
    /// since import time".
    func currentModelDir() -> URL? {
        q.sync { loadedModelDir }
    }

    /// Encode `text` into Int32 token ids on the inference serial
    /// queue. Returns nil when no model has been loaded yet (and
    /// therefore no tokenizer is available). Used by the document
    /// import flow so BPE encoding doesn't block the main actor and
    /// doesn't have to pass a non-Sendable tokenizer across an
    /// `await`.
    func tokenize(_ text: String) async -> [Int32]? {
        await withCheckedContinuation {
            (cont: CheckedContinuation<[Int32]?, Never>) in
            q.async {
                guard let tok = self._tokenizer else {
                    cont.resume(returning: nil)
                    return
                }
                let ids = tok.encode(text).map(Int32.init)
                cont.resume(returning: ids)
            }
        }
    }

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
                    self._tokenizer = tok
                    self.loadedConfig = cfg
                    self.loadedModelDir = url
                    // A model swap renders every cached KV state
                    // invalid (different weight tensors → different
                    // attention outputs).
                    self.cacheImage = nil
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
    /// Tokenize the full chat history through the V4 template. Used
    /// by `ChatStore` on first turn (or after a mode change) to
    /// produce the canonical `encodedTokens` baseline.
    ///
    /// `toolSchemasJSON` is forwarded to EncodingDSV4 which folds
    /// the tools block into the system message.
    ///
    /// `systemPromptOverride`, when non-nil, is injected as a
    /// `Message(role: .system, content: …)` at the head of the
    /// history (or merges with an existing leading system message
    /// from the transcript) so an Agent's preset prompt shows up
    /// before the user's first turn.
    func tokenizeFullHistory(_ history: [Message],
                              mode: ThinkingMode,
                              toolSchemasJSON: String? = nil,
                              systemPromptOverride: String? = nil) async -> [Int32]? {
        await withCheckedContinuation {
            (cont: CheckedContinuation<[Int32]?, Never>) in
            q.async {
                guard let tok = self._tokenizer else {
                    cont.resume(returning: nil); return
                }
                var effectiveHistory = history
                if let extra = systemPromptOverride,
                   !extra.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                {
                    if let firstIdx = effectiveHistory.firstIndex(where: {
                        $0.role == .system
                    }) {
                        // Prepend the agent prompt to whatever the
                        // transcript already had; keeps both visible
                        // to the model.
                        effectiveHistory[firstIdx].content =
                            extra + "\n\n" + effectiveHistory[firstIdx].content
                    } else {
                        effectiveHistory.insert(
                            Message(role: .system, content: extra),
                            at: 0)
                    }
                }
                let prompt = EncodingDSV4.encodeMessages(
                    effectiveHistory, mode: mode,
                    toolSchemasJSON: toolSchemasJSON)
                let ids = tok.encode(prompt).map(Int32.init)
                cont.resume(returning: ids)
            }
        }
    }

    /// Build the BPE prompt of a fresh chat that has a Project
    /// attached. Emits, in id space (no string concat / re-encode):
    ///
    ///   bos
    ///   system text…
    ///   ⟨begin_of_repo_name⟩ projectName ⟨end_of_repo_name⟩
    ///   for each file:
    ///       ⟨begin_of_file_name⟩ path ⟨end_of_file_name⟩
    ///       ⟨begin_of_file⟩ <file's pre-tokenized ids> ⟨end_of_file⟩
    ///   ⟨User⟩ userText ⟨Assistant⟩ ⟨think_marker⟩
    ///
    /// The per-file token streams are pulled from
    /// `DocumentLibrary.tokens(of:)` so we don't re-BPE multi-MB
    /// source files at chat-time — they were tokenized once when the
    /// project was indexed.
    ///
    /// Returns nil when no model is loaded (no tokenizer to resolve
    /// the special-token ids with). The "first turn" caller falls
    /// back to plain tokenizeFullHistory on nil.
    func tokenizeFirstTurnWithProject(systemText: String,
                                       projectName: String,
                                       files: [(path: String, tokens: [Int32])],
                                       userText: String,
                                       mode: ThinkingMode,
                                       toolSchemasJSON: String? = nil) async -> [Int32]? {
        await withCheckedContinuation {
            (cont: CheckedContinuation<[Int32]?, Never>) in
            q.async {
                guard let tok = self._tokenizer else {
                    cont.resume(returning: nil); return
                }
                // Resolve every special token to its id via a single
                // encode call each. `BPETokenizer.encode` pre-splits
                // on added_tokens, so a string that contains only an
                // added_token returns `[id]`. Any "0 results" (token
                // not in vocab) trips a nil return so the caller
                // falls back to the plain path.
                func id(_ s: String) -> Int32? {
                    let ids = tok.encode(s)
                    guard ids.count == 1 else { return nil }
                    return Int32(ids[0])
                }
                guard let bosId       = id(EncodingDSV4.bosToken),
                      let userId      = id(EncodingDSV4.userToken),
                      let assistantId = id(EncodingDSV4.assistantToken),
                      let beginRepoN  = id(EncodingDSV4.beginOfRepoName),
                      let endRepoN    = id(EncodingDSV4.endOfRepoName),
                      let beginFileN  = id(EncodingDSV4.beginOfFileName),
                      let endFileN    = id(EncodingDSV4.endOfFileName),
                      let beginFile   = id(EncodingDSV4.beginOfFile),
                      let endFile     = id(EncodingDSV4.endOfFile)
                else {
                    cont.resume(returning: nil); return
                }
                let thinkMarker = (mode == .chat)
                    ? EncodingDSV4.thinkClose
                    : EncodingDSV4.thinkOpen

                var out: [Int32] = []
                out.append(bosId)
                // System prefix: same precedence the chat template
                // uses — tools block first (so the model sees the
                // contract before the user's instructions), then the
                // original system text. Keeping the order consistent
                // with EncodingDSV4.injectSystemAdditions avoids
                // surprises when comparing first-turn output to a
                // re-encoded full history later.
                var systemAug = ""
                if let schemas = toolSchemasJSON, !schemas.isEmpty {
                    systemAug += EncodingDSV4.toolsBlock(toolSchemasJSON: schemas)
                }
                systemAug += systemText
                if !systemAug.isEmpty {
                    out.append(contentsOf:
                        tok.encode(systemAug).map(Int32.init))
                }
                out.append(beginRepoN)
                out.append(contentsOf:
                    tok.encode(projectName).map(Int32.init))
                out.append(endRepoN)
                for (path, tokens) in files {
                    out.append(beginFileN)
                    out.append(contentsOf: tok.encode(path).map(Int32.init))
                    out.append(endFileN)
                    out.append(beginFile)
                    out.append(contentsOf: tokens)
                    out.append(endFile)
                }
                out.append(userId)
                out.append(contentsOf: tok.encode(userText).map(Int32.init))
                out.append(assistantId)
                out.append(contentsOf: tok.encode(thinkMarker).map(Int32.init))
                cont.resume(returning: out)
            }
        }
    }

    /// Tokenize the *delta* that splices tool execution results
    /// back onto the cached prefix and re-opens the assistant
    /// turn so the model can keep going:
    ///
    ///   `<eos>` (closes the just-finished assistant turn —
    ///           the decode loop breaks before sampling its eos)
    ///   `<｜tool▁outputs▁begin｜>` …per-output frames…
    ///                          `<｜tool▁outputs▁end｜>`
    ///   `<Assistant><think_marker>` (re-opens for the next turn)
    ///
    /// The per-output frames carry the qualified tool name via
    /// `<｜tool▁sep｜>` so the model can disambiguate when several
    /// tools were called in one turn.
    func tokenizeToolOutputsDelta(callNames: [String],
                                   outputs: [String],
                                   mode: ThinkingMode) async -> [Int32]? {
        await withCheckedContinuation {
            (cont: CheckedContinuation<[Int32]?, Never>) in
            q.async {
                guard let tok = self._tokenizer else {
                    cont.resume(returning: nil); return
                }
                let thinkMarker = (mode == .chat)
                    ? EncodingDSV4.thinkClose
                    : EncodingDSV4.thinkOpen
                let block = EncodingDSV4.encodeToolOutputs(
                    callNames: callNames, outputs: outputs)
                let deltaText = EncodingDSV4.eosToken
                    + block
                    + EncodingDSV4.assistantToken
                    + thinkMarker
                let ids = tok.encode(deltaText).map(Int32.init)
                cont.resume(returning: ids)
            }
        }
    }

    /// Tokenize the *delta* needed to extend a cached prompt with one
    /// more user turn:
    ///   `<eos><User>userContent<Assistant><think_marker>`
    ///
    /// Why `<eos>` is in front: the decode loop breaks the moment a
    /// stop token (eos / EOT) is sampled, *before* feeding it back
    /// into the model. So neither `Conversation.encodedTokens` nor
    /// the live GPU KV cache contains the eos that closes the
    /// previous assistant turn. The delta supplies it so the chat
    /// template is well-formed when re-tokenizing this turn alone.
    ///
    /// `BPETokenizer.encode` pre-splits on special tokens before BPE
    /// merging, so `<eos>` will always emit as a single id regardless
    /// of what precedes/follows it — the concatenation of the
    /// previously-cached prefix and this delta is bit-identical to
    /// what `tokenizeFullHistory` would produce.
    func tokenizeUserTurnDelta(_ userContent: String,
                                mode: ThinkingMode) async -> [Int32]? {
        await withCheckedContinuation {
            (cont: CheckedContinuation<[Int32]?, Never>) in
            q.async {
                guard let tok = self._tokenizer else {
                    cont.resume(returning: nil); return
                }
                let thinkMarker = (mode == .chat)
                    ? EncodingDSV4.thinkClose
                    : EncodingDSV4.thinkOpen
                let deltaText = EncodingDSV4.eosToken
                    + EncodingDSV4.userToken
                    + userContent
                    + EncodingDSV4.assistantToken
                    + thinkMarker
                let ids = tok.encode(deltaText).map(Int32.init)
                cont.resume(returning: ids)
            }
        }
    }

    func generate(history: [Message],
                   mode: ThinkingMode,
                   options: SamplingOptions,
                   maxTokens: Int
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        return AsyncThrowingStream { continuation in
            q.async { [weak self] in
                guard let self else { continuation.finish(); return }
                guard let model = self.transformer,
                      let tok = self._tokenizer else {
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
                    continuation.yield(.token(text: piece, id: Int32(nextId)))

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
                let promptOut = promptIds.map(Int32.init)
                let generatedOut = generated.map(Int32.init)
                continuation.yield(.done(final: final,
                                          promptTokens: promptOut,
                                          generatedTokens: generatedOut))
                continuation.finish()
            }
        }
    }

    /// Fast-path variant: skip the chat-template encoding step and
    /// feed a pre-tokenized prompt straight into prefill + decode.
    /// Callers (currently `ChatStore`) compose `promptTokens` as
    /// `Conversation.encodedTokens` + `tokenizeUserTurnDelta(...)`,
    /// so the only BPE work per turn is on the *new* user content.
    func generateFromPrompt(promptTokens: [Int32],
                             mode: ThinkingMode,
                             options: SamplingOptions,
                             maxTokens: Int
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        return AsyncThrowingStream { continuation in
            q.async { [weak self] in
                guard let self else { continuation.finish(); return }
                guard let model = self.transformer,
                      let tok = self._tokenizer else {
                    continuation.finish(throwing: NSError(
                        domain: "InferenceService", code: 1, userInfo: [
                            NSLocalizedDescriptionKey:
                                "No model loaded yet — pick a folder first."
                        ]))
                    return
                }
                if promptTokens.isEmpty {
                    continuation.finish(throwing: NSError(
                        domain: "InferenceService", code: 2, userInfo: [
                            NSLocalizedDescriptionKey:
                                "Empty pre-tokenized prompt."
                        ]))
                    return
                }
                self.resetCancelFlag()
                model.releaseCache()

                // Prefill.
                let promptIds = promptTokens.map(Int.init)
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
                let stops = tok.stopTokenIds

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
                    continuation.yield(.token(text: piece, id: Int32(nextId)))

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

                let final = EncodingDSV4.parseCompletion(generatedText,
                                                           mode: mode)
                let generatedOut = generated.map(Int32.init)
                continuation.yield(.done(final: final,
                                          promptTokens: promptTokens,
                                          generatedTokens: generatedOut))
                continuation.finish()
            }
        }
    }

    /// Same as `generateFromPrompt`, but reuses the model's KV cache
    /// across consecutive turns of the same `conversationID`. When the
    /// previous turn's tokens are a prefix of the new `promptTokens`
    /// (and the mode hasn't changed), we skip `releaseCache()` and
    /// prefill only the *delta*. Mismatch falls back to a full reset
    /// + multi-token prefill.
    ///
    /// Multi-token prefill from `startPos > 0` is blocked by the
    /// `precondition(S == 1)` in `MLA.callAsFunction`'s decode branch,
    /// so the incremental prefill runs the delta token-by-token (the
    /// same path the decode loop already uses). For a typical
    /// `<eos><User>…<Assistant><think>` delta of ~10-50 tokens this
    /// is a small fixed cost compared to re-prefilling the full
    /// transcript every turn.
    func generateForConversation(promptTokens: [Int32],
                                  conversationID: UUID,
                                  mode: ThinkingMode,
                                  options: SamplingOptions,
                                  maxTokens: Int
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        return AsyncThrowingStream { continuation in
            q.async { [weak self] in
                guard let self else { continuation.finish(); return }
                guard let model = self.transformer,
                      let tok = self._tokenizer else {
                    continuation.finish(throwing: NSError(
                        domain: "InferenceService", code: 1, userInfo: [
                            NSLocalizedDescriptionKey:
                                "No model loaded yet — pick a folder first."
                        ]))
                    return
                }
                if promptTokens.isEmpty {
                    continuation.finish(throwing: NSError(
                        domain: "InferenceService", code: 2, userInfo: [
                            NSLocalizedDescriptionKey:
                                "Empty pre-tokenized prompt."
                        ]))
                    return
                }
                self.resetCancelFlag()

                // Decide reuse vs. full reset. The match is strict:
                // same conversation + same mode + cached tokens are a
                // strict prefix of the new prompt. Anything else (a
                // shorter new prompt = user edited backwards, a
                // diverging prefix = template changed) goes through
                // releaseCache() to be safe.
                let canReuse: Bool
                let cachedCount: Int
                if let img = self.cacheImage,
                   img.conversationID == conversationID,
                   img.mode == mode,
                   img.tokens.count > 0,
                   img.tokens.count < promptTokens.count,
                   img.tokens[...] == promptTokens.prefix(img.tokens.count)
                {
                    canReuse = true
                    cachedCount = img.tokens.count
                } else {
                    canReuse = false
                    cachedCount = 0
                    model.releaseCache()
                    self.cacheImage = nil
                }

                let deltaTokens = Array(promptTokens.suffix(
                    promptTokens.count - cachedCount))
                continuation.yield(.prefillStart(promptTokens: deltaTokens.count))
                let prefillStart = Date()

                var logits: Tensor
                if canReuse {
                    // Incremental prefill: feed the delta one token at
                    // a time. Same kernel path the decode loop uses,
                    // so we don't need any new attention codepath.
                    // Final logits come from the *last* delta token —
                    // that's the position the sampler reads from to
                    // produce the assistant's first new token.
                    var lastLogits: Tensor? = nil
                    for (i, t) in deltaTokens.enumerated() {
                        if self.isCancelled() { break }
                        lastLogits = model.forward(
                            inputIds: [[Int(t)]],
                            startPos: cachedCount + i)
                    }
                    guard let ll = lastLogits else {
                        continuation.finish(throwing: NSError(
                            domain: "InferenceService", code: 3, userInfo: [
                                NSLocalizedDescriptionKey:
                                    "Cancelled before prefill could produce logits."
                            ]))
                        return
                    }
                    logits = ll
                } else {
                    // Cold path: full multi-token prefill from startPos 0.
                    let ids = deltaTokens.map(Int.init)
                    logits = model.forward(inputIds: [ids], startPos: 0)
                }

                let prefillElapsed = Date().timeIntervalSince(prefillStart)
                let prefillTPM = prefillElapsed > 0
                    ? Double(deltaTokens.count) / prefillElapsed * 60
                    : 0
                continuation.yield(.prefillDone(
                    promptTokens: deltaTokens.count,
                    elapsed: prefillElapsed,
                    tokPerMin: prefillTPM))

                var opts = options
                let stops = tok.stopTokenIds

                var generated: [Int] = []
                var generatedText = ""
                let decodeStart = Date()
                var lastSample = decodeStart
                // The decode loop continues from where prefill stopped.
                // startPos for the *first* sampled token's forward is
                // promptTokens.count (the cache holds 0..<promptTokens.count
                // after the prefill step above, regardless of which path
                // we took).
                for step in 0..<maxTokens {
                    if self.isCancelled() { break }

                    let nextId = Sampler.sample(logits,
                                                  history: generated,
                                                  options: &opts)
                    if stops.contains(nextId) { break }
                    generated.append(nextId)

                    let piece = tok.decode([nextId])
                    generatedText += piece
                    continuation.yield(.token(text: piece, id: Int32(nextId)))

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
                    let startPos = promptTokens.count + step
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

                let final = EncodingDSV4.parseCompletion(generatedText,
                                                           mode: mode)
                let generatedOut = generated.map(Int32.init)

                // Stamp the image so the next turn of this same
                // conversation can extend us. Skipping this on
                // cancellation would also work, but the cache may
                // still be coherent up to `promptTokens.count`, so we
                // keep it — at worst the next turn re-prefills.
                self.cacheImage = CacheImage(
                    conversationID: conversationID,
                    tokens: promptTokens + generatedOut,
                    mode: mode)

                continuation.yield(.done(final: final,
                                          promptTokens: promptTokens,
                                          generatedTokens: generatedOut))
                continuation.finish()
            }
        }
    }
}
