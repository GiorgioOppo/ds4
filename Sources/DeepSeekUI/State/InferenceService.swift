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
    /// A decoded chunk of the prompt the model is about to see. Only
    /// emitted on cold prefill (cachedCount == 0) and only when the
    /// `showPrefillTrace` AppStorage flag is on. The chunks together
    /// reconstruct the full prompt text — system message (with tools
    /// block), conversation history, and the just-typed user turn —
    /// streamed to the UI so it can render a collapsible gray
    /// "what the model saw" block between the user message and the
    /// assistant reply. Decoded with the same tokenizer that produced
    /// the IDs, so the reconstruction is faithful to what the
    /// transformer ingests (token boundaries collapsed back into
    /// readable text).
    case prefillToken(text: String)
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
    /// Resolved chat template for the loaded model. Defaults to
    /// `DSV4Template()` because the engine is V4-coupled today; the
    /// dispatcher in `TokenizerLoader.load(tokenizerDir:)` will pick
    /// `JinjaChatTemplate` for any model directory that ships a
    /// `tokenizer_config.json` with a `chat_template` field.
    /// Production V4 flow still goes through `EncodingDSV4.*`
    /// directly — this property is exposed to callers (sub-agent
    /// renderers, future remote prompt builders) that want the
    /// model's official format without a V4 assumption.
    private(set) var chatTemplate: ChatTemplate = DSV4Template()
    private(set) var isDSV4Template: Bool = true
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

    /// Default fallback se nessun modello è ancora caricato (es.
    /// chiamata prematura). Per V4 con ratios `[0,0,4,128,4,128,4,0]`
    /// il LCM = 128. Il valore vero viene letto runtime da
    /// `self.loadedConfig.compressRatioLCM` quando disponibile.
    static let defaultCompressRatioLCM = 128

    /// LCM dei compressRatios del modello correntemente caricato.
    /// Letto dinamicamente da `loadedConfig` così supportiamo
    /// modelli con configurazione diversa da V4 standard senza
    /// bisogno di hard-code.
    private var currentCompressRatioLCM: Int {
        return self.loadedConfig?.compressRatioLCM
            ?? Self.defaultCompressRatioLCM
    }

    // MARK: - Cross-restart KV cache (ds4-style)

    /// Conversation attualmente "live" (l'ultima il cui state KV è
    /// nei MTLBuffer del modello). Aggiornata da
    /// `generateForConversation` ad ogni turn. `nil` significa
    /// "nessuna conversation specifica" (per es. dopo
    /// `releaseCache()` o prima del primo turn).
    private var activeConversationID: UUID? = nil

    /// Lifecycle del KV cache persistente. Setup in `loadModel` se
    /// `crossRestartKVCache` AppStorage flag è ON. Niente se OFF.
    private var kvLifecycle: KVCacheLifecycle? = nil

    /// Tokens attualmente nel KV cache (= `cacheImage.tokens` di
    /// solito, ma conservato qui per essere accessibile al lifecycle
    /// save closure senza dover hop al main).
    private var kvLiveTokens: [Int32] = []

    /// Setup `kvLifecycle` + closure di save se il flag AppStorage
    /// `crossRestartKVCache` è ON. Chiamato da `loadModel` dopo che
    /// transformer + tokenizer sono ready. Niente in caso di flag OFF.
    fileprivate func setupKVLifecycle() {
        let enabled = UserDefaults.standard.bool(
            forKey: AppSettingsKey.crossRestartKVCache)
        guard enabled else {
            self.kvLifecycle = nil
            return
        }
        let lifecycle = KVCacheLifecycle()
        // weak self per evitare retain cycle (lifecycle può
        // sopravvivere all'unload del modello durante shutdown).
        lifecycle.save = { [weak self] trigger in
            guard let self = self else { return }
            await self.persistKVCache(trigger: trigger)
        }
        self.kvLifecycle = lifecycle
    }

    /// Salva lo snapshot della KV cache + manifest su disco.
    /// Chiamato dalle closure del lifecycle. Eseguita sul queue
    /// dedicato di inferenza (`q`) per non confliggere col forward
    /// in corso.
    fileprivate func persistKVCache(trigger: KVCacheLifecycle.SaveTrigger) async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            self.q.async { [weak self] in
                guard let self = self,
                      let model = self.transformer,
                      let convID = self.activeConversationID
                else {
                    cont.resume(); return
                }
                do {
                    let url = try PersistencePaths.kvCacheURL(id: convID)
                    let snap = model.snapshotKVCache()

                    // Leggi compression dal AppStorage.
                    let compRaw = UserDefaults.standard.string(
                        forKey: AppSettingsKey.kvCacheCompression) ?? "f32"
                    let compression: KVCacheSnapshot.DiskCompression = {
                        switch compRaw {
                        case "f16":  return .f16
                        case "bf16": return .bf16
                        default:     return .f32
                        }
                    }()

                    try snap.save(to: url, compression: compression)

                    // Manifest v2: tokens + chunk alignment.
                    // Per il `cold` trigger applichiamo cold-save
                    // alignment a 2048 token (ds4 convention).
                    let tokens = self.kvLiveTokens
                    let alignedCount: Int? = trigger == .cold
                        ? min(KVCacheFile.coldSaveAlignedCount(tokens.count),
                              tokens.count)
                        : nil
                    let savedTokens = alignedCount ?? tokens.count
                    let alignedSlice = Array(tokens.prefix(savedTokens))

                    let manifestURL = url.appendingPathExtension("manifest")
                    let manifestData = self.buildManifestData(
                        tokens: alignedSlice,
                        chunkAlignment: alignedCount.map { _ in 2048 })
                    try manifestData.write(to: manifestURL, options: .atomic)

                    let logMsg = "[kvcache] saved (\(trigger.rawValue)): "
                        + "\(savedTokens) tokens, \(snap.totalBytes) bytes"
                        + " (\(compRaw))\n"
                    FileHandle.standardError.write(Data(logMsg.utf8))
                } catch {
                    let errMsg = "[kvcache] save failed (\(trigger.rawValue)): "
                        + "\(error.localizedDescription)\n"
                    FileHandle.standardError.write(Data(errMsg.utf8))
                }
                cont.resume()
            }
        }
    }

    /// Costruisce il blob binario del manifest v2 in-line (evita di
    /// dover esporre `KVCacheFile.ManifestData` come Codable
    /// pubblico). Layout: vedi `KVCacheFile.writeManifestFull`.
    private func buildManifestData(tokens: [Int32],
                                     chunkAlignment: Int?) -> Data {
        let useV2 = chunkAlignment != nil
        let version: UInt32 = useV2 ? 2 : 1
        var data = Data()
        var magic = KVCacheFile.manifestMagic.littleEndian
        var verLE = version.littleEndian
        var count = UInt64(tokens.count).littleEndian
        withUnsafeBytes(of: &magic) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &verLE) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &count) { data.append(contentsOf: $0) }
        tokens.withUnsafeBufferPointer { p in
            let byteCount = p.count * MemoryLayout<Int32>.stride
            let bytePtr = UnsafeRawPointer(p.baseAddress!)
                .assumingMemoryBound(to: UInt8.self)
            data.append(bytePtr, count: byteCount)
        }
        if useV2 {
            var logitsLen: UInt32 = 0  // logits non salvati qui
            withUnsafeBytes(of: &logitsLen) { data.append(contentsOf: $0) }
            var align = UInt32(chunkAlignment ?? 0).littleEndian
            withUnsafeBytes(of: &align) { data.append(contentsOf: $0) }
        }
        return data
    }

    /// Prova a ripristinare la KV cache di una conversation dal
    /// disco (se esiste un file `.kvcache` valido + manifest il cui
    /// hash dei token iniziale matcha). Ritorna `nil` se non c'è
    /// match o se il restore fallisce. Quando ritorna non-nil, il
    /// modello è stato `restoreKVCache(_:)` con lo snapshot e
    /// `activeConversationID` + `kvLiveTokens` sono settati.
    /// Caller può fare prefill solo del delta (`promptTokens -
    /// returnedTokens`) invece di cold prefill.
    ///
    /// Eseguito sul queue di inferenza dal chiamante.
    fileprivate func tryRestoreKVCache(for convID: UUID,
                                         model: Transformer)
        -> [Int32]?
    {
        guard self.kvLifecycle != nil else { return nil }
        do {
            let url = try PersistencePaths.kvCacheURL(id: convID)
            guard FileManager.default.fileExists(atPath: url.path) else {
                return nil
            }
            guard let snap = KVCacheSnapshot.load(from: url) else {
                return nil
            }
            // Leggi il manifest per ottenere i token IDs.
            let manifestURL = url.appendingPathExtension("manifest")
            guard let mdata = try? Data(contentsOf: manifestURL),
                  mdata.count >= 16 else {
                return nil
            }
            var cursor = 0
            guard let m1 = readU32(mdata, &cursor),
                  m1 == KVCacheFile.manifestMagic,
                  let _ver = readU32(mdata, &cursor),
                  _ver == 1 || _ver == 2,
                  let n = readU64(mdata, &cursor)
            else { return nil }
            _ = _ver
            let needed = Int(n) * 4
            guard cursor + needed <= mdata.count else { return nil }
            let tokenPtr = mdata.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                raw.baseAddress!.advanced(by: cursor)
                    .assumingMemoryBound(to: Int32.self)
            }
            let tokens = Array(UnsafeBufferPointer(start: tokenPtr,
                                                     count: Int(n)))
            // Restore in memory.
            model.restoreKVCache(snap)
            let okMsg = "[kvcache] restored from disk: "
                + "\(tokens.count) tokens, \(snap.totalBytes) bytes\n"
            FileHandle.standardError.write(Data(okMsg.utf8))
            return tokens
        } catch {
            let failMsg = "[kvcache] restore failed: "
                + "\(error.localizedDescription)\n"
            FileHandle.standardError.write(Data(failMsg.utf8))
            return nil
        }
    }

    /// Triggera shutdown sync — chiamato da app delegate quando l'app
    /// termina. Eseguito synchronously per garantire che il save
    /// finisca prima dell'exit.
    public func saveKVCacheOnShutdown() {
        guard let lifecycle = self.kvLifecycle else { return }
        let group = DispatchGroup()
        group.enter()
        Task {
            await lifecycle.triggerShutdown()
            group.leave()
        }
        _ = group.wait(timeout: .now() + 5.0)
    }

    /// Quando true, `generateForConversation` accetta anche
    /// common-prefix match (non solo strict-prefix). Se il nuovo
    /// prompt diverge dal cached dopo K token (K > 0), il
    /// `model.rewindKVTo` resetta esplicitamente kvState/scoreState
    /// di tutti i compressor al window boundary più vicino
    /// (pSafe = round_down(K, 128)). Il forward successivo
    /// ricostruisce coerentemente da pSafe in poi.
    ///
    /// **Caso d'uso**: user edita il proprio ultimo messaggio. Il
    /// prefisso comune (system + history + first user msg tokens)
    /// viene preservato.
    ///
    /// **Window boundary**: round-down al LCM(compressRatios)=128
    /// per V4. Significa che common=200 → riusa 128 token e
    /// ri-prefilla 72; common=150 → riusa 128 e ri-prefilla 22.
    /// L'overhead è < ratio_max=128 tokens nel caso peggiore.
    ///
    /// **Safety**: il rewind è esplicito (zero kvState, -inf
    /// scoreState ai window boundary). Niente "silent degrade"
    /// da forward overwrite implicito. Se ANCHE UN layer rifiuta
    /// il rewind, fallback automatico a `releaseCache()` + cold
    /// prefill.
    ///
    /// Default OFF perché non testato end-to-end su GPU reale.
    /// Quando attivato, monitorare l'output per regressioni.
    public var enableCommonPrefixRewind: Bool = false

    /// Holds the snapshot + CacheImage captured around an active
    /// sub-agent delegation. The host's KV cache is snapshotted
    /// into RAM before the sub-agent runs (which would otherwise
    /// clobber the buffers + invalidate `cacheImage`), and
    /// restored verbatim when the host resumes — so the host
    /// agent's next turn keeps the fast-delta path instead of
    /// paying a cold prefill.
    ///
    /// Keyed by an opaque UUID returned to the caller; entries are
    /// drained on `endDelegation`. The map size stays bounded as
    /// long as begin/end calls pair up — if the app dies between
    /// the two, the map is in RAM only and goes away with it.
    private var savedDelegations: [UUID: (KVCacheSnapshot, CacheImage?)] = [:]

    /// Snapshot the transformer's KV cache + the `cacheImage`
    /// shadow that tracks it, returning an opaque token to pass to
    /// `endDelegation` after the sub-agent run is done. Returns
    /// nil when no model is loaded (caller should run the
    /// sub-agent un-snapshotted in that case — there's nothing
    /// to preserve).
    ///
    /// O(cache bytes) — for V4 with windowSize 4096 that's a few
    /// hundred MB of RAM held until `endDelegation`. Only call it
    /// around real delegations, not as a "save point" pattern.
    func beginDelegation() async -> UUID? {
        await withCheckedContinuation {
            (cont: CheckedContinuation<UUID?, Never>) in
            q.async {
                guard let model = self.transformer else {
                    cont.resume(returning: nil); return
                }
                let snap = model.snapshotKVCache()
                let token = UUID()
                self.savedDelegations[token] = (snap, self.cacheImage)
                cont.resume(returning: token)
            }
        }
    }

    /// Wipe the sub-agent's pollution from the live KV cache and
    /// re-populate it from the snapshot taken at `beginDelegation`,
    /// also restoring the `cacheImage` so subsequent fast-path
    /// matching sees the host's pre-delegation prefix. No-op when
    /// the token is unknown (e.g. begin returned nil, or the
    /// caller paired tokens wrong).
    func endDelegation(_ token: UUID) async {
        await withCheckedContinuation {
            (cont: CheckedContinuation<Void, Never>) in
            q.async {
                guard let saved = self.savedDelegations.removeValue(forKey: token)
                else { cont.resume(returning: ()); return }
                guard let model = self.transformer else {
                    // Model was unloaded mid-delegation — nothing
                    // to restore into. Drop the snapshot.
                    cont.resume(returning: ()); return
                }
                model.releaseCache()
                model.restoreKVCache(saved.0)
                self.cacheImage = saved.1
                cont.resume(returning: ())
            }
        }
    }

    /// Snapshot of the active tokenizer for read-only use outside the
    /// generation path (e.g. the document import flow). Reads the
    /// stored reference on the serial queue so it never races a load.
    /// Returns nil until a model has been loaded.
    func currentTokenizer() -> Tokenizer? {
        q.sync { _tokenizer }
    }

    /// Non-blocking "is a model loaded?" check. Protetto da
    /// `stateLock` invece che dalla coda di inferenza `q`, così le
    /// view body possono chiamarlo senza bloccarsi per la durata di
    /// una generation in volo. Aggiornato dentro `loadModel` /
    /// `unloadModel` al momento di settare/svuotare il transformer.
    /// Le view che hanno solo bisogno di "è il modello pronto?"
    /// dovrebbero usare questo invece di `currentTokenizer() == nil`.
    private let stateLock = NSLock()
    private var _isModelLoaded: Bool = false
    func isModelLoaded() -> Bool {
        stateLock.lock(); defer { stateLock.unlock() }
        return _isModelLoaded
    }
    private func setModelLoaded(_ loaded: Bool) {
        stateLock.lock(); defer { stateLock.unlock() }
        _isModelLoaded = loaded
    }

    /// Snapshot of the active model directory. The document library
    /// uses it as a fingerprint to detect "different model selected
    /// since import time".
    func currentModelDir() -> URL? {
        q.sync { loadedModelDir }
    }

    /// Snapshot the currently-loaded tokenizer + config so external
    /// callers (T3: `LocalServerRoutes.makeChatCompletionsHandler`)
    /// can compile a `SchemaMask` without juggling actor isolation.
    /// Runs through the inference queue to respect the
    /// "private state guarded by q" invariant on `_tokenizer` /
    /// `loadedConfig`. Returns nil if no model is loaded yet.
    func snapshotTokenizerAndConfig() -> (Tokenizer, ModelConfig)? {
        return q.sync { [weak self] in
            guard let self,
                  let tok = self._tokenizer,
                  let cfg = self.loadedConfig
            else { return nil }
            return (tok, cfg)
        }
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

    /// Cancellazione per-conversazione. Il prefill/decode loop legge
    /// `isCancelled(for: convID)` fra un token e l'altro; se è in set
    /// (o se `globalCancel` è on) esce dopo aver finito il token
    /// corrente. Multi-track safe: cancellare la chat A non ferma
    /// la chat B che sta aspettando dietro di lei sulla q seriale.
    /// `globalCancel` resta come backstop per chiamate legacy
    /// (`cancelCurrent()` senza id) — useremmo unloadModel per uno
    /// stop più drastico ma manteniamo la semantica esistente.
    private var cancelledConvIDs = Set<UUID>()
    private var globalCancel = false
    private let cancelLock = NSLock()

    init() {}

    /// Marca questa conversazione come cancellata; se id è nil,
    /// alza il flag globale (vecchio comportamento `cancelCurrent()`
    /// senza argomenti — equivalente a "stop whatever is current").
    /// La conversation viene tolta dal set automaticamente al prossimo
    /// `resetCancelFlag(for:)` (= prima dell'inizio del prossimo
    /// generate per quella conv).
    func cancelCurrent(conversationID: UUID? = nil) {
        cancelLock.lock(); defer { cancelLock.unlock() }
        if let id = conversationID {
            cancelledConvIDs.insert(id)
        } else {
            globalCancel = true
        }
    }

    private func resetCancelFlag(for conversationID: UUID? = nil) {
        cancelLock.lock(); defer { cancelLock.unlock() }
        if let id = conversationID {
            cancelledConvIDs.remove(id)
        }
        // `globalCancel` lo droppiamo sempre quando un generate parte:
        // se era stato alzato per fermare il run precedente e quello
        // ha già finito (o se l'utente l'ha alzato per errore), il
        // nuovo run riparte pulito.
        globalCancel = false
    }

    private func isCancelled(for conversationID: UUID? = nil) -> Bool {
        cancelLock.lock(); defer { cancelLock.unlock() }
        if globalCancel { return true }
        if let id = conversationID, cancelledConvIDs.contains(id) {
            return true
        }
        return false
    }

    /// Tear down whatever model is currently in memory. Releases
    /// the transformer + tokenizer + cache shadow on the serial
    /// queue so a generation in flight finishes cleanly before
    /// the buffers go away. Idempotent — calling on an already-
    /// empty service is a no-op.
    ///
    /// Used by the in-chat model picker's "Unload" affordance and
    /// implicitly when the user picks a different model from the
    /// picker (the load path calls unload first so the old
    /// weights don't linger alongside the new ones).
    func unloadModel() async {
        await withCheckedContinuation {
            (cont: CheckedContinuation<Void, Never>) in
            q.async {
                if let model = self.transformer {
                    model.releaseCache()
                }
                self.transformer = nil
                self._tokenizer = nil
                self.setModelLoaded(false)
                self.loadedConfig = nil
                self.loadedModelDir = nil
                self.cacheImage = nil
                self.savedDelegations.removeAll()
                cont.resume(returning: ())
            }
        }
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
                    // Try the full dispatcher first so a Mistral / Llama /
                    // Gemma directory loads its `chat_template` Jinja
                    // automatically. Fall back to the legacy V4 path if
                    // the directory lacks a chat_template AND isn't V4 —
                    // this preserves the historical behaviour where
                    // `TokenizerLoader.load(from:)` accepted any
                    // `tokenizer.json` and assumed V4.
                    let tokURL = url.appendingPathComponent("tokenizer.json")
                    let tok: Tokenizer
                    do {
                        let loaded = try TokenizerLoader.load(tokenizerDir: url)
                        tok = loaded.tokenizer
                        self.chatTemplate = loaded.chatTemplate
                        self.isDSV4Template = loaded.isDSV4
                    } catch {
                        tok = try TokenizerLoader.load(from: tokURL)
                        self.chatTemplate = DSV4Template()
                        self.isDSV4Template = true
                    }

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

                    // Leggi le preference che influenzano il load.
                    // `warmupOnLoad` da AppStorage; saltato automaticamente
                    // dal loader se la RAM disponibile è insufficiente.
                    // `useMapShared` esperimentale, fallback a MAP_PRIVATE
                    // se mmap fallisce.
                    let warmup = UserDefaults.standard.bool(
                        forKey: AppSettingsKey.warmupOnLoad)
                    let useShared = UserDefaults.standard.bool(
                        forKey: AppSettingsKey.useMapSharedWeights)

                    let model = try Transformer.load(
                        config: cfg, from: url,
                        strategyOverride: strategyOverride,
                        forceLoad: forceLoad,
                        warmupOnLoad: warmup,
                        useMapSharedWeights: useShared)

                    self.transformer = model
                    self._tokenizer = tok
                    self.loadedConfig = cfg
                    self.loadedModelDir = url
                    self.setModelLoaded(true)
                    // A model swap renders every cached KV state
                    // invalid (different weight tensors → different
                    // attention outputs).
                    self.cacheImage = nil
                    // Setup del cross-restart KV cache (se abilitato)
                    // — instantiate il lifecycle e inietta la save
                    // closure. Niente se il flag è OFF.
                    self.setupKVLifecycle()
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

                // Diagnostic: DEEPSEEK_PROMPT_DEBUG=1 dumps the
                // assembled prompt String to stderr right before
                // tokenization. Pairs with a separate dump of the
                // raw toolSchemasJSON so we can tell whether a
                // weird-looking prompt body is the JSON itself
                // misbehaving or the template substitution dropping
                // text. Gated so production prefills aren't noisy.
                if ProcessInfo.processInfo
                    .environment["DEEPSEEK_PROMPT_DEBUG"] == "1" {
                    let header = "[prompt-debug] tokenizeFullHistory " +
                                 "history=\(effectiveHistory.count) " +
                                 "schemas=\(toolSchemasJSON?.count ?? -1) " +
                                 "prompt=\(prompt.count)\n"
                    FileHandle.standardError.write(Data(header.utf8))
                    if let schemas = toolSchemasJSON {
                        let line = "[prompt-debug] >>> toolSchemasJSON >>>\n" +
                                   schemas +
                                   "\n[prompt-debug] <<< toolSchemasJSON <<<\n"
                        FileHandle.standardError.write(Data(line.utf8))
                    }
                    let line = "[prompt-debug] >>> prompt >>>\n" +
                               prompt +
                               "\n[prompt-debug] <<< prompt <<<\n"
                    FileHandle.standardError.write(Data(line.utf8))
                }

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
                self.resetCancelFlag(for: conversationID)
                // Sync user preference for common-prefix rewind. Letta
                // ogni turn così l'utente può abilitarla/disabilitarla
                // dal Settings senza riavviare.
                self.enableCommonPrefixRewind = UserDefaults.standard.bool(
                    forKey: AppSettingsKey.commonPrefixRewind)

                // Aggiorna `activeConversationID` per il lifecycle save
                // closure. Se cambia (evict trigger condizione), salva
                // lo stato della conversation precedente prima del
                // releaseCache.
                let previousConvID = self.activeConversationID
                if previousConvID != conversationID,
                   let lifecycle = self.kvLifecycle,
                   previousConvID != nil
                {
                    let group = DispatchGroup()
                    group.enter()
                    Task {
                        await lifecycle.triggerEvict()
                        group.leave()
                    }
                    _ = group.wait(timeout: .now() + 5.0)
                    lifecycle.reset()
                }
                self.activeConversationID = conversationID

                // Cross-restart resume tentative: se non c'è
                // cacheImage live (= primo turn in questa session)
                // e il flag crossRestartKVCache è attivo, prova a
                // ripristinare da disco.
                if self.cacheImage == nil, self.kvLifecycle != nil {
                    if let restoredTokens = self.tryRestoreKVCache(
                        for: conversationID, model: model)
                    {
                        self.cacheImage = CacheImage(
                            conversationID: conversationID,
                            tokens: restoredTokens,
                            mode: mode)
                    }
                }

                // Decide reuse vs. full reset.
                //
                // **Strict-prefix** (always supported): cached tokens
                // are an exact prefix of the new prompt. Same
                // conversation + same mode + new is longer than cached
                // → reuse all cachedCount tokens, prefill the delta.
                //
                // **Common-prefix** (opt-in, `enableCommonPrefixRewind`):
                // even when cached diverges from new at some
                // position K (e.g., user edited their last message),
                // reuse the first K tokens and prefill `new[K..<]`
                // starting from startPos=K. The KV cache slots at
                // K..cached.count-1 will be silently overwritten by
                // the new forward pass. Risky for edge cases (see
                // class doc on `enableCommonPrefixRewind`).
                //
                // Mismatch on any axis (different conversation, mode
                // change, empty cache, common < threshold) → full
                // reset + cold prefill.
                let canReuse: Bool
                let cachedCount: Int
                if let img = self.cacheImage,
                   img.conversationID == conversationID,
                   img.mode == mode,
                   img.tokens.count > 0
                {
                    // Compute common prefix length once.
                    let limit = min(img.tokens.count, promptTokens.count)
                    var common = 0
                    while common < limit && img.tokens[common] == promptTokens[common] {
                        common += 1
                    }
                    let isStrictPrefix =
                        common == img.tokens.count
                        && img.tokens.count < promptTokens.count

                    if isStrictPrefix {
                        // Caso preferito: niente rewind, solo append.
                        canReuse = true
                        cachedCount = img.tokens.count
                    } else if self.enableCommonPrefixRewind
                                && common >= self.currentCompressRatioLCM
                                && common < promptTokens.count
                    {
                        // Common-prefix con rewind robusto. Round down
                        // al multiplo del LCM dei compressRatio del
                        // modello (letto dinamicamente dal config:
                        // 128 per V4 standard 0/4/128 ratios).
                        // Inizio-window garantito per tutti i layer →
                        // `rewindKVTo` può zerare scoreState/kvState
                        // senza orfani mid-window.
                        let lcm = self.currentCompressRatioLCM
                        let pSafe = (common / lcm) * lcm
                        if pSafe >= lcm
                            && model.rewindKVTo(pos: pSafe)
                        {
                            canReuse = true
                            cachedCount = pSafe
                            let discarded = img.tokens.count - pSafe
                            let extra = common - pSafe
                            continuation.yield(.status(
                                "Common-prefix rewind: keep \(pSafe) cached " +
                                "tokens (rounded down from \(common) to " +
                                "multiple of \(lcm)), discard \(discarded) " +
                                "stale, re-prefill \(extra) extra tokens " +
                                "to realign window boundary."))
                        } else {
                            // Rewind rifiutato da uno o più layer →
                            // stato potenzialmente parziale, force
                            // cold reset.
                            canReuse = false
                            cachedCount = 0
                            model.releaseCache()
                            self.cacheImage = nil
                        }
                    } else {
                        canReuse = false
                        cachedCount = 0
                        model.releaseCache()
                        self.cacheImage = nil
                    }
                } else {
                    canReuse = false
                    cachedCount = 0
                    model.releaseCache()
                    self.cacheImage = nil
                }

                let deltaTokens = Array(promptTokens.suffix(
                    promptTokens.count - cachedCount))
                continuation.yield(.prefillStart(promptTokens: deltaTokens.count))

                // Prefill trace: solo turn cold (KV cache vuota),
                // perché il delta dei turn incrementali è giusto il
                // nuovo user message + marker — niente di
                // ispezionabile in più. Decodifichiamo tutto il
                // delta in un colpo (round-trip safe sul byte-BPE
                // del V4: i token vengono da text tokenizzato pochi
                // ms fa), poi splittiamo in chunk fissi e yieldiamo
                // un evento per chunk con un micro-sleep in mezzo
                // così la UI vede il prompt scorrere invece di
                // apparire tutto insieme. Il flag è opt-out via
                // settings; default ON (`object(forKey:) == nil`
                // significa "non scritto ancora" → trattalo come
                // attivo).
                let traceFlag = UserDefaults.standard.object(
                    forKey: AppSettingsKey.showPrefillTrace) as? Bool ?? true
                if traceFlag, cachedCount == 0 {
                    let fullText = tok.decode(deltaTokens.map(Int.init))
                    if !fullText.isEmpty {
                        let chars = Array(fullText)
                        let chunkSize = 24
                        let totalChunks = max(1, (chars.count + chunkSize - 1) / chunkSize)
                        // Cap il delay sintetico a 1.5 s totali su
                        // prompt enormi così la prefill non si
                        // ferma dietro al render.
                        let perChunkDelay = min(0.004, 1.5 / Double(totalChunks))
                        var i = 0
                        while i < chars.count {
                            if self.isCancelled(for: conversationID) { break }
                            let end = min(i + chunkSize, chars.count)
                            continuation.yield(.prefillToken(
                                text: String(chars[i..<end])))
                            i = end
                            if perChunkDelay > 0 {
                                Thread.sleep(forTimeInterval: perChunkDelay)
                            }
                        }
                    }
                }

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
                        if self.isCancelled(for: conversationID) { break }
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

                // Cold-save trigger: dopo il primo prefill, salva lo
                // snapshot della KV cache su disco. Skip se il flag
                // crossRestartKVCache è OFF (lifecycle nil).
                self.kvLiveTokens = promptTokens
                if let lifecycle = self.kvLifecycle {
                    let group = DispatchGroup()
                    group.enter()
                    Task {
                        await lifecycle.triggerCold()
                        group.leave()
                    }
                    _ = group.wait(timeout: .now() + 10.0)
                }

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
                    if self.isCancelled(for: conversationID) { break }

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

                        // Continued-save trigger: la lifecycle ha il
                        // suo throttle interno (default 128 token o 5s);
                        // qui chiamiamo a ogni progress tick ma il save
                        // vero parte solo se la threshold è raggiunta.
                        // kvLiveTokens aggiornato col delta corrente
                        // (= promptTokens originali + token generati).
                        if let lifecycle = self.kvLifecycle {
                            let liveCount = promptTokens.count + generated.count
                            self.kvLiveTokens = promptTokens + generated
                                .map { Int32($0) }
                            // Task fire-and-forget — il save è async,
                            // throttle dentro il lifecycle. Non bloccare
                            // il decode loop sul I/O.
                            Task {
                                await lifecycle.triggerContinued(
                                    currentTokenCount: liveCount)
                            }
                        }
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

// MARK: - Byte parsing helpers for KV cache restore

@inline(__always)
fileprivate func readU32(_ data: Data, _ cursor: inout Int) -> UInt32? {
    guard cursor + 4 <= data.count else { return nil }
    let v = data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> UInt32 in
        raw.load(fromByteOffset: cursor, as: UInt32.self)
    }
    cursor += 4
    return UInt32(littleEndian: v)
}

@inline(__always)
fileprivate func readU64(_ data: Data, _ cursor: inout Int) -> UInt64? {
    guard cursor + 8 <= data.count else { return nil }
    let v = data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> UInt64 in
        raw.load(fromByteOffset: cursor, as: UInt64.self)
    }
    cursor += 8
    return UInt64(littleEndian: v)
}
