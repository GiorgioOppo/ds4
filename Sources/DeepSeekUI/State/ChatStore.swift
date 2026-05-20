import Foundation
import SwiftUI
import DeepSeekKit

/// Throughput readings the UI shows alongside the streamed text. Both
/// fields stay at 0 until the corresponding phase produces a sample.
struct GenerationMetrics: Equatable {
    var promptTokens: Int = 0
    var prefillElapsed: TimeInterval = 0
    var prefillTokPerMin: Double = 0
    var generatedTokens: Int = 0
    var generationElapsed: TimeInterval = 0
    var generationTokPerMin: Double = 0
    /// USD cost of the current turn as reported by the remote
    /// provider (OpenRouter's `usage.total_cost`). Nil for local
    /// generations and for remote providers that don't report
    /// cost. The ThroughputBar surfaces it when non-nil.
    var turnCostUSD: Double?
}

enum GenerationPhase: Equatable {
    case idle
    /// The prefill forward is running. `promptTokens` is fixed, `startTime`
    /// lets the UI animate elapsed seconds while waiting (prefill is a
    /// single synchronous forward that doesn't stream intermediate
    /// progress, so live elapsed is the most useful liveness signal).
    case prefilling(promptTokens: Int, startTime: Date)
    /// `buffer` is the visible assistant text. `reasoningBuffer` is
    /// the live thinking content (today streamed only via the remote
    /// path's `reasoning_content` delta; local generation emits it
    /// only at `.done` once the `<think>` block has been split off
    /// the token stream). Empty by default so older callsites that
    /// construct `.streaming(buffer:, status:, metrics:)` without
    /// reasoning continue to compile via the new init helpers below.
    case streaming(buffer: String, reasoningBuffer: String,
                    status: String, metrics: GenerationMetrics)
    case error(String)

    /// Shorthand factory matching the pre-T4-followup signature.
    /// Existing call sites use this form; it just leaves the new
    /// `reasoningBuffer` empty.
    static func streaming(buffer: String, status: String,
                           metrics: GenerationMetrics) -> GenerationPhase {
        .streaming(buffer: buffer, reasoningBuffer: "",
                    status: status, metrics: metrics)
    }
}

/// Multi-chat store. Owns `[Conversation]` indexed by id, a selection,
/// per-conversation generation state, and the shared `InferenceService`.
/// Persists each conversation as one JSON file under
/// `~/Library/Application Support/<appName>/conversations/`.
@MainActor
final class ChatStore: ObservableObject {
    @Published var conversations: [Conversation] = []
    @Published var selectedID: UUID?
    @Published private(set) var phases: [UUID: GenerationPhase] = [:]

    let service: InferenceService
    /// Path of the model currently loaded into `service`, sampled
    /// fresh every time the store needs to stamp it onto a new
    /// `Conversation` (creation, switch). Empty when no model is
    /// loaded — newly-created chats in that state record an empty
    /// `modelDirPath` and the first send will fail on tokenizer
    /// lookup until the user picks a model from the toolbar.
    /// Used to live as a stored `let` populated at init time, back
    /// Path del local model attualmente caricato nel service.
    /// Letto dal mirror `@Published` di ModelState (main-actor,
    /// non-blocking) invece che da `service.currentModelDir()` —
    /// quest'ultimo passa per `q.sync` sulla coda di inferenza e
    /// bloccherebbe il main thread per minuti se una generation è
    /// in volo. Usato dai costruttori di `Conversation` (newChat
    /// inclusa) per popolare il campo `modelDirPath`; per le chat
    /// remote viene "" e va bene così.
    /// Used to live as a stored `let` populated at init time, back
    /// when the load step gated the chat UI; in-chat picking
    /// turns it into a derived value.
    var modelDirPath: String {
        modelState.loadedLocalModelDir?.path ?? ""
    }
    /// Library references handed down from the App scene. Used at
    /// "first turn of a chat with a project attached" time to pull
    /// the project's pre-tokenized files and assemble the prompt
    /// with native repo/file delimiter tokens. Both are weak-ish
    /// (unowned semantics implicit via @MainActor reference holding)
    /// but kept as strong refs to avoid lifetime surprises — the
    /// libraries live for the whole app session.
    let documents: DocumentLibrary
    let projects: ProjectLibrary
    /// Live MCP pool. Used by `send` to snapshot the current tool
    /// schemas and fold them into the system block of the prompt.
    /// Snapshot-on-send semantics means a server enabled mid-chat
    /// won't appear to the model until the cached prefix is
    /// invalidated (mode change or new chat) — same trade-off the
    /// project-context path already makes.
    let mcpPool: MCPClientPool
    /// Agent presets. Resolved by `send` via
    /// `Conversation.agentID` — when matched, the agent's system
    /// prompt is injected and its `allowedToolNames` filters the
    /// MCP tool catalogue.
    let agents: AgentLibrary
    /// Reactive view of which model (local or remote) is currently
    /// loaded. `send` reads `loadedEndpoint` and dispatches to the
    /// local token-pipeline or the OpenRouter HTTP path
    /// accordingly. Kept as a reference so a model swap mid-chat
    /// takes effect without recreating the store.
    let modelState: ModelState

    private let saveDebounce: TimeInterval = 0.5
    private var pendingSaves: [UUID: Task<Void, Never>] = [:]
    /// Disk IO for the v2 lazy-loading chat layout (per-chat folder,
    /// manifest + per-turn + per-round files). In PR 2 this layer is
    /// dual-write only: `newChat()` creates the folder + writes
    /// `chat.json`, and `flushSave` mirrors the manifest alongside
    /// the legacy `{UUID}.json`. PR 3 will move reads onto this path
    /// and stop writing the legacy file for new chats.
    private let chatPersistence = ChatPersistence()

    /// Round payload cache shared by every chat (capacity is across
    /// the app, not per chat). Pinned entries — the in-flight
    /// streaming round and any round whose disclosure is open —
    /// stay warm; everything else can be evicted under memory
    /// pressure. The legacy synth path doesn't populate the cache:
    /// it re-derives rounds from `c.messages` every call because
    /// the in-memory transcript is already the source of truth.
    private var roundLRU = RoundLRUCache(capacity: 64)

    /// Memoisation of the legacy synth pipeline: keep the last
    /// `[StoredMessage]` we computed `[TurnSummary]` for, keyed by
    /// chat id. A simple ObjectIdentifier-style identity comparison
    /// won't work on a Swift value-type array; we use a content
    /// fingerprint (count + last id + last content length) which is
    /// O(1) and stable enough to cut 99% of re-synthesis cost
    /// during streaming, where only the last message's content
    /// grows token-by-token.
    private struct SynthCacheEntry {
        var fingerprint: SynthFingerprint
        var turns: [TurnSummary]
    }
    private var synthCache: [UUID: SynthCacheEntry] = [:]
    /// Per-conversation count of how many tool-call -> generate
    /// continuations have fired since the latest user turn. Cleared
    /// the moment a `.done` arrives without tool calls (= final
    /// reply) or the cap is hit. The cap prevents the obvious
    /// failure mode of a model that keeps calling tools forever.
    private var toolRoundtrips: [UUID: Int] = [:]
    private let maxToolRoundtripsPerTurn: Int = 21

    /// Task in volo per ogni conversation, keyed by conv id. Permette
    /// (a) di cancellare un singolo run senza toccare gli altri,
    /// (b) di sapere quali conversation hanno una generation pending
    /// — informazione che la UI usa per mostrare "Queued behind chat
    /// X…" quando una local-vs-local serializza dietro alla q dell'
    /// InferenceService. Cleanup nel `defer` del Task body così
    /// completion/error/cancel rimuovono sempre l'entry.
    private var generationTasks: [UUID: Task<Void, Never>] = [:]
    /// How many *levels* of sub-agent invocation are allowed past
    /// the host. Cap = N means the host can delegate (level 1),
    /// that sub can delegate (level 2), … up to level N. A
    /// sub-agent at level N does NOT receive the
    /// `__delegate_to_agent` schema in its tools block, so the
    /// limit is enforced structurally. Cycle prevention via
    /// `chain` adds a second layer of safety: an agent already in
    /// the call stack can't be delegated to again, regardless of
    /// depth. With cap=3 the worst-case memory is 3 KV-snapshots
    /// in RAM at once (~few-hundred-MB each).
    private let maxDelegationDepth: Int = 3

    /// Per-conversation stack of live delegations. Top of the
    /// stack is the deepest sub-agent currently running. The UI
    /// reads this to render the live delegation chain (target
    /// agent's identity + the task it was handed + a streaming
    /// buffer that grows as the sub-agent's tokens arrive). Empty
    /// when no delegation is in flight; cleared on the way back
    /// up as each frame's `runSubAgentToCompletionInner` returns.
    @Published private(set) var activeDelegations: [UUID: [DelegationFrame]] = [:]

    /// Per-frame cancellation handles (TODO §4 follow-up: stop a
    /// single sub-agent from the chain UI). When the delegation
    /// chain pushes a frame, the runner stashes its `Task`-cancel
    /// closure here; the UI's per-frame stop button calls
    /// `cancelDelegation(frameID:)` which invokes the closure +
    /// records a marker so the inner loop can report graceful
    /// cancellation.
    private var delegationCancellations: [UUID: () -> Void] = [:]
    /// Frames the user explicitly cancelled (vs. natural completion
    /// or error). Read by the inner loop right after the stream
    /// drains so the persisted assistant turn reflects the reason
    /// instead of looking like an empty reply.
    @Published private(set) var cancelledDelegations: Set<UUID> = []
    /// Sampler parameters captured at the start of `send`; reused
    /// by `runToolCallsAndContinue` so a tool-output continuation
    /// inherits the same temperature / topK / topP / maxTokens
    /// the user picked for the current turn.
    private var lastSamplingOptions: [UUID: (opts: SamplingOptions,
                                              maxTokens: Int)] = [:]

    /// Host della suite di tool nativi (read/write/edit/grep/shell/…
    /// 16 tool definiti in `Sources/DeepSeekTools/Tools/`). Era già
    /// inizializzato in `DeepSeekUIApp` e cablato al Settings, ma il
    /// chat path non lo agganciava — i suoi schemi non finivano nel
    /// system block e le tool call con nome nativo non venivano
    /// routate. Adesso `composeToolSchemasJSON` / `composeOpenAITools`
    /// emettono gli schemi con prefisso `native__<name>` e i dispatch
    /// site indirizzano le invocazioni `native__*` a
    /// `nativeTools.dispatch(...)`. Tool-call routing in `runRemoteLoop`
    /// e local execute path controlla il prefisso per decidere fra
    /// `NativeToolHost.dispatch` e `MCPClientPool.invokeQualified`.
    let nativeTools: NativeToolHost

    init(service: InferenceService,
         documents: DocumentLibrary,
         projects: ProjectLibrary,
         mcpPool: MCPClientPool,
         agents: AgentLibrary,
         modelState: ModelState,
         nativeTools: NativeToolHost) {
        self.service = service
        self.documents = documents
        self.projects = projects
        self.mcpPool = mcpPool
        self.agents = agents
        self.modelState = modelState
        self.nativeTools = nativeTools
        loadFromDisk()
        if conversations.isEmpty {
            let first = Conversation(modelDirPath: modelDirPath,
                                       endpoint: modelState.loadedEndpoint)
            conversations.append(first)
            selectedID = first.id
            scheduleSave(first.id)
        } else {
            selectedID = conversations.first?.id
        }
    }

    // ----- selection / mutation -----

    var selectedConversation: Conversation? {
        guard let id = selectedID else { return nil }
        return conversations.first(where: { $0.id == id })
    }

    func newChat() {
        // Cattura l'endpoint corrente di ModelState come default per
        // questa chat — l'utente può poi cambiarlo via picker senza
        // toccare gli altri chat in volo. Se è una remote, la chat
        // non occuperà il local model in RAM; se è local, condivide
        // il transformer caricato con le altre chat locali.
        let c = Conversation(modelDirPath: modelDirPath,
                              endpoint: modelState.loadedEndpoint)
        conversations.insert(c, at: 0)
        selectedID = c.id
        // PR 2: stamp this chat as v2 right away. The synchronous
        // manifest write below creates the per-chat folder + the
        // `chat.json` file before `scheduleSave` fires, so the
        // first `flushSave` already sees `isV2Chat == true` and
        // dual-writes the manifest. The legacy `{UUID}.json` stays
        // the source of truth for reads; PR 3 will flip that.
        try? chatPersistence.writeManifestImmediate(manifest(from: c))
        scheduleSave(c.id)
    }

    /// Attach (or detach when `pid == nil`) a project to a conversation
    /// and persist. The reference is metadata for now; the chat path
    /// reads it only for the toolbar label until Step 3 of the KV
    /// pipeline pulls the project's documents into the prefill.
    func setProject(_ pid: UUID?, for id: UUID) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }) else { return }
        guard conversations[idx].projectID != pid else { return }
        conversations[idx].projectID = pid
        scheduleSave(id)
    }

    /// Attach (or detach when `aid == nil`) an agent preset to a
    /// conversation. Affects subsequent turns whose prompt is
    /// rebuilt — the cached prefix path (fast delta) inherits the
    /// previous agent setting since the system block is part of
    /// the cached prefix.
    func setAgent(_ aid: UUID?, for id: UUID) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }) else { return }
        guard conversations[idx].agentID != aid else { return }
        conversations[idx].agentID = aid
        scheduleSave(id)
    }

    /// Cambia l'endpoint usato da questa chat per l'inferenza. Non
    /// tocca il `ModelState` globale né il modello caricato nel
    /// service — un'altra chat locale può continuare a usare il
    /// transformer corrente mentre questa passa a una remote. Il
    /// cambio prende effetto dal prossimo `send` (il turn corrente,
    /// se in volo, non viene interrotto).
    /// Invalida `encodedTokens` quando si cambia il *tipo* di
    /// endpoint (local↔remote) perché la token stream cached non è
    /// più valida — le rappresentazioni interne dei due backend non
    /// sono compatibili.
    func setEndpoint(_ endpoint: ModelEndpoint?, for id: UUID) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }) else { return }
        let old = conversations[idx].endpoint
        guard old != endpoint else { return }
        // Cambio tipo backend → la cache di token locale è
        // inutile per il backend nuovo, droppiamola così il primo
        // send rifà l'encoding da capo.
        if old?.isRemote != endpoint?.isRemote {
            conversations[idx].encodedTokens = nil
            conversations[idx].lastEncodedMode = nil
        }
        conversations[idx].endpoint = endpoint
        scheduleSave(id)
    }

    /// Endpoint effettivamente usato da questa chat per l'inferenza
    /// (vedi `Conversation.endpoint`). Fallback su `modelState.loadedEndpoint`
    /// per BWC su chat pre-migration. Esposto pubblicamente così la
    /// toolbar / ComposerView / banner-stato possono renderizzare
    /// l'endpoint giusto per la chat selezionata invece di leggere
    /// sempre lo stato globale di ModelState (che ora rappresenta
    /// "il modello locale caricato in RAM", non più "l'endpoint
    /// corrente di tutte le chat").
    func endpoint(of id: UUID) -> ModelEndpoint? {
        guard let c = conversations.first(where: { $0.id == id }) else {
            return modelState.loadedEndpoint
        }
        return c.endpoint ?? modelState.loadedEndpoint
    }

    func delete(_ id: UUID) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }) else { return }
        // Cancella un'eventuale generation in corso per questa
        // chat — altrimenti il Task continuerebbe a girare sul
        // service finché non finisce maxTokens, sprecando RAM/GPU
        // su una conversation che non esiste più.
        cancel(of: id)
        conversations.remove(at: idx)
        phases.removeValue(forKey: id)
        // Best-effort: remove the on-disk files. Both the JSON
        // transcript and the persistent KV cache live next to each
        // other; wipe both so a future conversation reusing this
        // UUID never inherits stale prefill bytes.
        if let url = try? PersistencePaths.conversationURL(id: id) {
            try? FileManager.default.removeItem(at: url)
        }
        if let url = try? PersistencePaths.kvCacheURL(id: id) {
            try? FileManager.default.removeItem(at: url)
        }
        // PR 2: also wipe the v2 per-chat folder if it exists.
        // Single FS call covers manifest, KV cache, pending
        // snapshot, and any turn/round files PR 3 will write here.
        try? chatPersistence.deleteChat(id: id)
        // PR 3a: drop synth/LRU entries for this chat so a recycled
        // UUID can't ever read stale summaries.
        synthCache.removeValue(forKey: id)
        for key in Array(roundLRU.allKeys) where key.chatID == id {
            roundLRU.remove(key)
        }
        if selectedID == id {
            selectedID = conversations.first?.id
        }
    }

    func phase(of id: UUID) -> GenerationPhase {
        phases[id] ?? .idle
    }

    // ----- streaming -----

    /// Send a user message to the currently-selected conversation.
    ///
    /// Fast path: when the conversation already carries an
    /// `encodedTokens` cache produced under the same mode, we only
    /// BPE-encode the new user turn (the "delta": `<User>text<Assistant>
    /// <think_marker>`) and concatenate. The model's prefill therefore
    /// re-runs over the full prompt — that part is unavoidable since
    /// `releaseCache()` is called per turn — but the *tokenization*
    /// of the history happens once and is reused turn after turn.
    /// On mode change or first turn the prompt is rebuilt from
    /// scratch via `tokenizeFullHistory`.
    func send(text: String,
               mode: ThinkingMode,
               options: SamplingOptions,
               maxTokens: Int) {
        guard let id = selectedID,
              let idx = conversations.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        switch phase(of: id) {
        case .streaming, .prefilling: return
        default: break
        }

        // Endpoint effettivo per QUESTA chat. `Conversation.endpoint`
        // è la fonte primaria: permette `chat A locale + chat B
        // remota` in parallelo perché ogni chat decide il proprio
        // backend invece di leggere lo stato globale di ModelState.
        // Fallback sul `loadedEndpoint` globale per le chat
        // pre-migration (campo nil) — al primo send con un
        // ModelState valido il caller può anche fare opportunistic
        // backfill via `setEndpoint(_:for:)`, ma il fallback è
        // sufficiente per la BWC.
        let effectiveEndpoint = conversations[idx].endpoint
            ?? modelState.loadedEndpoint

        // Remote endpoints take a completely different code path —
        // no tokenizer, no KV cache, no fast-delta. Branch early so
        // the local pipeline below operates only on the case it was
        // designed for.
        if case .openRouter(let modelID) = effectiveEndpoint {
            sendRemote(text: trimmed,
                        conversationIndex: idx,
                        provider: .openRouter,
                        modelID: modelID,
                        mode: mode,
                        options: options,
                        maxTokens: maxTokens)
            return
        }
        if case .anthropic(let modelID) = modelState.loadedEndpoint {
            sendRemote(text: trimmed,
                        conversationIndex: idx,
                        provider: .anthropic,
                        modelID: modelID,
                        mode: mode,
                        options: options,
                        maxTokens: maxTokens)
            return
        }

        let modeRaw = mode.rawValue
        let userMessage = StoredMessage(role: .user, content: trimmed)
        conversations[idx].messages.append(userMessage)
        conversations[idx].retitleIfNeeded()
        // Status iniziale: se c'è già un'altra conversation locale
        // attiva, questa generation andrà in coda dietro la q seriale
        // dell'InferenceService. Dirlo all'utente esplicitamente
        // invece di mostrargli "Encoding prompt…" per minuti mentre
        // sta in realtà aspettando.
        let initialStatus: String = otherLocalGenerationInFlight(excluding: id)
            ? "Waiting — another local chat is using the model…"
            : "Encoding prompt…"
        phases[id] = .streaming(buffer: "", status: initialStatus,
                                 metrics: GenerationMetrics())
        // Tool-call continuations (M3b) re-use these — capture them
        // before the Task that drives the model run is spawned so
        // the value is stable for the whole multi-roundtrip turn.
        lastSamplingOptions[id] = (options, maxTokens)
        toolRoundtrips[id] = 0
        scheduleSave(id)

        let placeholderId = UUID()
        conversations[idx].messages.append(
            StoredMessage(id: placeholderId, role: .assistant, content: ""))

        // Snapshot the bits we need off the main actor.
        let cachedPrefix = conversations[idx].encodedTokens
        let cachedMode   = conversations[idx].lastEncodedMode
        let canReusePrefix = (cachedPrefix != nil && cachedMode == modeRaw)
        let historyForFullEncode = conversations[idx].messages.map {
            $0.asKitMessage()
        }
        // Resolve the attached agent (if any) so its system prompt
        // and tool-filter apply to this turn's prompt build. The
        // snapshot is taken now and treated as immutable for the
        // turn — editing the agent mid-stream doesn't retroactively
        // alter the prompt.
        let agentConfig: AgentConfig? = conversations[idx].agentID
            .flatMap { agents.agent(id: $0) }
        let agentSystemPrompt = agentConfig?.systemPrompt

        // Snapshot the MCP tool catalogue once, off the main actor's
        // critical path. Filtered through the agent's allowlist when
        // one is in effect (nil = all tools, empty set = none,
        // explicit set = allowlist of qualified names).
        //
        // The composed JSON also includes a synthetic
        // `__delegate_to_agent` tool whenever there's more than one
        // registered agent the model could hand off to — the
        // attached agent is filtered out so it can't recurse into
        // itself. Snapshot semantics: edits to the AgentLibrary
        // mid-turn don't affect the in-flight generation.
        let delegableAgents = agents.agents.filter { $0.id != agentConfig?.id }
        let toolSchemasJSON = composeToolSchemasJSON(
            mcpAllowed: agentConfig?.allowedToolNames,
            delegableAgents: delegableAgents)

        // First-turn-with-project info: solo la prima user turn
        // di una chat (encodedTokens == nil) paga il costo di
        // costruzione del contesto progetto. Le turn successive
        // estendono il prefisso cached via fast-path delta.
        let projectContext: FirstTurnProjectContext? = {
            guard cachedPrefix == nil,
                  let pid = conversations[idx].projectID,
                  let project = projects.project(id: pid) else { return nil }
            switch project.effectiveContextMode {
            case .indexedContent:
                let docs = documents.documents(for: pid)
                if !docs.isEmpty {
                    var files: [(path: String, tokens: [Int32])] = []
                    files.reserveCapacity(docs.count)
                    for d in docs {
                        guard let toks = try? documents.tokens(of: d.id)
                        else { continue }
                        files.append((path: d.displayPath ?? d.name,
                                      tokens: toks))
                    }
                    return .indexedContent(name: project.name, files: files)
                }
                // Indexed mode ma nessun doc indicizzato (utente non
                // ha mai indicizzato il progetto): fall-through al
                // path inventory così il modello ha comunque
                // qualcosa di concreto su cui ragionare.
                fallthrough
            case .pathsOnly:
                let inventory = ProjectInventoryBuilder.build(
                    project,
                    maxFiles: project.effectiveMaxInventoryFiles)
                return inventory.entries.isEmpty
                    ? nil
                    : .pathsOnly(inventory: inventory)
            }
        }()

        // Cancella un'eventuale entry rimasta (sicurezza per
        // dev-loop: in produzione la gate `phase(of: id)` sopra
        // garantisce che non ci sia già un task per questa conv).
        generationTasks[id]?.cancel()
        let task = Task { [weak self] in
            defer {
                // Cleanup dell'entry per questa conversation alla
                // fine del run (success/throw/cancel). Defer
                // garantisce esecuzione anche se il body solleva o
                // viene cancellato. Siamo su @MainActor (ereditato
                // da ChatStore), quindi accesso diretto alla mappa.
                self?.generationTasks.removeValue(forKey: id)
            }
            guard let self else { return }
            do {
                let promptTokens = try await self.buildPromptTokens(
                    canReusePrefix: canReusePrefix,
                    cachedPrefix: cachedPrefix,
                    userText: trimmed,
                    mode: mode,
                    fullHistory: historyForFullEncode,
                    projectContext: projectContext,
                    toolSchemasJSON: toolSchemasJSON,
                    agentSystemPrompt: agentSystemPrompt)
                // Bail-out se l'utente ha cancellato mentre stavamo
                // aspettando la tokenize (la q seriale può tenerci
                // bloccati per la durata di un altro generate).
                // Throw così cade nel catch CancellationError che
                // resetta il phase a .idle invece di lasciarlo nello
                // status "Waiting…".
                if Task.isCancelled { throw CancellationError() }
                // Snapshot the pending turn on disk immediately. If
                // the app dies during the prefill (which can take
                // minutes for a project context), the user's prompt
                // is preserved and the resume path can still pick
                // up — without any tokens emitted yet.
                if let cIdx = self.conversations.firstIndex(where: { $0.id == id }) {
                    self.conversations[cIdx].pendingTurn = PendingTurn(
                        assistantMessageID: placeholderId,
                        promptTokens: promptTokens,
                        generatedTokens: [],
                        mode: modeRaw)
                    self.scheduleSave(id)
                }
                for try await event in self.service.generateForConversation(
                    promptTokens: promptTokens,
                    conversationID: id,
                    mode: mode,
                    options: options,
                    maxTokens: maxTokens)
                {
                    self.apply(event: event,
                                to: id,
                                placeholderId: placeholderId,
                                userMessageId: userMessage.id,
                                mode: modeRaw)
                }
            } catch is CancellationError {
                // Cancellazione esplicita: phase passa a .idle, non
                // .error — l'utente ha chiesto stop, non c'è un
                // errore da mostrare.
                self.phases[id] = .idle
                self.scheduleSave(id)
            } catch {
                self.phases[id] = .error((error as? LocalizedError)?.errorDescription
                                     ?? error.localizedDescription)
                self.scheduleSave(id)
            }
        }
        generationTasks[id] = task
    }

    /// True se almeno una conversation locale (esclusa `excluding`,
    /// di solito quella che stiamo per avviare) sta correndo nel
    /// `service` o è in coda davanti. Usato dal `send` per scegliere
    /// lo status iniziale del phase (chiaro vs "in attesa"). La q
    /// dell'InferenceService è seriale: due locali sullo stesso
    /// modello serializzano per forza. Le remote girano via HTTP
    /// — non contendono nulla, quindi non contano.
    private func otherLocalGenerationInFlight(excluding id: UUID) -> Bool {
        for (otherID, task) in generationTasks {
            guard otherID != id, !task.isCancelled else { continue }
            // Filtra solo gli active phase: una entry in
            // `generationTasks` può persistere per qualche istante
            // dopo che il Task body è uscito (prima che il `defer`
            // rimuova la chiave), e quei task non bloccano più la
            // q. Controlla il phase per essere sicuri.
            switch phases[otherID] ?? .idle {
            case .idle, .error:
                continue
            case .prefilling, .streaming:
                // Solo i local endpoint passano per la q seriale.
                // Le remote (OpenRouter) girano via HTTP, non
                // bloccano nulla. Leggiamo l'endpoint effettivo
                // (Conversation.endpoint con fallback global)
                // perché post-refactor multi-endpoint i due tipi
                // coesistono nella stessa app.
                let otherEp = self.endpoint(of: otherID)
                if case .localDirectory = otherEp {
                    return true
                }
            }
        }
        return false
    }

    /// Restart a generation that was cut short (crash, app-quit,
    /// power loss). The conversation already carries a `pendingTurn`
    /// snapshot describing the prompt the model saw and every token
    /// sampled before the interruption; we feed that prompt+tokens
    /// back into the model as a fresh prefill so the KV cache is
    /// rebuilt, then the decode loop picks up from
    /// `startPos == promptTokens.count + generatedTokens.count`.
    ///
    /// Sampling parameters are taken from current Settings (we
    /// don't persist them per-turn) — the partial text already on
    /// screen ensures resumed sampling stays consistent enough that
    /// the user can keep reading without seeing a regressed style.
    func resumePendingTurn(of id: UUID,
                             options: SamplingOptions,
                             maxTokens: Int) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }),
              let pt = conversations[idx].pendingTurn else { return }
        switch phase(of: id) {
        case .streaming, .prefilling: return
        default: break
        }
        // Re-derive ThinkingMode from the persisted raw value. Falls
        // back to .chat for forward-compat (a future enum case we
        // don't recognise yet shouldn't lock the user out of
        // resuming an interrupted reply).
        let mode = ThinkingMode(rawValue: pt.mode) ?? .chat
        // Stitch prompt + already-generated ids into one prompt:
        // the model will prefill the whole thing, the decode loop
        // then samples token #(generatedTokens.count + 1).
        let resumedPrompt = pt.promptTokens + pt.generatedTokens

        // Restore the live streaming UI state from the persisted
        // partial content. Tokens-per-minute resets to zero — we
        // don't carry it across restarts.
        let buffer: String
        if let mIdx = conversations[idx].messages.firstIndex(
            where: { $0.id == pt.assistantMessageID }) {
            buffer = conversations[idx].messages[mIdx].content
        } else {
            buffer = ""
        }
        phases[id] = .streaming(buffer: buffer,
                                 status: "Resuming…",
                                 metrics: GenerationMetrics())

        let placeholderId = pt.assistantMessageID
        let modeRaw = pt.mode
        // The most recent user message — used by `apply` to stamp
        // tokenCount once the turn finishes.
        let userMessageId = conversations[idx].messages
            .last(where: { $0.role == .user })?.id ?? placeholderId

        Task {
            do {
                for try await event in service.generateForConversation(
                    promptTokens: resumedPrompt,
                    conversationID: id,
                    mode: mode,
                    options: options,
                    maxTokens: maxTokens)
                {
                    apply(event: event,
                           to: id,
                           placeholderId: placeholderId,
                           userMessageId: userMessageId,
                           mode: modeRaw)
                }
            } catch {
                phases[id] = .error((error as? LocalizedError)?.errorDescription
                                     ?? error.localizedDescription)
                scheduleSave(id)
            }
        }
    }

    /// Carries the snapshotted project context across the actor hop
    /// into `buildPromptTokens`. Constructed only when the very
    /// first turn of a chat with an attached project is about to
    /// fire (cachedPrefix == nil, projectID != nil).
    ///
    /// Due varianti:
    /// - `.indexedContent`: vecchio comportamento — splice di token
    ///   pre-calcolati per ogni documento indicizzato (richiede
    ///   docs in `DocumentLibrary`).
    /// - `.pathsOnly`: nuovo default — albero gerarchico di path
    ///   nel system prompt; il modello esplora con tool.
    private enum FirstTurnProjectContext {
        case indexedContent(name: String,
                             files: [(path: String, tokens: [Int32])])
        case pathsOnly(inventory: ProjectInventory)
    }

    /// Assemble the BPE prompt:
    ///   - fast path: extend cached prefix with the new turn's delta;
    ///   - first turn with a project attached: emit native repo / file
    ///     delimiter ids and splice in each file's pre-tokenized stream;
    ///   - cold path: full re-encode of the entire history.
    ///
    /// `toolSchemasJSON` and `agentSystemPrompt` only affect the
    /// non-fast-path branches — those produce the system block
    /// from scratch. The delta fast path inherits both from the
    /// cached prefix.
    private func buildPromptTokens(canReusePrefix: Bool,
                                    cachedPrefix: [Int32]?,
                                    userText: String,
                                    mode: ThinkingMode,
                                    fullHistory: [Message],
                                    projectContext: FirstTurnProjectContext?,
                                    toolSchemasJSON: String?,
                                    agentSystemPrompt: String?
    ) async throws -> [Int32] {
        if canReusePrefix, let prefix = cachedPrefix {
            guard let delta = await service.tokenizeUserTurnDelta(
                userText, mode: mode)
            else {
                throw NSError(domain: "ChatStore", code: 10, userInfo: [
                    NSLocalizedDescriptionKey:
                        "Tokenizer unavailable (no model loaded?)."
                ])
            }
            return prefix + delta
        }
        // First turn con un progetto attaccato: scegli la
        // strategia in base alla modalità del progetto.
        if let ctx = projectContext {
            switch ctx {
            case .indexedContent(let name, let files):
                // Path legacy: splice nativo dei token pre-calcolati.
                let history = fullHistory.dropLast()
                let transcriptSystem = history
                    .first(where: { $0.role == .system })?
                    .content ?? ""
                let systemText: String
                if let agent = agentSystemPrompt,
                   !agent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                {
                    systemText = transcriptSystem.isEmpty
                        ? agent
                        : agent + "\n\n" + transcriptSystem
                } else {
                    systemText = transcriptSystem
                }
                if let tokens = await service.tokenizeFirstTurnWithProject(
                    systemText: systemText,
                    projectName: name,
                    files: files,
                    userText: userText,
                    mode: mode,
                    toolSchemasJSON: toolSchemasJSON)
                {
                    return tokens
                }
                // Fall through al cold path se i token speciali
                // non sono risolvibili.

            case .pathsOnly(let inventory):
                // Path nuovo default: prepend l'albero al system
                // prompt e usa il cold path standard. Il modello
                // dovrà invocare `read`/`glob`/`grep` per leggere
                // i singoli file.
                let inventoryBlock = inventory.renderTree()
                let history = fullHistory.dropLast()
                let transcriptSystem = history
                    .first(where: { $0.role == .system })?
                    .content ?? ""
                let baseSystem: String
                if let agent = agentSystemPrompt?
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                   !agent.isEmpty
                {
                    baseSystem = transcriptSystem.isEmpty
                        ? agent
                        : agent + "\n\n" + transcriptSystem
                } else {
                    baseSystem = transcriptSystem
                }
                let combinedSystem: String = baseSystem.isEmpty
                    ? inventoryBlock
                    : inventoryBlock + "\n" + baseSystem
                guard let tokens = await service.tokenizeFullHistory(
                    Array(history), mode: mode,
                    toolSchemasJSON: toolSchemasJSON,
                    systemPromptOverride: combinedSystem)
                else {
                    throw NSError(domain: "ChatStore", code: 11, userInfo: [
                        NSLocalizedDescriptionKey:
                            "Tokenizer unavailable (no model loaded?)."
                    ])
                }
                return tokens
            }
        }
        // Cold path / mode changed: full re-encode of the entire
        // conversation. Drops the last message (the just-appended
        // empty assistant placeholder), then asks for the trailing
        // assistant marker via mode.
        let history = fullHistory.dropLast()
        guard let tokens = await service.tokenizeFullHistory(
            Array(history), mode: mode,
            toolSchemasJSON: toolSchemasJSON,
            systemPromptOverride: agentSystemPrompt)
        else {
            throw NSError(domain: "ChatStore", code: 11, userInfo: [
                NSLocalizedDescriptionKey:
                    "Tokenizer unavailable (no model loaded?)."
            ])
        }
        return tokens
    }


    /// Cancella la generation in volo per `id` (default: la chat
    /// attualmente selezionata). Cancella il Task swift — che
    /// termina il `for try await event in …` loop — e alza il flag
    /// per-conv nel service così se il run è già dentro al
    /// prefill/decode esce dopo il token corrente. Le altre chat in
    /// volo (locale + remote) restano intatte.
    func cancel(of id: UUID? = nil) {
        guard let target = id ?? selectedID else { return }
        generationTasks[target]?.cancel()
        service.cancelCurrent(conversationID: target)
    }

    /// User chose to throw away the interrupted assistant turn. The
    /// partial content stays in the transcript (so they don't lose
    /// what they already read), but the model's view of the chat
    /// rolls back to the last completed turn — i.e. `encodedTokens`
    /// is untouched and the next `send` starts fresh.
    func discardPendingTurn(of id: UUID) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }),
              conversations[idx].pendingTurn != nil else { return }
        conversations[idx].pendingTurn = nil
        scheduleSave(id)
    }

    private func apply(event: GenerationEvent,
                        to id: UUID,
                        placeholderId: UUID,
                        userMessageId: UUID,
                        mode: String) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }) else { return }
        switch event {
        case .token(let piece, let tokenId):
            // Once tokens start flowing we leave `.prefilling` if we
            // were still in it (e.g. very small prompts where
            // prefillDone arrived after the first token sample). Carry
            // the metrics forward.
            let m = currentMetrics(of: id)
            let buffer = currentBuffer(of: id)
            let newBuffer = buffer + piece
            phases[id] = .streaming(buffer: newBuffer, status: "", metrics: m)
            if let mIdx = conversations[idx].messages.firstIndex(
                where: { $0.id == placeholderId }) {
                conversations[idx].messages[mIdx].content = newBuffer
            }
            // Append the raw id to the on-disk pendingTurn so a
            // crash mid-stream replays this token exactly. The save
            // is debounced (~500 ms) so we don't pay disk I/O at
            // the model's sampling rate.
            conversations[idx].pendingTurn?.generatedTokens.append(tokenId)
            scheduleSave(id)

        case .status(let s):
            let m = currentMetrics(of: id)
            let buffer = currentBuffer(of: id)
            phases[id] = .streaming(buffer: buffer, status: s, metrics: m)

        case .prefillStart(let promptTokens):
            phases[id] = .prefilling(promptTokens: promptTokens,
                                      startTime: Date())
            // Resetta il prefillTrace sul placeholder: su un resume
            // dopo crash il placeholder rientra con il trace del
            // turn precedente; senza azzeramento i nuovi
            // `.prefillToken` ci appenderebbero sopra producendo un
            // trace duplicato. Sul flusso normale (placeholder
            // appena creato) è un no-op.
            if let mIdx = conversations[idx].messages.firstIndex(
                where: { $0.id == placeholderId }) {
                conversations[idx].messages[mIdx].prefillTrace = nil
            }

        case .prefillToken(let text):
            // Append al `prefillTrace` del placeholder. Cresce
            // live durante il prefill (la UI legge da
            // `placeholder.prefillTrace` per disegnare il blocco
            // grigio collassabile fra l'user e l'assistente).
            // Persistito nel conversation JSON al `.done` via
            // `scheduleSave`, ma scriviamo subito qui per il
            // crash-recovery — il debounce di scheduleSave evita
            // di sgranchire il disco a ogni chunk.
            if let mIdx = conversations[idx].messages.firstIndex(
                where: { $0.id == placeholderId }) {
                let prev = conversations[idx].messages[mIdx].prefillTrace ?? ""
                conversations[idx].messages[mIdx].prefillTrace = prev + text
                scheduleSave(id)
            }

        case .prefillDone(let promptTokens, let elapsed, let tokPerMin):
            var m = currentMetrics(of: id)
            m.promptTokens = promptTokens
            m.prefillElapsed = elapsed
            m.prefillTokPerMin = tokPerMin
            phases[id] = .streaming(buffer: currentBuffer(of: id),
                                     status: "", metrics: m)

        case .generationProgress(let generated, let elapsed, let tokPerMin):
            var m = currentMetrics(of: id)
            m.generatedTokens = generated
            m.generationElapsed = elapsed
            m.generationTokPerMin = tokPerMin
            phases[id] = .streaming(buffer: currentBuffer(of: id),
                                     status: "", metrics: m)

        case .done(let final, let promptTokens, let generatedTokens):
            // Finalize the assistant placeholder with the parsed
            // structure (reasoning + tool calls split out).
            // Preserva l'eventuale prefillTrace accumulato durante
            // il cold prefill — l'init non lo prende dal Message
            // (il kit non lo conosce) quindi va riassegnato dal
            // valore precedente.
            if let mIdx = conversations[idx].messages.firstIndex(
                where: { $0.id == placeholderId }) {
                let prevTrace = conversations[idx].messages[mIdx].prefillTrace
                conversations[idx].messages[mIdx] = StoredMessage(
                    id: placeholderId,
                    role: .assistant,
                    content: final.content,
                    reasoningContent: final.reasoningContent,
                    toolCalls: final.toolCalls.map(StoredToolCall.init),
                    tokenCount: generatedTokens.count,
                    prefillTrace: prevTrace)
            }
            // Stamp the user message with the share of the prompt it
            // contributed. Approximation: prompt length minus the
            // bytes already accounted for by previous messages.
            if let uIdx = conversations[idx].messages.firstIndex(
                where: { $0.id == userMessageId }) {
                let prevPromptTokens = conversations[idx].encodedTokens?.count ?? 0
                conversations[idx].messages[uIdx].tokenCount =
                    max(promptTokens.count - prevPromptTokens, 0)
            }
            // Persist the canonical tokenized history: full prompt
            // plus everything we just sampled. Next turn's send()
            // will append a delta to this — no full re-encode.
            conversations[idx].encodedTokens = promptTokens + generatedTokens
            conversations[idx].lastEncodedMode = mode
            // The generation completed cleanly; drop the crash-
            // recovery snapshot so the UI stops offering Resume.
            conversations[idx].pendingTurn = nil

            // Model asked to call one or more tools? Run them, splice
            // the outputs back into the prompt, and let the model
            // continue. The iteration guard stops the obvious
            // infinite-recursion failure mode (model keeps emitting
            // calls without ever producing a final reply).
            if !final.toolCalls.isEmpty,
               toolRoundtrips[id, default: 0] < maxToolRoundtripsPerTurn {
                toolRoundtrips[id, default: 0] += 1
                // `mode` arrives here as the raw String (so `apply`
                // can be called from `send` and from the resume
                // path with the same signature). Round-trip it
                // back to the enum the continuation needs.
                let modeEnum = ThinkingMode(rawValue: mode) ?? .chat
                runToolCallsAndContinue(
                    conversationID: id,
                    finishedMessageID: placeholderId,
                    calls: final.toolCalls,
                    mode: modeEnum)
                scheduleSave(id)
                return
            }
            toolRoundtrips[id] = nil
            lastSamplingOptions[id] = nil

            phases[id] = .idle
            scheduleSave(id)
        }
    }

    /// Dispatch every tool call the model just emitted to MCPClientPool,
    /// stash the outputs on the originating assistant message, then
    /// build the `<eos>…<tool_outputs>…<Assistant>` delta and feed
    /// the model the result of its own work so it can finish the
    /// reply (or call more tools, bounded by `maxToolRoundtripsPerTurn`).
    private func runToolCallsAndContinue(conversationID id: UUID,
                                          finishedMessageID: UUID,
                                          calls: [ToolCall],
                                          mode: ThinkingMode) {
        // Surface that we're between turns — the buffer is empty
        // (the previous placeholder is finalised), and the next
        // streaming view-state will be created by the
        // generateForConversation stream below.
        phases[id] = .streaming(buffer: "",
                                 status: "Running \(calls.count) tool\(calls.count == 1 ? "" : "s")…",
                                 metrics: currentMetrics(of: id))

        // Capture references the detached Task needs.
        let pool = self.mcpPool

        Task { [weak self] in
            guard let self else { return }
            // Run sequentially — most MCP servers are single-process
            // and an interleaved fan-out doesn't buy much. Errors
            // are flattened into the output text by `invokeQualified`
            // so this loop never throws.
            // Seed the cycle-prevention chain with whatever agent
            // currently drives this conversation — that way a
            // delegated sub can't loop back and delegate to the
            // host itself. The chain is then extended at each
            // nesting level by `runSubAgentToCompletion`.
            let hostAgentID = self.conversations
                .first(where: { $0.id == id })?.agentID

            var outputs: [String] = []
            outputs.reserveCapacity(calls.count)
            for call in calls {
                let result = await self.executeToolCall(
                    call,
                    conversationID: id,
                    hostAgentID: hostAgentID)
                outputs.append(result)
            }

            // Persist the outputs on the assistant turn that asked
            // for them — that's what the chat template's
            // tool_outputs block hangs off.
            if let cIdx = self.conversations.firstIndex(where: { $0.id == id }),
               let mIdx = self.conversations[cIdx].messages.firstIndex(
                where: { $0.id == finishedMessageID })
            {
                self.conversations[cIdx].messages[mIdx].toolOutputs = outputs
            }

            // Build the delta to feed the model. Same shape as the
            // user-turn fast path, but instead of `<User>...` we
            // splice the tool_outputs block between `<eos>` and the
            // re-opened `<Assistant>` turn. The cache prefix already
            // includes everything up to (and including) the call-
            // emitting assistant content, so cacheImage will match
            // and the model continues with only the new delta
            // tokens to chew on.
            guard let cachedPrefix = self.conversations.first(where: { $0.id == id })?
                .encodedTokens else {
                self.phases[id] = .error(
                    "Tool-call continuation: missing cached prefix")
                self.toolRoundtrips[id] = nil
                self.scheduleSave(id)
                return
            }
            guard let delta = await self.service.tokenizeToolOutputsDelta(
                callNames: calls.map(\.name),
                outputs: outputs,
                mode: mode)
            else {
                self.phases[id] = .error(
                    "Tokenizer unavailable while building tool-output delta")
                self.toolRoundtrips[id] = nil
                self.scheduleSave(id)
                return
            }
            let promptTokens = cachedPrefix + delta

            // Append a fresh placeholder for the model's next reply
            // and re-arm pendingTurn so a crash here is recoverable.
            let newPlaceholderId = UUID()
            if let cIdx = self.conversations.firstIndex(where: { $0.id == id }) {
                self.conversations[cIdx].messages.append(
                    StoredMessage(id: newPlaceholderId,
                                   role: .assistant,
                                   content: ""))
                self.conversations[cIdx].pendingTurn = PendingTurn(
                    assistantMessageID: newPlaceholderId,
                    promptTokens: promptTokens,
                    generatedTokens: [],
                    mode: mode.rawValue)
                self.scheduleSave(id)
            }

            // Reuse the same sampling parameters the user picked for
            // this turn (captured in `lastSamplingOptions` at the
            // start of `send`). Fallback to sensible-but-tame
            // defaults if somehow the slot was cleared mid-flight.
            let stored = self.lastSamplingOptions[id]
            let opts = stored?.opts ?? SamplingOptions(
                temperature: 0.7,
                topK: 0, topP: 1.0,
                repetitionPenalty: 1.0)
            let maxTok = stored?.maxTokens ?? 4096
            do {
                for try await event in self.service.generateForConversation(
                    promptTokens: promptTokens,
                    conversationID: id,
                    mode: mode,
                    options: opts,
                    maxTokens: maxTok)
                {
                    self.apply(event: event,
                                to: id,
                                placeholderId: newPlaceholderId,
                                userMessageId: finishedMessageID,
                                mode: mode.rawValue)
                }
            } catch {
                self.phases[id] = .error(
                    (error as? LocalizedError)?.errorDescription
                    ?? error.localizedDescription)
                self.toolRoundtrips[id] = nil
                self.scheduleSave(id)
            }
        }
    }

    /// Build the JSON tools array that goes into the system block:
    /// every connected MCP tool the agent is allowed to see, plus a
    /// synthetic `__delegate_to_agent` schema when there's at least
    /// one other agent the model could hand work off to.
    ///
    /// Returns nil when the resulting set is empty (no MCP tools
    /// allowed AND no other agents) so the chat template can skip
    /// the tools block entirely.
    /// Esegue una tool call invocata dal modello. Tre branch:
    ///   - `__delegate_to_agent` → spawn sub-agent;
    ///   - `native__<name>` → `nativeTools.dispatch` (lo strip del
    ///     prefisso è l'unica cosa che separa il nome wire dal nome
    ///     registro);
    ///   - tutto il resto → `mcpPool.invokeQualified` (deve avere
    ///     forma `server__tool` o l'MCP pool rifiuta).
    /// Centralizza il routing così i tre call site
    /// (runToolCallsAndContinue, runRemoteLoop, runSubAgentToCompletionInner)
    /// non duplicano la logica.
    private func executeToolCall(_ call: ToolCall,
                                  conversationID: UUID,
                                  hostAgentID: UUID?) async -> String {
        if call.name == EncodingDSV4.delegateToolName {
            return await executeSubAgentDelegation(
                argsJSON: call.args,
                hostAgentID: hostAgentID,
                hostConvID: conversationID)
        }
        if call.name.hasPrefix("native__") {
            let nativeName = String(call.name.dropFirst("native__".count))
            return await invokeNativeTool(
                name: nativeName,
                argsJSON: call.args,
                conversationID: conversationID)
        }
        // Tool name che non match nessuno dei branch supportati e
        // non ha il `__` separator MCP. Senza questa guardia,
        // `mcpPool.invokeQualified` ritorna un errore generico
        // "[error: tool name not qualified as <server>__<tool>: X]"
        // che è criptico per il modello. Qui invece elenchiamo
        // esplicitamente quali nomi avrebbe potuto usare così
        // l'output del tool message è auto-esplicativo e il modello
        // recupera invece di ripetere lo stesso errore.
        if call.name.range(of: "__") == nil {
            return unknownToolError(call.name)
        }
        return await mcpPool.invokeQualified(call.name, argsJSON: call.args)
    }

    /// Costruisce un messaggio "tool not found" auto-descrittivo che
    /// elenca i nomi qualificati che il modello dovrebbe usare al
    /// posto dello sbagliato. Tornare un errore ricco invece del
    /// silenzio (o di un errore generico) è ciò che rompe il loop
    /// agentico quando il modello rinomina i tool a casaccio.
    private func unknownToolError(_ name: String) -> String {
        let nativeNames = nativeTools.schemas
            .map { "native__\($0.name)" }
            .sorted()
        let mcpNames = mcpPool.allTools()
            .map { $0.qualifiedName }
            .sorted()
        var hint = "Available tool names:"
        if !nativeNames.isEmpty {
            hint += " " + nativeNames.joined(separator: ", ")
        }
        if !mcpNames.isEmpty {
            hint += (nativeNames.isEmpty ? " " : ", ")
                + mcpNames.joined(separator: ", ")
        }
        hint += ", __delegate_to_agent."
        return "[error: tool '\(name)' is not registered. \(hint)]"
    }

    /// Decodifica gli args JSON, risolve mode + rootDirectory dalla
    /// chat (Project attaccato → primo sourcePath; fallback → home),
    /// invoca `NativeToolHost.dispatch` e ritorna l'output testuale
    /// per la tool_outputs block. Errori (JSON malformed, esecuzione
    /// fallita) tornano come stringa con marker `[error: ...]` così
    /// il modello può recuperare invece di crashare.
    private func invokeNativeTool(name: String,
                                    argsJSON: String,
                                    conversationID: UUID) async -> String {
        let input: [String: Any]
        if argsJSON.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            input = [:]
        } else if let data = argsJSON.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data)
                    as? [String: Any] {
            input = obj
        } else {
            return "[error: native tool '\(name)' got malformed JSON args]"
        }

        let conv = conversations.first(where: { $0.id == conversationID })
        let agent = conv?.agentID.flatMap { agents.agent(id: $0) }
        let mode = agent?.agentMode ?? .build
        let rootDir: URL = {
            if let pid = conv?.projectID,
               let project = projects.project(id: pid),
               let root = ProjectRootBuilder.ensureBuilt(project) {
                return root
            }
            return FileManager.default.homeDirectoryForCurrentUser
        }()

        let out = await nativeTools.dispatch(
            name: name, input: input, mode: mode, rootDirectory: rootDir)
        if out.isError {
            return "[error: \(out.output)]"
        }
        // Output vuoto = la tool è andata a buon fine ma non ha
        // trovato niente (glob con pattern malformato, grep senza
        // match, read di un file vuoto, ecc.). Senza un marker il
        // modello vede una stringa vuota e non capisce se la tool
        // ha funzionato — di solito ritenta con varianti e blocca
        // il loop agentico. Un marker esplicito gli dice "operazione
        // OK, zero risultati" così può cambiare strategia.
        let trimmed = out.output.trimmingCharacters(
            in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return "(empty output / 0 results — tool '\(name)' " +
                "ran successfully but produced no content; " +
                "consider whether the arguments are correct)"
        }
        return out.output
    }

    private func composeToolSchemasJSON(
        mcpAllowed: Set<String>?,
        delegableAgents: [AgentConfig]
    ) -> String? {
        var schemas: [[String: Any]] = []

        // Native tools (DeepSeekTools): read/write/edit/grep/shell/glob/
        // apply_patch/task/todo/plan/web_fetch/web_search/lsp/repo_*.
        // Prefisso `native__<name>` per non collidere con tool MCP che
        // potrebbero avere nomi corti uguali; il dispatch site
        // (runToolCallsAndContinue, runSubAgentToCompletionInner,
        // runRemoteLoop) controlla il prefisso e instrada al
        // `nativeTools.dispatch` invece che a `mcpPool.invokeQualified`.
        // Filtro per `mcpAllowed`: se l'allowlist è non-nil ma vuota
        // la chat è "tools off" → niente nativi né MCP. Altrimenti i
        // nomi sono inclusi se l'allowlist li contiene esplicitamente
        // come `native__<name>`, oppure se non c'è allowlist.
        if !(mcpAllowed?.isEmpty == true) {
            let nativeSchemas = nativeTools.schemas
            for native in nativeSchemas {
                let qualified = "native__\(native.name)"
                if let allowed = mcpAllowed,
                   !allowed.contains(qualified) { continue }
                schemas.append([
                    "name": qualified,
                    "description": native.description,
                    "inputSchema": native.inputSchema.foundationValue
                ])
            }
        }

        // MCP tools (already filtered by agent allowlist semantics
        // when `mcpAllowed` is non-nil).
        if let allowed = mcpAllowed, allowed.isEmpty {
            // explicit "no tools" set — skip MCP entirely
        } else {
            for tool in mcpPool.allTools() {
                if let allowed = mcpAllowed,
                   !allowed.contains(tool.qualifiedName) { continue }
                schemas.append([
                    "name": tool.qualifiedName,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                ])
            }
        }

        // Native tools (TODO §8 follow-up). The host's
        // `schemas` already include only tools the registry made
        // available for the current `AgentMode`; we prefix the
        // public name with `native__` so the dispatch side can
        // tell native from MCP without ambiguity. Agent allowlist
        // filtering of native tools is a separate follow-up.
        for s in nativeTools.schemas {
            schemas.append([
                "name": "native__\(s.name)",
                "description": s.description,
                "inputSchema": s.inputSchema.foundationValue,
            ])
        }

        if !delegableAgents.isEmpty {
            let roster = delegableAgents
                .map { "- \($0.name): \($0.summary.isEmpty ? "(no summary)" : $0.summary)" }
                .joined(separator: "\n")
            schemas.append([
                "name": EncodingDSV4.delegateToolName,
                "description":
                    """
                    Delegate a focused sub-task to another agent. The named agent will run independently with its own system prompt and produce a single textual reply that becomes this tool's output. Use it when a sub-task is better handled by a specialist agent. Available agents:
                    \(roster)
                    """,
                "inputSchema": [
                    "type": "object",
                    "properties": [
                        "agent_name": [
                            "type": "string",
                            "description": "Exact name of the agent to invoke."
                        ],
                        "task": [
                            "type": "string",
                            "description": "Self-contained instructions for the sub-agent. Include everything it needs — it doesn't see this conversation's history."
                        ]
                    ],
                    "required": ["agent_name", "task"]
                ]
            ])
        }

        if schemas.isEmpty { return nil }
        guard let data = try? JSONSerialization.data(
            withJSONObject: schemas,
            options: [.prettyPrinted, .sortedKeys])
        else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Host-level entry point: invoked from the chat's tool-call
    /// loop when the model emits `__delegate_to_agent`. Seeds the
    /// chain with the host's own agentID so cycles back through
    /// the host are refused, then hands off to `dispatchDelegation`
    /// — the same helper sub-agents use for their own (nested)
    /// delegate calls. `hostConvID` is forwarded so the UI's
    /// `activeDelegations` stack is keyed against the right
    /// conversation as the chain pushes / pops frames.
    private func executeSubAgentDelegation(argsJSON: String,
                                            hostAgentID: UUID?,
                                            hostConvID: UUID) async -> String {
        let initialChain: [UUID] = hostAgentID.map { [$0] } ?? []
        return await dispatchDelegation(argsJSON: argsJSON,
                                          depth: 1,
                                          chain: initialChain,
                                          hostConvID: hostConvID)
    }

    // MARK: - delegation UI frame helpers

    /// Push a fresh `DelegationFrame` onto the conversation's
    /// stack and return its id so the caller can later update
    /// the buffer + pop the frame from the same actor hop.
    private func pushDelegationFrame(_ frame: DelegationFrame,
                                       hostConvID: UUID) {
        activeDelegations[hostConvID, default: []].append(frame)
    }

    /// Register a cancellation closure for a freshly-pushed frame.
    /// Called by the inner runner once it's wrapped its decode loop
    /// in a `Task`. The closure typically just calls
    /// `task.cancel()`; the loop checks `Task.isCancelled` between
    /// yielded tokens (or via the AsyncThrowingStream
    /// `onTermination` hook).
    private func registerDelegationCancellation(
        frameID: UUID, cancel: @escaping () -> Void)
    {
        delegationCancellations[frameID] = cancel
    }

    /// User-facing entry point for the chain UI's per-frame Stop
    /// button (TODO §4 follow-up). Cancels the matching sub-agent
    /// run and records the frame id in `cancelledDelegations` so
    /// the persisted reply gets a `[cancelled]` marker.
    public func cancelDelegation(frameID: UUID) {
        cancelledDelegations.insert(frameID)
        if let cancel = delegationCancellations[frameID] {
            cancel()
        }
    }

    /// Effective agent catalog for the active conversation
    /// (TODO §11). When the chat is attached to a project that
    /// carries a `.deepseek/agents.json` overlay, those entries
    /// override / extend the global library. Returns the global
    /// list unchanged when no project / no overlay file is present.
    public func effectiveAgents() -> [AgentConfig] {
        guard let conv = selectedConversation,
              let projectID = conv.projectID,
              let project = projects.project(id: projectID),
              let firstPath = project.sourcePaths.first
        else { return agents.agents }
        let overlay = ProjectOverlayLoader.load(
            rootDirectory: URL(fileURLWithPath: firstPath))
        return ProjectOverlayLoader.mergeAgents(
            global: agents.agents, overlay: overlay.agents)
    }

    /// Snapshot the current project's overlay (or an empty one
    /// rooted at $HOME if no project is attached). Helper for UI
    /// surfaces that want to introspect what came from the
    /// project vs. the global library.
    func currentProjectOverlay() -> ProjectOverlay {
        guard let conv = selectedConversation,
              let projectID = conv.projectID,
              let project = projects.project(id: projectID),
              let firstPath = project.sourcePaths.first
        else {
            return .empty(rootDirectory: URL(fileURLWithPath: NSHomeDirectory()))
        }
        return ProjectOverlayLoader.load(
            rootDirectory: URL(fileURLWithPath: firstPath))
    }

    private func appendDelegationBuffer(hostConvID: UUID,
                                         frameID: UUID,
                                         text: String) {
        guard var stack = activeDelegations[hostConvID],
              let idx = stack.firstIndex(where: { $0.id == frameID })
        else { return }
        stack[idx].buffer.append(text)
        activeDelegations[hostConvID] = stack
    }

    /// Clean up the cancellation registry when a frame pops. The
    /// closure isn't retained past the frame's lifetime so the
    /// Task it points at can deinit cleanly.
    private func popDelegationCancellation(frameID: UUID) {
        delegationCancellations.removeValue(forKey: frameID)
        cancelledDelegations.remove(frameID)
    }

    private func popDelegationFrame(hostConvID: UUID, frameID: UUID) {
        activeDelegations[hostConvID]?.removeAll { $0.id == frameID }
        if activeDelegations[hostConvID]?.isEmpty == true {
            activeDelegations[hostConvID] = nil
        }
    }

    /// Parse a delegation tool-call payload (`{ agent_name, task }`),
    /// resolve the target agent, enforce both the structural
    /// depth cap and the chain-membership cycle check, and run the
    /// resolved agent through `runSubAgentToCompletion`. Used by
    /// both the host (via `executeSubAgentDelegation`) and the
    /// nested case (sub-agent's loop discovers a delegate call in
    /// `runSubAgentToCompletionInner`).
    ///
    /// Errors are flattened into "[error: …]" strings instead of
    /// thrown so the in-loop tool-output splice always has data
    /// to feed the model. The model can self-correct on a refused
    /// delegation by trying a different agent or just answering
    /// directly.
    private func dispatchDelegation(argsJSON: String,
                                     depth: Int,
                                     chain: [UUID],
                                     hostConvID: UUID) async -> String {
        if depth > maxDelegationDepth {
            return "[error: max delegation depth (\(maxDelegationDepth)) reached]"
        }
        guard let data = argsJSON.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return "[error: malformed delegation arguments]"
        }
        let agentName = (obj["agent_name"] as? String) ?? ""
        let task = (obj["task"] as? String) ?? ""
        guard !agentName.isEmpty, !task.isEmpty else {
            return "[error: delegation requires non-empty agent_name and task]"
        }
        guard let agent = agents.agents.first(where: { $0.name == agentName }) else {
            return "[error: no agent named '\(agentName)' is registered]"
        }
        if chain.contains(agent.id) {
            return "[error: agent '\(agentName)' is already in the delegation chain — refused to avoid a cycle]"
        }
        return await runSubAgentToCompletion(
            agent: agent, task: task,
            depth: depth, chain: chain,
            hostConvID: hostConvID)
    }

    /// Drive a sub-agent to completion, wrapped in a snapshot /
    /// restore pair so the caller's KV cache survives the
    /// delegation intact. The snapshot path is best-effort — if
    /// `beginDelegation` returns nil (no model loaded, somehow)
    /// the run still happens, just without the preservation
    /// benefit.
    ///
    /// `depth` is the level this sub-agent runs at (1 = first
    /// hop from the host). `chain` is the list of agentIDs
    /// already live on the call stack; the wrapper extends it
    /// with `agent.id` before passing to the inner so the
    /// cycle-prevention check covers the sub-agent itself too.
    ///
    /// All exit paths from the inner loop go back through
    /// `endDelegation` so a thrown / early-returned sub-agent
    /// doesn't leak the snapshot in `InferenceService.savedDelegations`.
    private func runSubAgentToCompletion(agent: AgentConfig,
                                          task: String,
                                          depth: Int,
                                          chain: [UUID],
                                          hostConvID: UUID) async -> String {
        let snapToken = await service.beginDelegation()
        let result = await runSubAgentToCompletionInner(
            agent: agent, task: task,
            depth: depth, chain: chain + [agent.id],
            hostConvID: hostConvID)
        if let token = snapToken {
            await service.endDelegation(token)
        }
        return result
    }

    /// Inner loop — see `runSubAgentToCompletion` for the wrapper
    /// that owns the snapshot/restore around it. Keep this function
    /// pure in the sense that every exit returns a String (no
    /// throw, no continuation leak) so the wrapper can always pair
    /// its `endDelegation`.
    ///
    /// `chain` here already includes this sub-agent's id (the
    /// wrapper appended it). When the depth is still under the
    /// cap, the sub-agent gets its own `__delegate_to_agent`
    /// schema with the roster of agents *not* in the chain — so
    /// it can hand off further, but never back through an
    /// already-active agent.
    private func runSubAgentToCompletionInner(agent: AgentConfig,
                                                task: String,
                                                depth: Int,
                                                chain: [UUID],
                                                hostConvID: UUID) async -> String {
        let mode = ThinkingMode(rawValue: agent.defaultMode) ?? .chat
        let opts = SamplingOptions(
            temperature: Float(min(1.0, max(0.5, agent.temperature))),
            topK: agent.topK,
            topP: Float(agent.topP),
            minP: Float(agent.minP),
            tailFree: Float(agent.tailFree),
            typical: Float(agent.typical),
            repetitionPenalty: Float(agent.repetitionPenalty),
            frequencyPenalty: Float(agent.frequencyPenalty),
            presencePenalty: Float(agent.presencePenalty),
            mirostatTau: Float(agent.mirostatTau),
            mirostatEta: Float(agent.mirostatEta),
            mirostatMu: Float(2.0 * agent.mirostatTau))

        // Push a live UI frame so the user sees the delegation
        // unfolding. The frame stays until this function exits
        // (popped in the defer below); buffer grows on every
        // token event.
        let frame = DelegationFrame(
            agentID: agent.id,
            agentName: agent.name,
            agentIconName: agent.iconName,
            agentTint: agent.tint,
            task: task,
            depth: depth)
        let frameID = frame.id
        pushDelegationFrame(frame, hostConvID: hostConvID)
        defer { popDelegationFrame(hostConvID: hostConvID, frameID: frameID) }

        // Composed schema: sub-agent's allowed MCP tools, plus the
        // delegate tool (with a roster of every agent not already
        // on the call stack) when we haven't hit the depth cap.
        let delegableAgents: [AgentConfig] = (depth < maxDelegationDepth)
            ? agents.agents.filter { !chain.contains($0.id) }
            : []
        let toolJson = composeToolSchemasJSON(
            mcpAllowed: agent.allowedToolNames,
            delegableAgents: delegableAgents)

        guard var promptTokens = await service.tokenizeFullHistory(
            [Message(role: .user, content: task)],
            mode: mode,
            toolSchemasJSON: toolJson,
            systemPromptOverride: agent.systemPrompt)
        else {
            return "[error: tokenizer unavailable]"
        }

        let subID = UUID() // isolated cacheImage namespace
        let maxIterations = 21
        var iteration = 0
        var lastContent = ""

        while iteration < maxIterations {
            iteration += 1

            var finalMessage: Message?
            var capturedPrompt: [Int32] = []
            var capturedGenerated: [Int32] = []

            do {
                for try await event in service.generateForConversation(
                    promptTokens: promptTokens,
                    conversationID: subID,
                    mode: mode,
                    options: opts,
                    maxTokens: agent.maxTokens)
                {
                    switch event {
                    case .token(let text, _):
                        // Stream every decoded token into the live
                        // UI frame so the user can watch the
                        // sub-agent write its reply in real time.
                        appendDelegationBuffer(
                            hostConvID: hostConvID,
                            frameID: frameID,
                            text: text)
                    case .done(let f, let p, let g):
                        finalMessage = f
                        capturedPrompt = p
                        capturedGenerated = g
                    default:
                        break
                    }
                }
            } catch {
                return "[error: sub-agent generation failed: \((error as? LocalizedError)?.errorDescription ?? error.localizedDescription)]"
            }

            guard let final = finalMessage else {
                return "[sub-agent generation didn't finalize]"
            }
            lastContent = final.content

            if final.toolCalls.isEmpty {
                return lastContent
            }

            // Sub-agent emitted tool calls — execute them. Route
            // delegate calls back through `dispatchDelegation`
            // (recursive nesting, bounded by depth + chain) and
            // everything else through the MCP pool.
            var outputs: [String] = []
            outputs.reserveCapacity(final.toolCalls.count)
            for call in final.toolCalls {
                let out: String
                if call.name == EncodingDSV4.delegateToolName {
                    // Recursive sub-agent: tracking depth + chain
                    // qui non è riutilizzabile da `executeToolCall`
                    // (che è chiamato solo dal turn host), quindi
                    // teniamo il branch dedicato.
                    out = await dispatchDelegation(
                        argsJSON: call.args,
                        depth: depth + 1,
                        chain: chain,
                        hostConvID: hostConvID)
                } else if call.name.hasPrefix("native__") {
                    let nativeName = String(
                        call.name.dropFirst("native__".count))
                    out = await invokeNativeTool(
                        name: nativeName,
                        argsJSON: call.args,
                        conversationID: hostConvID)
                } else if call.name.range(of: "__") == nil {
                    // Stesso fallback dell'host path: nome senza
                    // qualificatore = tool non registrato → errore
                    // ricco con elenco dei nomi disponibili così il
                    // modello recupera invece di girare in tondo.
                    out = unknownToolError(call.name)
                } else {
                    out = await mcpPool.invokeQualified(
                        call.name, argsJSON: call.args)
                }
                outputs.append(out)
            }

            guard let delta = await service.tokenizeToolOutputsDelta(
                callNames: final.toolCalls.map(\.name),
                outputs: outputs,
                mode: mode)
            else {
                return "[error: failed to build tool-output delta in sub-agent]"
            }
            promptTokens = capturedPrompt + capturedGenerated + delta
        }

        if !lastContent.isEmpty {
            return lastContent
                + "\n\n[note: sub-agent hit \(maxIterations) tool-call iterations and was cut off]"
        }
        return "[sub-agent hit \(maxIterations) tool-call iterations without producing a final reply]"
    }

    // MARK: - Remote (OpenRouter) send path

    /// Companion to `send` for chats whose loaded endpoint is a
    /// remote OpenAI-compatible API. Different enough from the
    /// local path that it gets its own function: no tokenizer
    /// access, no KV cache image, no fast-delta arithmetic. The
    /// public surface that `apply` exposes (`phases[id]`, the
    /// placeholder message lifecycle, `pendingTurn`) is mimicked
    /// here so the chat UI doesn't have to special-case remote
    /// chats at the view layer.
    /// Which remote backend `sendRemote` / `runRemoteLoop` dispatch
    /// to. Threaded through so both the Keychain account and the
    /// streaming client pick the right provider — same shape on the
    /// consumer side (`OpenAIStreamChunk`) so the iteration loop
    /// stays unified.
    enum RemoteProvider {
        case openRouter
        case anthropic

        fileprivate var keychainAccount: String {
            switch self {
            case .openRouter: return KeychainAccount.openRouterAPIKey
            case .anthropic:  return KeychainAccount.anthropicAPIKey
            }
        }

        fileprivate var displayName: String {
            switch self {
            case .openRouter: return "OpenRouter"
            case .anthropic:  return "Anthropic"
            }
        }
    }

    private func sendRemote(text: String,
                              conversationIndex idx: Int,
                              provider: RemoteProvider,
                              modelID: String,
                              mode: ThinkingMode,
                              options: SamplingOptions,
                              maxTokens: Int) {
        let id = conversations[idx].id

        // API key is read at send time so a key rotation in
        // Settings takes effect on the next turn without
        // restarting the app.
        let apiKey = KeychainStore.get(account: provider.keychainAccount) ?? ""
        guard !apiKey.isEmpty else {
            phases[id] = .error(
                "\(provider.displayName) API key not configured. "
                + "Add it from Settings → API Keys.")
            return
        }

        let userMessage = StoredMessage(role: .user, content: text)
        let placeholder = StoredMessage(role: .assistant, content: "")
        conversations[idx].messages.append(userMessage)
        conversations[idx].messages.append(placeholder)
        conversations[idx].retitleIfNeeded()
        phases[id] = .streaming(buffer: "",
                                 status: "Calling \(modelID)…",
                                 metrics: GenerationMetrics())
        // TODO §4 follow-up: persist remote-side pendingTurn so a
        // crash mid-call surfaces a retry affordance on next launch.
        // Cleared in `finalizeRemoteIteration` and in the catch
        // branch of `runRemoteLoop`.
        conversations[idx].remotePendingTurn = RemotePendingTurn(
            assistantMessageID: placeholder.id,
            userMessageID: userMessage.id,
            userText: text,
            mode: mode.rawValue,
            issuedAt: Date())
        lastSamplingOptions[id] = (options, maxTokens)
        toolRoundtrips[id] = 0
        scheduleSave(id)

        let userMessageID = userMessage.id
        let firstPlaceholderID = placeholder.id

        // Stesso pattern del local-send: registriamo il Task nella
        // mappa per permettere cancellazione granulare e cleanup
        // automatico. Cancella un eventuale residuo per questa
        // conv prima di sostituire (dovrebbe essere già nil per la
        // gate sopra, ma è una rete di sicurezza per dev-loop).
        generationTasks[id]?.cancel()
        let task = Task { [weak self] in
            defer {
                self?.generationTasks.removeValue(forKey: id)
            }
            guard let self else { return }
            await self.runRemoteLoop(conversationID: id,
                                       initialPlaceholderID: firstPlaceholderID,
                                       userMessageID: userMessageID,
                                       provider: provider,
                                       modelID: modelID,
                                       mode: mode,
                                       options: options,
                                       maxTokens: maxTokens,
                                       apiKey: apiKey)
        }
        generationTasks[id] = task
    }

    /// Drive a remote chat to completion through (up to)
    /// `maxToolRoundtripsPerTurn` HTTP round-trips. Each iteration
    /// snapshots the conversation history, builds the OpenAI
    /// request, streams the response into the current placeholder,
    /// and — if the model emitted tool calls — executes them
    /// through `mcpPool`, appends a fresh placeholder, and loops.
    /// Bails out cleanly on errors / cap hit / no more tool calls.
    private func runRemoteLoop(conversationID id: UUID,
                                 initialPlaceholderID: UUID,
                                 userMessageID: UUID,
                                 provider: RemoteProvider,
                                 modelID: String,
                                 mode: ThinkingMode,
                                 options: SamplingOptions,
                                 maxTokens: Int,
                                 apiKey: String) async {
        let openRouterClient = OpenRouterClient()
        let anthropicClient = AnthropicClient()
        var currentPlaceholderID = initialPlaceholderID
        var iteration = 0

        // Costruisci l'inventario di progetto una sola volta per
        // tutta la durata della loop tool-roundtrip. Per il path
        // remoto la modalità `indexedContent` non è applicabile —
        // un modello remoto non può ricevere splice di token native
        // — quindi si fa sempre fallback a `pathsOnly`.
        let projectInventory: ProjectInventory? = await MainActor.run {
            guard let idx = self.conversations.firstIndex(where: { $0.id == id }),
                  let pid = self.conversations[idx].projectID,
                  let project = self.projects.project(id: pid)
            else { return nil }
            let inventory = ProjectInventoryBuilder.build(
                project,
                maxFiles: project.effectiveMaxInventoryFiles)
            return inventory.entries.isEmpty ? nil : inventory
        }

        while iteration < maxToolRoundtripsPerTurn {
            iteration += 1

            // Snapshot what the next HTTP call needs from the
            // store, on the main actor — the conversation could
            // have been mutated between iterations.
            let snapshot: (history: [StoredMessage],
                           agent: AgentConfig?)? = await MainActor.run {
                guard let idx = self.conversations.firstIndex(where: { $0.id == id })
                else { return nil }
                let history = Array(self.conversations[idx]
                    .messages.dropLast())  // drop the in-progress placeholder
                let agent = self.conversations[idx].agentID
                    .flatMap { self.agents.agent(id: $0) }
                return (history, agent)
            }
            guard let snapshot else { return }

            let openAIMessages = self.buildOpenAIMessages(
                history: snapshot.history,
                agent: snapshot.agent,
                projectInventory: projectInventory)
            // Stessa logica del local path (vedi `send` →
            // `composeToolSchemasJSON`): l'agent attivo viene filtrato
            // fuori dalla roster delegabile per evitare di mandarsi
            // a se stesso. Senza un agent attaccato, tutti gli agent
            // registrati sono potenzialmente delegabili. ChatStore è
            // @MainActor → accesso sincrono diretto.
            let delegableAgents = self.agents.agents
                .filter { $0.id != snapshot.agent?.id }
            let toolsArray = self.composeOpenAITools(
                mcpAllowed: snapshot.agent?.allowedToolNames,
                delegableAgents: delegableAgents)

            var body: [String: Any] = [
                "model": modelID,
                "messages": openAIMessages.map { $0.toJSON() },
                "temperature": Double(options.temperature),
                "top_p": Double(options.topP),
                "max_tokens": maxTokens,
                // OpenRouter: ask for usage stats in the final chunk
                // so the cost banner has data without a second call.
                "usage": ["include": true]
            ]
            if options.topK > 0 { body["top_k"] = options.topK }
            if let toolsArray, !toolsArray.isEmpty {
                body["tools"] = toolsArray
                body["tool_choice"] = "auto"
            }
            switch mode {
            case .max:  body["reasoning"] = ["effort": "high"]
            case .high: body["reasoning"] = ["effort": "medium"]
            case .chat: break
            }

            // Prompt trace remoto: dump del body JSON inviato al
            // provider — system message, tools array (formato OpenAI
            // o Anthropic), tool_choice/tools, messages (history +
            // tool outputs delle iterazioni precedenti), sampler.
            // Ogni iterazione del loop ha il suo placeholder fresco
            // e quindi il suo dump dedicato, così l'utente può
            // verificare lungo la timeline cosa il modello ha
            // effettivamente visto a ogni passo (in particolare: il
            // marker "(empty output)" appare nei tool messages delle
            // iterazioni che seguono una tool call senza risultati).
            // In background per non ritardare la HTTP request. Per
            // Anthropic dumpiamo la forma OpenAI-shape `body` perché
            // mantiene 1:1 le info (system + messages + tools +
            // sampler) senza dover esporre l'API Anthropic dettagliata
            // — il trace è per debugging, non per wire-match.
            let bodyForTrace = body
            Task { [weak self] in
                await self?.emitRemotePromptTrace(
                    conversationID: id,
                    placeholderID: currentPlaceholderID,
                    body: bodyForTrace)
            }

            // Stream + accumulate. Both providers return the same
            // `OpenAIStreamChunk` shape — for Anthropic, that's a
            // local translation inside `AnthropicClient` — so the
            // accumulator below is provider-agnostic.
            let stream: AsyncThrowingStream<OpenAIStreamChunk, Error>
            switch provider {
            case .openRouter:
                stream = openRouterClient.streamChatCompletion(
                    apiKey: apiKey, body: body)
            case .anthropic:
                let anthropicBody = AnthropicMessageBuilder.buildBody(
                    model: modelID,
                    maxTokens: maxTokens,
                    history: snapshot.history,
                    agentSystem: snapshot.agent?.systemPrompt,
                    tools: AnthropicMessageBuilder.translateTools(toolsArray),
                    temperature: options.temperature,
                    topP: options.topP)
                stream = anthropicClient.streamMessages(
                    apiKey: apiKey, body: anthropicBody)
            }
            var contentBuf = ""
            var reasoningBuf = ""
            var toolCallsAccum: [Int: (id: String?, name: String, args: String)] = [:]
            var usage: OpenAIUsage?
            let startedAt = Date()
            var lastProgressAt = startedAt
            var generatedTokens = 0

            do {
                for try await chunk in stream {
                    if let choice = chunk.choices.first, let delta = choice.delta {
                        if let c = delta.content, !c.isEmpty {
                            contentBuf.append(c)
                            generatedTokens += 1
                            await MainActor.run {
                                self.updateRemoteBuffer(
                                    conversationID: id, buffer: contentBuf)
                            }
                            let now = Date()
                            if now.timeIntervalSince(lastProgressAt) > 0.5 {
                                lastProgressAt = now
                                let elapsed = now.timeIntervalSince(startedAt)
                                let tpm = elapsed > 0
                                    ? Double(generatedTokens) / elapsed * 60.0
                                    : 0
                                await MainActor.run {
                                    self.updateRemoteProgress(
                                        conversationID: id,
                                        generated: generatedTokens,
                                        elapsed: elapsed,
                                        tokPerMin: tpm)
                                }
                            }
                        }
                        if let r = delta.reasoningContent, !r.isEmpty {
                            reasoningBuf.append(r)
                            // TODO §4 follow-up: push the running
                            // thinking buffer to the UI bubble so the
                            // reasoning is visible mid-stream instead
                            // of only at .done.
                            let snapshotReasoning = reasoningBuf
                            await MainActor.run {
                                self.updateRemoteReasoningBuffer(
                                    conversationID: id,
                                    reasoning: snapshotReasoning)
                            }
                        }
                        if let tcs = delta.toolCalls {
                            for tc in tcs {
                                var existing = toolCallsAccum[tc.index]
                                    ?? (id: nil, name: "", args: "")
                                if let cid = tc.id { existing.id = cid }
                                if let n = tc.function?.name { existing.name.append(n) }
                                if let a = tc.function?.arguments { existing.args.append(a) }
                                toolCallsAccum[tc.index] = existing
                            }
                        }
                    }
                    if let u = chunk.usage { usage = u }
                }
            } catch {
                let msg = (error as? LocalizedError)?.errorDescription
                    ?? error.localizedDescription
                await MainActor.run {
                    self.phases[id] = .error(msg)
                    self.toolRoundtrips[id] = nil
                    self.lastSamplingOptions[id] = nil
                    self.scheduleSave(id)
                }
                return
            }

            let toolCalls: [ToolCall] = toolCallsAccum
                .sorted { $0.key < $1.key }
                .map { (_, v) in
                    ToolCall(name: v.name,
                              args: v.args.isEmpty ? "{}" : v.args,
                              id: v.id)
                }
            let final = Message(
                role: .assistant,
                content: contentBuf,
                reasoningContent: reasoningBuf.isEmpty ? nil : reasoningBuf,
                toolCalls: toolCalls,
                toolOutputs: [])
            let capturedUsage = usage

            // Finalize this iteration's placeholder + record cost
            // on the main actor. When toolCalls is empty we also
            // mark the chat idle and exit the loop.
            let isFinal = final.toolCalls.isEmpty
            await MainActor.run {
                self.finalizeRemoteIteration(
                    conversationID: id,
                    placeholderID: currentPlaceholderID,
                    userMessageID: userMessageID,
                    final: final,
                    usage: capturedUsage,
                    isFinal: isFinal)
            }
            if isFinal { return }

            // Execute tool calls. Delegation isn't supported on
            // remote chats (yet) — that case returns a structured
            // error string so the model can self-correct on the
            // next iteration. MCP calls route through the same
            // pool the local path uses. Native calls (TODO §8)
            // route to NativeToolHost.dispatch via the prefix
            // check.
            var outputs: [String] = []
            outputs.reserveCapacity(final.toolCalls.count)
            // Snapshot dell'host agent ID per la cycle prevention
            // della delegation (lo stesso check del local path).
            // `executeToolCall` instrada internamente delegation /
            // native__* / MCP via il loro prefisso, quindi anche il
            // remote loop usa lo stesso router. ChatStore è
            // @MainActor → accesso sincrono diretto.
            let hostAgentID = self.conversations
                .first(where: { $0.id == id })?.agentID
            for call in final.toolCalls {
                let result = await self.executeToolCall(
                    call,
                    conversationID: id,
                    hostAgentID: hostAgentID)
                outputs.append(result)
            }

            // Store outputs on the just-finalised placeholder and
            // append a fresh placeholder for the next iteration's
            // streaming target.
            let nextPlaceholderID = await MainActor.run { () -> UUID in
                guard let idx = self.conversations.firstIndex(where: { $0.id == id })
                else { return UUID() }
                if let mIdx = self.conversations[idx].messages.firstIndex(
                    where: { $0.id == currentPlaceholderID })
                {
                    self.conversations[idx].messages[mIdx].toolOutputs = outputs
                }
                let next = StoredMessage(role: .assistant, content: "")
                self.conversations[idx].messages.append(next)
                self.phases[id] = .streaming(
                    buffer: "",
                    status: "Calling \(modelID) again with tool results…",
                    metrics: GenerationMetrics())
                self.scheduleSave(id)
                return next.id
            }
            currentPlaceholderID = nextPlaceholderID
        }

        // Cap hit — surface the truncation via a banner-style
        // error string so the user knows why the model stopped.
        await MainActor.run {
            self.phases[id] = .error(
                "Reached \(self.maxToolRoundtripsPerTurn) tool-call iterations without a final reply — the model may be looping.")
            self.toolRoundtrips[id] = nil
            self.lastSamplingOptions[id] = nil
            self.scheduleSave(id)
        }
    }

    private func updateRemoteBuffer(conversationID id: UUID, buffer: String) {
        guard case .streaming(_, let reasoning, let status, let metrics) =
                phases[id] else { return }
        phases[id] = .streaming(buffer: buffer, reasoningBuffer: reasoning,
                                 status: status, metrics: metrics)
    }

    /// TODO §4 follow-up: live reasoning content in the bubble.
    /// Called by `runRemoteLoop` whenever the upstream emits more
    /// `reasoning_content` so the UI can render the running
    /// thinking buffer before the turn finalizes at `.done`. Local
    /// generation can't separate think tokens mid-stream, so this
    /// path is remote-only today.
    private func updateRemoteReasoningBuffer(conversationID id: UUID,
                                               reasoning: String) {
        guard case .streaming(let buffer, _, let status, let metrics) =
                phases[id] else { return }
        phases[id] = .streaming(buffer: buffer, reasoningBuffer: reasoning,
                                 status: status, metrics: metrics)
    }

    private func updateRemoteProgress(conversationID id: UUID,
                                       generated: Int,
                                       elapsed: TimeInterval,
                                       tokPerMin: Double) {
        guard case .streaming(let buffer, let reasoning, _, var metrics) =
                phases[id] else { return }
        metrics.generatedTokens = generated
        metrics.generationElapsed = elapsed
        metrics.generationTokPerMin = tokPerMin
        phases[id] = .streaming(buffer: buffer, reasoningBuffer: reasoning,
                                 status: "Streaming…", metrics: metrics)
    }

    /// Write the finalised assistant turn back to the store,
    /// stamp token counts + cost on the relevant messages, bump
    /// the conversation's cumulativeCostUSD, and either mark the
    /// chat idle (`isFinal == true`) or leave it in a transient
    /// streaming state ready for the next loop iteration.
    private func finalizeRemoteIteration(conversationID id: UUID,
                                          placeholderID: UUID,
                                          userMessageID: UUID,
                                          final: Message,
                                          usage: OpenAIUsage?,
                                          isFinal: Bool) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }) else { return }
        if let mIdx = conversations[idx].messages.firstIndex(
            where: { $0.id == placeholderID })
        {
            // Preserva il prefillTrace accumulato durante lo
            // streaming remote (dump del body OpenRouter). L'init
            // dello StoredMessage non lo prende dal `Message` del
            // kit — il backend remoto non lo conosce — quindi va
            // letto dal valore precedente e ripassato esplicitamente
            // così non viene azzerato dalla riscrittura del
            // placeholder.
            let prevTrace = conversations[idx].messages[mIdx].prefillTrace
            conversations[idx].messages[mIdx] = StoredMessage(
                id: placeholderID,
                role: .assistant,
                content: final.content,
                reasoningContent: final.reasoningContent,
                toolCalls: final.toolCalls.map(StoredToolCall.init),
                tokenCount: usage?.completionTokens,
                toolOutputs: nil,
                prefillTrace: prevTrace)
        }
        if let uIdx = conversations[idx].messages.firstIndex(
            where: { $0.id == userMessageID })
        {
            // Stamp prompt-token count on the user message that
            // kicked off this turn — only the first iteration's
            // usage carries the real prompt size; later iterations
            // re-bill the same prompt + tool outputs, but the user
            // message hasn't changed so we keep the first value.
            if conversations[idx].messages[uIdx].tokenCount == nil {
                conversations[idx].messages[uIdx].tokenCount = usage?.promptTokens
            }
        }
        // Track cost: add to conversation total, surface this
        // iteration's cost in the metrics for the ThroughputBar.
        if let cost = usage?.totalCost {
            conversations[idx].cumulativeCostUSD =
                (conversations[idx].cumulativeCostUSD ?? 0) + cost
        }
        // Reset transient fast-path state — remote never uses it.
        conversations[idx].encodedTokens = nil
        conversations[idx].lastEncodedMode = nil
        conversations[idx].pendingTurn = nil
        if isFinal {
            var metrics = currentMetrics(of: id)
            metrics.turnCostUSD = usage?.totalCost
            phases[id] = .streaming(buffer: final.content,
                                     status: "",
                                     metrics: metrics)
            toolRoundtrips[id] = nil
            lastSamplingOptions[id] = nil
            // TODO §4 follow-up: the remote turn finished, drop
            // the recovery breadcrumb.
            conversations[idx].remotePendingTurn = nil
            phases[id] = .idle
        }
        scheduleSave(id)
    }

    /// Translate the active MCP tool catalogue (filtered through
    /// the agent's allowlist, if any) into the OpenAI-shape
    /// `tools` array OpenRouter expects in the request body.
    /// Returns nil when no tools should be sent — keeps the JSON
    /// minimal and avoids triggering tool-aware paths on the
    /// provider side for chats that don't need them.
    ///
    /// Delegation isn't included here: cross-backend sub-agent
    /// invocation would need a remote sub-agent loop that doesn't
    /// exist yet. A `__delegate_to_agent` schema would just lure
    /// the model into calls we have to refuse with an error.
    /// Emette il body JSON di una OpenRouter / Anthropic request
    /// nel campo `prefillTrace` del placeholder, a chunk per dare
    /// l'effetto streaming come per il prefill trace locale. Gated
    /// dallo stesso flag `showPrefillTrace`. Reset del trace al
    /// start così un retry/iterazione sovrascrive invece di
    /// appendere. Si esegue su un Task in background (non blocca
    /// l'HTTP) — il dump è solo informativo e la response arriva
    /// comunque.
    private func emitRemotePromptTrace(conversationID: UUID,
                                         placeholderID: UUID,
                                         body: [String: Any]) async {
        let traceFlag = UserDefaults.standard.object(
            forKey: AppSettingsKey.showPrefillTrace) as? Bool ?? true
        guard traceFlag else { return }

        let pretty: String? = {
            // .sortedKeys per output deterministico (utile per
            // confrontare due trace). .withoutEscapingSlashes evita
            // che gli URL dentro al body diventino illeggibili.
            let opts: JSONSerialization.WritingOptions =
                [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            guard let data = try? JSONSerialization.data(
                withJSONObject: body, options: opts) else { return nil }
            return String(data: data, encoding: .utf8)
        }()
        guard let text = pretty else { return }

        // Reset on this placeholder prima di scrivere (no doppione
        // se per qualche motivo entriamo qua due volte).
        await MainActor.run {
            if let cIdx = self.conversations.firstIndex(where: { $0.id == conversationID }),
               let mIdx = self.conversations[cIdx].messages.firstIndex(
                where: { $0.id == placeholderID }) {
                self.conversations[cIdx].messages[mIdx].prefillTrace = nil
            }
        }

        let chars = Array(text)
        let chunkSize = 40
        var i = 0
        while i < chars.count {
            if Task.isCancelled { break }
            let end = min(i + chunkSize, chars.count)
            let chunk = String(chars[i..<end])
            await MainActor.run {
                guard let cIdx = self.conversations.firstIndex(where: { $0.id == conversationID }),
                      let mIdx = self.conversations[cIdx].messages.firstIndex(
                        where: { $0.id == placeholderID })
                else { return }
                let prev = self.conversations[cIdx].messages[mIdx].prefillTrace ?? ""
                self.conversations[cIdx].messages[mIdx].prefillTrace = prev + chunk
                self.scheduleSave(conversationID)
            }
            i = end
            // ~4 ms fra chunk — totalizza ~1 s per body da 10 KB.
            // Trascurabile vs latenza HTTP (centinaia di ms al
            // first-byte del provider).
            try? await Task.sleep(nanoseconds: 4_000_000)
        }
    }

    /// Costruisce l'array `tools` nel formato OpenAI canonico
    /// (`{type: "function", function: {name, description, parameters}}`)
    /// che OpenRouter / Anthropic inietta nel system prompt
    /// server-side. Specchio remoto di `composeToolSchemasJSON` per
    /// il local DSV4 — stessi input, output diverso. Include native
    /// tools (prefisso `native__`), MCP tools (prefisso
    /// `server__`), e il sintetico `__delegate_to_agent` quando ci
    /// sono altri agent invocabili; senza quest'ultimo, un chat
    /// remota con agent configurati non saprebbe come delegare e
    /// l'asimmetria local↔remote sarebbe visibile.
    private func composeOpenAITools(
        mcpAllowed: Set<String>?,
        delegableAgents: [AgentConfig]
    ) -> [[String: Any]]? {
        var schemas: [[String: Any]] = []

        // Native tools (DeepSeekTools) — wrappate in OpenAI function
        // format. Stesso prefisso `native__<name>` del path locale così
        // i dispatch site sanno instradarle a `nativeTools.dispatch`
        // invece che a `mcpPool.invokeQualified`. JSONValue ha bisogno
        // di `foundationValue` per restituire `[String: Any]` ad
        // JSONSerialization.
        let allEmpty = (mcpAllowed?.isEmpty == true)
        if !allEmpty {
            for native in nativeTools.schemas {
                let qualified = "native__\(native.name)"
                if let allowed = mcpAllowed,
                   !allowed.contains(qualified) { continue }
                schemas.append([
                    "type": "function",
                    "function": [
                        "name": qualified,
                        "description": native.description,
                        "parameters": native.inputSchema.foundationValue
                    ]
                ])
            }
        }

        // MCP tools (filtered by the attached agent's allowlist
        // when one is in effect — same precedence the local path
        // uses in composeToolSchemasJSON).
        let mcpAllEmpty = (mcpAllowed?.isEmpty == true)
        if !mcpAllEmpty {
            for tool in mcpPool.allTools() {
                if let allowed = mcpAllowed,
                   !allowed.contains(tool.qualifiedName) {
                    continue
                }
                schemas.append([
                    "type": "function",
                    "function": [
                        "name": tool.qualifiedName,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    ]
                ])
            }
        }

        // Sub-agent delegation. Mirrors composeToolSchemasJSON:1075
        // — same name, same description shape, same parameters; only
        // the wrapping differs (OpenAI `function` block vs DSV4
        // `inputSchema` field). Il dispatch site in `runRemoteLoop`
        // routes `call.name == EncodingDSV4.delegateToolName`
        // attraverso `executeSubAgentDelegation`.
        if !delegableAgents.isEmpty {
            let roster = delegableAgents
                .map { "- \($0.name): \($0.summary.isEmpty ? "(no summary)" : $0.summary)" }
                .joined(separator: "\n")
            schemas.append([
                "type": "function",
                "function": [
                    "name": EncodingDSV4.delegateToolName,
                    "description":
                        """
                        Delegate a focused sub-task to another agent. The named agent will run independently with its own system prompt and produce a single textual reply that becomes this tool's output. Use it when a sub-task is better handled by a specialist agent. Available agents:
                        \(roster)
                        """,
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "agent_name": [
                                "type": "string",
                                "description": "Exact name of the agent to invoke."
                            ],
                            "task": [
                                "type": "string",
                                "description": "Clear, self-contained description of what the sub-agent should do."
                            ]
                        ],
                        "required": ["agent_name", "task"]
                    ]
                ]
            ])
        }

        return schemas.isEmpty ? nil : schemas
    }

    /// Convert this app's stored transcript into the OpenAI-style
    /// `messages` array OpenRouter expects. Agent system prompt
    /// (when an agent is attached) is prepended to whatever
    /// transcript-side system message(s) the chat carries.
    /// Assistant turns with `toolCalls` emit the tool_calls field;
    /// each `toolOutput` becomes a follow-up `{role: "tool",
    /// tool_call_id, content}` message so the upstream model can
    /// thread the answer back to its original call.
    private func buildOpenAIMessages(history: [StoredMessage],
                                       agent: AgentConfig?,
                                       projectInventory: ProjectInventory? = nil)
        -> [OpenAIMessage]
    {
        var out: [OpenAIMessage] = []

        let agentSystem = agent?.systemPrompt
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let transcriptSystem = history
            .filter { $0.role == .system }
            .map(\.content)
            .joined(separator: "\n\n")
        var systemContent = ""
        if let s = agentSystem, !s.isEmpty {
            systemContent = s
        }
        if !transcriptSystem.isEmpty {
            if !systemContent.isEmpty { systemContent += "\n\n" }
            systemContent += transcriptSystem
        }
        // Prepend l'albero dei path del progetto. Il modello dovrà
        // usare i tool (read/glob/grep) per ottenere il contenuto.
        if let inventory = projectInventory {
            let block = inventory.renderTree()
            systemContent = systemContent.isEmpty
                ? block
                : block + "\n" + systemContent
        }
        if !systemContent.isEmpty {
            out.append(OpenAIMessage(role: "system", content: systemContent))
        }

        for msg in history where msg.role != .system {
            switch msg.role {
            case .user:
                out.append(OpenAIMessage(role: "user", content: msg.content))
            case .assistant:
                var m = OpenAIMessage(role: "assistant",
                                        content: msg.content.isEmpty ? nil : msg.content)
                if let r = msg.reasoningContent, !r.isEmpty {
                    m.reasoningContent = r
                }
                if !msg.toolCalls.isEmpty {
                    m.toolCalls = msg.toolCalls.enumerated().map { (i, tc) in
                        OpenAIToolCall(
                            id: tc.id ?? "call_\(msg.id.uuidString)_\(i)",
                            name: tc.name,
                            arguments: tc.args.isEmpty ? "{}" : tc.args)
                    }
                }
                out.append(m)
                if let outputs = msg.toolOutputs {
                    for (i, output) in outputs.enumerated()
                    where i < msg.toolCalls.count {
                        let callID = msg.toolCalls[i].id
                            ?? "call_\(msg.id.uuidString)_\(i)"
                        out.append(OpenAIMessage(
                            role: "tool",
                            content: output,
                            toolCallID: callID))
                    }
                }
            case .system:
                break
            }
        }
        return out
    }

    private func currentMetrics(of id: UUID) -> GenerationMetrics {
        if case .streaming(_, _, _, let m) = phase(of: id) { return m }
        return GenerationMetrics()
    }

    private func currentBuffer(of id: UUID) -> String {
        if case .streaming(let b, _, _, _) = phase(of: id) { return b }
        return ""
    }

    // ----- persistence -----

    private func loadFromDisk() {
        guard let dir = try? PersistencePaths.conversationsDir() else { return }
        let files = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var loaded: [Conversation] = []
        for url in files where url.pathExtension == "json" {
            if let data = try? Data(contentsOf: url),
               let c = try? decoder.decode(Conversation.self, from: data) {
                loaded.append(c)
            }
        }
        loaded.sort { $0.createdAt > $1.createdAt }
        conversations = loaded
    }

    private func scheduleSave(_ id: UUID) {
        pendingSaves[id]?.cancel()
        pendingSaves[id] = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(nanoseconds: UInt64(self.saveDebounce * 1_000_000_000))
            if Task.isCancelled { return }
            await self.flushSave(id)
        }
    }

    private func flushSave(_ id: UUID) async {
        guard let c = conversations.first(where: { $0.id == id }) else { return }
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(c)
            let url = try PersistencePaths.conversationURL(id: id)
            try data.write(to: url, options: .atomic)
        } catch {
            // Surface persistence errors as a status note on the
            // conversation; non-fatal — the in-memory state still
            // reflects everything correctly.
            phases[id] = .error("Save failed: \(error.localizedDescription)")
        }
        // PR 2: dual-write the v2 manifest for any chat that has a
        // per-chat folder. The legacy `{UUID}.json` above remains
        // the source of truth for reads; PR 3 will flip that.
        if PersistencePaths.isV2Chat(id: id) {
            scheduleManifestForV2(c)
        }
    }

    /// Build a v2 `ChatManifest` snapshot of `c`. `turnIDs` stays
    /// empty in PR 2 — turn/round persistence lands in PR 3, when
    /// the synthesizer that derives summaries from `c.messages`
    /// goes live. Until then, the manifest carries only the chat
    /// metadata (title, dates, endpoint, agent/project, cost).
    private func manifest(from c: Conversation) -> ChatManifest {
        ChatManifest(
            id: c.id,
            title: c.title,
            createdAt: c.createdAt,
            modelDirPath: c.modelDirPath,
            endpoint: c.endpoint,
            projectID: c.projectID,
            agentID: c.agentID,
            cumulativeCostUSD: c.cumulativeCostUSD,
            lastEncodedMode: c.lastEncodedMode,
            turnIDs: [],
            schemaVersion: 2)
    }

    /// Stage a debounced manifest dual-write for a v2 chat.
    private func scheduleManifestForV2(_ c: Conversation) {
        chatPersistence.scheduleManifestSave(manifest(from: c))
    }

    // MARK: - v2 lazy-loading API (PR 3a — dead code)
    //
    // These two methods are the read surface the UI will consume in
    // PR 3b: `turns(of:)` returns the per-turn summary list and
    // `loadRound(_:)` fetches one round's full payload. They work
    // for BOTH legacy and v2 chats: legacy chats synthesise summaries
    // from the in-memory `[StoredMessage]` array on every call
    // (memoised by a content fingerprint), v2 chats will read
    // pre-written summary / round files once PR 3b adds the write
    // path. For now the v2 read paths fall back on the same synth
    // so the UI doesn't observe a regression when PR 3b lands.

    /// Per-turn summaries for `chatID`, in chronological order.
    /// Synth-derived from `conversations[idx].messages` for both
    /// legacy and v2 chats in PR 3a; PR 3b will read summary files
    /// for v2 chats directly. Returns an empty array when the chat
    /// id isn't known to the store.
    func turns(of chatID: UUID) -> [TurnSummary] {
        guard let c = conversations.first(where: { $0.id == chatID }) else {
            return []
        }
        let fingerprint = SynthFingerprint(messages: c.messages)
        if let cached = synthCache[chatID],
           cached.fingerprint == fingerprint
        {
            return cached.turns
        }
        let turns = Self.synthesizeTurns(from: c.messages)
        synthCache[chatID] = SynthCacheEntry(
            fingerprint: fingerprint, turns: turns)
        return turns
    }

    /// Full payload for one round. Synth-derived in PR 3a; PR 3b
    /// adds a disk read path keyed on `RoundKey`, with the
    /// `roundLRU` cache in front of it. Returns nil when the
    /// `(chatID, turnID, roundID)` triple isn't present in the
    /// in-memory transcript.
    func loadRound(_ key: RoundKey) -> StoredRound? {
        if let cached = roundLRU.get(key) { return cached }
        guard let c = conversations.first(where: { $0.id == key.chatID })
        else { return nil }
        guard let round = Self.synthesizeRound(
            from: c.messages, turnID: key.turnID, roundID: key.roundID)
        else { return nil }
        return round
    }

    // MARK: - synthesizer (legacy ↔ v2 bridge)

    /// Walk `messages` once and bucket consecutive assistants into
    /// the turn opened by the preceding user (or system) message.
    /// One `TurnSummary` per bucket; each summary's `roundIDs`
    /// captures the assistant `StoredMessage.id`s in order so a
    /// later `loadRound(_:)` can locate the right entry back.
    ///
    /// Mirrors the previous `ChatView.groupedItems` grouping so
    /// existing chats render identically under the new pipeline:
    /// system messages get their own `.isSystem` turn with no
    /// rounds; user messages start a turn whose rounds are filled
    /// by every assistant message until the next user / system.
    nonisolated static func synthesizeTurns(
        from messages: [StoredMessage]
    ) -> [TurnSummary] {
        var out: [TurnSummary] = []
        var buffer: [StoredMessage] = []
        var lead: StoredMessage?

        func flush() {
            guard let leadMsg = lead else {
                // Orphan assistant messages with no preceding user —
                // shouldn't happen in practice but we render them as
                // a synthetic empty-user turn so they don't vanish.
                if !buffer.isEmpty {
                    out.append(makeTurn(
                        lead: StoredMessage(role: .user, content: ""),
                        rounds: buffer))
                    buffer.removeAll(keepingCapacity: true)
                }
                return
            }
            out.append(makeTurn(lead: leadMsg, rounds: buffer))
            buffer.removeAll(keepingCapacity: true)
            lead = nil
        }

        for m in messages {
            switch m.role {
            case .assistant:
                buffer.append(m)
            case .user:
                flush()
                lead = m
            case .system:
                // System messages always stand alone — flush whatever
                // came before, emit the system turn, and reset.
                flush()
                out.append(TurnSummary(
                    id: m.id,
                    createdAt: .now,
                    userMessageID: m.id,
                    userText: m.content,
                    userTokenCount: m.tokenCount,
                    finalContentPreview: "",
                    finalContentIsTruncated: false,
                    roundIDs: [],
                    flags: [.isSystem],
                    toolCallCount: 0,
                    totalGeneratedTokens: 0,
                    turnCostUSD: nil))
            }
        }
        flush()
        return out
    }

    /// Construct one `TurnSummary` from a `(user, assistantRounds)`
    /// pair. Aggregates flags, tool-call count, and a preview of
    /// the final round's content. Used by both `synthesizeTurns`
    /// and (in PR 3b) the live-update path that builds summaries
    /// incrementally during streaming.
    nonisolated static func makeTurn(
        lead: StoredMessage, rounds: [StoredMessage]
    ) -> TurnSummary {
        var flags: TurnFlags = []
        var toolCallCount = 0
        var totalGeneratedTokens = 0
        for r in rounds {
            if let rc = r.reasoningContent, !rc.isEmpty {
                flags.insert(.hasReasoning)
            }
            if !r.toolCalls.isEmpty {
                flags.insert(.hasToolCalls)
                toolCallCount += r.toolCalls.count
                if r.toolCalls.contains(where: {
                    $0.name == EncodingDSV4.delegateToolName
                }) {
                    flags.insert(.hasDelegation)
                }
            }
            if let pt = r.prefillTrace, !pt.isEmpty {
                flags.insert(.hasPrefillTrace)
            }
            if let tc = r.tokenCount { totalGeneratedTokens += tc }
        }
        let finalContent = rounds.last?.content ?? ""
        let (preview, truncated) = Self.previewSlice(finalContent)
        return TurnSummary(
            id: lead.id,
            createdAt: .now,
            userMessageID: lead.id,
            userText: lead.content,
            userTokenCount: lead.tokenCount,
            finalContentPreview: preview,
            finalContentIsTruncated: truncated,
            roundIDs: rounds.map(\.id),
            flags: flags,
            toolCallCount: toolCallCount,
            totalGeneratedTokens: totalGeneratedTokens,
            turnCostUSD: nil)
    }

    /// Look up one round inside a synthesised transcript. The
    /// `turnID` matches the lead user/system message id (the turn's
    /// stable identifier); `roundID` matches the assistant message
    /// id in `roundIDs`. Returns the assistant message rebuilt as a
    /// `StoredRound`, or nil when either id doesn't resolve.
    nonisolated static func synthesizeRound(
        from messages: [StoredMessage],
        turnID: UUID,
        roundID: UUID
    ) -> StoredRound? {
        // Find the lead message + the index in `messages` that opens
        // this turn. Walk forward across consecutive assistants and
        // pick the one whose id matches `roundID`.
        guard let leadIdx = messages.firstIndex(where: { $0.id == turnID })
        else { return nil }
        var i = leadIdx + 1
        var roundIndex = 0
        while i < messages.count, messages[i].role == .assistant {
            if messages[i].id == roundID {
                return Self.makeRound(from: messages[i],
                                       roundIndex: roundIndex)
            }
            roundIndex += 1
            i += 1
        }
        return nil
    }

    nonisolated static func makeRound(
        from m: StoredMessage, roundIndex: Int
    ) -> StoredRound {
        StoredRound(
            id: m.id,
            roundIndex: roundIndex,
            content: m.content,
            reasoningContent: m.reasoningContent,
            toolCalls: m.toolCalls,
            toolOutputs: m.toolOutputs,
            prefillTrace: m.prefillTrace,
            tokenCount: m.tokenCount)
    }

    /// Truncate `content` to roughly 2 KB so the summary stays
    /// light. The cutoff is character-based (not byte-based) to
    /// keep the UI predictable; the small overshoot in UTF-8 byte
    /// count is fine — the file is still kilobyte-scale.
    nonisolated private static func previewSlice(
        _ content: String
    ) -> (String, Bool) {
        let limit = 2048
        if content.count <= limit { return (content, false) }
        let cutoff = content.index(content.startIndex, offsetBy: limit)
        return (String(content[..<cutoff]), true)
    }
}

/// Lightweight content fingerprint of a `[StoredMessage]` array.
/// `Equatable` so the synth memoisation cache can decide whether a
/// re-synth is needed; intentionally NOT a full hash of every
/// message (that would walk the whole transcript on every render).
/// What's captured is enough to detect:
///   * append (count changed),
///   * mutation of the streaming target (last message's id + its
///     content length changed),
///   * tool-output landing on the last assistant (last message's
///     toolOutputs count changed).
struct SynthFingerprint: Equatable {
    let count: Int
    let lastID: UUID?
    let lastContentCount: Int
    let lastToolOutputsCount: Int

    init(messages: [StoredMessage]) {
        self.count = messages.count
        self.lastID = messages.last?.id
        self.lastContentCount = messages.last?.content.count ?? 0
        self.lastToolOutputsCount =
            messages.last?.toolOutputs?.count ?? 0
    }
}
