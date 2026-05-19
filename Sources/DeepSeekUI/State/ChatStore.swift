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
    case streaming(buffer: String, status: String, metrics: GenerationMetrics)
    case error(String)
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
    /// when the load step gated the chat UI; in-chat picking
    /// turns it into a derived value.
    var modelDirPath: String {
        service.currentModelDir()?.path ?? ""
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
    /// Per-conversation count of how many tool-call -> generate
    /// continuations have fired since the latest user turn. Cleared
    /// the moment a `.done` arrives without tool calls (= final
    /// reply) or the cap is hit. The cap prevents the obvious
    /// failure mode of a model that keeps calling tools forever.
    private var toolRoundtrips: [UUID: Int] = [:]
    private let maxToolRoundtripsPerTurn: Int = 8

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
    /// Sampler parameters captured at the start of `send`; reused
    /// by `runToolCallsAndContinue` so a tool-output continuation
    /// inherits the same temperature / topK / topP / maxTokens
    /// the user picked for the current turn.
    private var lastSamplingOptions: [UUID: (opts: SamplingOptions,
                                              maxTokens: Int)] = [:]

    init(service: InferenceService,
         documents: DocumentLibrary,
         projects: ProjectLibrary,
         mcpPool: MCPClientPool,
         agents: AgentLibrary,
         modelState: ModelState) {
        self.service = service
        self.documents = documents
        self.projects = projects
        self.mcpPool = mcpPool
        self.agents = agents
        self.modelState = modelState
        loadFromDisk()
        if conversations.isEmpty {
            let first = Conversation(modelDirPath: modelDirPath)
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
        let c = Conversation(modelDirPath: modelDirPath)
        conversations.insert(c, at: 0)
        selectedID = c.id
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

        // Remote endpoints take a completely different code path —
        // no tokenizer, no KV cache, no fast-delta. Branch early so
        // the local pipeline below operates only on the case it was
        // designed for.
        if case .openRouter(let modelID) = modelState.loadedEndpoint {
            sendRemote(text: trimmed,
                        conversationIndex: idx,
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
                // Solo i local model usano la q dell'InferenceService.
                // Le chat remote (modelDirPath vuoto) girano via HTTP
                // — non bloccano nulla.
                if let c = conversations.first(where: { $0.id == otherID }),
                   !c.modelDirPath.isEmpty {
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
                let result: String
                if call.name == EncodingDSV4.delegateToolName {
                    // Synthetic "delegate to another agent" tool.
                    // Handled in-process (no MCP server involved) by
                    // spawning a sub-agent run.
                    result = await self.executeSubAgentDelegation(
                        argsJSON: call.args,
                        hostAgentID: hostAgentID,
                        hostConvID: id)
                } else {
                    result = await pool.invokeQualified(
                        call.name, argsJSON: call.args)
                }
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
    private func composeToolSchemasJSON(
        mcpAllowed: Set<String>?,
        delegableAgents: [AgentConfig]
    ) -> String? {
        var schemas: [[String: Any]] = []

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

    private func appendDelegationBuffer(hostConvID: UUID,
                                         frameID: UUID,
                                         text: String) {
        guard var stack = activeDelegations[hostConvID],
              let idx = stack.firstIndex(where: { $0.id == frameID })
        else { return }
        stack[idx].buffer.append(text)
        activeDelegations[hostConvID] = stack
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
        let maxIterations = 8
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
                    out = await dispatchDelegation(
                        argsJSON: call.args,
                        depth: depth + 1,
                        chain: chain,
                        hostConvID: hostConvID)
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
    private func sendRemote(text: String,
                              conversationIndex idx: Int,
                              modelID: String,
                              mode: ThinkingMode,
                              options: SamplingOptions,
                              maxTokens: Int) {
        let id = conversations[idx].id

        // API key is read at send time so a key rotation in
        // Settings takes effect on the next turn without
        // restarting the app.
        let apiKey = KeychainStore.get(account: KeychainAccount.openRouterAPIKey) ?? ""
        guard !apiKey.isEmpty else {
            phases[id] = .error(
                "OpenRouter API key not configured. Add it from Settings → API Keys.")
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
                                 modelID: String,
                                 mode: ThinkingMode,
                                 options: SamplingOptions,
                                 maxTokens: Int,
                                 apiKey: String) async {
        let client = OpenRouterClient()
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
            let toolsArray = self.composeOpenAITools(
                mcpAllowed: snapshot.agent?.allowedToolNames)

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

            // Stream + accumulate.
            let stream = client.streamChatCompletion(
                apiKey: apiKey, body: body)
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
            // pool the local path uses.
            var outputs: [String] = []
            outputs.reserveCapacity(final.toolCalls.count)
            for call in final.toolCalls {
                let result: String
                if call.name == EncodingDSV4.delegateToolName {
                    result = "[error: cross-agent delegation is not yet supported on remote models]"
                } else {
                    result = await self.mcpPool.invokeQualified(
                        call.name, argsJSON: call.args)
                }
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
        guard case .streaming(_, let status, let metrics) = phases[id] else { return }
        phases[id] = .streaming(buffer: buffer, status: status, metrics: metrics)
    }

    private func updateRemoteProgress(conversationID id: UUID,
                                       generated: Int,
                                       elapsed: TimeInterval,
                                       tokPerMin: Double) {
        guard case .streaming(let buffer, _, var metrics) = phases[id] else { return }
        metrics.generatedTokens = generated
        metrics.generationElapsed = elapsed
        metrics.generationTokPerMin = tokPerMin
        phases[id] = .streaming(buffer: buffer,
                                 status: "Streaming…",
                                 metrics: metrics)
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
            conversations[idx].messages[mIdx] = StoredMessage(
                id: placeholderID,
                role: .assistant,
                content: final.content,
                reasoningContent: final.reasoningContent,
                toolCalls: final.toolCalls.map(StoredToolCall.init),
                tokenCount: usage?.completionTokens,
                toolOutputs: nil)
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
    private func composeOpenAITools(
        mcpAllowed: Set<String>?
    ) -> [[String: Any]]? {
        if let allowed = mcpAllowed, allowed.isEmpty { return nil }
        var schemas: [[String: Any]] = []
        for tool in mcpPool.allTools() {
            if let allowed = mcpAllowed, !allowed.contains(tool.qualifiedName) {
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
        if case .streaming(_, _, let m) = phase(of: id) { return m }
        return GenerationMetrics()
    }

    private func currentBuffer(of id: UUID) -> String {
        if case .streaming(let b, _, _) = phase(of: id) { return b }
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
    }
}
