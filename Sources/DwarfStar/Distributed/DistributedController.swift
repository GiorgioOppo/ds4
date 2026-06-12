import Foundation
import DS4Engine
import DS4Core

/// Drives distributed inference from the UI. A node is either a WORKER (owns a
/// layer slice, listens for the coordinator) or the COORDINATOR (connects to the
/// workers and runs a multi-turn CHAT across the cluster). Heavy work runs off
/// the main actor; logs/tokens stream back via AsyncStream channels.
@MainActor
@Observable
final class DistributedController {
    // Model path/context come from the shared settings (each role still loads
    // its own engine; the mmap weights are shared).
    let settings: AppSettings
    var modelPath: String {
        get { settings.modelPath }
        set { settings.modelPath = newValue }
    }
    var contextSize: Int {
        get { settings.contextSize }
        set { settings.contextSize = newValue }
    }
    var activationBits = 32

    init(settings: AppSettings) { self.settings = settings }

    // Worker (Distribuito sidebar tab).
    var port = 9100
    var layerStart = 0
    var layerEnd = DistEngine.modelLayers - 1
    var hasOutput = true
    var modelLayers: Int { DistEngine.modelLayers }
    var workerRunning = false
    var workerLoading = false
    var workerLog = ""

    // Coordinator (Chat tab → Distribuito).
    var peersText = "127.0.0.1:9100"
    var prefillChunk = 32
    var forwardEnabled = false
    var returnHost = ""
    var returnPort = 9099
    var think = false
    var maxTokens = 512
    var connected = false        // route established
    var coordLoading = false
    var coordLog = ""

    // Coordinator chat.
    var messages: [UIMessage] = []
    var chatInput = ""
    var isGenerating = false
    var status = ""              // live prefill/decode progress (last log line)
    /// Authoritative conversation as engine turns (incl. tool calls/results):
    /// re-rendered in full on every send (stateless coordinator).
    private var turns: [ChatTurn] = []
    private var toolRounds = 0
    /// Agentic roles (write tools) get a larger budget — but distributed rounds
    /// re-prefill the whole conversation, so keep it tighter than local.
    private var maxToolRounds: Int {
        selectedAgent.toolNames.contains("project_write") ? 10 : 4
    }

    // Agent (role): same library as the local chat, own selection. Tools run
    // LOCALLY on this (coordinator) Mac — incl. project_* against the active project.
    var agents: [AgentProfile] = ChatStore.loadAgents()
    var selectedAgentId: String = UserDefaults.standard.string(forKey: "DS4SelectedAgentDist") ?? "generale" {
        didSet { UserDefaults.standard.set(selectedAgentId, forKey: "DS4SelectedAgentDist") }
    }
    var selectedAgent: AgentProfile { agents.first { $0.id == selectedAgentId } ?? agents[0] }
    func selectAgent(_ id: String) {
        selectedAgentId = id
        newChat()                 // fresh conversation with the new role (like local)
    }

    private var worker: DistWorker?
    private var coordinator: DistCoordinator?
    private var coordTask: Task<Void, Never>?      // connect / generation task
    private var workerLogTask: Task<Void, Never>?
    private var coordLogTask: Task<Void, Never>?
    private var eventTask: Task<Void, Never>?

    var workerSummary: String { "worker :\(port) · layer \(layerStart)…\(layerEnd)\(hasOutput ? " +output" : "")" }

    // MARK: Worker

    func startWorker() {
        guard !workerRunning, !workerLoading else { return }
        workerLoading = true; workerLog = "Caricamento modello (worker)…\n"
        let cfg = DistWorker.Config(modelPath: ProcessStream.absolutePath(modelPath),
                                    port: UInt16(clamping: port), layerStart: layerStart,
                                    layerEnd: layerEnd, hasOutput: hasOutput, contextSize: contextSize)
        let (logStream, logCont) = AsyncStream<String>.makeStream()
        workerLogTask?.cancel()
        workerLogTask = Task { [weak self] in for await s in logStream { self?.workerLog += s } }
        let onLog: @Sendable (String) -> Void = { logCont.yield($0) }
        let loadTask = Task.detached { () -> DistWorker in
            let w = try DistWorker(config: cfg, onLog: onLog)
            try w.start()
            return w
        }
        Task {
            do { self.worker = try await loadTask.value; self.workerLoading = false; self.workerRunning = true }
            catch { logCont.yield("avvio worker fallito: \(error)\n"); self.workerLoading = false; self.workerRunning = false }
        }
    }

    func stopWorker() {
        worker?.stop(); worker = nil
        workerRunning = false; workerLoading = false
        workerLog += "[worker fermato]\n"
    }

    // MARK: Coordinator — connect / chat

    func connectCoordinator() {
        guard !connected, !coordLoading else { return }
        let peers = parsePeers()
        guard !peers.isEmpty else { coordLog += "nessun worker indicato\n"; return }
        if forwardEnabled, returnHost.trimmingCharacters(in: .whitespaces).isEmpty {
            coordLog += "inoltro: indica l'host di ritorno (IP LAN di questo Mac)\n"; return
        }
        coordLoading = true; coordLog = "Caricamento modello (coordinatore)…\n"
        let cfg = DistCoordinator.Config(modelPath: ProcessStream.absolutePath(modelPath),
                                         contextSize: contextSize, peers: peers,
                                         activationBits: activationBits, prefillChunk: prefillChunk,
                                         forward: forwardEnabled,
                                         returnHost: returnHost.trimmingCharacters(in: .whitespaces),
                                         returnPort: UInt16(clamping: returnPort))
        let (logStream, logCont) = AsyncStream<String>.makeStream()
        coordLogTask?.cancel()
        coordLogTask = Task { [weak self] in for await s in logStream { self?.coordLog += s } }
        let onLog: @Sendable (String) -> Void = { logCont.yield($0) }

        coordTask = Task {
            do {
                let coord = try await Task.detached { try DistCoordinator(config: cfg) }.value
                try await coord.connect(onLog: onLog)
                self.coordinator = coord
                self.coordLoading = false; self.connected = true
            } catch {
                logCont.yield("connessione fallita: \(error)\n")
                self.coordLoading = false; self.connected = false
            }
        }
    }

    func disconnectCoordinator() {
        coordTask?.cancel(); coordTask = nil
        let coord = coordinator
        Task.detached { coord?.disconnect() }
        coordinator = nil
        connected = false; isGenerating = false
        coordLog += "[disconnesso]\n"
    }

    /// Send the current input as a chat turn and stream the reply, running the
    /// tool loop: DSML calls are executed LOCALLY (ToolRegistry, incl. project_*)
    /// and fed back as .toolResult turns until the model answers with text.
    func sendChat() {
        let text = chatInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard connected, !isGenerating, !text.isEmpty, coordinator != nil else { return }
        chatInput = ""
        messages.append(UIMessage(role: .user, text: text))
        turns.append(.user(text))
        toolRounds = 0
        isGenerating = true
        generateTurn()
    }

    /// One generation round (assistant reply or tool call) + tool continuation.
    private func generateTurn() {
        guard let coord = coordinator else { isGenerating = false; return }
        let index = messages.count
        messages.append(UIMessage(role: .assistant, text: ""))

        let agent = selectedAgent
        // Immutable snapshot for the detached closure (capturing a mutable local
        // trips Swift 6 region analysis: "sending parameter risks data races").
        let sendTurns: [ChatTurn] = (agent.systemPrompt.isEmpty ? [] : [.system(agent.systemPrompt)]) + turns
        let tools = agent.toolNames.isEmpty ? [] : ToolRegistry.specs(enabled: Set(agent.toolNames))
        let wantThink = think, maxT = maxTokens, samp = SamplingParams()

        enum Ev: Sendable { case log(String), progress(String), reasoning(String), token(String) }
        let (stream, cont) = AsyncStream<Ev>.makeStream()
        eventTask?.cancel()
        eventTask = Task { [weak self] in
            for await e in stream {
                guard let self, index < self.messages.count else { continue }
                switch e {
                case .log(let s): self.coordLog += s
                case .progress(let s): self.status = s     // live "N token · X tok/s"
                case .reasoning(let s): self.messages[index].reasoning += s
                case .token(let s): self.messages[index].text += s
                }
            }
        }
        let onLog: @Sendable (String) -> Void = { cont.yield(.log($0)) }
        let onProgress: @Sendable (String) -> Void = { cont.yield(.progress($0)) }
        let onReasoning: @Sendable (String) -> Void = { cont.yield(.reasoning($0)) }
        let onToken: @Sendable (String) -> Void = { cont.yield(.token($0)) }

        coordTask = Task {
            // Explicit capture list: only Sendable copies cross into the detached
            // task (no main-actor state in the closure's region).
            let work = Task.detached { [coord, sendTurns, tools, wantThink, maxT, samp,
                                        onLog, onProgress, onReasoning, onToken] () -> Result<[ToolCall], Error> in
                do {
                    let calls = try await coord.send(turns: sendTurns, tools: tools, think: wantThink,
                                                     maxTokens: maxT, sampling: samp,
                                                     onLog: onLog, onProgress: onProgress,
                                                     onReasoning: onReasoning, onToken: onToken)
                    return .success(calls)
                } catch { return .failure(error) }
            }
            let result = await work.value
            cont.finish()
            switch result {
            case .failure(let error):
                if !(error is CancellationError) { self.coordLog += "errore: \(error)\n" }
                self.isGenerating = false; self.status = ""
            case .success(let calls):
                self.finishTurn(index: index, calls: calls)
            }
        }
    }

    /// Record the assistant turn; execute tool calls locally and continue, or stop.
    private func finishTurn(index: Int, calls: [ToolCall]) {
        guard index < messages.count else { isGenerating = false; status = ""; return }
        let visible = ToolCallParser.stripLeakedMarkup(messages[index].text, markup: .dsv4)
        messages[index].text = visible
        messages[index].toolCalls = calls
        turns.append(.assistant(text: visible, toolCalls: calls))

        guard !calls.isEmpty else { isGenerating = false; status = ""; return }
        toolRounds += 1
        guard toolRounds <= maxToolRounds else {
            messages.append(UIMessage(role: .tool, text: "⚠️ troppi round di tool (\(maxToolRounds)) — interrotto."))
            isGenerating = false; status = ""
            return
        }
        for c in calls {
            let out = ToolRegistry.execute(c)
                ?? ToolOutput(callId: c.id, name: c.name,
                              content: #"{"error":"tool non integrato: non supportato in distribuito"}"#)
            messages.append(UIMessage(role: .tool, text: "\(c.name) → \(out.content)"))
            turns.append(.toolResult(callId: out.callId, name: out.name, content: out.content))
        }
        generateTurn()      // continue with the tool results
    }

    func stopGeneration() {
        coordTask?.cancel()
        isGenerating = false
        coordLog += "[interrotto] (la prossima domanda riparte da capo)\n"
    }

    func newChat() {
        messages.removeAll()
        turns.removeAll()
        toolRounds = 0
        agents = ChatStore.loadAgents()   // pick up edits from the Agenti tab
    }

    // MARK: Helpers

    func parsePeers() -> [DistCoordinator.Peer] {
        peersText.split(whereSeparator: \.isNewline).compactMap { line in
            let parts = line.trimmingCharacters(in: .whitespaces).split(separator: ":")
            guard parts.count == 2, let port = UInt16(parts[1]) else { return nil }
            return DistCoordinator.Peer(host: String(parts[0]), port: port)
        }
    }
}
