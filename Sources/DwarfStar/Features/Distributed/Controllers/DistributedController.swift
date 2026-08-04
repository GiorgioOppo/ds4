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
        set {
            settings.modelPath = newValue
            modelLayers = nil
            modelExperts = nil
        }
    }
    var contextSize: Int {
        get { settings.contextSize }
        set { settings.contextSize = newValue }
    }
    var activationBits = 32

    init(settings: AppSettings) { self.settings = settings }

    // Worker (Distribuito sidebar tab). The slice/model/context are NOT set
    // here: the worker starts idle and the coordinator ASSIGNs its whole job
    // (gguf, settings, layer range) at connect time.
    var port = 9100
    private(set) var modelLayers: Int?
    private(set) var modelExperts: Int?
    var modelLayersLabel: String { modelLayers.map(String.init) ?? "dal GGUF" }
    var workerRunning = false
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
    private var previousToolRoundFingerprints: Set<String> = []
    /// Bumped by Stop / New Chat: an async tool round (MCP call in flight)
    /// captured under an older epoch must drop its results instead of
    /// appending to a conversation the user has ended or cleared.
    private var chatEpoch = 0
    /// Finite safety budgets: a degraded local model must not repeat expensive
    /// or mutating calls until the cluster context is exhausted.
    private let maxToolRounds = 16
    private let maxToolCallsPerRound = 8

    /// One entry for every model-emitted call. Rejected entries carry a ready
    /// result, while executable entries are resolved asynchronously. Keeping a
    /// single ordered list prevents fast rejections from overtaking earlier MCP
    /// calls in the conversation transcript.
    private struct PlannedToolCall: Sendable {
        let call: ToolCall
        let presetOutput: ToolOutput?
        let presetMessage: String?
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

    var workerSummary: String { "worker :\(port) · job assigned by the coordinator" }

    /// Metadata-only refresh for labels shown before a coordinator is built.
    /// The authoritative values are overwritten from DistCoordinator after its
    /// full GGUF validation, so the UI never assumes Flash's 43 layers.
    func refreshModelGeometry() async {
        let requestedPath = ProcessStream.absolutePath(modelPath)
        guard !requestedPath.isEmpty else {
            modelLayers = nil; modelExperts = nil
            return
        }
        let layers: Int? = await Task.detached(priority: .utility) {
            (try? InferenceService.inspectModel(path: requestedPath))?.layerCount
        }.value
        guard requestedPath == ProcessStream.absolutePath(modelPath) else { return }
        modelLayers = layers
        modelExperts = nil
    }

    /// The connected coordinator, exposed so the Benchmark panel can reuse the live
    /// route instead of opening a second connection (nil unless connected). The
    /// benchmark resets the cluster KV, so it must not overlap a chat generation.
    var connectedCoordinator: DistCoordinator? { connected ? coordinator : nil }

    /// Set by the Benchmark panel while it drives the shared coordinator, so a chat
    /// turn can't interleave frames on the same connections (mutual exclusion).
    var benchmarkActive = false

    // MARK: Worker

    func startWorker() {
        guard !workerRunning else { return }
        // No model load here: the worker starts IDLE and loads its engine when
        // the coordinator sends the ASSIGN (gguf + settings + layer slice).
        workerLog = "Worker idle: waiting for the coordinator's assignment...\n"
        let cfg = DistWorker.Config(port: UInt16(clamping: port),
                                    localModelPath: ProcessStream.absolutePath(modelPath))
        let (logStream, logCont) = AsyncStream<String>.makeStream()
        workerLogTask?.cancel()
        workerLogTask = Task { [weak self] in for await s in logStream { self?.workerLog += s } }
        let onLog: @Sendable (String) -> Void = { logCont.yield($0) }
        let w = DistWorker(config: cfg, onLog: onLog)
        do {
            try w.start()
            worker = w
            workerRunning = true
        } catch {
            logCont.yield("worker start failed: \(error)\n")
            workerRunning = false
        }
    }

    func stopWorker() {
        worker?.stop(); worker = nil
        workerRunning = false
        workerLog += "[worker stopped]\n"
    }

    // MARK: Coordinator — connect / chat

    func connectCoordinator() {
        guard !connected, !coordLoading else { return }
        let peers = parsePeers()
        guard !peers.isEmpty else { coordLog += "no workers specified\n"; return }
        if forwardEnabled, returnHost.trimmingCharacters(in: .whitespaces).isEmpty {
            coordLog += "forwarding: enter the return host (this Mac's LAN IP)\n"; return
        }
        coordLoading = true; coordLog = "Loading model (coordinator)...\n"
        // The coordinator defines each worker's job: same gguf/context as this
        // Mac, plus the local expert slot-cache budget and — when the local
        // disk-KV toggle is on — a per-shard checkpoint budget (same keys as
        // the local chat settings).
        let slots = UserDefaults.standard.object(forKey: "DS4ExpertCacheSlots") as? Int ?? 0
        let kvOn = (UserDefaults.standard.object(forKey: "DS4DiskKV") as? Bool) ?? true
        let kvKTok = UserDefaults.standard.object(forKey: "DS4DiskKVBudgetKTok") as? Int ?? 1000
        let q4 = (UserDefaults.standard.object(forKey: "DS4DenseQ4") as? Bool) ?? true
        let cfg = DistCoordinator.Config(modelPath: ProcessStream.absolutePath(modelPath),
                                         contextSize: contextSize, peers: peers,
                                         activationBits: activationBits, prefillChunk: prefillChunk,
                                         forward: forwardEnabled,
                                         returnHost: returnHost.trimmingCharacters(in: .whitespaces),
                                         returnPort: UInt16(clamping: returnPort),
                                         workerCacheSlots: slots,
                                         diskKVBudgetTokens: kvOn ? kvKTok * 1000 : 0,
                                         useExpertBundle: false, useDenseQ4: q4)
        let (logStream, logCont) = AsyncStream<String>.makeStream()
        coordLogTask?.cancel()
        coordLogTask = Task { [weak self] in for await s in logStream { self?.coordLog += s } }
        let onLog: @Sendable (String) -> Void = { logCont.yield($0) }

        let vertical = verticalEnabled
        coordTask = Task {
            do {
                let coord = try await Task.detached { try DistCoordinator(config: cfg) }.value
                self.modelLayers = coord.modelLayers
                self.modelExperts = coord.modelExperts
                if vertical {
                    try await coord.connectVertical(onLog: onLog)
                } else {
                    try await coord.connect(onLog: onLog)
                }
                self.coordinator = coord
                self.coordLoading = false; self.connected = true
            } catch {
                logCont.yield("connection failed: \(error)\n")
                self.coordLoading = false; self.connected = false
            }
        }
    }

    // MARK: Verticale (expert parallelism, Fase C)

    /// Scissione VERTICALE: i worker ricevono shard di ESPERTI (mask dimensionata
    /// dal GGUF: 256 Flash, 384 Pro)
    /// e il backbone denso (route/attention/KV/head) gira su QUESTO Mac.
    /// Prerequisito misurato: RTT < 1 ms tra i nodi (Thunderbolt/Ethernet
    /// diretta) — su Wi-Fi i ~41 round-trip per token costano più del guadagno.
    var verticalEnabled: Bool = UserDefaults.standard.bool(forKey: "DS4DistVertical") {
        didSet { UserDefaults.standard.set(verticalEnabled, forKey: "DS4DistVertical") }
    }

    /// Benchmark della route verticale (prompt sintetico 96 + 28 di decode);
    /// il decode blocca sui round-trip di rete, quindi gira detached.
    func runVerticalBenchmark() {
        guard connected, let coord = coordinator, !benchmarkActive else { return }
        benchmarkActive = true
        coordLog += "benchmark verticale (96 prefill + 28 decode)…\n"
        Task {
            do {
                let p = try await Task.detached {
                    try coord.verticalBenchmark(contextTokens: 96, genTokens: 28)
                }.value
                self.coordLog += String(format: "verticale: prefill %.2f tok/s · decode regime %.2f tok/s (media %.2f)\n",
                                        p.prefillTps, p.genTpsP99, p.genTps)
            } catch {
                self.coordLog += "benchmark verticale fallito: \(error)\n"
            }
            self.benchmarkActive = false
        }
    }

    func disconnectCoordinator() {
        coordTask?.cancel(); coordTask = nil
        let coord = coordinator
        Task.detached { coord?.disconnect() }
        coordinator = nil
        connected = false; isGenerating = false
        coordLog += "[disconnected]\n"
    }

    /// Send the current input as a chat turn and stream the reply, running the
    /// tool loop: DSML calls are executed LOCALLY (ToolRegistry, incl. project_*)
    /// and fed back as .toolResult turns until the model answers with text.
    func sendChat() {
        let text = chatInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard connected, !isGenerating, !benchmarkActive, !text.isEmpty, coordinator != nil else {
            if benchmarkActive { coordLog += "benchmark running: wait for it to finish before chatting.\n" }
            return
        }
        chatInput = ""
        messages.append(UIMessage(role: .user, text: text))
        turns.append(.user(text))
        toolRounds = 0
        previousToolRoundFingerprints = []
        isGenerating = true
        generateTurn()
    }

    /// One generation round (assistant reply or tool call) + tool continuation.
    private func generateTurn() {
        guard let coord = coordinator else { isGenerating = false; return }
        let epoch = chatEpoch
        let index = messages.count
        messages.append(UIMessage(role: .assistant, text: ""))

        let agent = selectedAgent.withActiveProjectContext(
            ProjectCache.shared.agentContext())
        // Immutable snapshot for the detached closure (capturing a mutable local
        // trips Swift 6 region analysis: "sending parameter risks data races").
        let sendTurns: [ChatTurn] = (agent.systemPrompt.isEmpty ? [] : [.system(agent.systemPrompt)]) + turns
        let tools = agent.toolNames.isEmpty ? [] : ToolRegistry.autoSpecs(enabled: Set(agent.toolNames))
        // This is the authoritative capability snapshot for this exact model
        // invocation. Agent selection and MCP availability can change while a
        // decode is running; neither may broaden what its output can execute.
        let declaredToolNames = Set(tools.map(\.name))
        let wantThink = think, maxT = maxTokens, samp = SamplingParams()

        enum Ev: Sendable { case log(String), progress(String), reasoning(String), token(String) }
        let (stream, cont) = AsyncStream<Ev>.makeStream()
        eventTask?.cancel()
        let eventConsumer = Task { [weak self] in
            for await e in stream {
                guard !Task.isCancelled,
                      let self,
                      epoch == self.chatEpoch else { break }
                guard index < self.messages.count else { continue }
                switch e {
                case .log(let s): self.coordLog += s
                case .progress(let s): self.status = s     // live "N token · X tok/s"
                case .reasoning(let s): self.messages[index].reasoning += s
                case .token(let s): self.messages[index].text += s
                }
            }
        }
        eventTask = eventConsumer
        let onLog: @Sendable (String) -> Void = { cont.yield(.log($0)) }
        let onProgress: @Sendable (String) -> Void = { cont.yield(.progress($0)) }
        let onReasoning: @Sendable (String) -> Void = { cont.yield(.reasoning($0)) }
        let onToken: @Sendable (String) -> Void = { cont.yield(.token($0)) }

        let vertical = verticalEnabled
        coordTask = Task {
            // Explicit capture list: only Sendable copies cross into the detached
            // task (no main-actor state in the closure's region).
            let work = Task.detached { [coord, sendTurns, tools, wantThink, maxT, samp,
                                        onLog, onProgress, onReasoning, onToken] () -> Result<[ToolCall], Error> in
                do {
                    // Verticale: stesso contratto, decode sul backbone locale
                    // (la FFN routed viaggia sui worker dentro ogni forward).
                    let calls = vertical
                        ? try await coord.sendVertical(turns: sendTurns, tools: tools, think: wantThink,
                                                       maxTokens: maxT, sampling: samp,
                                                       onLog: onLog, onProgress: onProgress,
                                                       onReasoning: onReasoning, onToken: onToken)
                        : try await coord.send(turns: sendTurns, tools: tools, think: wantThink,
                                               maxTokens: maxT, sampling: samp,
                                               onLog: onLog, onProgress: onProgress,
                                               onReasoning: onReasoning, onToken: onToken)
                    return .success(calls)
                } catch { return .failure(error) }
            }
            // A detached task does NOT inherit cancellation: forward Stop
            // (coordTask.cancel) explicitly, or the cluster generation would
            // keep running — and streaming tokens — to completion.
            let result = await withTaskCancellationHandler { await work.value }
                                                  onCancel: { work.cancel() }
            cont.finish()
            // Drain all queued stream events before reading the assembled
            // assistant text. This also makes the epoch check cover the final
            // tokens that raced with decode completion.
            await eventConsumer.value
            guard !Task.isCancelled, epoch == self.chatEpoch else { return }
            switch result {
            case .failure(let error):
                if !(error is CancellationError) { self.coordLog += "error: \(error)\n" }
                self.isGenerating = false; self.status = ""
            case .success(let calls):
                self.finishTurn(index: index, calls: calls,
                                declaredToolNames: declaredToolNames,
                                epoch: epoch)
            }
        }
    }

    /// Record the assistant turn; execute tool calls locally and continue, or stop.
    private func finishTurn(index: Int, calls: [ToolCall],
                            declaredToolNames: Set<String>, epoch: Int) {
        guard !Task.isCancelled, epoch == chatEpoch else { return }
        guard index < messages.count else { isGenerating = false; status = ""; return }
        let visible = ToolCallParser.stripLeakedMarkup(messages[index].text, markup: .dsv4)
        messages[index].text = visible
        messages[index].toolCalls = calls
        turns.append(.assistant(text: visible, toolCalls: calls))

        guard !calls.isEmpty else { isGenerating = false; status = ""; return }
        toolRounds += 1
        guard toolRounds <= maxToolRounds else {
            messages.append(UIMessage(role: .tool, text: "Too many tool rounds (\(maxToolRounds)); stopped."))
            isGenerating = false; status = ""
            return
        }
        // MCP calls make this round ASYNC (the old all-builtin round was
        // synchronous, so Stop / New Chat could never interleave). The Task is
        // stored in coordTask so stopGeneration() cancels it, and the epoch
        // guard drops the round if Stop or New Chat ran during an await —
        // otherwise a finished MCP call would append stale turns and silently
        // resume a generation the user ended.
        let policy = ToolExecutionPolicy(allowedToolNames: declaredToolNames)
        var currentFingerprints = Set<String>()
        var plannedCalls: [PlannedToolCall] = []
        plannedCalls.reserveCapacity(calls.count)
        for (offset, call) in calls.enumerated() {
            if offset >= maxToolCallsPerRound {
                let content = #"{"error":"tool_call_batch_limit","message":"call not executed: too many calls in one round"}"#
                let output = ToolOutput(callId: call.id, name: call.name, content: content)
                plannedCalls.append(PlannedToolCall(
                    call: call,
                    presetOutput: output,
                    presetMessage: "✗ \(call.name) not executed: per-round limit \(maxToolCallsPerRound)"))
                continue
            }

            let firstInRound = currentFingerprints.insert(call.fingerprint).inserted
            if firstInRound, !previousToolRoundFingerprints.contains(call.fingerprint) {
                plannedCalls.append(PlannedToolCall(call: call,
                                                    presetOutput: nil,
                                                    presetMessage: nil))
            } else {
                let content = #"{"error":"duplicate_tool_call","message":"identical consecutive tool call rejected"}"#
                let output = ToolOutput(callId: call.id, name: call.name, content: content)
                plannedCalls.append(PlannedToolCall(
                    call: call,
                    presetOutput: output,
                    presetMessage: "✗ \(call.name) not executed: duplicate consecutive call"))
            }
        }
        previousToolRoundFingerprints = currentFingerprints
        coordTask = Task {
            for planned in plannedCalls {
                guard !Task.isCancelled, epoch == chatEpoch else { return }

                let out: ToolOutput
                let displayText: String
                if let preset = planned.presetOutput {
                    out = preset
                    displayText = planned.presetMessage ?? "\(planned.call.name) → \(preset.content)"
                } else {
                    let call = planned.call
                    if MCPManager.shared.isMCPTool(named: call.name) { status = "MCP: \(call.name)…" }
                    out = await ToolRegistry.executeAuto(call, policy: policy)
                        ?? ToolOutput(callId: call.id, name: call.name,
                                      content: #"{"error":"unknown tool: not built in and not provided by a connected MCP server"}"#)
                    displayText = "\(call.name) → \(out.content)"
                }
                guard !Task.isCancelled, epoch == chatEpoch else { return }
                messages.append(UIMessage(role: .tool, text: displayText))
                turns.append(.toolResult(callId: out.callId, name: out.name, content: out.content))
            }
            guard !Task.isCancelled, epoch == chatEpoch else { return }
            generateTurn()      // continue with the tool results
        }
    }

    func stopGeneration() {
        chatEpoch += 1                    // invalidate any in-flight async tool round
        coordTask?.cancel()
        coordTask = nil
        eventTask?.cancel()
        eventTask = nil
        isGenerating = false
        status = ""
        coordLog += "[stopped] (the next question starts from scratch)\n"
    }

    func newChat() {
        chatEpoch += 1                    // an in-flight tool round must not touch the fresh chat
        coordTask?.cancel()
        coordTask = nil
        eventTask?.cancel()
        eventTask = nil
        isGenerating = false
        status = ""
        messages.removeAll()
        turns.removeAll()
        toolRounds = 0
        previousToolRoundFingerprints = []
        agents = ChatStore.loadAgents()
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
