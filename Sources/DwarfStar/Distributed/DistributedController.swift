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
    enum Role: String, CaseIterable, Identifiable {
        case worker = "Worker"
        case coordinator = "Coordinatore"
        var id: String { rawValue }
    }

    // Shared.
    var role: Role = .worker
    var modelPath = AppEnvironment.defaultModelPath
    var contextSize = 8192
    var activationBits = 32

    // Worker.
    var port = 9100
    var layerStart = 0
    var layerEnd = DistEngine.modelLayers - 1
    var hasOutput = true
    var modelLayers: Int { DistEngine.modelLayers }

    // Coordinator.
    var peersText = "127.0.0.1:9100"
    var prefillChunk = 32
    var forwardEnabled = false
    var returnHost = ""
    var returnPort = 9099
    var think = false
    var maxTokens = 512

    // Coordinator chat.
    var messages: [UIMessage] = []
    var chatInput = ""
    var connected = false        // route established
    var isGenerating = false

    // Live state.
    var log = ""
    var isRunning = false        // worker listening OR coordinator connected
    var isLoading = false

    private var worker: DistWorker?
    private var coordinator: DistCoordinator?
    private var coordTask: Task<Void, Never>?      // connect / generation task
    private var logTask: Task<Void, Never>?
    private var eventTask: Task<Void, Never>?

    var endpointSummary: String {
        role == .worker ? "worker :\(port) · layer \(layerStart)…\(layerEnd)\(hasOutput ? " +output" : "")"
                        : "coordinatore · \(parsePeers().count) worker"
    }

    // MARK: Worker

    func startWorker() {
        guard !isRunning, !isLoading else { return }
        isLoading = true; log = "Caricamento modello (worker)…\n"
        let cfg = DistWorker.Config(modelPath: ProcessStream.absolutePath(modelPath),
                                    port: UInt16(clamping: port), layerStart: layerStart,
                                    layerEnd: layerEnd, hasOutput: hasOutput, contextSize: contextSize)
        let (logStream, logCont) = AsyncStream<String>.makeStream()
        drainLog(logStream)
        let onLog: @Sendable (String) -> Void = { logCont.yield($0) }
        let loadTask = Task.detached { () -> DistWorker in
            let w = try DistWorker(config: cfg, onLog: onLog)
            try w.start()
            return w
        }
        Task {
            do { self.worker = try await loadTask.value; self.isLoading = false; self.isRunning = true }
            catch { logCont.yield("avvio worker fallito: \(error)\n"); self.isLoading = false; self.isRunning = false }
        }
    }

    func stopWorker() {
        worker?.stop(); worker = nil
        isRunning = false; isLoading = false
        log += "[worker fermato]\n"
    }

    // MARK: Coordinator — connect / chat

    func connectCoordinator() {
        guard !connected, !isLoading else { return }
        let peers = parsePeers()
        guard !peers.isEmpty else { log += "nessun worker indicato\n"; return }
        if forwardEnabled, returnHost.trimmingCharacters(in: .whitespaces).isEmpty {
            log += "inoltro: indica l'host di ritorno (IP LAN di questo Mac)\n"; return
        }
        isLoading = true; log = "Caricamento modello (coordinatore)…\n"
        let cfg = DistCoordinator.Config(modelPath: ProcessStream.absolutePath(modelPath),
                                         contextSize: contextSize, peers: peers,
                                         activationBits: activationBits, prefillChunk: prefillChunk,
                                         forward: forwardEnabled,
                                         returnHost: returnHost.trimmingCharacters(in: .whitespaces),
                                         returnPort: UInt16(clamping: returnPort))
        let (logStream, logCont) = AsyncStream<String>.makeStream(); drainLog(logStream)
        let onLog: @Sendable (String) -> Void = { logCont.yield($0) }

        coordTask = Task {
            do {
                let coord = try await Task.detached { try DistCoordinator(config: cfg) }.value
                try await coord.connect(onLog: onLog)
                self.coordinator = coord
                self.isLoading = false; self.connected = true; self.isRunning = true
            } catch {
                logCont.yield("connessione fallita: \(error)\n")
                self.isLoading = false; self.connected = false
            }
        }
    }

    func disconnectCoordinator() {
        coordTask?.cancel(); coordTask = nil
        let coord = coordinator
        Task.detached { coord?.disconnect() }
        coordinator = nil
        connected = false; isRunning = false; isGenerating = false
        log += "[disconnesso]\n"
    }

    /// Send the current input as a chat turn and stream the reply.
    func sendChat() {
        let text = chatInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard connected, !isGenerating, !text.isEmpty, let coord = coordinator else { return }
        chatInput = ""
        messages.append(UIMessage(role: .user, text: text))
        let index = messages.count
        messages.append(UIMessage(role: .assistant, text: ""))
        isGenerating = true

        // Snapshot the conversation as engine turns.
        let turns: [ChatTurn] = messages.dropLast().map { m in
            m.role == .user ? .user(m.text) : .assistant(text: m.text, toolCalls: [])
        }
        let wantThink = think, maxT = maxTokens, samp = SamplingParams()

        enum Ev: Sendable { case log(String), reasoning(String), token(String) }
        let (stream, cont) = AsyncStream<Ev>.makeStream()
        eventTask?.cancel()
        eventTask = Task { [weak self] in
            for await e in stream {
                guard let self, index < self.messages.count else { continue }
                switch e {
                case .log(let s): self.log += s
                case .reasoning(let s): self.messages[index].reasoning += s
                case .token(let s): self.messages[index].text += s
                }
            }
        }
        let onLog: @Sendable (String) -> Void = { cont.yield(.log($0)) }
        let onReasoning: @Sendable (String) -> Void = { cont.yield(.reasoning($0)) }
        let onToken: @Sendable (String) -> Void = { cont.yield(.token($0)) }

        coordTask = Task {
            let work = Task.detached { () -> String? in
                do {
                    try await coord.send(turns: turns, think: wantThink, maxTokens: maxT, sampling: samp,
                                         onLog: onLog, onReasoning: onReasoning, onToken: onToken)
                    return nil
                } catch is CancellationError { return nil }
                catch { return "\(error)" }
            }
            if let err = await work.value { cont.yield(.log("errore: \(err)\n")) }
            cont.finish()
            self.isGenerating = false
        }
    }

    func stopGeneration() {
        coordTask?.cancel()
        isGenerating = false
        log += "[interrotto] (la prossima domanda riparte da capo)\n"
    }

    func newChat() { messages.removeAll() }

    // MARK: Helpers

    private func drainLog(_ stream: AsyncStream<String>) {
        logTask?.cancel()
        logTask = Task { [weak self] in for await line in stream { self?.log += line } }
    }

    func parsePeers() -> [DistCoordinator.Peer] {
        peersText.split(whereSeparator: \.isNewline).compactMap { line in
            let parts = line.trimmingCharacters(in: .whitespaces).split(separator: ":")
            guard parts.count == 2, let port = UInt16(parts[1]) else { return nil }
            return DistCoordinator.Peer(host: String(parts[0]), port: port)
        }
    }
}
