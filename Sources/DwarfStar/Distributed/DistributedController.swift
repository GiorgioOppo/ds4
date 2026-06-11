import Foundation
import DS4Engine

/// Drives distributed inference from the UI. A node is either a WORKER (owns a
/// layer slice, listens for the coordinator) or the COORDINATOR (connects to the
/// workers, owns the prompt + sampling). Heavy model loading and generation run
/// off the main actor; logs/tokens stream back via AsyncStream channels.
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
    var activationBits = 32          // 32 / 16 / 8

    // Worker.
    var port = 9100
    var layerStart = 0
    var layerEnd = 20
    var hasOutput = false

    // Coordinator.
    var peersText = "127.0.0.1:9100"  // one host:port per line, in layer order
    var prompt = ""
    var maxTokens = 256
    var prefillChunk = 32             // tokens per WORK frame during prefill
    var forwardEnabled = false        // worker→worker forwarding (needs returnHost)
    var returnHost = ""               // this Mac's LAN address, as the workers see it
    var returnPort = 9099
    var output = ""

    // Live state.
    var log = ""
    var isRunning = false
    var isLoading = false

    private var worker: DistWorker?
    private var coordTask: Task<Void, Never>?
    private var logTask: Task<Void, Never>?
    private var tokenTask: Task<Void, Never>?

    var endpointSummary: String {
        role == .worker ? "worker :\(port) · layer \(layerStart)…\(layerEnd)\(hasOutput ? " +output" : "")"
                        : "coordinatore · \(parsePeers().count) worker"
    }

    func start() { role == .worker ? startWorker() : startCoordinator() }

    func stop() {
        worker?.stop(); worker = nil
        coordTask?.cancel(); coordTask = nil
        logTask?.cancel(); logTask = nil
        tokenTask?.cancel(); tokenTask = nil
        isRunning = false; isLoading = false
        log += "[fermato]\n"
    }

    // MARK: Worker

    private func startWorker() {
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
            do {
                self.worker = try await loadTask.value
                self.isLoading = false; self.isRunning = true
            } catch {
                logCont.yield("avvio worker fallito: \(error)\n")
                self.isLoading = false; self.isRunning = false
            }
        }
    }

    // MARK: Coordinator

    private func startCoordinator() {
        guard !isRunning, !isLoading else { return }
        let peers = parsePeers()
        guard !peers.isEmpty else { log += "nessun worker indicato\n"; return }
        isLoading = true; output = ""; log = "Caricamento modello (coordinatore)…\n"

        if forwardEnabled && returnHost.trimmingCharacters(in: .whitespaces).isEmpty {
            log += "inoltro worker→worker: indica l'host di ritorno (l'indirizzo LAN di questo Mac)\n"
            isLoading = false
            return
        }
        let cfg = DistCoordinator.Config(modelPath: ProcessStream.absolutePath(modelPath),
                                         contextSize: contextSize, peers: peers,
                                         activationBits: activationBits, prefillChunk: prefillChunk,
                                         forward: forwardEnabled,
                                         returnHost: returnHost.trimmingCharacters(in: .whitespaces),
                                         returnPort: UInt16(clamping: returnPort))
        let (logStream, logCont) = AsyncStream<String>.makeStream(); drainLog(logStream)
        let (tokStream, tokCont) = AsyncStream<String>.makeStream(); drainTokens(tokStream)
        let onLog: @Sendable (String) -> Void = { logCont.yield($0) }
        let onToken: @Sendable (String) -> Void = { tokCont.yield($0) }
        let promptText = prompt, maxT = maxTokens

        coordTask = Task {
            self.isLoading = false; self.isRunning = true
            let work = Task.detached { () -> String? in
                do {
                    let coord = try DistCoordinator(config: cfg)
                    try await coord.generate(system: nil, prompt: promptText, maxTokens: maxT,
                                             sampling: SamplingParams(), onLog: onLog, onToken: onToken)
                    return nil
                } catch { return "\(error)" }
            }
            if let err = await work.value { logCont.yield("errore: \(err)\n") }
            self.isRunning = false
            logCont.finish(); tokCont.finish()
        }
    }

    // MARK: Log/token drains (run on the main actor)

    private func drainLog(_ stream: AsyncStream<String>) {
        logTask?.cancel()
        logTask = Task { [weak self] in for await line in stream { self?.log += line } }
    }
    private func drainTokens(_ stream: AsyncStream<String>) {
        tokenTask?.cancel()
        tokenTask = Task { [weak self] in for await t in stream { self?.output += t } }
    }

    /// Parse the peer list: one `host:port` per line, in layer order.
    func parsePeers() -> [DistCoordinator.Peer] {
        peersText.split(whereSeparator: \.isNewline).compactMap { line in
            let parts = line.trimmingCharacters(in: .whitespaces).split(separator: ":")
            guard parts.count == 2, let port = UInt16(parts[1]) else { return nil }
            return DistCoordinator.Peer(host: String(parts[0]), port: port)
        }
    }
}
