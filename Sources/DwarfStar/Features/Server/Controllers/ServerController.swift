import Foundation
import DS4Engine

/// Native, in-process HTTP server (`LocalServer`) exposing the model over an
/// OpenAI-compatible API. It does NOT load its own engine: it wraps THE single
/// shared `InferenceService` loaded in Settings (same weights, same KV, same
/// resident-Q4/mlock buffers — a second engine would double wired memory and
/// OOM on 16 GB). The shared actor serializes chat and server calls; the server
/// remains one-request-at-a-time.
@MainActor
@Observable
final class ServerController {
    // Configuration (editable before Start).
    let settings: AppSettings
    let store: ChatStore
    var modelPath: String { settings.modelPath }
    /// The one model id the API serves: the loaded GGUF's basename. There is no
    /// model choice over HTTP — it's whatever was loaded in Settings.
    var modelId: String {
        ((modelPath as NSString).lastPathComponent as NSString).deletingPathExtension
    }

    init(settings: AppSettings, store: ChatStore) { self.settings = settings; self.store = store }
    var host = "127.0.0.1"
    var port = 8000
    var maxTokens = 1024
    var cors = false
    /// Optional shared secret (Bearer / x-api-key). Empty = no authentication.
    var apiKey = ""

    // Live state.
    var log = ""
    var isRunning = false
    var isLoading = false
    var isStopping = false

    private var server: LocalServer?
    private var logTask: Task<Void, Never>?
    private var serverStartTask: Task<LocalServer, Error>?
    private var shutdownTask: Task<Void, Never>?
    private var startEpoch: UInt64 = 0
    private var engineLease: EngineActivityGate.Lease?

    var endpoint: String { "http://\(host):\(port)/v1" }

    func start() {
        guard !isRunning, !isLoading, !isStopping else { return }
        // The listener shares the chat engine and therefore its mutable KV.
        // Do not acquire the server lease in the middle of a generation, tool
        // continuation, manual tool-result pause, or role/cache warmup that was
        // already started before the lease existed.
        guard !store.isGenerating,
              store.generation == nil,
              store.pendingManualCalls.isEmpty,
              store.engineSetupTask == nil else {
            log = "The chat engine is still generating, awaiting tool results, or warming up. Finish or stop that work before starting the server.\n"
            return
        }
        let engine = store.sharedEngine
        let glmEngine = store.isReady ? store.glmService : nil
        guard engine != nil || glmEngine != nil else {
            log = "No model loaded. Load the model in Settings first — the server exposes that single shared engine.\n"
            return
        }
        let activityGate = EngineActivityGate.shared
        guard let lease = activityGate.acquire(.server) else {
            let owner = activityGate.activeOwner?.displayName ?? "another engine operation"
            log = "The engine is busy with \(owner). Stop it before starting the server.\n"
            return
        }
        engineLease = lease
        startEpoch &+= 1
        let epoch = startEpoch
        isLoading = true
        log = "Starting server on the shared engine...\n"

        let path = ProcessStream.absolutePath(modelPath)
        let name = (path as NSString).lastPathComponent
        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        let cfg = LocalServer.Config(host: host, port: UInt16(clamping: port),
                                     cors: cors, maxTokens: maxTokens,
                                     apiKey: key.isEmpty ? nil : key)

        // Sendable log channel: the server (any thread) yields lines; we drain them
        // on the main actor (Swift 6 concurrency-safe, no @MainActor capture).
        let (logStream, logCont) = AsyncStream<String>.makeStream()
        logTask?.cancel()
        logTask = Task { [weak self] in
            for await line in logStream { self?.log += line }
        }
        let onLog: @Sendable (String) -> Void = { logCont.yield($0) }

        // Only bind the listening socket off the main actor; the engine (and its
        // disk-KV, set when the chat loaded it) is already live and shared.
        let startTask = Task.detached { () -> LocalServer in
            // Idempotente (no-op se la chat ha già scaldato il motore): la
            // prima richiesta API non deve pagare la creazione dei pool
            // esperti e i kernel freddi in mezzo al prefill del client.
            try Task.checkCancellation()
            onLog("riscaldamento motore (no-op se già caldo)…\n")
            await engine?.warmup()
            _ = await glmEngine?.warmup()
            try Task.checkCancellation()
            let srv = LocalServer(engine: engine, glmEngine: glmEngine,
                                  modelName: name, config: cfg, onLog: onLog)
            try srv.start()
            return srv
        }
        serverStartTask = startTask
        Task { [weak self] in
            do {
                let startedServer = try await startTask.value
                guard let self else {
                    await startedServer.stop()
                    EngineActivityGate.shared.release(lease)
                    return
                }
                guard self.startEpoch == epoch, self.engineLease == lease else {
                    await startedServer.stop()
                    self.serverStartTask = nil
                    if self.isStopping {
                        self.completeStop(expected: lease)
                    } else {
                        self.releaseEngineLease(expected: lease)
                    }
                    return
                }
                self.serverStartTask = nil
                self.server = startedServer
                self.isLoading = false
                self.isRunning = true
            } catch {
                guard let self else {
                    EngineActivityGate.shared.release(lease)
                    return
                }
                guard self.startEpoch == epoch, self.engineLease == lease else {
                    self.serverStartTask = nil
                    if self.isStopping {
                        self.completeStop(expected: lease)
                    } else {
                        self.releaseEngineLease(expected: lease)
                    }
                    return
                }
                logCont.yield("start failed: \(error)\n")
                self.serverStartTask = nil
                self.isLoading = false
                self.isRunning = false
                self.releaseEngineLease(expected: lease)
            }
        }
    }

    func stop() {
        guard !isStopping else { return }
        guard isRunning || isLoading || server != nil || serverStartTask != nil || engineLease != nil else {
            return
        }

        startEpoch &+= 1
        isStopping = true
        isRunning = false
        isLoading = false
        log += "[server stopping: draining accepted requests…]\n"

        let runningServer = server
        server = nil                  // the shared engine stays alive for the chat

        // A cancelled startup retains the lease until its detached task has
        // actually drained; its completion branch performs the same barrier.
        if let startup = serverStartTask {
            startup.cancel()
            return
        }

        guard let runningServer else {
            completeStop()
            return
        }

        let lease = engineLease
        shutdownTask = Task { [weak self] in
            await runningServer.stop()
            guard let self else {
                if let lease { EngineActivityGate.shared.release(lease) }
                return
            }
            self.completeStop(expected: lease)
        }
    }

    /// Runs only after listener, accepted request tasks, inference and model I/O
    /// have drained. Keeping the lease until this point prevents auto-tune or a
    /// benchmark from overlapping a cancelled HTTP generation.
    private func completeStop(expected lease: EngineActivityGate.Lease? = nil) {
        shutdownTask = nil
        serverStartTask = nil
        isStopping = false
        isRunning = false
        isLoading = false
        logTask?.cancel()
        logTask = nil
        log += "[server stopped]\n"
        releaseEngineLease(expected: lease)
    }

    private func releaseEngineLease(expected: EngineActivityGate.Lease? = nil) {
        guard let lease = engineLease,
              expected == nil || expected == lease else { return }
        engineLease = nil
        EngineActivityGate.shared.release(lease)
    }
}
