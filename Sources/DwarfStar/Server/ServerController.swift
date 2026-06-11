import Foundation
import DS4Engine

/// Owns a native, in-process HTTP server (`LocalServer`) that exposes the model
/// over an OpenAI-compatible API. Unlike the old C `ds4-server` subprocess, this
/// loads its OWN `InferenceService` in-process: the GGUF weights are no-copy mmap
/// views, so the OS page cache shares them with the chat engine (no second copy
/// of the weights in RAM). Only the KV cache + Metal scratch are independent.
@MainActor
@Observable
final class ServerController {
    // Configuration (editable before Start).
    var modelPath = AppEnvironment.defaultModelPath
    var host = "127.0.0.1"
    var port = 8000
    var contextSize = 8192
    var maxTokens = 1024
    var cors = false

    // Live state.
    var log = ""
    var isRunning = false
    var isLoading = false

    private var engine: InferenceService?
    private var server: LocalServer?
    private var logTask: Task<Void, Never>?

    var endpoint: String { "http://\(host):\(port)/v1" }

    func start() {
        guard !isRunning, !isLoading else { return }
        isLoading = true
        log = "Caricamento modello in-process…\n"

        let path = ProcessStream.absolutePath(modelPath)
        let name = (path as NSString).lastPathComponent
        let ctx = contextSize
        let cfg = LocalServer.Config(host: host, port: UInt16(clamping: port),
                                     cors: cors, maxTokens: maxTokens)

        // Sendable log channel: the server (any thread) yields lines; we drain them
        // on the main actor. Avoids capturing this @MainActor object in a @Sendable
        // closure (Swift 6 concurrency-safe).
        let (logStream, logCont) = AsyncStream<String>.makeStream()
        logTask?.cancel()
        logTask = Task { [weak self] in
            for await line in logStream { self?.log += line }
        }
        let onLog: @Sendable (String) -> Void = { logCont.yield($0) }

        // Load the model OFF the main thread (mmap + Metal setup is heavy). The
        // detached task captures only Sendable values (no self), then we hop back
        // to the main actor to publish state.
        let loadTask = Task.detached { () -> (InferenceService, LocalServer) in
            let eng = try InferenceService(modelPath: path, contextSize: ctx, systemPrompt: nil)
            let srv = LocalServer(engine: eng, modelName: name, config: cfg, onLog: onLog)
            try srv.start()
            return (eng, srv)
        }
        Task {
            do {
                let (eng, srv) = try await loadTask.value
                self.engine = eng
                self.server = srv
                self.isLoading = false
                self.isRunning = true
            } catch {
                logCont.yield("avvio fallito: \(error)\n")
                self.isLoading = false
                self.isRunning = false
            }
        }
    }

    func stop() {
        server?.stop()
        server = nil
        engine = nil                  // release the model (KV + scratch)
        logTask?.cancel()
        logTask = nil
        isRunning = false
        isLoading = false
        log += "[server fermato]\n"
    }
}
