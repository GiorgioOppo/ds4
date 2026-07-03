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

    private var server: LocalServer?
    private var logTask: Task<Void, Never>?

    var endpoint: String { "http://\(host):\(port)/v1" }

    func start() {
        guard !isRunning, !isLoading else { return }
        guard let engine = store.sharedEngine else {
            log = "No model loaded. Load the model in Settings first — the server exposes that single shared engine.\n"
            return
        }
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
            let srv = LocalServer(engine: engine, modelName: name, config: cfg, onLog: onLog)
            try srv.start()
            return srv
        }
        Task {
            do {
                self.server = try await startTask.value
                self.isLoading = false
                self.isRunning = true
            } catch {
                logCont.yield("start failed: \(error)\n")
                self.isLoading = false
                self.isRunning = false
            }
        }
    }

    func stop() {
        server?.stop()
        server = nil                  // the shared engine stays alive for the chat
        logTask?.cancel()
        logTask = nil
        isRunning = false
        isLoading = false
        log += "[server stopped]\n"
    }
}
