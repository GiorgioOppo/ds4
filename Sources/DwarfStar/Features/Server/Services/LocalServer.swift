import Foundation
@preconcurrency import Network
import DS4Core
import DS4Engine

/// A minimal native HTTP server (Network.framework) that exposes the in-process
/// `InferenceService` over an OpenAI-compatible API. This replaces the removed C
/// `ds4-server` binary: no subprocess, no second copy of the weights — the same
/// loaded engine serves both the chat UI and HTTP clients.
///
/// The model is a single actor that holds one KV cache, so requests are
/// SERIALIZED (one generation at a time) via `RequestGate`. Each request is
/// stateless: the full message list is rendered fresh (OpenAI semantics).
///
/// Endpoints (faithful to ds4_server.c's wire format):
///   OPTIONS *                    → 204 (CORS preflight)
///   GET  /v1/models              → {"object":"list","data":[…]}
///   GET  /v1/models/{id}         → one model object
///   POST /v1/chat/completions    → chat.completion (or SSE chat.completion.chunk when stream:true)
final class LocalServer: @unchecked Sendable {
    struct Config: Sendable {
        var host: String
        var port: UInt16
        var cors: Bool
        var maxTokens: Int
        /// Optional shared secret: when set, every /v1 request must carry
        /// `Authorization: Bearer <key>` (OpenAI style) or `x-api-key: <key>`
        /// (Anthropic style). The transport stays plaintext HTTP — this guards
        /// against other local processes, not network eavesdroppers.
        var apiKey: String? = nil
    }

    /// Largest accepted request body (the whole transcript is re-sent each call,
    /// so this is generous; anything bigger is a client bug or abuse) → 413.
    static let maxBodyBytes = 32 * 1024 * 1024
    /// A client must deliver its full request within this window, or the
    /// connection is dropped — stalled sockets can't accumulate forever.
    static let readTimeout: UInt64 = 60 * 1_000_000_000

    /// Il motore che la chat ha caricato, dietro il contratto comune —
    /// DeepSeek o GLM, indifferente per il server.
    let backend: any ChatBackend
    let modelName: String         // display name (the GGUF file)
    /// The ONE model id the API advertises: the loaded GGUF's basename. There
    /// is no model choice — the server wraps the single engine loaded in
    /// Settings, so the request's "model" field is informational only.
    let modelId: String
    let config: Config
    let onLog: @Sendable (String) -> Void
    let queue = DispatchQueue(label: "ds4.localserver", qos: .userInitiated)
    let gate = RequestGate()

    /// Listener and accepted-request ownership are protected together so a
    /// connection is either rejected after shutdown begins, or is present in
    /// the exact task snapshot that `stop()` cancels and awaits.
    private struct ActiveRequest {
        let connection: NWConnection
        let task: Task<Void, Never>
    }
    private let lifecycleLock = NSLock()
    private var acceptingConnections = false
    private var listener: NWListener?
    private var activeRequests: [UUID: ActiveRequest] = [:]

    init(backend: any ChatBackend, modelName: String, config: Config,
         onLog: @escaping @Sendable (String) -> Void) {
        self.backend = backend
        self.modelName = modelName
        self.modelId = (modelName as NSString).deletingPathExtension
        self.config = config
        self.onLog = onLog
    }

    /// The engine can only serve the loaded model: a different requested id is
    /// logged and overridden, and every response reports the REAL model.
    func resolveModel(_ requested: String?) -> String {
        if let requested, requested != modelId {
            onLog("campo model \"\(requested)\" ignorato: il motore condiviso serve \(modelId)\n")
        }
        return modelId
    }

    // MARK: Lifecycle

    func start() throws {
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        guard let port = NWEndpoint.Port(rawValue: config.port) else {
            throw ServerError.badPort
        }
        params.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(config.host), port: port)

        let l = try NWListener(using: params)
        l.stateUpdateHandler = { [onLog, config, weak l] state in
            switch state {
            case .ready:
                let p = l?.port?.rawValue ?? config.port
                onLog("In ascolto su http://\(config.host):\(p)/v1\n")
            case .failed(let e): onLog("listener fallito: \(e)\n")
            default: break
            }
        }
        l.newConnectionHandler = { [weak self] conn in self?.accept(conn) }

        lifecycleLock.lock()
        acceptingConnections = true
        listener = l
        l.start(queue: queue)
        lifecycleLock.unlock()
    }

    /// Stop accepting work and wait until every accepted request, its inference
    /// task, and model-owned GPU/I/O work have drained.  This is the hard
    /// barrier used before `ServerController` releases the shared-engine lease.
    func stop() async {
        let (listenerToCancel, requests): (NWListener?, [ActiveRequest]) = lifecycleLock.withLock {
            acceptingConnections = false
            let listenerToCancel = listener
            listener = nil
            return (listenerToCancel, Array(activeRequests.values))
        }

        listenerToCancel?.cancel()
        if !requests.isEmpty {
            onLog("arresto server: annullo e dreno \(requests.count) richiesta/e attive…\n")
        }
        for request in requests {
            request.connection.cancel()
            request.task.cancel()
        }
        for request in requests {
            await request.task.value
        }

        // `InferenceService.run` owns the actual generation task behind an
        // AsyncThrowingStream. Dropping/cancelling the stream requests its
        // cancellation; this actor hop waits for that synchronous Metal loop
        // and all decoder/disk-KV background work to finish as well.
        await backend.quiesceForTeardown()
    }

    func accept(_ conn: NWConnection) {
        lifecycleLock.lock()
        guard acceptingConnections else {
            lifecycleLock.unlock()
            conn.cancel()
            return
        }

        let id = UUID()
        conn.start(queue: queue)
        let task = Task(priority: .userInitiated) { [weak self] in
            guard let self else {
                conn.cancel()
                return
            }
            await self.serve(conn)
            self.requestDidFinish(id)
        }
        // The lock remains held from the acceptance check through insertion.
        // A very short request may finish immediately, but its removal waits
        // for this insertion before taking the same lock.
        activeRequests[id] = ActiveRequest(connection: conn, task: task)
        lifecycleLock.unlock()
    }

    private func requestDidFinish(_ id: UUID) {
        lifecycleLock.lock()
        activeRequests.removeValue(forKey: id)
        lifecycleLock.unlock()
    }

    // MARK: Connection handling

    func serve(_ conn: NWConnection) async {
        do {
            guard let req = try await readRequest(conn) else { conn.cancel(); return }
            try await route(conn, req)
        } catch ServerError.timeout {
            // stalled client: the read already cancelled the connection
        } catch ServerError.bodyTooLarge {
            try? await send(conn, Self.httpError(413, "request body too large", cors: config.cors))
        } catch is CancellationError {
            // Expected during the async shutdown barrier.
        } catch is NWError {
            // client went away mid-response (SSE disconnects land here) — the
            // broken `for try await` already cancelled the generation via
            // onTermination; nothing useful can be sent on a dead socket.
        } catch let e as InferenceError {
            // Errore SEMANTICO (es. prompt oltre il contesto): il client deve
            // leggere il motivo — un 503 anonimo fa sembrare rotto il server
            // quando il problema è la richiesta.
            onLog("errore richiesta: \(e)\n")
            try? await send(conn, Self.httpError(400, "\(e)", cors: config.cors))
        } catch {
            onLog("errore richiesta: \(error)\n")
            try? await send(conn, Self.httpError(503, "internal error", cors: config.cors))
        }
        conn.cancel()
    }

    func route(_ conn: NWConnection, _ req: HTTPRequest) async throws {
        if req.method == "OPTIONS" {
            try await send(conn, Self.response(204, contentType: nil, body: "", cors: config.cors))
            return
        }
        if let key = config.apiKey, !key.isEmpty {
            let bearer = req.headers["authorization"] ?? ""
            let xKey = req.headers["x-api-key"] ?? ""
            guard bearer == "Bearer \(key)" || xKey == key else {
                onLog("401 \(req.method) \(req.path) (API key mancante o errata)\n")
                try await send(conn, Self.httpError(401, "invalid or missing API key", cors: config.cors))
                return
            }
        }
        // Probe dei client "locally hosted" (Xcode Intelligence, ecc.): la
        // base URL e /health devono rispondere 200, non 404 — alcuni client
        // scartano il provider al primo probe fallito.
        if req.method == "GET", req.path == "/" || req.path == "/health" {
            try await send(conn, Self.response(200, contentType: "application/json",
                                               body: "{\"status\":\"ok\",\"server\":\"dwarfstar\",\"model\":\(jsonString(modelId))}",
                                               cors: config.cors))
            return
        }
        if req.method == "GET", req.path == "/v1/models" {
            try await send(conn, Self.response(200, contentType: "application/json",
                                               body: modelsJSON(), cors: config.cors))
            return
        }
        let modelPrefix = "/v1/models/"
        if req.method == "GET", req.path.hasPrefix(modelPrefix) {
            // Qualunque id: il motore condiviso serve UN solo modello — un id
            // diverso (Xcode può normalizzare/troncare il nome) viene loggato
            // e risolto su quello caricato, come nel body delle completions.
            let id = String(req.path.dropFirst(modelPrefix.count))
            try await send(conn, Self.response(200, contentType: "application/json",
                                               body: modelJSON(resolveModel(id.isEmpty ? nil : id)),
                                               cors: config.cors))
            return
        }
        if req.method == "POST", req.path == "/v1/chat/completions" {
            try await handleChat(conn, body: req.body)
            return
        }
        if req.method == "POST", req.path == "/v1/messages" {
            try await handleAnthropic(conn, body: req.body)
            return
        }
        if req.method == "POST", req.path == "/v1/completions" {
            try await handleCompletions(conn, body: req.body)
            return
        }
        if req.method == "POST", req.path == "/v1/responses" {
            try await handleResponses(conn, body: req.body)
            return
        }
        try await send(conn, Self.httpError(404, "unknown endpoint", cors: config.cors))
    }
}
