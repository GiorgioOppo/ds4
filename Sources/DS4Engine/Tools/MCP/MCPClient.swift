import Foundation

// One MCP client connection: owns the transport, performs the initialize
// handshake, routes JSON-RPC responses back to their awaiting requests, and
// exposes the two operations the app needs — list tools and call a tool.

enum MCPClientError: Error, CustomStringConvertible {
    case server(String)          // JSON-RPC error object from the server
    case timeout(String)         // no response within the deadline
    case disconnected(String)    // transport closed underneath us
    case protocolError(String)   // malformed / unexpected payload
    var description: String {
        switch self {
        case .server(let m): return "server error: \(m)"
        case .timeout(let m): return "timeout waiting for \(m)"
        case .disconnected(let m): return m.isEmpty ? "disconnected" : m
        case .protocolError(let m): return "protocol error: \(m)"
        }
    }
}

actor MCPClient {
    private let config: MCPServerConfig
    private var transport: (any MCPTransport)?
    private var nextID = 1
    /// In-flight requests by id: the continuation (resumed with the RAW response
    /// frame — Data keeps the payload Sendable and avoids a re-serialization)
    /// plus its deadline task, cancelled the moment the request settles.
    private var pending: [Int: (cont: CheckedContinuation<Data, Error>,
                                deadline: Task<Void, Never>)] = [:]
    private var closedReason: String?
    /// Fired once when the transport dies underneath us (process exit); NOT
    /// fired on shutdown(). Passes `self` so the owner can ignore stale clients.
    private let onDisconnect: @Sendable (MCPClient, String) -> Void

    init(config: MCPServerConfig, onDisconnect: @escaping @Sendable (MCPClient, String) -> Void) {
        self.config = config
        self.onDisconnect = onDisconnect
    }

    /// stderr tail of a stdio server (best-effort diagnostics for error states).
    var stderrTail: String {
        (transport as? MCPStdioTransport)?.stderrSnapshot ?? ""
    }

    // MARK: Lifecycle

    /// Open the transport, run `initialize` + `notifications/initialized`, and
    /// fetch the full tool list. Returns the server's name/version string.
    func connect() async throws -> (serverInfo: String, tools: [MCPToolInfo]) {
        guard transport == nil else { throw MCPClientError.protocolError("already connected") }
        let t: any MCPTransport
        switch config.transport {
        case .stdio(let command, let arguments, let environment):
            t = MCPStdioTransport(command: command, arguments: arguments, environment: environment)
        case .http(let url, let headers):
            t = try MCPHTTPTransport(urlString: url, headers: headers)
        }
        transport = t
        try t.start(
            onMessage: { [weak self] frame in
                guard let self else { return }
                Task { await self.handle(frame) }
            },
            onClose: { [weak self] reason in
                guard let self else { return }
                Task { await self.transportClosed(reason) }
            })

        let initFrame = try await request(method: "initialize",
                                          params: MCPProtocol.initializeParams(),
                                          timeout: 30)
        let info = try Self.resultDict(initFrame)
        let serverInfo = (info["serverInfo"] as? [String: Any]).map {
            "\(($0["name"] as? String) ?? "?") \(($0["version"] as? String) ?? "")"
                .trimmingCharacters(in: .whitespaces)
        } ?? ""
        try await t.send(MCPProtocol.notification(method: "notifications/initialized"))

        // Fetch every page of tools/list. Both bounds guard against a
        // misbehaving server: one that pages forever, and one that returns
        // empty pages with a nextCursor (tools.count would never grow).
        var tools: [MCPToolInfo] = []
        var cursor: String?
        var pages = 0
        repeat {
            let params: [String: Any]? = cursor.map { ["cursor": $0] }
            let page = try Self.resultDict(try await request(method: "tools/list",
                                                             params: params, timeout: 30))
            let (pageTools, next) = MCPProtocol.parseToolsList(page)
            tools += pageTools
            cursor = next
            pages += 1
        } while cursor != nil && pages < 16 && tools.count < 512

        return (serverInfo, tools)
    }

    /// Call a tool with the model's raw JSON arguments; returns the flattened
    /// text result (errors from the server are thrown, `isError` results are
    /// returned as "Tool error: …" text so the model can react).
    func callTool(name: String, argumentsJSON: String) async throws -> String {
        var arguments: [String: Any] = [:]
        if let data = argumentsJSON.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            arguments = obj
        }
        let frame = try await request(method: "tools/call",
                                      params: ["name": name, "arguments": arguments],
                                      timeout: 120)
        return MCPProtocol.flattenCallResult(try Self.resultDict(frame))
    }

    /// Close the transport and fail whatever is still in flight. Idempotent.
    func shutdown() {
        failAllPending(with: MCPClientError.disconnected("client shut down"))
        closedReason = closedReason ?? "client shut down"
        transport?.close()
        transport = nil
    }

    // MARK: Internals

    /// One JSON-RPC round-trip, returning the raw response frame. Cancellation-
    /// aware: cancelling the awaiting task (the user's Stop) settles the request
    /// immediately with CancellationError instead of blocking until the timeout.
    private func request(method: String, params: [String: Any]?,
                         timeout: TimeInterval) async throws -> Data {
        try Task.checkCancellation()   // already-cancelled caller: don't even send
        if let closedReason { throw MCPClientError.disconnected(closedReason) }
        guard let transport else { throw MCPClientError.disconnected("no transport") }
        let id = nextID
        nextID += 1
        let frame = MCPProtocol.request(id: id, method: method, params: params)

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { cont in
                let deadline = Task { [weak self] in
                    try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                    await self?.settle(id: id, with: .failure(MCPClientError.timeout(method)))
                }
                pending[id] = (cont, deadline)
                Task {
                    do { try await transport.send(frame) }
                    catch { self.settle(id: id, with: .failure(error)) }
                }
            }
        } onCancel: {
            Task { await self.settle(id: id, with: .failure(CancellationError())) }
        }
    }

    private func handle(_ frame: Data) {
        switch MCPProtocol.classify(frame) {
        case .response(let id, _, let errorMessage):
            if let errorMessage {
                settle(id: id, with: .failure(MCPClientError.server(errorMessage)))
            } else {
                settle(id: id, with: .success(frame))
            }
        case .serverRequest(let id, let method):
            // Answer pings; refuse anything else (sampling, roots, elicitation).
            let reply = method == "ping" ? MCPProtocol.emptyResponse(id: id)
                                         : MCPProtocol.methodNotFound(id: id, method: method)
            let t = transport
            Task { try? await t?.send(reply) }
        case .notification, .invalid:
            break   // logging / list_changed etc. — nothing to route
        }
    }

    private func settle(id: Int, with result: Result<Data, Error>) {
        guard let entry = pending.removeValue(forKey: id) else { return }
        entry.deadline.cancel()
        entry.cont.resume(with: result)
    }

    private func transportClosed(_ reason: String) {
        guard closedReason == nil else { return }
        closedReason = reason
        failAllPending(with: MCPClientError.disconnected(reason))
        transport = nil
        onDisconnect(self, reason)
    }

    private func failAllPending(with error: Error) {
        let entries = pending.values
        pending.removeAll()
        for e in entries {
            e.deadline.cancel()
            e.cont.resume(throwing: error)
        }
    }

    /// Extract the JSON-RPC `result` object from a raw response frame.
    private static func resultDict(_ frame: Data) throws -> [String: Any] {
        guard let obj = try? JSONSerialization.jsonObject(with: frame) as? [String: Any],
              let result = obj["result"] as? [String: Any] else {
            throw MCPClientError.protocolError("result is not a JSON object")
        }
        return result
    }
}
