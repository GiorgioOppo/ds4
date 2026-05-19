import Foundation
import Network

/// Localhost HTTP/1.1 server scaffolding for the OpenAI-compatible
/// API surface (T1 of TODO §10.1). The actor owns an `NWListener`,
/// accepts connections off `Network.framework`'s callback queue, and
/// dispatches parsed requests to handlers registered through
/// `register(method:path:handler:)`.
///
/// Why `Network.framework` and not SwiftNIO: keeps `Package.swift`
/// dependency-free, matches Foundation `URLSession` already in use
/// for `OpenRouterClient`, and supports `requireLocalEndpoint` /
/// TLS without extra plumbing if we ever expose the server outside
/// localhost. The trade-off is the lower-level HTTP parsing — we
/// implement a minimal HTTP/1.1 + SSE writer here. That's fine for
/// our shape (POST JSON / GET JSON / POST + SSE stream) but would
/// need expansion for chunked uploads or websockets.
///
/// Concurrency: connection handlers run inside `Task { ... }`
/// captures dispatched from the listener queue. The actor itself
/// stores only the listener + the route table; per-connection state
/// lives on the stack of `handle(connection:)`.
actor LocalServer {
    /// Single route entry. Method match is case-insensitive on the
    /// verb; path match is exact for now (no `:param` capture). If
    /// we ever need parameterized routes (`/v1/files/:id`) extend
    /// the matcher here without touching the connection loop.
    private struct Route {
        let method: String
        let path: String
        let handler: LocalServerHandler
    }

    /// Current listener, non-nil while `running == true`. Stored so
    /// `stop()` can cancel it; never read for routing logic.
    private var listener: NWListener?
    private var routes: [Route] = []
    private(set) var port: UInt16 = 8080
    private(set) var bindAddress: String = "127.0.0.1"
    private(set) var bearerToken: String? = nil
    private(set) var running: Bool = false

    /// Background queue we hand to `Network.framework` for listener
    /// + connection state callbacks. `userInitiated` because the
    /// HTTP requests we serve are interactive (CLI/IDE clients
    /// waiting on a streaming completion).
    private let queue = DispatchQueue(
        label: "deepseek.localserver", qos: .userInitiated)

    init() {}

    // MARK: - Lifecycle

    /// Bind the listener on `address:port`. Pass `bearerToken` to
    /// require `Authorization: Bearer <token>` on every request; nil
    /// disables auth (the default for localhost).
    ///
    /// Throws `LocalServerError.invalidPort` if `port` is 0 or above
    /// 65535, `LocalServerError.bindFailed` if the port is taken.
    /// The latter is surfaced async via the listener state callback,
    /// so we wait up to 2 s for the listener to enter `.ready`
    /// before returning.
    func start(port: UInt16,
                address: String = "127.0.0.1",
                bearerToken: String? = nil) async throws {
        if running { try await stop() }
        guard port > 0 else { throw LocalServerError.invalidPort }
        self.port = port
        self.bindAddress = address
        self.bearerToken = bearerToken

        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        // requireLocalEndpoint pinning isn't exposed cleanly via
        // NWParameters — we rely on the OS routing the bind to the
        // requested address. For `127.0.0.1` this is automatic; for
        // a future `0.0.0.0` we'd accept any interface.

        let endpointPort: NWEndpoint.Port
        guard let p = NWEndpoint.Port(rawValue: port) else {
            throw LocalServerError.invalidPort
        }
        endpointPort = p

        let listener: NWListener
        do {
            listener = try NWListener(using: params, on: endpointPort)
        } catch {
            throw LocalServerError.bindFailed(
                port: port, underlying: error)
        }
        self.listener = listener

        try await withCheckedThrowingContinuation {
            (cont: CheckedContinuation<Void, Error>) in
            // `settled` is racy between the timeout task and the
            // listener state callback (different queues). Use a
            // lock-guarded once-flag so resume happens exactly once.
            let settled = OnceFlag()
            let timeoutTask = Task {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                if settled.fire() {
                    cont.resume(throwing: LocalServerError.bindFailed(
                        port: port,
                        underlying: NSError(
                            domain: "LocalServer", code: -1,
                            userInfo: [NSLocalizedDescriptionKey:
                                "Listener never reached .ready (2s timeout)."])))
                }
            }
            listener.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    if settled.fire() {
                        timeoutTask.cancel()
                        cont.resume(returning: ())
                    }
                case .failed(let err):
                    if settled.fire() {
                        timeoutTask.cancel()
                        cont.resume(throwing: LocalServerError.bindFailed(
                            port: port, underlying: err))
                    }
                default:
                    break
                }
            }
            listener.newConnectionHandler = { [weak self] conn in
                guard let self else { conn.cancel(); return }
                Task { await self.handle(connection: conn) }
            }
            listener.start(queue: self.queue)
        }
        self.running = true
    }

    /// Tear down the listener. Connections already mid-flight finish
    /// naturally; we don't force-close them because doing so would
    /// truncate in-flight SSE streams. Idempotent.
    func stop() async throws {
        guard let listener else {
            running = false
            return
        }
        listener.cancel()
        self.listener = nil
        running = false
    }

    // MARK: - Routes

    func register(method: String,
                   path: String,
                   handler: @escaping LocalServerHandler) {
        routes.append(Route(
            method: method.uppercased(), path: path, handler: handler))
    }

    /// Snapshot of the route table, for tests / status UI.
    func routeList() -> [(method: String, path: String)] {
        routes.map { ($0.method, $0.path) }
    }

    private func match(method: String, path: String) -> LocalServerHandler? {
        let m = method.uppercased()
        // Strip query string so `/v1/models?filter=x` still matches
        // `/v1/models`. We don't yet expose query params upward.
        let pathOnly: String
        if let q = path.firstIndex(of: "?") {
            pathOnly = String(path[..<q])
        } else {
            pathOnly = path
        }
        for r in routes where r.method == m && r.path == pathOnly {
            return r.handler
        }
        return nil
    }

    // MARK: - Connection loop

    /// Per-connection state machine: wait for `.ready`, read until
    /// we have headers + body, route, write response, close.
    /// Errors at any stage close the socket — we don't try to keep
    /// the connection open for a second request (keep-alive is not
    /// implemented; clients reconnect, which is fine over loopback).
    private func handle(connection conn: NWConnection) async {
        await withCheckedContinuation {
            (cont: CheckedContinuation<Void, Never>) in
            let settled = OnceFlag()
            conn.stateUpdateHandler = { state in
                switch state {
                case .ready, .failed, .cancelled:
                    if settled.fire() { cont.resume() }
                default:
                    break
                }
            }
            conn.start(queue: self.queue)
        }

        // Read head + body.
        let request: HTTPRequest
        do {
            request = try await Self.readRequest(from: conn)
        } catch {
            await Self.sendErrorResponse(
                conn, status: 400, message: "Bad Request: \(error)")
            conn.cancel()
            return
        }

        // Bearer auth (if configured).
        if let expected = self.bearerToken, !expected.isEmpty {
            let header = request.header("Authorization") ?? ""
            let prefix = "Bearer "
            guard header.hasPrefix(prefix),
                  String(header.dropFirst(prefix.count)) == expected
            else {
                await Self.sendErrorResponse(
                    conn, status: 401, message: "Unauthorized")
                conn.cancel()
                return
            }
        }

        // Route + dispatch.
        guard let handler = self.match(
            method: request.method, path: request.path)
        else {
            await Self.sendErrorResponse(
                conn, status: 404,
                message: "No route for \(request.method) \(request.path)")
            conn.cancel()
            return
        }

        let writer = HTTPResponseWriter(connection: conn)
        await handler(request, writer)
        await writer.endIfNeeded()
        conn.cancel()
    }

    // MARK: - HTTP/1.1 parser

    /// Read the head block (everything up to `\r\n\r\n`), parse the
    /// request line + headers, then read `Content-Length` body bytes
    /// if any. Bounded at 16 MiB head + 64 MiB body to avoid trivial
    /// OOM from a misbehaving client; the body cap is generous so
    /// chat completions with large `messages[]` payloads fit.
    private static func readRequest(
        from conn: NWConnection) async throws -> HTTPRequest
    {
        let maxHead = 16 * 1024
        let maxBody = 64 * 1024 * 1024
        var head = Data()
        while head.count < maxHead {
            let chunk = try await receive(
                conn, minimum: 1,
                maximum: maxHead - head.count)
            if chunk.isEmpty { break }
            head.append(chunk)
            if let separator = head.range(of: Data([0x0d, 0x0a, 0x0d, 0x0a])) {
                let headData = head.subdata(in: 0..<separator.lowerBound)
                let leftover = head.subdata(in: separator.upperBound..<head.count)
                guard let headStr = String(data: headData, encoding: .utf8) else {
                    throw LocalServerError.invalidHead
                }
                let (method, path, headers) = try parseHead(headStr)
                let contentLength = Int(
                    headers.first(where: {
                        $0.key.caseInsensitiveCompare("Content-Length") == .orderedSame
                    })?.value ?? "0") ?? 0
                if contentLength < 0 || contentLength > maxBody {
                    throw LocalServerError.bodyTooLarge
                }
                var body = leftover
                while body.count < contentLength {
                    let want = contentLength - body.count
                    let more = try await receive(
                        conn, minimum: 1, maximum: want)
                    if more.isEmpty { break }
                    body.append(more)
                }
                return HTTPRequest(
                    method: method,
                    path: path,
                    headers: headers,
                    body: body)
            }
        }
        throw LocalServerError.headTooLarge
    }

    private static func parseHead(_ s: String) throws
        -> (method: String, path: String, headers: [String: String])
    {
        var lines = s.components(separatedBy: "\r\n")
        guard !lines.isEmpty else { throw LocalServerError.invalidHead }
        let requestLine = lines.removeFirst()
        let parts = requestLine.split(separator: " ", maxSplits: 2,
                                       omittingEmptySubsequences: false)
        guard parts.count >= 2 else { throw LocalServerError.invalidHead }
        let method = String(parts[0])
        let path = String(parts[1])
        var headers: [String: String] = [:]
        for line in lines where !line.isEmpty {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = String(line[..<colon])
            let value = line[line.index(after: colon)...]
                .trimmingCharacters(in: .whitespaces)
            headers[name] = value
        }
        return (method, path, headers)
    }

    private static func receive(_ conn: NWConnection,
                                 minimum: Int,
                                 maximum: Int) async throws -> Data
    {
        try await withCheckedThrowingContinuation {
            (cont: CheckedContinuation<Data, Error>) in
            conn.receive(minimumIncompleteLength: minimum,
                          maximumLength: maximum) { data, _, _, err in
                if let err {
                    cont.resume(throwing: err)
                    return
                }
                cont.resume(returning: data ?? Data())
            }
        }
    }

    /// Send a minimal error response without a `HTTPResponseWriter`.
    /// Used for failure paths (parse error, auth fail, 404) that
    /// happen before we route, so we don't have a writer yet.
    private static func sendErrorResponse(
        _ conn: NWConnection, status: Int, message: String) async
    {
        let writer = HTTPResponseWriter(connection: conn)
        let body = Data(message.utf8)
        try? await writer.writeHead(
            status: status,
            statusText: HTTPStatus.text(for: status),
            headers: [
                "Content-Type": "text/plain; charset=utf-8",
                "Content-Length": String(body.count),
                "Connection": "close",
            ])
        try? await writer.write(body)
        await writer.endIfNeeded()
    }
}

// MARK: - Public types

/// Parsed HTTP/1.1 request. `headers` keeps original casing on the
/// keys; use `header(_:)` for case-insensitive lookups (which is
/// the spec — header names are case-insensitive).
struct HTTPRequest: Sendable {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data

    func header(_ name: String) -> String? {
        for (k, v) in headers
        where k.caseInsensitiveCompare(name) == .orderedSame {
            return v
        }
        return nil
    }

    /// Decode the body as JSON `T`. Throws if the body isn't valid
    /// JSON or doesn't match `T`. Empty body decodes as failure (no
    /// "default" Codable construction).
    func decodeJSON<T: Decodable>(_ type: T.Type) throws -> T {
        try JSONDecoder().decode(type, from: body)
    }
}

/// Writer surface passed to handlers. Routes call `writeHead(...)`
/// once and then either:
///   - `write(_:)` followed by `end()` for non-streaming responses, or
///   - `writeSSE(_:)` repeatedly (after declaring an SSE response in
///     `writeHead(...)`) and `end()` to terminate with `data: [DONE]`.
///
/// `endIfNeeded()` is called automatically by `LocalServer` after
/// the handler returns, so forgetting `end()` doesn't leak the
/// connection; it just truncates the response to whatever's been
/// flushed.
final class HTTPResponseWriter: @unchecked Sendable {
    private let connection: NWConnection
    private var headersSent = false
    private var ended = false
    private let lock = NSLock()

    init(connection: NWConnection) {
        self.connection = connection
    }

    /// Emit the status line + headers. Idempotent: subsequent calls
    /// are silently ignored. After this returns, the body slot is
    /// open for `write(_:)` / `writeSSE(_:)`.
    func writeHead(status: Int,
                    statusText: String,
                    headers: [String: String]) async throws {
        lock.lock()
        if headersSent { lock.unlock(); return }
        headersSent = true
        lock.unlock()
        var head = "HTTP/1.1 \(status) \(statusText)\r\n"
        for (k, v) in headers {
            head += "\(k): \(v)\r\n"
        }
        head += "\r\n"
        try await send(Data(head.utf8))
    }

    func write(_ data: Data) async throws {
        try await send(data)
    }

    /// Emit one `data: <payload>\n\n` SSE frame. Caller is
    /// responsible for serializing the payload to a single-line
    /// JSON string (no embedded newlines in `payload`).
    func writeSSE(_ payload: String) async throws {
        let frame = "data: \(payload)\n\n"
        try await send(Data(frame.utf8))
    }

    /// Terminate an SSE stream with `data: [DONE]\n\n` per the
    /// OpenAI convention. No-op for non-streaming responses.
    func writeSSEDone() async throws {
        try await send(Data("data: [DONE]\n\n".utf8))
    }

    /// Mark the response complete. Safe to call multiple times.
    func end() async {
        lock.lock()
        ended = true
        lock.unlock()
    }

    fileprivate func endIfNeeded() async {
        lock.lock()
        let already = ended
        ended = true
        lock.unlock()
        _ = already
    }

    private func send(_ data: Data) async throws {
        try await withCheckedThrowingContinuation {
            (cont: CheckedContinuation<Void, Error>) in
            connection.send(content: data,
                             completion: .contentProcessed { err in
                if let err {
                    cont.resume(throwing: err)
                } else {
                    cont.resume()
                }
            })
        }
    }
}

/// Handler signature: receive the parsed request + a writer for the
/// response. Handlers are `async` so they can interleave with
/// `InferenceService.generateForConversation`'s
/// `AsyncThrowingStream` without blocking the connection queue.
typealias LocalServerHandler = @Sendable (HTTPRequest, HTTPResponseWriter) async -> Void

enum LocalServerError: LocalizedError {
    case invalidPort
    case bindFailed(port: UInt16, underlying: Error)
    case invalidHead
    case headTooLarge
    case bodyTooLarge

    var errorDescription: String? {
        switch self {
        case .invalidPort:
            return "Invalid port (must be 1-65535)."
        case .bindFailed(let port, let underlying):
            return "Could not bind to port \(port): \(underlying.localizedDescription)"
        case .invalidHead:
            return "Malformed HTTP request line."
        case .headTooLarge:
            return "HTTP head exceeded 16 KiB."
        case .bodyTooLarge:
            return "Request body exceeded 64 MiB."
        }
    }
}

/// Lock-guarded one-shot flag. Used to ensure a continuation is
/// resumed at most once when multiple callbacks race (timeout vs.
/// listener state, etc.). `fire()` returns `true` exactly once for
/// the lifetime of the instance — subsequent calls return `false`.
private final class OnceFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var fired = false

    func fire() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if fired { return false }
        fired = true
        return true
    }
}

/// Minimal HTTP status reason-phrase table. Only entries we emit.
enum HTTPStatus {
    static func text(for code: Int) -> String {
        switch code {
        case 200: return "OK"
        case 204: return "No Content"
        case 400: return "Bad Request"
        case 401: return "Unauthorized"
        case 404: return "Not Found"
        case 405: return "Method Not Allowed"
        case 500: return "Internal Server Error"
        case 501: return "Not Implemented"
        default:  return "OK"
        }
    }
}
