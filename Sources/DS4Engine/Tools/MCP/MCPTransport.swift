import Foundation

// MCP transports: how JSON-RPC frames physically reach a server. Both
// implementations push every INCOMING frame through `onMessage` (set once,
// before the first send), so MCPClient can treat stdio and HTTP uniformly as
// a stream of frames routed by request id.

enum MCPTransportError: Error, CustomStringConvertible {
    case notStarted(String)
    case processExited
    case badURL(String)
    case http(Int, String)
    case network(String)
    var description: String {
        switch self {
        case .notStarted(let m): return "server not started: \(m)"
        case .processExited: return "server process exited"
        case .badURL(let u): return "invalid URL: \(u)"
        case .http(let code, let body):
            return "HTTP \(code)" + (body.isEmpty ? "" : ": \(String(body.prefix(200)))")
        case .network(let m): return "network error: \(m)"
        }
    }
}

protocol MCPTransport: AnyObject, Sendable {
    /// Install the incoming-frame / closed callbacks, then open the transport.
    func start(onMessage: @escaping @Sendable (Data) -> Void,
               onClose: @escaping @Sendable (String) -> Void) throws
    /// Send one JSON-RPC frame (any response frames arrive via `onMessage`).
    func send(_ frame: Data) async throws
    func close()
}

// MARK: - stdio

/// Spawns the server as a child process and speaks newline-delimited JSON-RPC
/// over its stdin/stdout (the MCP stdio transport). stderr is kept in a small
/// tail buffer so a crashing server's diagnostics can be shown to the user.
final class MCPStdioTransport: MCPTransport, @unchecked Sendable {
    private let command: String
    private let arguments: [String]
    private let environment: [String: String]

    private let lock = NSLock()
    private var process: Process?
    private var stdinPipe: Pipe?
    private var buffer = Data()
    private var stderrTail = ""
    private var onMessage: (@Sendable (Data) -> Void)?
    private var onClose: (@Sendable (String) -> Void)?
    private var closed = false

    /// Hard cap on one stdout frame: a non-MCP binary streaming newline-free
    /// output would otherwise grow `buffer` without bound.
    private static let maxFrameBytes = 8 * 1024 * 1024

    init(command: String, arguments: [String], environment: [String: String]) {
        self.command = command; self.arguments = arguments; self.environment = environment
    }

    /// Last stderr lines of the child (error diagnostics for the UI).
    var stderrSnapshot: String { lock.lock(); defer { lock.unlock() }; return stderrTail }

    func start(onMessage: @escaping @Sendable (Data) -> Void,
               onClose: @escaping @Sendable (String) -> Void) throws {
        self.onMessage = onMessage
        self.onClose = onClose

        let p = Process()
        // Resolve the command through /usr/bin/env: MCP servers are typically
        // launched as `npx`/`uvx`/`node`, and a GUI app's PATH lacks the usual
        // package-manager prefixes — extend it before resolving.
        p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        p.arguments = [command] + arguments
        var env = ProcessInfo.processInfo.environment
        let extraPaths = ["/usr/local/bin", "/opt/homebrew/bin", "/opt/homebrew/sbin",
                          (NSHomeDirectory() as NSString).appendingPathComponent(".local/bin")]
        let basePath = env["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
        let current = Set(basePath.split(separator: ":").map(String.init))
        env["PATH"] = (extraPaths.filter { !current.contains($0) } + [basePath]).joined(separator: ":")
        for (k, v) in environment { env[k] = v }
        p.environment = env

        let inPipe = Pipe(), outPipe = Pipe(), errPipe = Pipe()
        p.standardInput = inPipe
        p.standardOutput = outPipe
        p.standardError = errPipe

        outPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard let self else { return }
            if data.isEmpty { handle.readabilityHandler = nil; return }   // EOF
            self.consume(data)
        }
        errPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty { handle.readabilityHandler = nil; return }   // EOF: stop the callbacks
            guard let self, let s = String(data: data, encoding: .utf8) else { return }
            self.lock.lock()
            self.stderrTail = String((self.stderrTail + s).suffix(2000))
            self.lock.unlock()
        }
        p.terminationHandler = { [weak self] _ in
            guard let self else { return }
            self.lock.lock()
            let wasClosed = self.closed
            self.closed = true
            let tail = self.stderrTail.trimmingCharacters(in: .whitespacesAndNewlines)
            self.lock.unlock()
            if !wasClosed {
                onClose(tail.isEmpty ? "server process exited"
                                     : "server process exited: \(String(tail.suffix(300)))")
            }
        }

        do { try p.run() } catch {
            throw MCPTransportError.notStarted("\(command): \(error.localizedDescription)")
        }
        lock.lock(); process = p; stdinPipe = inPipe; lock.unlock()
    }

    /// Split the stdout byte stream into newline-delimited frames.
    private func consume(_ data: Data) {
        lock.lock()
        buffer.append(data)
        var frames: [Data] = []
        while let nl = buffer.firstIndex(of: 0x0A) {
            let line = buffer.subdata(in: buffer.startIndex..<nl)
            buffer.removeSubrange(buffer.startIndex...nl)
            if !line.isEmpty { frames.append(line) }
        }
        let overflow = buffer.count > Self.maxFrameBytes
        let handler = onMessage
        let closer = onClose
        lock.unlock()
        for f in frames { handler?(f) }
        if overflow {   // not speaking line-delimited JSON-RPC: tear down
            close()
            closer?("server output is not MCP framing (single frame exceeded \(Self.maxFrameBytes / (1024 * 1024)) MB)")
        }
    }

    func send(_ frame: Data) async throws {
        lock.lock()
        let pipe = stdinPipe
        let alive = (process?.isRunning ?? false) && !closed
        lock.unlock()
        guard let pipe, alive else { throw MCPTransportError.processExited }
        var line = frame
        line.append(0x0A)
        // write(contentsOf:) raises an ObjC exception on a broken pipe; the
        // POSIX write on the descriptor returns an error we can check instead.
        // Loop: a pipe can accept fewer bytes than asked on large frames.
        let fd = pipe.fileHandleForWriting.fileDescriptor
        let ok = line.withUnsafeBytes { buf -> Bool in
            var off = 0
            while off < buf.count {
                let n = write(fd, buf.baseAddress!.advanced(by: off), buf.count - off)
                if n <= 0 {
                    if errno == EINTR { continue }
                    return false
                }
                off += n
            }
            return true
        }
        guard ok else { throw MCPTransportError.processExited }
    }

    func close() {
        lock.lock()
        closed = true
        let p = process
        process = nil
        let pipe = stdinPipe
        stdinPipe = nil
        lock.unlock()
        try? pipe?.fileHandleForWriting.close()   // EOF on stdin: polite shutdown
        guard let p, p.isRunning else { return }
        p.terminate()
        // Escalate to SIGKILL if the server ignores SIGTERM.
        DispatchQueue.global().asyncAfter(deadline: .now() + 3) {
            if p.isRunning { kill(p.processIdentifier, SIGKILL) }
        }
    }
}

// MARK: - Streamable HTTP

/// Streamable-HTTP transport: each frame is POSTed to the endpoint; the
/// response is either a single JSON message or a short SSE stream whose
/// `data:` events are the response frames. The `Mcp-Session-Id` issued by the
/// server on `initialize` is echoed on every later request.
final class MCPHTTPTransport: MCPTransport, @unchecked Sendable {
    private let url: URL
    private let headers: [String: String]
    private let session: URLSession

    private let lock = NSLock()
    private var sessionID: String?
    private var onMessage: (@Sendable (Data) -> Void)?

    init(urlString: String, headers: [String: String]) throws {
        guard let u = URL(string: urlString), let scheme = u.scheme?.lowercased(),
              scheme == "http" || scheme == "https",
              let host = u.host, !host.isEmpty else {   // "http://" alone parses as a URL
            throw MCPTransportError.badURL(urlString)
        }
        self.url = u
        self.headers = headers
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 120   // tool calls can be slow
        self.session = URLSession(configuration: cfg)
    }

    func start(onMessage: @escaping @Sendable (Data) -> Void,
               onClose: @escaping @Sendable (String) -> Void) throws {
        lock.lock(); self.onMessage = onMessage; lock.unlock()
        // Nothing to open: connections are per-POST. onClose is unused — an
        // HTTP failure surfaces on the send that hit it.
    }

    func send(_ frame: Data) async throws {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.httpBody = frame
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("application/json, text/event-stream", forHTTPHeaderField: "Accept")
        req.setValue(MCP.protocolVersion, forHTTPHeaderField: "MCP-Protocol-Version")
        lock.lock(); let sid = sessionID; lock.unlock()
        if let sid { req.setValue(sid, forHTTPHeaderField: "Mcp-Session-Id") }
        for (k, v) in headers { req.setValue(v, forHTTPHeaderField: k) }

        let data: Data, resp: URLResponse
        do { (data, resp) = try await session.data(for: req) }
        catch { throw MCPTransportError.network(error.localizedDescription) }
        guard let http = resp as? HTTPURLResponse else {
            throw MCPTransportError.network("no HTTP response")
        }
        if let sid = http.value(forHTTPHeaderField: "Mcp-Session-Id") {
            lock.lock(); sessionID = sid; lock.unlock()
        }
        guard (200..<300).contains(http.statusCode) else {
            throw MCPTransportError.http(http.statusCode,
                                         String(data: data, encoding: .utf8) ?? "")
        }
        // 202/empty body: an accepted notification — nothing to deliver.
        guard !data.isEmpty else { return }
        lock.lock(); let handler = onMessage; lock.unlock()
        let contentType = (http.value(forHTTPHeaderField: "Content-Type") ?? "").lowercased()
        if contentType.contains("text/event-stream") {
            for frame in Self.sseFrames(String(data: data, encoding: .utf8) ?? "") {
                handler?(Data(frame.utf8))
            }
        } else {
            handler?(data)
        }
    }

    func close() { session.invalidateAndCancel() }

    /// Extract the `data:` payloads from an SSE body (multi-line data fields are
    /// joined with "\n" per the SSE spec; events are separated by blank lines).
    static func sseFrames(_ body: String) -> [String] {
        var frames: [String] = []
        var current: [String] = []
        for rawLine in body.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.hasSuffix("\r") ? String(rawLine.dropLast()) : String(rawLine)
            if line.isEmpty {
                if !current.isEmpty { frames.append(current.joined(separator: "\n")); current = [] }
                continue
            }
            guard line.hasPrefix("data:") else { continue }   // ignore event:/id:/retry:
            var payload = String(line.dropFirst(5))
            if payload.hasPrefix(" ") { payload.removeFirst() }
            current.append(payload)
        }
        if !current.isEmpty { frames.append(current.joined(separator: "\n")) }
        return frames
    }
}
