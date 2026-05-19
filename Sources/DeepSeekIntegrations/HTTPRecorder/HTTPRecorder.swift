import Foundation

/// Record and replay HTTP traffic for the chat layer. Inspired by
/// opencode's `http-recorder` package: useful for reproducible
/// integration tests against OpenRouter (and any future provider)
/// and for offline replays of regressions.
///
/// Wire it via `HTTPRecorderURLProtocol`: callers register the
/// protocol on a `URLSessionConfiguration.protocolClasses` and
/// configure the active `HTTPRecorder.shared` instance. The
/// protocol intercepts every request on that session, forwards it
/// to a "real" `URLSession.shared` (when `.record`) or stops short
/// and replays a stored response (when `.replay`).
///
/// Recording writes one `<sequence>.json` per request into the
/// `directory`; replay reads them back in order.
public final class HTTPRecorder: @unchecked Sendable {
    /// Process-wide singleton. The `URLProtocol` subclass below
    /// reads its mode + directory from here; the GUI / CLI flips
    /// the mode via `setMode(_:)` and `configure(directory:mode:)`.
    /// Single global instance because URLProtocol subclasses can't
    /// carry per-instance state across `init(request:)` /
    /// `startLoading()` boundaries cleanly.
    public static let shared = HTTPRecorder(
        directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("DeepSeekHTTPRecorder"),
        mode: .off)

    /// Replace the singleton's recording directory + mode in one
    /// call. Useful so callers don't have to thread the
    /// `URLProtocol` registration through their setup separately.
    public func configure(directory: URL, mode: Mode) {
        lock.lock()
        self.directory = directory
        self.sequence = 0
        lock.unlock()
        setMode(mode)
    }
    public enum Mode: String, Sendable {
        case off, record, replay
    }

    public private(set) var directory: URL
    public private(set) var mode: Mode

    /// Live snapshot used by `HTTPRecorderURLProtocol` so its
    /// per-request handlers don't have to capture `self`. Same
    /// values as `directory` / `mode`; the indirection keeps
    /// reads lock-free on the hot path.
    fileprivate var activeDirectory: URL { directory }

    private var sequence: Int = 0
    private let lock = NSLock()

    public init(directory: URL, mode: Mode = .off) {
        self.directory = directory
        self.mode = mode
    }

    public func setMode(_ mode: Mode) {
        lock.lock(); defer { lock.unlock() }
        self.mode = mode
        if mode != .off {
            try? FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
        }
    }

    /// Persist one recorded interaction. The on-disk format is a
    /// hand-rolled JSON object: `{ "method": ..., "url": ...,
    /// "headers": {…}, "body": "<base64>", "response": {…} }`.
    /// Kept hand-rolled (no Codable) so future-proof against schema
    /// drift across runs.
    public func recordInteraction(method: String,
                                  url: URL,
                                  headers: [String: String],
                                  body: Data?,
                                  statusCode: Int,
                                  responseHeaders: [String: String],
                                  responseBody: Data) throws {
        lock.lock(); defer { lock.unlock() }
        sequence += 1
        let filename = String(format: "%04d.json", sequence)
        let target = directory.appendingPathComponent(filename)
        let payload: [String: Any] = [
            "method": method,
            "url": url.absoluteString,
            "headers": headers,
            "body": body?.base64EncodedString() ?? NSNull(),
            "response": [
                "statusCode": statusCode,
                "headers": responseHeaders,
                "body": responseBody.base64EncodedString(),
            ],
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload,
            options: [.prettyPrinted, .sortedKeys])
        try data.write(to: target, options: .atomic)
    }

    /// Sequence counter, exposed so the URLProtocol replay path
    /// can detect "ran out of recordings" without re-scanning.
    public var currentSequence: Int {
        lock.lock(); defer { lock.unlock() }
        return sequence
    }

    /// Load the next replay entry from disk. Returns `nil` when the
    /// directory has no further entries.
    public func nextReplay() throws -> (statusCode: Int,
                                        headers: [String: String],
                                        body: Data)? {
        lock.lock(); defer { lock.unlock() }
        sequence += 1
        let filename = String(format: "%04d.json", sequence)
        let target = directory.appendingPathComponent(filename)
        guard FileManager.default.fileExists(atPath: target.path) else { return nil }
        let data = try Data(contentsOf: target)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let resp = root["response"] as? [String: Any],
              let status = resp["statusCode"] as? Int,
              let headers = resp["headers"] as? [String: String],
              let bodyB64 = resp["body"] as? String,
              let body = Data(base64Encoded: bodyB64) else {
            throw NSError(domain: "HTTPRecorder", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "bad replay entry"])
        }
        return (status, headers, body)
    }
}

// MARK: - URLProtocol bridge

/// Hooks `HTTPRecorder.shared` into `URLSession` traffic. Register
/// it on a session's `protocolClasses` to intercept every request
/// the session issues. Behaviour:
///
///   `.off`     → no-op; the protocol claims it can't handle the
///                 request and the standard URL loader runs.
///   `.record`  → forwards the request to `URLSession.shared`
///                 (a fresh session with the recorder unregistered,
///                 to avoid recursion), captures the response, and
///                 persists both via `HTTPRecorder.shared
///                 .recordInteraction(...)`.
///   `.replay`  → reads the next stored interaction via
///                 `HTTPRecorder.shared.nextReplay()` and synthesises
///                 the response without touching the network. When
///                 no more recordings are available, fails the
///                 request with `URLError.resourceUnavailable`.
public final class HTTPRecorderURLProtocol: URLProtocol {
    public override class func canInit(with request: URLRequest) -> Bool {
        // Don't recurse into ourselves: a marker header that we
        // attach on the forward request distinguishes the "real"
        // outbound call from the original.
        if request.value(forHTTPHeaderField: Self.forwardMarker) != nil {
            return false
        }
        return HTTPRecorder.shared.mode != .off
    }

    public override class func canonicalRequest(for request: URLRequest)
        -> URLRequest
    {
        request
    }

    public override func startLoading() {
        let recorder = HTTPRecorder.shared
        switch recorder.mode {
        case .off:
            // Shouldn't happen — canInit returned false. Bail
            // defensively rather than hang.
            client?.urlProtocol(self, didFailWithError:
                URLError(.unknown))
        case .record:
            forwardAndRecord()
        case .replay:
            replayNext()
        }
    }

    public override func stopLoading() {
        // Best-effort cancel of any pending forward task.
        forwardTask?.cancel()
    }

    private var forwardTask: URLSessionDataTask?
    private static let forwardMarker = "X-DeepSeek-Recorder-Forward"

    private func forwardAndRecord() {
        var forward = request
        forward.setValue("1", forHTTPHeaderField: Self.forwardMarker)
        // Use a session that DOESN'T have us registered, otherwise
        // the recorder would recurse on its own forward.
        let cfg = URLSessionConfiguration.ephemeral
        cfg.protocolClasses = []
        let session = URLSession(configuration: cfg)
        forwardTask = session.dataTask(with: forward) { [weak self] data, response, error in
            guard let self else { return }
            if let error = error {
                self.client?.urlProtocol(self, didFailWithError: error)
                return
            }
            guard let http = response as? HTTPURLResponse,
                  let body = data
            else {
                self.client?.urlProtocol(self, didFailWithError:
                    URLError(.badServerResponse))
                return
            }
            // Persist the interaction. The marker header is
            // stripped before recording.
            var recordedHeaders = self.request.allHTTPHeaderFields ?? [:]
            recordedHeaders.removeValue(forKey: Self.forwardMarker)
            let respHeaders = (http.allHeaderFields as? [String: String])
                ?? [:]
            try? HTTPRecorder.shared.recordInteraction(
                method: self.request.httpMethod ?? "GET",
                url: self.request.url ?? URL(string: "about:blank")!,
                headers: recordedHeaders,
                body: self.request.httpBody,
                statusCode: http.statusCode,
                responseHeaders: respHeaders,
                responseBody: body)

            // Feed the captured response back to the original
            // client as if nothing happened.
            self.client?.urlProtocol(self, didReceive: http,
                                       cacheStoragePolicy: .notAllowed)
            self.client?.urlProtocol(self, didLoad: body)
            self.client?.urlProtocolDidFinishLoading(self)
        }
        forwardTask?.resume()
    }

    private func replayNext() {
        do {
            guard let entry = try HTTPRecorder.shared.nextReplay() else {
                client?.urlProtocol(self, didFailWithError:
                    URLError(.resourceUnavailable))
                return
            }
            let url = request.url ?? URL(string: "about:blank")!
            let response = HTTPURLResponse(
                url: url, statusCode: entry.statusCode,
                httpVersion: "HTTP/1.1",
                headerFields: entry.headers) ?? HTTPURLResponse()
            client?.urlProtocol(self, didReceive: response,
                                  cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: entry.body)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }
}
