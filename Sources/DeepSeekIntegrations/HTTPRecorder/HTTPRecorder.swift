import Foundation

/// Record and replay HTTP traffic for the chat layer. Inspired by
/// opencode's `http-recorder` package: useful for reproducible
/// integration tests against OpenRouter (and any future provider)
/// and for offline replays of regressions.
///
/// Wire it as a `URLProtocol` subclass on a custom `URLSession`
/// configuration so it intercepts only the sessions the host opts in.
/// Recording writes one `<sequence>.json` per request into the
/// `directory`; replay reads them back in order.
///
/// Status: scaffolded. The interceptor isn't wired into
/// `OpenRouterAPI` yet — that's a follow-up. Listed in TODO.md.
public final class HTTPRecorder: @unchecked Sendable {
    public enum Mode: String, Sendable {
        case off, record, replay
    }

    public let directory: URL
    public private(set) var mode: Mode

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
