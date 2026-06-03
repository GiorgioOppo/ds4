import Foundation
import DeepSeekIntegrations

// MARK: - Types

/// One model entry as returned by `GET /api/v1/models`. Only the
/// fields the picker / cost banner actually use are decoded; the
/// rest of the (large) catalog payload is ignored to keep the
/// in-memory size bounded.
struct OpenRouterModel: Codable, Identifiable, Hashable {
    let id: String              // e.g. "anthropic/claude-3.5-sonnet"
    let name: String?           // human-readable display name
    let description: String?
    let contextLength: Int?
    let pricing: Pricing?

    struct Pricing: Codable, Hashable {
        /// Per-token cost in USD. OpenRouter returns these as
        /// strings (sometimes with scientific notation like "1e-6")
        /// — kept as String to avoid Double precision loss in the
        /// JSON round-trip, parsed lazily by the cost banner.
        let prompt: String?
        let completion: String?

        var promptPerToken: Double? { prompt.flatMap(Double.init) }
        var completionPerToken: Double? { completion.flatMap(Double.init) }
    }

    enum CodingKeys: String, CodingKey {
        case id, name, description, pricing
        case contextLength = "context_length"
    }

    /// Short label for picker rows when `name` is missing.
    /// Strips the provider prefix so the model is recognisable at
    /// a glance: "anthropic/claude-3.5-sonnet" → "claude-3.5-sonnet".
    var displayName: String {
        if let n = name, !n.isEmpty { return n }
        if let slash = id.firstIndex(of: "/") {
            return String(id[id.index(after: slash)...])
        }
        return id
    }
}

private struct OpenRouterModelsResponse: Decodable {
    let data: [OpenRouterModel]
}

// MARK: - Chat completion DTOs

/// OpenAI-compatible message envelope. We use `[String: Any]` for
/// the request body (built per-call by `OpenRouterClient`) so the
/// caller has full control over optional fields without forcing
/// every Codable property to be Optional. Response decoding stays
/// strongly typed below.
struct OpenAIMessage {
    var role: String                // "system" | "user" | "assistant" | "tool"
    var content: String?
    var name: String?
    var toolCalls: [OpenAIToolCall]?
    var toolCallID: String?         // present when role == "tool"
    var reasoningContent: String?   // OpenRouter extension for R1/o1-style models

    /// Serialise to the dict shape OpenRouter expects.
    func toJSON() -> [String: Any] {
        var obj: [String: Any] = ["role": role]
        if let c = content { obj["content"] = c }
        if let n = name { obj["name"] = n }
        if let tc = toolCalls, !tc.isEmpty {
            obj["tool_calls"] = tc.map { $0.toJSON() }
        }
        if let id = toolCallID { obj["tool_call_id"] = id }
        if let r = reasoningContent { obj["reasoning_content"] = r }
        return obj
    }
}

struct OpenAIToolCall {
    var id: String?
    var type: String = "function"
    var name: String
    var arguments: String   // JSON-encoded args

    func toJSON() -> [String: Any] {
        var obj: [String: Any] = [
            "type": type,
            "function": [
                "name": name,
                "arguments": arguments
            ]
        ]
        if let id { obj["id"] = id }
        return obj
    }
}

// MARK: - Streaming response (typed)

struct OpenAIStreamChunk: Decodable {
    let id: String?
    let choices: [OpenAIStreamChoice]
    let usage: OpenAIUsage?
}

struct OpenAIStreamChoice: Decodable {
    let index: Int
    let delta: OpenAIDelta?
    let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case index, delta
        case finishReason = "finish_reason"
    }
}

struct OpenAIDelta: Decodable {
    let role: String?
    let content: String?
    let reasoningContent: String?
    let toolCalls: [OpenAIToolCallDelta]?

    enum CodingKeys: String, CodingKey {
        case role, content
        case reasoningContent = "reasoning_content"
        case toolCalls = "tool_calls"
    }
}

struct OpenAIToolCallDelta: Decodable {
    let index: Int
    let id: String?
    let type: String?
    let function: OpenAIFunctionCallDelta?
}

struct OpenAIFunctionCallDelta: Decodable {
    let name: String?
    let arguments: String?
}

struct OpenAIUsage: Decodable {
    let promptTokens: Int
    let completionTokens: Int
    /// OpenRouter-specific: dollar cost of the call, summed across
    /// prompt + completion at the model's per-token rate. Some
    /// upstream providers omit it; treat as bonus information.
    let totalCost: Double?

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalCost = "total_cost"
    }
}

// MARK: - Errors

enum OpenRouterError: LocalizedError {
    case missingAPIKey
    case invalidURL
    case invalidResponse
    case http(code: Int, body: String?)
    case unauthorized
    case decodingFailed(String)

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "OpenRouter API key not configured. Open Settings to add it."
        case .invalidURL:
            return "Invalid OpenRouter base URL"
        case .invalidResponse:
            return "Unexpected response from OpenRouter"
        case .http(let code, let body):
            if code == 402 {
                // 402 bodies embed the key's dashboard URL (which
                // includes the key id). The auto-retry path
                // already tried lowering `max_tokens`; if we got
                // here the retry didn't help — show a clean
                // message and keep the raw URL out of the chat
                // transcript so it doesn't end up in screenshots
                // / model context / log files.
                return "OpenRouter rejected the request (HTTP 402): the API "
                    + "key's per-request budget is below the requested "
                    + "max_tokens. Lower max_tokens in Settings → Generation, "
                    + "or top up the key at openrouter.ai/keys."
            }
            if let body, !body.isEmpty {
                return "OpenRouter HTTP \(code): \(Self.redact(body))"
            }
            return "OpenRouter HTTP \(code)"
        case .unauthorized:
            return "OpenRouter rejected the API key. Check Settings."
        case .decodingFailed(let msg):
            return "OpenRouter response decode failed: \(msg)"
        }
    }

    /// Strip out the `openrouter.ai/workspaces/.../keys/<hex>` URLs
    /// some error bodies embed. The hex segment is the key id —
    /// useful to OpenRouter for support, but a credential leak
    /// when echoed into the chat, screenshots, or model logs.
    private static let keyURLRegex: NSRegularExpression? = try? NSRegularExpression(
        pattern: #"https?://openrouter\.ai/[^\s"']+keys/[A-Za-z0-9]+"#)

    static func redact(_ body: String) -> String {
        guard let regex = keyURLRegex else { return body }
        let ns = body as NSString
        let range = NSRange(location: 0, length: ns.length)
        return regex.stringByReplacingMatches(
            in: body, range: range,
            withTemplate: "<openrouter dashboard URL redacted>")
    }
}

// MARK: - Client

/// Thin async wrapper over `URLSession` that talks to OpenRouter's
/// OpenAI-compatible REST surface. No retries here — the caller
/// owns user-visible behaviour around transient errors. Streaming
/// uses `URLSession.bytes(for:)` and parses the SSE line protocol
/// directly so we never have to load the whole completion into
/// memory before yielding the first token.
final class OpenRouterClient: @unchecked Sendable {
    static let defaultBaseURL = URL(string: "https://openrouter.ai/api/v1")!
    /// Sent as `HTTP-Referer` per OpenRouter's etiquette guide so
    /// requests are attributable in the dashboards. Not strictly
    /// required.
    static let attributionURL = "https://github.com/giorgiooppo/DeepSeek-V4-Pro-MacOS"
    static let appName = "DeepSeek-V4-Pro-MacOS"

    let baseURL: URL
    private let session: URLSession

    init(baseURL: URL = OpenRouterClient.defaultBaseURL) {
        self.baseURL = baseURL
        let cfg = URLSessionConfiguration.default
        // Generous timeouts: cold-start a 200k-context model on
        // upstream and you can wait 20-30 s before the first
        // chunk lands. Streaming itself is the gate, not the
        // initial response.
        cfg.timeoutIntervalForRequest = 60
        cfg.timeoutIntervalForResource = 600
        cfg.waitsForConnectivity = true
        // TODO §9: opt-in HTTPRecorder. The protocol decides per-
        // request whether to intercept based on
        // `HTTPRecorder.shared.mode`. When off (the default) the
        // protocol's canInit returns false and we hit the standard
        // URL loader — zero overhead.
        cfg.protocolClasses = [HTTPRecorderURLProtocol.self]
            + (cfg.protocolClasses ?? [])
        self.session = URLSession(configuration: cfg)
    }

    // MARK: - models

    /// Fetch the full catalog. Cheap (~50 KB JSON, no auth needed
    /// strictly but we send the key anyway so per-user rate limits
    /// apply correctly).
    func fetchModels(apiKey: String?) async throws -> [OpenRouterModel] {
        var req = URLRequest(url: baseURL.appendingPathComponent("models"))
        if let apiKey, !apiKey.isEmpty {
            req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        addStandardHeaders(&req)
        let (data, response) = try await session.data(for: req)
        try Self.checkStatus(response, body: data)
        do {
            return try JSONDecoder().decode(
                OpenRouterModelsResponse.self, from: data).data
        } catch {
            throw OpenRouterError.decodingFailed(String(describing: error))
        }
    }

    /// Validate the API key by hitting `/auth/key`. Returns the
    /// decoded auth info on success, throws on rejection.
    func validateKey(_ key: String) async throws {
        var req = URLRequest(url: baseURL.appendingPathComponent("auth/key"))
        req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        addStandardHeaders(&req)
        let (data, response) = try await session.data(for: req)
        try Self.checkStatus(response, body: data)
        // Body shape isn't important for the validation path —
        // status 200 is enough.
    }

    // MARK: - chat completion (streaming)

    /// Build the SSE async stream of parsed `OpenAIStreamChunk`s
    /// for one chat completion request. `body` is the raw dict
    /// the caller assembles (messages, tools, sampler, …) so the
    /// client doesn't have to know about every optional field.
    /// `stream: true` is forced regardless of what the caller put
    /// in — non-streaming completions live in their own helper
    /// (not added until something needs them).
    ///
    /// Errors surface through the stream's `finish(throwing:)`
    /// path. Heartbeat / non-data SSE lines are skipped silently;
    /// only `data: { … }` payloads decode to `OpenAIStreamChunk`.
    func streamChatCompletion(apiKey: String,
                                body: [String: Any])
        -> AsyncThrowingStream<OpenAIStreamChunk, Error>
    {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var bodyToSend = body
                    bodyToSend["stream"] = true
                    var triedBudgetRetry = false

                    // Single attempt loop. We re-enter at most once
                    // on HTTP 402: OpenRouter rejects when the key's
                    // per-request budget can't cover the requested
                    // `max_tokens`, but the error body tells us how
                    // many tokens the key CAN afford. We trim the
                    // body and retry transparently so the user
                    // doesn't see a flaky-looking failure when the
                    // fix is mechanical.
                    requestLoop:
                    while true {
                        var req = URLRequest(
                            url: baseURL.appendingPathComponent(
                                "chat/completions"))
                        req.httpMethod = "POST"
                        req.setValue("application/json",
                                      forHTTPHeaderField: "Content-Type")
                        req.setValue("text/event-stream",
                                      forHTTPHeaderField: "Accept")
                        req.setValue("Bearer \(apiKey)",
                                      forHTTPHeaderField: "Authorization")
                        Self.addStandardHeaders(&req)
                        req.httpBody = try JSONSerialization.data(
                            withJSONObject: bodyToSend, options: [])

                        let (bytes, response) = try await session.bytes(for: req)

                        if let http = response as? HTTPURLResponse,
                           http.statusCode != 200 {
                            // Drain the body for a useful error message.
                            var collected = Data()
                            for try await b in bytes {
                                collected.append(b)
                                if collected.count > 4096 { break }
                            }
                            let body = String(data: collected, encoding: .utf8)
                            if http.statusCode == 401 || http.statusCode == 403 {
                                throw OpenRouterError.unauthorized
                            }
                            if http.statusCode == 402,
                               !triedBudgetRetry,
                               let affordable = Self.parseAffordableTokens(
                                from: body),
                               affordable >= 256
                            {
                                // Subtract a small safety margin so a
                                // retry that lands on a SLIGHTLY
                                // pricier provider doesn't 402 again.
                                let target = max(256,
                                                 Int(Double(affordable) * 0.95))
                                bodyToSend["max_tokens"] = target
                                triedBudgetRetry = true
                                continue requestLoop
                            }
                            throw OpenRouterError.http(
                                code: http.statusCode, body: body)
                        }

                        let decoder = JSONDecoder()
                        for try await rawLine in bytes.lines {
                            if Task.isCancelled {
                                continuation.finish()
                                return
                            }
                            // SSE: blank lines separate events; comments
                            // start with ':'. We only care about
                            // `data: …` payloads.
                            if rawLine.isEmpty || rawLine.hasPrefix(":") {
                                continue
                            }
                            guard rawLine.hasPrefix("data: ") else { continue }
                            let payload = String(rawLine.dropFirst("data: ".count))
                            if payload == "[DONE]" {
                                continuation.finish()
                                return
                            }
                            guard let data = payload.data(using: .utf8) else { continue }
                            do {
                                let chunk = try decoder.decode(
                                    OpenAIStreamChunk.self, from: data)
                                continuation.yield(chunk)
                            } catch {
                                // OpenRouter occasionally interleaves
                                // provider-specific events (e.g. router
                                // upstream switches) we don't model.
                                // Skip rather than fail the whole stream.
                                continue
                            }
                        }
                        // Stream ended cleanly without an explicit
                        // [DONE] terminator — finish the continuation
                        // and exit the retry loop.
                        continuation.finish()
                        break requestLoop
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    // MARK: - helpers

    /// Pull the "you can only afford N tokens" hint out of an
    /// OpenRouter 402 body. The error is structured like:
    ///
    /// ```json
    /// { "error": {
    ///     "code": 402,
    ///     "message": "...can only afford 3630. To increase...",
    ///     "metadata": { "previous_errors": [
    ///         {"code":402,"message":"...afford 4967..."}, …
    ///     ]}
    /// }}
    /// ```
    ///
    /// Multiple provider attempts can land in `previous_errors`
    /// with different limits; the **top-level** message is the
    /// last attempt and the one we should target on retry. Falls
    /// back to a free-text scan when the JSON shape isn't what we
    /// expect — OpenRouter occasionally returns slightly different
    /// wrappers, and a regex over the body is still better than
    /// throwing the whole error away.
    ///
    /// Visible for testing; not stable API.
    static func parseAffordableTokens(from body: String?) -> Int? {
        guard let body, !body.isEmpty else { return nil }
        if let data = body.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data)
            as? [String: Any],
           let err = obj["error"] as? [String: Any],
           let message = err["message"] as? String,
           let parsed = extractAffordFromText(message)
        {
            return parsed
        }
        return extractAffordFromText(body)
    }

    private static let affordRegex: NSRegularExpression? = try? NSRegularExpression(
        pattern: #"can only afford (\d+)"#)

    private static func extractAffordFromText(_ text: String) -> Int? {
        guard let regex = affordRegex else { return nil }
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        guard let match = regex.firstMatch(in: text, range: range)
        else { return nil }
        let group = match.range(at: 1)
        guard group.location != NSNotFound else { return nil }
        return Int(ns.substring(with: group))
    }

    private static func addStandardHeaders(_ req: inout URLRequest) {
        req.setValue(attributionURL, forHTTPHeaderField: "HTTP-Referer")
        req.setValue(appName,        forHTTPHeaderField: "X-Title")
    }

    private func addStandardHeaders(_ req: inout URLRequest) {
        Self.addStandardHeaders(&req)
    }

    /// Map HTTP status to a thrown `OpenRouterError` for the
    /// non-streaming paths. Body is captured into the error for
    /// the diagnostic banner.
    private static func checkStatus(_ response: URLResponse,
                                     body: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw OpenRouterError.invalidResponse
        }
        switch http.statusCode {
        case 200...299: return
        case 401, 403: throw OpenRouterError.unauthorized
        default:
            let bodyString = String(data: body, encoding: .utf8)
            throw OpenRouterError.http(code: http.statusCode, body: bodyString)
        }
    }
}
