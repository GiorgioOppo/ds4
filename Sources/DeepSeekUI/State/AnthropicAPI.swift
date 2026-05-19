import Foundation

/// Anthropic Messages API client (TODO §10.4 / T4). Mirrors
/// `OpenRouterClient`'s shape — same `AsyncThrowingStream<OpenAIStreamChunk,
/// Error>` return type from the streaming entrypoint — so `ChatStore`'s
/// remote loop can consume Anthropic and OpenRouter streams through
/// the same code path. The Anthropic-specific event shape is
/// translated to OpenAI delta chunks inside this file.
///
/// Why a native driver instead of routing through OpenRouter:
///
///   1. `cache_control` is not exposed in OpenRouter's body, so
///      prompt caching (cache hit ≈ 90 % discount on Sonnet) is
///      unreachable through them. This client auto-injects
///      `cache_control: {type:"ephemeral"}` on the system block
///      (when long enough to be cacheable) and the last
///      `tool_result` block, mirroring Anthropic's recommended
///      hot-cache pattern.
///   2. Skips the OpenRouter margin on the most-used models.
///
/// API contract reference:
///   https://docs.anthropic.com/en/api/messages
///   https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
final class AnthropicClient: @unchecked Sendable {
    static let defaultBaseURL = URL(string: "https://api.anthropic.com")!
    static let apiVersion = "2023-06-01"

    let baseURL: URL
    private let session: URLSession

    init(baseURL: URL = AnthropicClient.defaultBaseURL) {
        self.baseURL = baseURL
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 60
        cfg.timeoutIntervalForResource = 600
        cfg.waitsForConnectivity = true
        self.session = URLSession(configuration: cfg)
    }

    // MARK: - Validation + catalog

    /// Cheap key-validity probe. Anthropic doesn't expose an
    /// `/auth/key`-style endpoint, so we hit `/v1/models` which is
    /// authenticated but free.
    func validateKey(_ key: String) async throws {
        var req = URLRequest(url: baseURL.appendingPathComponent("v1/models"))
        Self.addAuthHeaders(&req, apiKey: key)
        let (data, response) = try await session.data(for: req)
        try Self.checkStatus(response, body: data)
    }

    /// Fetch the public model catalog. The shape is `{data: [{id, ...}]}`
    /// — we strongly-type only `id` and `display_name`, the rest of
    /// each row goes ignored.
    func fetchModels(apiKey: String) async throws -> [AnthropicModel] {
        var req = URLRequest(url: baseURL.appendingPathComponent("v1/models"))
        Self.addAuthHeaders(&req, apiKey: apiKey)
        let (data, response) = try await session.data(for: req)
        try Self.checkStatus(response, body: data)
        struct Catalog: Decodable { let data: [AnthropicModel] }
        do {
            return try JSONDecoder().decode(Catalog.self, from: data).data
        } catch {
            throw AnthropicError.decodingFailed(String(describing: error))
        }
    }

    // MARK: - Messages (streaming)

    /// Stream a `POST /v1/messages` call. `body` carries the request
    /// payload already shaped for Anthropic (built by
    /// `AnthropicMessageBuilder`). `stream: true` is forced.
    ///
    /// Emits `OpenAIStreamChunk`s with the same field semantics as
    /// `OpenRouterClient.streamChatCompletion`: a first delta carries
    /// `role: "assistant"`, subsequent deltas carry `content` (text)
    /// and/or `toolCalls` (with `function.arguments` streaming via
    /// `input_json_delta` accumulation), and a final delta carries
    /// `finish_reason` translated from Anthropic's `stop_reason`.
    ///
    /// Errors surface through `finish(throwing:)`. Anthropic returns
    /// rate-limit / overload errors as JSON `{type: "error", error: {
    /// type, message }}` in the body even on 4xx/5xx — we surface the
    /// message verbatim via `AnthropicError.http`.
    func streamMessages(apiKey: String,
                         body: [String: Any])
        -> AsyncThrowingStream<OpenAIStreamChunk, Error>
    {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var req = URLRequest(
                        url: baseURL.appendingPathComponent("v1/messages"))
                    req.httpMethod = "POST"
                    req.setValue("application/json",
                                  forHTTPHeaderField: "Content-Type")
                    req.setValue("text/event-stream",
                                  forHTTPHeaderField: "Accept")
                    Self.addAuthHeaders(&req, apiKey: apiKey)
                    var bodyToSend = body
                    bodyToSend["stream"] = true
                    req.httpBody = try JSONSerialization.data(
                        withJSONObject: bodyToSend, options: [])

                    let (bytes, response) = try await session.bytes(for: req)

                    if let http = response as? HTTPURLResponse,
                       http.statusCode != 200
                    {
                        var collected = Data()
                        for try await b in bytes {
                            collected.append(b)
                            if collected.count > 4096 { break }
                        }
                        let body = String(data: collected, encoding: .utf8)
                        if http.statusCode == 401 || http.statusCode == 403 {
                            throw AnthropicError.unauthorized
                        }
                        throw AnthropicError.http(
                            code: http.statusCode, body: body)
                    }

                    // Translator state. Anthropic streams content blocks
                    // by index; we map each `tool_use` block onto an
                    // OpenAI tool_calls[] entry (also indexed). The
                    // partial JSON for tool arguments arrives as a
                    // sequence of `input_json_delta.partial_json` strings
                    // that we forward verbatim (the consumer
                    // concatenates them just like OpenRouter does).
                    var chatID: String = "anthropic-chat"
                    var emittedRole = false
                    // Maps anthropic content_block.index → OpenAI
                    // tool_calls[] index for that block.
                    var toolBlockToOpenAIIndex: [Int: Int] = [:]
                    var nextToolIndex = 0

                    let decoder = JSONDecoder()
                    for try await rawLine in bytes.lines {
                        if Task.isCancelled {
                            continuation.finish()
                            return
                        }
                        // Anthropic prefixes each event with both an
                        // `event: <name>` line and a `data: <json>` line
                        // followed by a blank line. The `event:` lines
                        // duplicate `type` inside the JSON, so we ignore
                        // them and dispatch on the payload's `type`.
                        if rawLine.isEmpty || rawLine.hasPrefix(":") { continue }
                        if rawLine.hasPrefix("event:") { continue }
                        guard rawLine.hasPrefix("data: ") else { continue }
                        let payload = String(rawLine.dropFirst("data: ".count))
                        guard let data = payload.data(using: .utf8) else { continue }
                        guard let dict = try? JSONSerialization.jsonObject(with: data)
                                as? [String: Any],
                              let type = dict["type"] as? String
                        else { continue }

                        switch type {
                        case "message_start":
                            if let msg = dict["message"] as? [String: Any],
                               let id = msg["id"] as? String {
                                chatID = id
                            }
                            // Emit the role: "assistant" opener once
                            // for parity with the OpenAI delta shape.
                            let chunk = Self.makeChunk(
                                id: chatID,
                                delta: OpenAIDelta(role: "assistant",
                                                    content: nil,
                                                    reasoningContent: nil,
                                                    toolCalls: nil),
                                finishReason: nil)
                            emittedRole = true
                            continuation.yield(chunk)

                        case "content_block_start":
                            guard let idx = dict["index"] as? Int,
                                  let block = dict["content_block"]
                                    as? [String: Any],
                                  let blockType = block["type"] as? String
                            else { continue }
                            if blockType == "tool_use" {
                                let openAIIdx = nextToolIndex
                                toolBlockToOpenAIIndex[idx] = openAIIdx
                                nextToolIndex += 1
                                let id = block["id"] as? String
                                let name = block["name"] as? String ?? ""
                                let tcd = OpenAIToolCallDelta(
                                    index: openAIIdx,
                                    id: id,
                                    type: "function",
                                    function: OpenAIFunctionCallDelta(
                                        name: name, arguments: nil))
                                let chunk = Self.makeChunk(
                                    id: chatID,
                                    delta: OpenAIDelta(
                                        role: emittedRole ? nil : "assistant",
                                        content: nil,
                                        reasoningContent: nil,
                                        toolCalls: [tcd]),
                                    finishReason: nil)
                                continuation.yield(chunk)
                            }
                            // text blocks emit no opener — the first
                            // text_delta carries the content directly.

                        case "content_block_delta":
                            guard let idx = dict["index"] as? Int,
                                  let delta = dict["delta"] as? [String: Any],
                                  let deltaType = delta["type"] as? String
                            else { continue }
                            switch deltaType {
                            case "text_delta":
                                if let text = delta["text"] as? String {
                                    let chunk = Self.makeChunk(
                                        id: chatID,
                                        delta: OpenAIDelta(
                                            role: nil, content: text,
                                            reasoningContent: nil,
                                            toolCalls: nil),
                                        finishReason: nil)
                                    continuation.yield(chunk)
                                }
                            case "input_json_delta":
                                if let partial = delta["partial_json"] as? String,
                                   let openAIIdx = toolBlockToOpenAIIndex[idx]
                                {
                                    let tcd = OpenAIToolCallDelta(
                                        index: openAIIdx,
                                        id: nil,
                                        type: nil,
                                        function: OpenAIFunctionCallDelta(
                                            name: nil, arguments: partial))
                                    let chunk = Self.makeChunk(
                                        id: chatID,
                                        delta: OpenAIDelta(
                                            role: nil, content: nil,
                                            reasoningContent: nil,
                                            toolCalls: [tcd]),
                                        finishReason: nil)
                                    continuation.yield(chunk)
                                }
                            case "thinking_delta":
                                // Extended-thinking models (claude-3-7-thinking).
                                // Surface as reasoning_content so the UI's
                                // existing reasoning bubble path picks it up.
                                if let text = delta["thinking"] as? String {
                                    let chunk = Self.makeChunk(
                                        id: chatID,
                                        delta: OpenAIDelta(
                                            role: nil, content: nil,
                                            reasoningContent: text,
                                            toolCalls: nil),
                                        finishReason: nil)
                                    continuation.yield(chunk)
                                }
                            default:
                                continue
                            }

                        case "content_block_stop":
                            continue

                        case "message_delta":
                            // Final chunk: translate Anthropic's
                            // stop_reason to OpenAI's finish_reason
                            // vocabulary.
                            let stopReason: String? = {
                                guard let d = dict["delta"] as? [String: Any]
                                else { return nil }
                                return d["stop_reason"] as? String
                            }()
                            let openaiFinish = Self.translateStopReason(stopReason)
                            let chunk = Self.makeChunk(
                                id: chatID,
                                delta: OpenAIDelta(
                                    role: nil, content: nil,
                                    reasoningContent: nil,
                                    toolCalls: nil),
                                finishReason: openaiFinish)
                            continuation.yield(chunk)

                        case "message_stop":
                            continuation.finish()
                            return

                        case "ping":
                            continue

                        case "error":
                            // Mid-stream Anthropic error event.
                            let msg: String = {
                                if let err = dict["error"] as? [String: Any],
                                   let m = err["message"] as? String { return m }
                                return "Anthropic stream error"
                            }()
                            throw AnthropicError.http(code: 500, body: msg)

                        default:
                            continue
                        }
                        _ = decoder
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    // MARK: - Helpers

    /// Anthropic-style auth headers. `x-api-key` plus the required
    /// `anthropic-version` pin (otherwise the API returns 400) and
    /// `anthropic-beta` for prompt caching where the caller asked for
    /// it (no-op when the body doesn't carry `cache_control`).
    private static func addAuthHeaders(_ req: inout URLRequest,
                                        apiKey: String) {
        req.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        req.setValue(apiVersion, forHTTPHeaderField: "anthropic-version")
    }

    private static func checkStatus(_ response: URLResponse,
                                     body: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw AnthropicError.invalidResponse
        }
        switch http.statusCode {
        case 200...299: return
        case 401, 403: throw AnthropicError.unauthorized
        default:
            let bodyString = String(data: body, encoding: .utf8)
            throw AnthropicError.http(code: http.statusCode, body: bodyString)
        }
    }

    private static func translateStopReason(_ raw: String?) -> String? {
        guard let raw else { return nil }
        switch raw {
        case "end_turn":      return "stop"
        case "stop_sequence": return "stop"
        case "tool_use":      return "tool_calls"
        case "max_tokens":    return "length"
        default:              return raw
        }
    }

    private static func makeChunk(id: String,
                                    delta: OpenAIDelta,
                                    finishReason: String?) -> OpenAIStreamChunk
    {
        OpenAIStreamChunk(
            id: id,
            choices: [OpenAIStreamChoice(
                index: 0, delta: delta, finishReason: finishReason)],
            usage: nil)
    }
}

// MARK: - Models catalog row

struct AnthropicModel: Codable, Identifiable, Hashable {
    let id: String              // e.g. "claude-3-5-sonnet-20241022"
    let displayName: String?    // e.g. "Claude 3.5 Sonnet"

    enum CodingKeys: String, CodingKey {
        case id
        case displayName = "display_name"
    }

    var label: String { displayName ?? id }
}

// MARK: - Errors

enum AnthropicError: LocalizedError {
    case missingAPIKey
    case invalidResponse
    case http(code: Int, body: String?)
    case unauthorized
    case decodingFailed(String)

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "Anthropic API key not configured. Open Settings → API Keys to add it."
        case .invalidResponse:
            return "Unexpected response from Anthropic."
        case .http(let code, let body):
            if let body, !body.isEmpty { return "Anthropic HTTP \(code): \(body)" }
            return "Anthropic HTTP \(code)"
        case .unauthorized:
            return "Anthropic rejected the API key. Check Settings."
        case .decodingFailed(let msg):
            return "Anthropic response decode failed: \(msg)"
        }
    }
}
