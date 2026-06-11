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
    }

    private let engine: InferenceService
    private let modelName: String         // display name (the GGUF file)
    private let config: Config
    private let onLog: @Sendable (String) -> Void
    private let queue = DispatchQueue(label: "ds4.localserver", qos: .userInitiated)
    private let gate = RequestGate()
    private var listener: NWListener?

    /// The model aliases the API advertises (mirrors ds4_server.c).
    private static let aliases = ["deepseek-v4-flash", "deepseek-v4-pro"]

    init(engine: InferenceService, modelName: String, config: Config,
         onLog: @escaping @Sendable (String) -> Void) {
        self.engine = engine
        self.modelName = modelName
        self.config = config
        self.onLog = onLog
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
        l.start(queue: queue)
        listener = l
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }

    private func accept(_ conn: NWConnection) {
        conn.start(queue: queue)
        Task { [weak self] in await self?.serve(conn) }
    }

    // MARK: Connection handling

    private func serve(_ conn: NWConnection) async {
        do {
            guard let req = try await readRequest(conn) else { conn.cancel(); return }
            try await route(conn, req)
        } catch {
            try? await send(conn, Self.httpError(503, "internal error", cors: config.cors))
        }
        conn.cancel()
    }

    private func route(_ conn: NWConnection, _ req: HTTPRequest) async throws {
        if req.method == "OPTIONS" {
            try await send(conn, Self.response(204, contentType: nil, body: "", cors: config.cors))
            return
        }
        if req.method == "GET", req.path == "/v1/models" {
            try await send(conn, Self.response(200, contentType: "application/json",
                                               body: modelsJSON(), cors: config.cors))
            return
        }
        let modelPrefix = "/v1/models/"
        if req.method == "GET", req.path.hasPrefix(modelPrefix) {
            let id = String(req.path.dropFirst(modelPrefix.count))
            if Self.aliases.contains(id) {
                try await send(conn, Self.response(200, contentType: "application/json",
                                                   body: modelJSON(id), cors: config.cors))
                return
            }
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
        try await send(conn, Self.httpError(404, "unknown endpoint", cors: config.cors))
    }

    // MARK: /v1/chat/completions

    private func handleChat(_ conn: NWConnection, body: Data) async throws {
        guard let obj = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any] else {
            try await send(conn, Self.httpError(400, "invalid JSON body", cors: config.cors)); return
        }
        let parsed = ChatRequestParser.parse(obj, defaultTokens: config.maxTokens)
        guard !parsed.turns.isEmpty else {
            try await send(conn, Self.httpError(400, "no messages", cors: config.cors)); return
        }
        let model = parsed.model ?? "deepseek-v4-flash"
        let id = "chatcmpl-" + String(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))
        let created = Int(Date().timeIntervalSince1970)
        onLog("POST /v1/chat/completions (\(parsed.turns.count) msg, stream=\(parsed.stream))\n")

        // Serialize: only one generation runs against the single-model engine.
        await gate.acquire()
        defer { Task { await gate.release() } }

        let stream = await engine.complete(turns: parsed.turns, tools: parsed.tools,
                                            thinkMode: parsed.think, sampling: parsed.sampling,
                                            maxTokens: parsed.maxTokens)
        if parsed.stream {
            try await streamChat(conn, stream: stream, id: id, model: model, created: created)
        } else {
            try await bufferChat(conn, stream: stream, id: id, model: model, created: created)
        }
    }

    /// Streaming (SSE) chat.completion.chunk events, matching ds4_server.c.
    private func streamChat(_ conn: NWConnection, stream: AsyncThrowingStream<GenEvent, Error>,
                            id: String, model: String, created: Int) async throws {
        try await send(conn, Data(Self.sseHeader(cors: config.cors).utf8))
        let m = jsonString(model)
        func chunk(_ delta: String, finish: String) -> Data {
            Data(("data: {\"id\":\"\(id)\",\"object\":\"chat.completion.chunk\",\"created\":\(created),\"model\":\(m),\"choices\":[{\"index\":0,\"delta\":\(delta),\"finish_reason\":\(finish)}]}\n\n").utf8)
        }
        // First chunk announces the assistant role (OpenAI convention).
        try await send(conn, chunk("{\"role\":\"assistant\"}", finish: "null"))

        var finish = "stop"
        do {
            for try await event in stream {
                switch event {
                case .reasoning(let r):
                    try await send(conn, chunk("{\"reasoning_content\":\(jsonString(r))}", finish: "null"))
                case .text(let t):
                    try await send(conn, chunk("{\"content\":\(jsonString(t))}", finish: "null"))
                case .toolCall(let calls):
                    finish = "tool_calls"
                    try await send(conn, chunk("{\"tool_calls\":\(toolCallsJSON(calls))}", finish: "null"))
                case .toolStream, .progress:
                    break
                }
            }
        } catch is CancellationError {
            // client disconnected or generation stopped — close cleanly
        }
        try await send(conn, chunk("{}", finish: "\"\(finish)\""))
        try await send(conn, Data("data: [DONE]\n\n".utf8))
    }

    /// Non-streaming: collect the full reply into one chat.completion body.
    private func bufferChat(_ conn: NWConnection, stream: AsyncThrowingStream<GenEvent, Error>,
                            id: String, model: String, created: Int) async throws {
        var content = "", reasoning = "", finish = "stop"
        var calls: [ToolCall] = []
        do {
            for try await event in stream {
                switch event {
                case .reasoning(let r): reasoning += r
                case .text(let t): content += t
                case .toolCall(let c): calls = c; finish = "tool_calls"
                case .toolStream, .progress: break
                }
            }
        } catch is CancellationError {}

        var message: [String: Any] = ["role": "assistant", "content": content]
        if !reasoning.isEmpty { message["reasoning_content"] = reasoning }
        if !calls.isEmpty {
            message["content"] = NSNull()
            message["tool_calls"] = calls.map { [
                "id": $0.id, "type": "function",
                "function": ["name": $0.name, "arguments": $0.argumentsJSON]
            ] }
        }
        let payload: [String: Any] = [
            "id": id, "object": "chat.completion", "created": created, "model": model,
            "choices": [["index": 0, "message": message, "finish_reason": finish]],
            "usage": ["prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0],
        ]
        let data = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
        try await send(conn, Self.response(200, contentType: "application/json",
                                           body: String(data: data, encoding: .utf8) ?? "{}",
                                           cors: config.cors))
    }

    // MARK: /v1/messages (Anthropic Messages API)

    private func handleAnthropic(_ conn: NWConnection, body: Data) async throws {
        guard let obj = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any] else {
            try await send(conn, Self.anthropicError(400, "invalid JSON body", cors: config.cors)); return
        }
        let parsed = ChatRequestParser.parseAnthropic(obj, defaultTokens: config.maxTokens)
        guard !parsed.turns.isEmpty else {
            try await send(conn, Self.anthropicError(400, "no messages", cors: config.cors)); return
        }
        let model = parsed.model ?? "deepseek-v4-flash"
        let id = "msg_" + String(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))
        onLog("POST /v1/messages (\(parsed.turns.count) msg, stream=\(parsed.stream))\n")

        await gate.acquire()
        defer { Task { await gate.release() } }

        let stream = await engine.complete(turns: parsed.turns, tools: parsed.tools,
                                            thinkMode: parsed.think, sampling: parsed.sampling,
                                            maxTokens: parsed.maxTokens)
        if parsed.stream {
            try await streamAnthropic(conn, stream: stream, id: id, model: model)
        } else {
            try await bufferAnthropic(conn, stream: stream, id: id, model: model)
        }
    }

    /// Anthropic SSE: message_start → content_block_(start|delta|stop)* → message_delta → message_stop.
    /// Text/thinking stream as their block types; each tool call is one tool_use block whose
    /// arguments arrive as a single input_json_delta (valid: clients accumulate partial_json).
    private func streamAnthropic(_ conn: NWConnection, stream: AsyncThrowingStream<GenEvent, Error>,
                                 id: String, model: String) async throws {
        try await send(conn, Data(Self.sseHeader(cors: config.cors).utf8))
        let m = jsonString(model)
        try await sse(conn, "message_start",
            "{\"type\":\"message_start\",\"message\":{\"id\":\"\(id)\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\(m),\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}")

        enum Block: Equatable { case none, thinking, text }
        var current: Block = .none
        var index = 0
        var stopReason = "end_turn"

        func closeCurrent() async throws {
            if current != .none {
                try await sse(conn, "content_block_stop", "{\"type\":\"content_block_stop\",\"index\":\(index)}")
                index += 1
                current = .none
            }
        }

        do {
            for try await event in stream {
                switch event {
                case .reasoning(let r):
                    if current != .thinking {
                        try await closeCurrent()
                        try await sse(conn, "content_block_start",
                            "{\"type\":\"content_block_start\",\"index\":\(index),\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}")
                        current = .thinking
                    }
                    try await sse(conn, "content_block_delta",
                        "{\"type\":\"content_block_delta\",\"index\":\(index),\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\(jsonString(r))}}")
                case .text(let t):
                    if current != .text {
                        try await closeCurrent()
                        try await sse(conn, "content_block_start",
                            "{\"type\":\"content_block_start\",\"index\":\(index),\"content_block\":{\"type\":\"text\",\"text\":\"\"}}")
                        current = .text
                    }
                    try await sse(conn, "content_block_delta",
                        "{\"type\":\"content_block_delta\",\"index\":\(index),\"delta\":{\"type\":\"text_delta\",\"text\":\(jsonString(t))}}")
                case .toolCall(let calls):
                    stopReason = "tool_use"
                    try await closeCurrent()
                    for c in calls {
                        try await sse(conn, "content_block_start",
                            "{\"type\":\"content_block_start\",\"index\":\(index),\"content_block\":{\"type\":\"tool_use\",\"id\":\(jsonString(c.id)),\"name\":\(jsonString(c.name)),\"input\":{}}}")
                        try await sse(conn, "content_block_delta",
                            "{\"type\":\"content_block_delta\",\"index\":\(index),\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\(jsonString(c.argumentsJSON))}}")
                        try await sse(conn, "content_block_stop", "{\"type\":\"content_block_stop\",\"index\":\(index)}")
                        index += 1
                    }
                case .toolStream, .progress:
                    break
                }
            }
        } catch is CancellationError {}
        try await closeCurrent()
        try await sse(conn, "message_delta",
            "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"\(stopReason)\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":0}}")
        try await sse(conn, "message_stop", "{\"type\":\"message_stop\"}")
    }

    private func bufferAnthropic(_ conn: NWConnection, stream: AsyncThrowingStream<GenEvent, Error>,
                                 id: String, model: String) async throws {
        var content = "", reasoning = "", stopReason = "end_turn"
        var calls: [ToolCall] = []
        do {
            for try await event in stream {
                switch event {
                case .reasoning(let r): reasoning += r
                case .text(let t): content += t
                case .toolCall(let c): calls = c; stopReason = "tool_use"
                case .toolStream, .progress: break
                }
            }
        } catch is CancellationError {}

        var blocks: [String] = []
        if !reasoning.isEmpty {
            blocks.append("{\"type\":\"thinking\",\"thinking\":\(jsonString(reasoning)),\"signature\":\"\"}")
        }
        if !content.isEmpty {
            blocks.append("{\"type\":\"text\",\"text\":\(jsonString(content))}")
        }
        for c in calls {
            let input = c.argumentsJSON.isEmpty ? "{}" : c.argumentsJSON
            blocks.append("{\"type\":\"tool_use\",\"id\":\(jsonString(c.id)),\"name\":\(jsonString(c.name)),\"input\":\(input)}")
        }
        if blocks.isEmpty { blocks.append("{\"type\":\"text\",\"text\":\"\"}") }

        let body = "{\"id\":\"\(id)\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\(jsonString(model)),\"content\":[\(blocks.joined(separator: ","))],\"stop_reason\":\"\(stopReason)\",\"stop_sequence\":null,\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}"
        try await send(conn, Self.response(200, contentType: "application/json", body: body, cors: config.cors))
    }

    // MARK: /v1/completions (legacy text completion)

    private func handleCompletions(_ conn: NWConnection, body: Data) async throws {
        guard let obj = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any] else {
            try await send(conn, Self.httpError(400, "invalid JSON body", cors: config.cors)); return
        }
        let parsed = ChatRequestParser.parseCompletion(obj, defaultTokens: config.maxTokens)
        guard !parsed.turns.isEmpty else {
            try await send(conn, Self.httpError(400, "no prompt", cors: config.cors)); return
        }
        let model = parsed.model ?? "deepseek-v4-flash"
        let id = "cmpl-" + String(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))
        let created = Int(Date().timeIntervalSince1970)
        onLog("POST /v1/completions (stream=\(parsed.stream))\n")

        await gate.acquire()
        defer { Task { await gate.release() } }

        let stream = await engine.complete(turns: parsed.turns, tools: [],
                                            thinkMode: parsed.think, sampling: parsed.sampling,
                                            maxTokens: parsed.maxTokens)
        let m = jsonString(model)
        if parsed.stream {
            try await send(conn, Data(Self.sseHeader(cors: config.cors).utf8))
            do {
                for try await event in stream {
                    if case .text(let t) = event {
                        try await send(conn, Data(("data: {\"id\":\"\(id)\",\"object\":\"text_completion\",\"created\":\(created),\"model\":\(m),\"choices\":[{\"text\":\(jsonString(t)),\"index\":0,\"finish_reason\":null}]}\n\n").utf8))
                    }
                }
            } catch is CancellationError {}
            try await send(conn, Data(("data: {\"id\":\"\(id)\",\"object\":\"text_completion\",\"created\":\(created),\"model\":\(m),\"choices\":[{\"text\":\"\",\"index\":0,\"finish_reason\":\"stop\"}]}\n\n").utf8))
            try await send(conn, Data("data: [DONE]\n\n".utf8))
        } else {
            var text = ""
            do { for try await event in stream { if case .text(let t) = event { text += t } } }
            catch is CancellationError {}
            let payload: [String: Any] = [
                "id": id, "object": "text_completion", "created": created, "model": model,
                "choices": [["text": text, "index": 0, "finish_reason": "stop"]],
                "usage": ["prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0],
            ]
            let data = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
            try await send(conn, Self.response(200, contentType: "application/json",
                                               body: String(data: data, encoding: .utf8) ?? "{}",
                                               cors: config.cors))
        }
    }

    // MARK: JSON helpers

    /// Emit one Anthropic SSE event: `event: <name>\ndata: <json>\n\n`.
    private func sse(_ conn: NWConnection, _ event: String, _ data: String) async throws {
        try await send(conn, Data("event: \(event)\ndata: \(data)\n\n".utf8))
    }

    private func toolCallsJSON(_ calls: [ToolCall]) -> String {
        var parts: [String] = []
        for (i, c) in calls.enumerated() {
            parts.append("{\"index\":\(i),\"id\":\(jsonString(c.id)),\"type\":\"function\",\"function\":{\"name\":\(jsonString(c.name)),\"arguments\":\(jsonString(c.argumentsJSON))}}")
        }
        return "[" + parts.joined(separator: ",") + "]"
    }

    private func modelsJSON() -> String {
        "{\"object\":\"list\",\"data\":[" + Self.aliases.map { modelJSON($0) }.joined(separator: ",") + "]}"
    }

    private func modelJSON(_ id: String) -> String {
        "{\"id\":\(jsonString(id)),\"object\":\"model\",\"created\":1767225600,\"owned_by\":\"dwarfstar\",\"name\":\(jsonString(modelName)),\"max_completion_tokens\":\(config.maxTokens)}"
    }

    /// Quote + escape an arbitrary string as a JSON string literal.
    private func jsonString(_ s: String) -> String {
        guard let d = try? JSONSerialization.data(withJSONObject: [s]),
              let str = String(data: d, encoding: .utf8) else { return "\"\"" }
        return String(str.dropFirst().dropLast())   // strip the surrounding [ ]
    }

    // MARK: Low-level HTTP (async wrappers over NWConnection)

    private func send(_ conn: NWConnection, _ data: Data) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            conn.send(content: data, completion: .contentProcessed { error in
                if let error { cont.resume(throwing: error) } else { cont.resume() }
            })
        }
    }

    private func receive(_ conn: NWConnection) async throws -> Data? {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Data?, Error>) in
            conn.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { data, _, isComplete, error in
                if let error { cont.resume(throwing: error); return }
                if let data, !data.isEmpty { cont.resume(returning: data) }
                else { cont.resume(returning: isComplete ? nil : Data()) }
            }
        }
    }

    /// Read a full HTTP/1.1 request: headers up to CRLFCRLF, then `Content-Length` body bytes.
    private func readRequest(_ conn: NWConnection) async throws -> HTTPRequest? {
        var buf = Data()
        let sep = Data("\r\n\r\n".utf8)
        // Read until headers are complete.
        while buf.range(of: sep) == nil {
            guard let chunk = try await receive(conn) else { return nil }
            if chunk.isEmpty { continue }
            buf.append(chunk)
            if buf.count > 8 * 1024 * 1024 { return nil }   // guard runaway headers
        }
        guard let headerEnd = buf.range(of: sep) else { return nil }
        let headerData = buf.subdata(in: buf.startIndex..<headerEnd.lowerBound)
        guard let headerText = String(data: headerData, encoding: .utf8) else { return nil }

        var lines = headerText.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else { return nil }
        let comps = requestLine.split(separator: " ")
        guard comps.count >= 2 else { return nil }
        let method = String(comps[0])
        let path = String(comps[1].split(separator: "?").first ?? comps[1])

        lines.removeFirst()
        var contentLength = 0
        for line in lines where line.lowercased().hasPrefix("content-length:") {
            contentLength = Int(line.dropFirst("content-length:".count).trimmingCharacters(in: .whitespaces)) ?? 0
        }

        var body = buf.subdata(in: headerEnd.upperBound..<buf.endIndex)
        while body.count < contentLength {
            guard let chunk = try await receive(conn) else { break }
            if chunk.isEmpty { continue }
            body.append(chunk)
        }
        return HTTPRequest(method: method, path: path, body: body)
    }

    // MARK: Response builders (faithful to ds4_server.c)

    private static func corsHeaders(_ cors: Bool) -> String {
        cors ? "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n" : ""
    }

    static func response(_ status: Int, contentType: String?, body: String, cors: Bool) -> Data {
        let reason = statusText(status)
        var head = "HTTP/1.1 \(status) \(reason)\r\n"
        let bodyData = Data(body.utf8)
        if let contentType { head += "Content-Type: \(contentType)\r\n" }
        head += "Content-Length: \(bodyData.count)\r\n"
        head += corsHeaders(cors)
        head += "Connection: close\r\n\r\n"
        var out = Data(head.utf8)
        out.append(bodyData)
        return out
    }

    static func httpError(_ status: Int, _ message: String, cors: Bool) -> Data {
        let payload: [String: Any] = ["error": ["message": message, "type": "invalid_request_error"]]
        let body = (try? JSONSerialization.data(withJSONObject: payload)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        return response(status, contentType: "application/json", body: body, cors: cors)
    }

    static func anthropicError(_ status: Int, _ message: String, cors: Bool) -> Data {
        let payload: [String: Any] = ["type": "error",
                                      "error": ["type": "invalid_request_error", "message": message]]
        let body = (try? JSONSerialization.data(withJSONObject: payload)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        return response(status, contentType: "application/json", body: body, cors: cors)
    }

    static func sseHeader(cors: Bool) -> String {
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n" +
        corsHeaders(cors) + "Connection: close\r\n\r\n"
    }

    private static func statusText(_ code: Int) -> String {
        switch code {
        case 200: return "OK"
        case 204: return "No Content"
        case 400: return "Bad Request"
        case 404: return "Not Found"
        case 503: return "Service Unavailable"
        default:  return "OK"
        }
    }

    enum ServerError: Error { case badPort }
}

private struct HTTPRequest {
    let method: String
    let path: String
    let body: Data
}

/// Serializes generations: the single-model engine can run only one at a time.
private actor RequestGate {
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        if !busy { busy = true; return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func release() {
        if waiters.isEmpty { busy = false }
        else { waiters.removeFirst().resume() }
    }
}
