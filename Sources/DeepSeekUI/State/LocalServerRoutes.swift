import Foundation
import DeepSeekKit

/// OpenAI-compatible route handlers wired onto a `LocalServer` (T1
/// of TODO §10.1). Today's surface:
///
///   - `GET  /v1/models`              — returns the currently loaded
///                                       model as a singleton catalog
///   - `POST /v1/chat/completions`    — both `stream: true` (SSE) and
///                                       `stream: false` (single JSON)
///
/// Tools (`tools[]` passthrough → MCP / native registry) follow in a
/// later sub-task; for now any `tools` field in the request is
/// ignored and the loop stays in pure text mode.
///
/// Concurrency: each handler captures `service` by reference. The
/// underlying `InferenceService` dispatches every generation onto a
/// single serial queue, so concurrent HTTP requests naturally
/// serialize — first-come-first-served, no multiplexing. That's the
/// right semantics for single-user dev workflows; multi-tenant
/// serving would require `KV cache pool` (TODO §5) plus per-request
/// transformer instances.
enum LocalServerRoutes {
    /// Wire every supported route onto `server`. Call once at startup
    /// after `LocalServer.start(...)` succeeds.
    static func register(on server: LocalServer,
                          service: InferenceService) async {
        await server.register(method: "GET", path: "/v1/models",
                               handler: makeModelsHandler(service: service))
        await server.register(method: "POST", path: "/v1/chat/completions",
                               handler: makeChatCompletionsHandler(service: service))
    }

    // MARK: - /v1/models

    private static func makeModelsHandler(
        service: InferenceService) -> LocalServerHandler
    {
        return { _, writer in
            let modelName: String
            if let dir = service.loadedModelDir {
                modelName = dir.lastPathComponent
            } else {
                modelName = "no-model-loaded"
            }
            let now = Int(Date().timeIntervalSince1970)
            let payload: [String: Any] = [
                "object": "list",
                "data": [[
                    "id": modelName,
                    "object": "model",
                    "created": now,
                    "owned_by": "local",
                ]],
            ]
            await writeJSON(writer, status: 200, payload: payload)
        }
    }

    // MARK: - /v1/chat/completions

    private static func makeChatCompletionsHandler(
        service: InferenceService) -> LocalServerHandler
    {
        return { request, writer in
            // Decode the OpenAI request body.
            let req: OpenAIChatRequest
            do {
                req = try request.decodeJSON(OpenAIChatRequest.self)
            } catch {
                await writeError(writer, status: 400,
                                  message: "Invalid request body: \(error.localizedDescription)")
                return
            }

            // Bail early if the model isn't loaded — without a
            // tokenizer we can't even turn the prompt into ids.
            guard service.loadedModelDir != nil else {
                await writeError(writer, status: 503,
                                  message: "No local model loaded on the server. "
                                           + "Pick a model in the desktop app first.")
                return
            }

            // Map OpenAI messages → DeepSeekKit Message. Tool messages
            // (role="tool") are folded into the preceding assistant
            // turn's toolOutputs — first-cut handling until tools[]
            // passthrough lands.
            let history = mapMessages(req.messages)
            let mode = thinkingMode(from: req)
            let options = samplingOptions(from: req)
            let maxTokens = req.max_tokens ?? 1024

            guard let promptTokens = await service.tokenizeFullHistory(
                history, mode: mode, toolSchemasJSON: nil)
            else {
                await writeError(writer, status: 500,
                                  message: "Tokenizer unavailable.")
                return
            }

            // Synthesize a fresh UUID per request. This forces a
            // full reset of the InferenceService KV cache on each
            // call — correct but slow for repeated turns from the
            // same client. Session reuse is a future optimization
            // (see TODO §5 "KV cache pool").
            let conversationID = UUID()
            let stream = service.generateForConversation(
                promptTokens: promptTokens,
                conversationID: conversationID,
                mode: mode,
                options: options,
                maxTokens: maxTokens)

            let modelName = service.loadedModelDir?.lastPathComponent
                ?? "local-model"
            let chatID = "chatcmpl-\(UUID().uuidString)"
            let created = Int(Date().timeIntervalSince1970)

            if req.stream == true {
                await streamChatResponse(
                    writer: writer, chatID: chatID, model: modelName,
                    created: created, stream: stream)
            } else {
                await bufferChatResponse(
                    writer: writer, chatID: chatID, model: modelName,
                    created: created, stream: stream)
            }
        }
    }

    // MARK: - Streaming response

    /// Drain `stream` and emit one SSE `chat.completion.chunk` per
    /// sampled token. Terminates with `data: [DONE]` per the OpenAI
    /// convention. On stream error, emits a final chunk carrying
    /// `finish_reason: "stop"` rather than failing the connection
    /// mid-frame — most OpenAI clients react badly to torn streams.
    private static func streamChatResponse(
        writer: HTTPResponseWriter,
        chatID: String,
        model: String,
        created: Int,
        stream: AsyncThrowingStream<GenerationEvent, Error>) async
    {
        do {
            try await writer.writeHead(
                status: 200, statusText: "OK",
                headers: [
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                ])

            // Emit a first chunk with role: "assistant" so OpenAI
            // clients that key off the role transition open a new
            // message bubble.
            let openChunk = chunkPayload(
                id: chatID, model: model, created: created,
                delta: ["role": "assistant"],
                finishReason: nil)
            try await writer.writeSSE(jsonString(openChunk))

            var finishReason = "stop"
            do {
                for try await event in stream {
                    switch event {
                    case .token(let text, _):
                        let chunk = chunkPayload(
                            id: chatID, model: model, created: created,
                            delta: ["content": text],
                            finishReason: nil)
                        try await writer.writeSSE(jsonString(chunk))
                    case .done:
                        finishReason = "stop"
                    case .status, .prefillStart, .prefillDone,
                         .generationProgress:
                        // Status events aren't part of the OpenAI
                        // streaming protocol; swallow them.
                        continue
                    }
                }
            } catch {
                finishReason = "stop"
            }

            let finalChunk = chunkPayload(
                id: chatID, model: model, created: created,
                delta: [:],
                finishReason: finishReason)
            try await writer.writeSSE(jsonString(finalChunk))
            try await writer.writeSSEDone()
        } catch {
            // If writing fails mid-stream the client disconnected.
            // Nothing useful to do — connection.cancel() runs in
            // LocalServer.handle().
        }
    }

    // MARK: - Buffered response

    private static func bufferChatResponse(
        writer: HTTPResponseWriter,
        chatID: String,
        model: String,
        created: Int,
        stream: AsyncThrowingStream<GenerationEvent, Error>) async
    {
        var collected = ""
        var generatedTokens = 0
        var promptTokens = 0
        var finishReason = "stop"

        do {
            for try await event in stream {
                switch event {
                case .token(let text, _):
                    collected += text
                    generatedTokens += 1
                case .done(_, let prompt, _):
                    promptTokens = prompt.count
                    finishReason = "stop"
                case .status, .prefillStart, .prefillDone,
                     .generationProgress:
                    continue
                }
            }
        } catch {
            await writeError(writer, status: 500,
                              message: "Generation failed: \(error.localizedDescription)")
            return
        }

        let payload: [String: Any] = [
            "id": chatID,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [[
                "index": 0,
                "message": [
                    "role": "assistant",
                    "content": collected,
                ],
                "finish_reason": finishReason,
            ]],
            "usage": [
                "prompt_tokens": promptTokens,
                "completion_tokens": generatedTokens,
                "total_tokens": promptTokens + generatedTokens,
            ],
        ]
        await writeJSON(writer, status: 200, payload: payload)
    }

    // MARK: - Request mapping helpers

    private static func mapMessages(
        _ openai: [OpenAIChatRequest.Message]) -> [Message]
    {
        var out: [Message] = []
        for m in openai {
            let content = m.content ?? ""
            switch m.role {
            case "system":
                out.append(Message(role: .system, content: content))
            case "user":
                out.append(Message(role: .user, content: content))
            case "assistant":
                out.append(Message(role: .assistant, content: content))
            case "tool":
                // Fold tool results into the previous assistant's
                // toolOutputs slot. If there's no preceding assistant
                // (malformed request), drop the result silently — the
                // DSV4 encoder has no place for an orphan tool block.
                if let lastIdx = out.lastIndex(where: { $0.role == .assistant }) {
                    out[lastIdx].toolOutputs.append(content)
                }
            default:
                // Unknown role — pass through as user content so the
                // model at least sees the bytes rather than 400ing.
                out.append(Message(role: .user, content: content))
            }
        }
        return out
    }

    /// OpenAI doesn't model "thinking mode" natively. Treat
    /// `reasoning_effort: "medium"` / `"high"` as a hint to flip on
    /// `.high`; everything else stays `.chat`. (The desktop app's
    /// remote picker mirrors this convention.)
    private static func thinkingMode(
        from req: OpenAIChatRequest) -> ThinkingMode
    {
        guard let effort = req.reasoning_effort?.lowercased() else {
            return .chat
        }
        switch effort {
        case "high", "max": return .high
        case "medium":      return .high
        default:            return .chat
        }
    }

    private static func samplingOptions(
        from req: OpenAIChatRequest) -> SamplingOptions
    {
        SamplingOptions(
            temperature: req.temperature ?? 1.0,
            topK: 0,
            topP: req.top_p ?? 1.0,
            minP: 0.0,
            tailFree: 1.0,
            typical: 1.0,
            repetitionPenalty: 1.0,
            frequencyPenalty: req.frequency_penalty ?? 0.0,
            presencePenalty: req.presence_penalty ?? 0.0)
    }

    // MARK: - Output helpers

    private static func chunkPayload(
        id: String, model: String, created: Int,
        delta: [String: Any], finishReason: String?) -> [String: Any]
    {
        // JSONSerialization treats Swift `nil` in `[String: Any]` as
        // "key absent", but the OpenAI streaming protocol requires
        // the `finish_reason` slot to be present (with explicit JSON
        // `null` until the final chunk). Use NSNull() to force the
        // null literal.
        let fr: Any = finishReason ?? NSNull()
        return [
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [[
                "index": 0,
                "delta": delta,
                "finish_reason": fr,
            ]],
        ]
    }

    private static func jsonString(_ obj: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(
            withJSONObject: obj, options: [.withoutEscapingSlashes])
        else { return "{}" }
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    private static func writeJSON(_ writer: HTTPResponseWriter,
                                   status: Int,
                                   payload: [String: Any]) async
    {
        let body: Data
        if let data = try? JSONSerialization.data(
            withJSONObject: payload, options: [.withoutEscapingSlashes])
        {
            body = data
        } else {
            body = Data("{}".utf8)
        }
        try? await writer.writeHead(
            status: status,
            statusText: HTTPStatus.text(for: status),
            headers: [
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": String(body.count),
                "Connection": "close",
            ])
        try? await writer.write(body)
    }

    private static func writeError(_ writer: HTTPResponseWriter,
                                    status: Int,
                                    message: String) async
    {
        let payload: [String: Any] = [
            "error": [
                "message": message,
                "type": "invalid_request_error",
                "code": status,
            ],
        ]
        await writeJSON(writer, status: status, payload: payload)
    }
}

// MARK: - OpenAI request shape

/// Codable shape for `POST /v1/chat/completions`. Only models the
/// subset of fields we honor today; unknown fields are silently
/// ignored (the rest of the OpenAI surface — `tools`, `tool_choice`,
/// `logprobs`, `response_format`, etc. — lands in follow-up commits).
private struct OpenAIChatRequest: Decodable {
    let model: String?
    let messages: [Message]
    let stream: Bool?
    let temperature: Float?
    let top_p: Float?
    let max_tokens: Int?
    let frequency_penalty: Float?
    let presence_penalty: Float?
    /// OpenAI's reasoning-effort field for thinking-mode hint.
    let reasoning_effort: String?

    struct Message: Decodable {
        let role: String
        let content: String?
        let name: String?
        let tool_call_id: String?
    }
}
