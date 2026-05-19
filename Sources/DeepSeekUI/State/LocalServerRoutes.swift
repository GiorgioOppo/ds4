import Foundation
import DeepSeekKit

/// OpenAI-compatible route handlers wired onto a `LocalServer` (T1
/// of TODO §10.1). Today's surface:
///
///   - `GET  /v1/models`              — returns the currently loaded
///                                       model as a singleton catalog
///   - `POST /v1/chat/completions`    — both `stream: true` (SSE) and
///                                       `stream: false` (single JSON);
///                                       supports server-side `tools[]`
///                                       dispatch through `MCPClientPool`
///
/// Tool flow (mirrors the local-backend loop in `ChatStore`):
///   1. Build `toolSchemasJSON` from the controller's MCP registry,
///      optionally filtered by the `tools[*].function.name` array in
///      the request body.
///   2. Tokenize the user-supplied messages with that schema injected
///      into the DSV4 prompt.
///   3. Generate until `.done`. If the parsed assistant Message has
///      no `toolCalls`, emit the text to the client and we're done.
///   4. Otherwise: append the assistant turn to the in-memory
///      history, invoke each tool via the controller, slot the
///      results into `toolOutputs`, and loop. Cap at 8 round-trips
///      to bound runaway models — the cap matches `ChatStore`'s
///      `maxToolRoundtripsPerTurn`.
///
/// Limitation (documented): streaming clients see no SSE bytes while
/// intermediate tool rounds run. Only the final text-only round
/// streams. This is the same trade-off `ChatStore` makes on the local
/// path; OpenAI's `delta.tool_calls` streaming would require
/// formatting DSML tool calls back to OpenAI tool_calls JSON
/// per-token, which is a separate workstream.
enum LocalServerRoutes {
    /// Hard cap on tool-call iterations per request. Same constant
    /// as ChatStore's local loop.
    private static let maxToolRoundtripsPerTurn = 8

    /// Wire every supported route onto `server`. Call once after
    /// `LocalServer.start(...)` succeeds. `controller` brokers all
    /// `@MainActor`-isolated access (MCP pool, Keychain reads).
    static func register(on server: LocalServer,
                          service: InferenceService,
                          controller: LocalServerController) async {
        await server.register(method: "GET", path: "/v1/models",
                               handler: makeModelsHandler(service: service))
        await server.register(method: "POST", path: "/v1/chat/completions",
                               handler: makeChatCompletionsHandler(
                                service: service, controller: controller))
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
        service: InferenceService,
        controller: LocalServerController) -> LocalServerHandler
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

            guard service.loadedModelDir != nil else {
                await writeError(writer, status: 503,
                                  message: "No local model loaded on the server. "
                                           + "Pick a model in the desktop app first.")
                return
            }

            // Resolve which MCP tools to expose this turn:
            //   - request.tools nil → expose every registered tool
            //   - request.tools [] → opt out, no tools
            //   - request.tools non-empty → filter by name
            let allowedNames: Set<String>? = {
                guard let tools = req.tools else { return nil }
                let names = tools.compactMap { $0.function?.name }
                return Set(names)
            }()
            let toolSchemasJSON = await controller.composeToolSchemasJSON(
                allowedNames: allowedNames)

            // Map OpenAI messages → DeepSeekKit Message.
            var history = mapMessages(req.messages)
            let mode = thinkingMode(from: req)
            var options = samplingOptions(from: req)
            let maxTokens = req.max_tokens ?? 1024

            // T3: `response_format: {type:"json_schema", json_schema:
            // {schema: …}}` — compile the inner schema into a
            // SchemaMask and attach it to the sampler. Failures
            // surface as a 400 so the client knows the constraint
            // wasn't honored, rather than silently getting an
            // unconstrained response.
            if let schemaDict = extractResponseSchema(body: request.body) {
                guard let (tok, cfg) = service.snapshotTokenizerAndConfig()
                else {
                    await writeError(writer, status: 503,
                                      message: "response_format requested but "
                                               + "no model is loaded.")
                    return
                }
                do {
                    let mask = try SchemaCompiler.compile(
                        schema: schemaDict,
                        tokenizer: tok,
                        vocabSize: cfg.vocabSize)
                    options.schemaMask = mask
                } catch {
                    await writeError(writer, status: 400,
                                      message: "response_format: \(error.localizedDescription)")
                    return
                }
            }

            let conversationID = UUID()
            let modelName = service.loadedModelDir?.lastPathComponent
                ?? "local-model"
            let chatID = "chatcmpl-\(UUID().uuidString)"
            let created = Int(Date().timeIntervalSince1970)

            // Tool-call loop. Each non-final iteration appends an
            // assistant turn with toolCalls + toolOutputs and re-runs
            // the model. The final iteration (zero toolCalls in the
            // parsed Message) is the one that streams / buffers to
            // the client.
            var totalPromptTokens = 0
            var totalGeneratedTokens = 0

            for _ in 0..<maxToolRoundtripsPerTurn {
                guard let promptTokens = await service.tokenizeFullHistory(
                    history, mode: mode,
                    toolSchemasJSON: toolSchemasJSON)
                else {
                    await writeError(writer, status: 500,
                                      message: "Tokenizer unavailable.")
                    return
                }
                totalPromptTokens = promptTokens.count
                let stream = service.generateForConversation(
                    promptTokens: promptTokens,
                    conversationID: conversationID,
                    mode: mode,
                    options: options,
                    maxTokens: maxTokens)

                // Drain the stream. We need the final parsed Message
                // (with `toolCalls` populated by EncodingDSV4) before
                // we can decide whether to stream or loop again — so
                // intermediate iterations always buffer. Only the
                // last iteration (no tool calls) gets to stream live.
                var finalMessage: Message?

                do {
                    for try await event in stream {
                        switch event {
                        case .token:
                            totalGeneratedTokens += 1
                        case .done(let msg, _, _):
                            finalMessage = msg
                        case .status, .prefillToken, .prefillStart,
                             .prefillDone, .generationProgress:
                            continue
                        }
                    }
                } catch {
                    await writeError(writer, status: 500,
                                      message: "Generation failed: \(error.localizedDescription)")
                    return
                }

                guard let msg = finalMessage else {
                    await writeError(writer, status: 500,
                                      message: "Inference stream ended without a final message.")
                    return
                }

                if msg.toolCalls.isEmpty {
                    // Final round — emit to the client.
                    if req.stream == true {
                        await streamFinalResponse(
                            writer: writer, chatID: chatID,
                            model: modelName, created: created,
                            text: msg.content,
                            promptTokens: totalPromptTokens,
                            generatedTokens: totalGeneratedTokens)
                    } else {
                        await bufferedFinalResponse(
                            writer: writer, chatID: chatID,
                            model: modelName, created: created,
                            text: msg.content,
                            finishReason: "stop",
                            promptTokens: totalPromptTokens,
                            generatedTokens: totalGeneratedTokens)
                    }
                    return
                }

                // Tool-call round: extend history with the assistant
                // turn carrying the calls + outputs, then loop.
                var assistant = msg
                var outputs: [String] = []
                outputs.reserveCapacity(msg.toolCalls.count)
                for call in msg.toolCalls {
                    let result = await controller.invokeQualified(
                        call.name, argsJSON: call.args)
                    outputs.append(result)
                }
                assistant.toolOutputs = outputs
                history.append(assistant)
            }

            // Hit the round-trip cap with tools still outstanding.
            // Emit whatever the last assistant message produced with
            // finish_reason: "tool_calls" so OpenAI clients can take
            // over the tool dispatch themselves.
            let lastText = history.last?.content ?? ""
            await bufferedFinalResponse(
                writer: writer, chatID: chatID,
                model: modelName, created: created,
                text: lastText,
                finishReason: "tool_calls",
                promptTokens: totalPromptTokens,
                generatedTokens: totalGeneratedTokens)
        }
    }

    // MARK: - Response writers (final-round only)

    private static func streamFinalResponse(
        writer: HTTPResponseWriter,
        chatID: String, model: String, created: Int,
        text: String,
        promptTokens: Int, generatedTokens: Int) async
    {
        // promptTokens / generatedTokens are accepted for symmetry
        // with the buffered path; the OpenAI SSE protocol has no
        // standard slot for usage in stream chunks.
        _ = promptTokens
        _ = generatedTokens
        do {
            try await writer.writeHead(
                status: 200, statusText: "OK",
                headers: [
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                ])
            let opener = chunkPayload(
                id: chatID, model: model, created: created,
                delta: ["role": "assistant"], finishReason: nil)
            try await writer.writeSSE(jsonString(opener))
            if !text.isEmpty {
                let content = chunkPayload(
                    id: chatID, model: model, created: created,
                    delta: ["content": text], finishReason: nil)
                try await writer.writeSSE(jsonString(content))
            }
            let finalChunk = chunkPayload(
                id: chatID, model: model, created: created,
                delta: [:], finishReason: "stop")
            try await writer.writeSSE(jsonString(finalChunk))
            try await writer.writeSSEDone()
        } catch {
            // Client disconnected mid-flush — nothing useful to do.
        }
    }

    private static func bufferedFinalResponse(
        writer: HTTPResponseWriter,
        chatID: String, model: String, created: Int,
        text: String, finishReason: String,
        promptTokens: Int, generatedTokens: Int) async
    {
        let payload: [String: Any] = [
            "id": chatID,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [[
                "index": 0,
                "message": [
                    "role": "assistant",
                    "content": text,
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
                out.append(Message(role: .user, content: content))
            }
        }
        return out
    }

    /// Extract `response_format.json_schema.schema` from the raw
    /// request body. We re-parse the body with `JSONSerialization`
    /// because `OpenAIChatRequest` is `Codable` over a fixed shape
    /// and the schema dict is intentionally unstructured.
    /// Returns nil when the request doesn't carry a JSON-schema
    /// response_format (including the "text" / "json_object"
    /// variants we don't constrain).
    private static func extractResponseSchema(
        body: Data) -> [String: Any]?
    {
        guard let obj = try? JSONSerialization.jsonObject(with: body)
                as? [String: Any]
        else { return nil }
        guard let rf = obj["response_format"] as? [String: Any] else {
            return nil
        }
        if (rf["type"] as? String) != "json_schema" { return nil }
        let js = rf["json_schema"] as? [String: Any]
        return (js?["schema"] as? [String: Any]) ?? js
    }

    /// OpenAI doesn't model "thinking mode" natively. Treat
    /// `reasoning_effort: "medium"` / `"high"` as a hint to flip on
    /// `.high`; everything else stays `.chat`.
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
        // the `finish_reason` slot present (with explicit JSON `null`
        // until the final chunk). Use NSNull() to force the null literal.
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
/// ignored.
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
    /// Optional tool advertisement from the client. We honor it only
    /// as a name-filter against our own MCP registry — the schemas
    /// here are ignored; the authoritative schemas come from the MCP
    /// servers.
    let tools: [Tool]?

    struct Message: Decodable {
        let role: String
        let content: String?
        let name: String?
        let tool_call_id: String?
    }

    struct Tool: Decodable {
        let type: String?
        let function: Function?
        struct Function: Decodable {
            let name: String
        }
    }
}
