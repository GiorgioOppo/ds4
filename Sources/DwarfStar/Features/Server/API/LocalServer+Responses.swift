import Foundation
@preconcurrency import Network
import DS4Core
import DS4Engine

extension LocalServer {
// MARK: /v1/responses (OpenAI Responses API)

    func handleResponses(_ conn: NWConnection, body: Data) async throws {
        guard let obj = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any] else {
            try await send(conn, Self.httpError(400, "invalid JSON body", cors: config.cors)); return
        }
        let parsed = ChatRequestParser.parseResponses(obj, defaultTokens: config.maxTokens)
        guard !parsed.turns.isEmpty else {
            try await send(conn, Self.httpError(400, "no input", cors: config.cors)); return
        }
        do {
            try ToolHistoryValidator.validate(parsed.turns)
        } catch {
            try await send(conn, Self.httpError(
                400, "invalid tool history: \(error)", cors: config.cors))
            return
        }
        let model = resolveModel(parsed.model)
        let created = Int(Date().timeIntervalSince1970)
        func rid(_ p: String) -> String { p + String(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24)) }
        let respId = rid("resp_"), msgId = rid("msg_"), rsId = rid("rs_")
        onLog("POST /v1/responses (\(parsed.turns.count) item, stream=\(parsed.stream))\n")

        try await acquireGate()
        defer { gate.release() }

        let stream = await startCompletion(turns: parsed.turns, tools: parsed.tools,
                                           think: parsed.think, sampling: parsed.sampling,
                                           maxTokens: parsed.maxTokens)
        if parsed.stream {
            try await streamResponses(conn, stream: stream, model: model, created: created,
                                      respId: respId, msgId: msgId, rsId: rsId)
        } else {
            try await bufferResponses(conn, stream: stream, model: model, created: created,
                                      respId: respId, msgId: msgId, rsId: rsId)
        }
    }

    struct ToolItem { let fcId: String; let callId: String; let name: String; let args: String; let index: Int }

    func responseEnvelope(_ respId: String, model: String, created: Int,
                                  status: String, output: String, usage: Bool) -> String {
        var s = "{\"id\":\"\(respId)\",\"object\":\"response\",\"created_at\":\(created),\"status\":\"\(status)\",\"model\":\(jsonString(model)),\"output\":\(output)"
        if usage { s += ",\"usage\":{\"input_tokens\":0,\"output_tokens\":0,\"total_tokens\":0}" }
        return s + "}"
    }

    /// Full OpenAI Responses streaming lifecycle:
    /// created → (reasoning item) → (message item) → (function_call items) → completed.
    func streamResponses(_ conn: NWConnection, stream: AsyncThrowingStream<GenEvent, Error>,
                                 model: String, created: Int, respId: String, msgId: String, rsId: String) async throws {
        try await send(conn, Data(Self.sseHeader(cors: config.cors).utf8))
        var seq = 0
        func resp(_ type: String, _ rest: String) async throws {
            let body = rest.isEmpty ? "{\"type\":\"\(type)\",\"sequence_number\":\(seq)}"
                                    : "{\"type\":\"\(type)\",\"sequence_number\":\(seq),\(rest)}"
            seq += 1
            try await send(conn, Data("event: \(type)\ndata: \(body)\n\n".utf8))
        }

        var nextIndex = 0
        var reasoningOpen = false, reasoningEmitted = false, reasoningText = "", reasoningIndex = 0
        var messageOpen = false, messageEmitted = false, messageText = "", messageIndex = 0
        var toolItems: [ToolItem] = []

        try await resp("response.created",
            "\"response\":\(responseEnvelope(respId, model: model, created: created, status: "in_progress", output: "[]", usage: false))")

        func closeReasoning() async throws {
            guard reasoningOpen else { return }
            try await resp("response.reasoning_summary_text.done",
                "\"item_id\":\"\(rsId)\",\"output_index\":\(reasoningIndex),\"summary_index\":0,\"text\":\(jsonString(reasoningText))")
            try await resp("response.reasoning_summary_part.done",
                "\"item_id\":\"\(rsId)\",\"output_index\":\(reasoningIndex),\"summary_index\":0,\"part\":{\"type\":\"summary_text\",\"text\":\(jsonString(reasoningText))}")
            try await resp("response.output_item.done",
                "\"output_index\":\(reasoningIndex),\"item\":{\"id\":\"\(rsId)\",\"type\":\"reasoning\",\"status\":\"completed\",\"summary\":[{\"type\":\"summary_text\",\"text\":\(jsonString(reasoningText))}]}")
            reasoningOpen = false
        }
        func closeMessage() async throws {
            guard messageOpen else { return }
            try await resp("response.output_text.done",
                "\"item_id\":\"\(msgId)\",\"output_index\":\(messageIndex),\"content_index\":0,\"text\":\(jsonString(messageText))")
            try await resp("response.content_part.done",
                "\"item_id\":\"\(msgId)\",\"output_index\":\(messageIndex),\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\(jsonString(messageText)),\"annotations\":[]}")
            try await resp("response.output_item.done",
                "\"output_index\":\(messageIndex),\"item\":{\"id\":\"\(msgId)\",\"type\":\"message\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\(jsonString(messageText)),\"annotations\":[]}]}")
            messageOpen = false
        }

        do {
            for try await event in stream {
                switch event {
                case .reasoning(let r):
                    if !reasoningOpen {
                        reasoningIndex = nextIndex; nextIndex += 1
                        reasoningOpen = true; reasoningEmitted = true
                        try await resp("response.output_item.added",
                            "\"output_index\":\(reasoningIndex),\"item\":{\"id\":\"\(rsId)\",\"type\":\"reasoning\",\"status\":\"in_progress\",\"summary\":[]}")
                        try await resp("response.reasoning_summary_part.added",
                            "\"item_id\":\"\(rsId)\",\"output_index\":\(reasoningIndex),\"summary_index\":0,\"part\":{\"type\":\"summary_text\",\"text\":\"\"}")
                    }
                    reasoningText += r
                    try await resp("response.reasoning_summary_text.delta",
                        "\"item_id\":\"\(rsId)\",\"output_index\":\(reasoningIndex),\"summary_index\":0,\"delta\":\(jsonString(r))")
                case .text(let t):
                    try await closeReasoning()
                    if !messageOpen {
                        messageIndex = nextIndex; nextIndex += 1
                        messageOpen = true; messageEmitted = true
                        try await resp("response.output_item.added",
                            "\"output_index\":\(messageIndex),\"item\":{\"id\":\"\(msgId)\",\"type\":\"message\",\"status\":\"in_progress\",\"role\":\"assistant\",\"content\":[]}")
                        try await resp("response.content_part.added",
                            "\"item_id\":\"\(msgId)\",\"output_index\":\(messageIndex),\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}")
                    }
                    messageText += t
                    try await resp("response.output_text.delta",
                        "\"item_id\":\"\(msgId)\",\"output_index\":\(messageIndex),\"content_index\":0,\"delta\":\(jsonString(t))")
                case .toolCall(let calls):
                    try await closeReasoning()
                    try await closeMessage()
                    for c in calls {
                        let idx = nextIndex; nextIndex += 1
                        let fcId = "fc_" + String(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(20))
                        let item = ToolItem(fcId: fcId, callId: c.id, name: c.name, args: c.argumentsJSON, index: idx)
                        toolItems.append(item)
                        try await resp("response.output_item.added",
                            "\"output_index\":\(idx),\"item\":\(toolItemJSON(item, status: "in_progress", withArgs: false))")
                        try await resp("response.function_call_arguments.delta",
                            "\"item_id\":\"\(fcId)\",\"output_index\":\(idx),\"delta\":\(jsonString(c.argumentsJSON))")
                        try await resp("response.function_call_arguments.done",
                            "\"item_id\":\"\(fcId)\",\"output_index\":\(idx),\"name\":\(jsonString(c.name)),\"arguments\":\(jsonString(c.argumentsJSON))")
                        try await resp("response.output_item.done",
                            "\"output_index\":\(idx),\"item\":\(toolItemJSON(item, status: "completed", withArgs: true))")
                    }
                case .toolStream, .progress:
                    break
                }
            }
        } catch is CancellationError {}
        try await closeReasoning()
        try await closeMessage()

        var outItems: [String] = []
        if reasoningEmitted {
            outItems.append("{\"id\":\"\(rsId)\",\"type\":\"reasoning\",\"status\":\"completed\",\"summary\":[{\"type\":\"summary_text\",\"text\":\(jsonString(reasoningText))}]}")
        }
        if messageEmitted {
            outItems.append("{\"id\":\"\(msgId)\",\"type\":\"message\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\(jsonString(messageText)),\"annotations\":[]}]}")
        }
        for it in toolItems { outItems.append(toolItemJSON(it, status: "completed", withArgs: true)) }
        let output = "[" + outItems.joined(separator: ",") + "]"
        try await resp("response.completed",
            "\"response\":\(responseEnvelope(respId, model: model, created: created, status: "completed", output: output, usage: true))")
    }

    func bufferResponses(_ conn: NWConnection, stream: AsyncThrowingStream<GenEvent, Error>,
                                 model: String, created: Int, respId: String, msgId: String, rsId: String) async throws {
        var content = "", reasoning = ""
        var calls: [ToolCall] = []
        do {
            for try await event in stream {
                switch event {
                case .reasoning(let r): reasoning += r
                case .text(let t): content += t
                case .toolCall(let c): calls = c
                case .toolStream, .progress: break
                }
            }
        } catch is CancellationError {}

        var outItems: [String] = []
        if !reasoning.isEmpty {
            outItems.append("{\"id\":\"\(rsId)\",\"type\":\"reasoning\",\"status\":\"completed\",\"summary\":[{\"type\":\"summary_text\",\"text\":\(jsonString(reasoning))}]}")
        }
        if !content.isEmpty || calls.isEmpty {
            outItems.append("{\"id\":\"\(msgId)\",\"type\":\"message\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\(jsonString(content)),\"annotations\":[]}]}")
        }
        for (i, c) in calls.enumerated() {
            let item = ToolItem(fcId: "fc_\(i)", callId: c.id, name: c.name, args: c.argumentsJSON, index: i)
            outItems.append(toolItemJSON(item, status: "completed", withArgs: true))
        }
        let output = "[" + outItems.joined(separator: ",") + "]"
        let body = responseEnvelope(respId, model: model, created: created, status: "completed", output: output, usage: true)
        try await send(conn, Self.response(200, contentType: "application/json", body: body, cors: config.cors))
    }

    func toolItemJSON(_ it: ToolItem, status: String, withArgs: Bool) -> String {
        let args = withArgs ? jsonString(it.args.isEmpty ? "{}" : it.args) : "\"\""
        return "{\"id\":\"\(it.fcId)\",\"type\":\"function_call\",\"status\":\"\(status)\",\"name\":\(jsonString(it.name)),\"call_id\":\(jsonString(it.callId)),\"arguments\":\(args)}"
    }
}
