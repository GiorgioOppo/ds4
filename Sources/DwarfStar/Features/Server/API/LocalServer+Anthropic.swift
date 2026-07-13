import Foundation
@preconcurrency import Network
import DS4Core
import DS4Engine

extension LocalServer {
// MARK: /v1/messages (Anthropic Messages API)

    func handleAnthropic(_ conn: NWConnection, body: Data) async throws {
        guard let obj = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any] else {
            try await send(conn, Self.anthropicError(400, "invalid JSON body", cors: config.cors)); return
        }
        let parsed = ChatRequestParser.parseAnthropic(obj, defaultTokens: config.maxTokens)
        guard !parsed.turns.isEmpty else {
            try await send(conn, Self.anthropicError(400, "no messages", cors: config.cors)); return
        }
        let model = resolveModel(parsed.model)
        let id = "msg_" + String(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))
        onLog("POST /v1/messages (\(parsed.turns.count) msg, stream=\(parsed.stream))\n")

        await acquireGate()
        defer { Task { await gate.release() } }

        let stream = await startCompletion(turns: parsed.turns, tools: parsed.tools,
                                           think: parsed.think, sampling: parsed.sampling,
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
    func streamAnthropic(_ conn: NWConnection, stream: AsyncThrowingStream<GenEvent, Error>,
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

    func bufferAnthropic(_ conn: NWConnection, stream: AsyncThrowingStream<GenEvent, Error>,
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
}

