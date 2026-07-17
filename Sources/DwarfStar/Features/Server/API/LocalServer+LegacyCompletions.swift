import Foundation
@preconcurrency import Network
import DS4Core
import DS4Engine

extension LocalServer {
// MARK: /v1/completions (legacy text completion)

    func handleCompletions(_ conn: NWConnection, body: Data) async throws {
        guard let obj = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any] else {
            try await send(conn, Self.httpError(400, "invalid JSON body", cors: config.cors)); return
        }
        let parsed = ChatRequestParser.parseCompletion(obj, defaultTokens: config.maxTokens)
        guard !parsed.turns.isEmpty else {
            try await send(conn, Self.httpError(400, "no prompt", cors: config.cors)); return
        }
        let model = resolveModel(parsed.model)
        let id = "cmpl-" + String(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))
        let created = Int(Date().timeIntervalSince1970)
        onLog("POST /v1/completions (stream=\(parsed.stream))\n")

        try await acquireGate()
        defer { gate.release() }

        let stream = await startCompletion(turns: parsed.turns, tools: [],
                                           think: parsed.think, sampling: parsed.sampling,
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
}
