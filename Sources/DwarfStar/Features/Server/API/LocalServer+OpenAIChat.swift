import Foundation
@preconcurrency import Network
import DS4Core
import DS4Engine

extension LocalServer {
// MARK: /v1/chat/completions

    // MARK: Log ricco (pannello Server): cosa arriva, cosa generiamo, cosa esce

    /// Una riga di anteprima: newline collassate, troncata con ellissi.
    func preview(_ s: String, max n: Int = 140) -> String {
        let flat = s.replacingOccurrences(of: "\n", with: "⏎")
        return flat.count <= n ? flat : String(flat.prefix(n)) + "…"
    }

    /// L'ultimo messaggio utente della conversazione (per il log della richiesta).
    static func lastUserText(_ turns: [ChatTurn]) -> String? {
        for t in turns.reversed() { if case .user(let s) = t { return s } }
        return nil
    }

    /// Composizione del transcript per ruolo: rende verificabile il conteggio
    /// del prefill ("prefill 765 token" per un semplice "ciao" = il system
    /// prompt nascosto del client, non il messaggio visibile).
    static func turnsSummary(_ turns: [ChatTurn]) -> String {
        turns.map { t in
            switch t {
            case .system(let s):            return "system \(s.count) char"
            case .user(let s):              return "user \(s.count) char"
            case .assistant(let s, let c):  return "assistant \(s.count) char"
                                                   + (c.isEmpty ? "" : " (+\(c.count) tool call)")
            case .toolResult(_, _, let s):  return "tool_result \(s.count) char"
            }
        }.joined(separator: " · ")
    }

    /// Acquisisce il gate MISURANDO l'attesa: se il motore è occupato (altra
    /// richiesta o turno chat in corso) la richiesta resta in coda in silenzio —
    /// senza questa riga il tempo di coda sembra un hang del server.
    func acquireGate() async {
        let t0 = Date()
        await gate.acquire()
        let dt = Date().timeIntervalSince(t0)
        if dt > 0.5 {
            onLog(String(format: "… richiesta rimasta in coda %.1fs (motore occupato)\n", dt))
        }
    }

    /// Chiama `engine.complete` misurando quanto costa PRIMA che parta la
    /// generazione: hop sull'actor (che può essere occupato da un turno chat,
    /// fuori dal gate) + render + tokenizzazione dell'intero transcript.
    func startCompletion(turns: [ChatTurn], tools: [ToolSpec], think: DS4ThinkMode,
                                 sampling: SamplingParams, maxTokens: Int) async
        -> AsyncThrowingStream<GenEvent, Error> {
        let t0 = Date()
        let stream = await engine.complete(turns: turns, tools: tools, thinkMode: think,
                                           sampling: sampling, maxTokens: maxTokens)
        let dt = Date().timeIntervalSince(t0)
        if dt > 0.5 {
            onLog(String(format: "· preparazione prompt %.1fs (render+tokenizzazione, attesa motore inclusa)\n", dt))
        }
        return stream
    }

    func handleChat(_ conn: NWConnection, body: Data) async throws {
        guard let obj = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any] else {
            try await send(conn, Self.httpError(400, "invalid JSON body", cors: config.cors)); return
        }
        let parsed = ChatRequestParser.parse(obj, defaultTokens: config.maxTokens)
        guard !parsed.turns.isEmpty else {
            try await send(conn, Self.httpError(400, "no messages", cors: config.cors)); return
        }
        let model = resolveModel(parsed.model)
        let id = "chatcmpl-" + String(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))
        let created = Int(Date().timeIntervalSince1970)
        onLog("POST /v1/chat/completions (\(parsed.turns.count) msg · stream=\(parsed.stream) · "
              + "maxTok=\(parsed.maxTokens) · tools=\(parsed.tools.count) · body \(body.count) B)\n")
        if let u = Self.lastUserText(parsed.turns) {
            onLog("→ user: \"\(preview(u))\"\n")
        }
        onLog("→ transcript: \(Self.turnsSummary(parsed.turns))\n")

        // Serialize: only one generation runs against the single-model engine.
        await acquireGate()
        defer { Task { await gate.release() } }

        let stream = await startCompletion(turns: parsed.turns, tools: parsed.tools,
                                           think: parsed.think, sampling: parsed.sampling,
                                           maxTokens: parsed.maxTokens)
        if parsed.stream {
            try await streamChat(conn, stream: stream, id: id, model: model, created: created)
        } else {
            try await bufferChat(conn, stream: stream, id: id, model: model, created: created)
        }
    }

    /// Streaming (SSE) chat.completion.chunk events, matching ds4_server.c.
    func streamChat(_ conn: NWConnection, stream: AsyncThrowingStream<GenEvent, Error>,
                            id: String, model: String, created: Int) async throws {
        try await send(conn, Data(Self.sseHeader(cors: config.cors).utf8))
        let m = jsonString(model)
        func chunk(_ delta: String, finish: String) -> Data {
            Data(("data: {\"id\":\"\(id)\",\"object\":\"chat.completion.chunk\",\"created\":\(created),\"model\":\(m),\"choices\":[{\"index\":0,\"delta\":\(delta),\"finish_reason\":\(finish)}]}\n\n").utf8)
        }
        // First chunk announces the assistant role (OpenAI convention).
        try await send(conn, chunk("{\"role\":\"assistant\"}", finish: "null"))

        var finish = "stop"
        // Contatori per il log: quanto stiamo generando, a che ritmo, e una
        // coda della risposta per l'anteprima a fine stream.
        let t0 = Date()
        var textChars = 0, reasonChars = 0, toolCallCount = 0
        var tail = ""
        // distantPast: la PRIMA riga di progresso ("prefill N token…",
        // "ripristino KV…") deve uscire subito — con Date() veniva soppressa
        // dal throttle e con chunk da 512 token la prima riga visibile
        // arrivava dopo minuti: prefill in corso ma log muto = "tempo morto".
        var lastNote = Date.distantPast
        func note() {
            guard Date().timeIntervalSince(lastNote) >= 2 else { return }
            lastNote = Date()
            let dt = Date().timeIntervalSince(t0)
            onLog(String(format: "… generazione: %d char testo (+%d reasoning) · %.1fs\n",
                         textChars, reasonChars, dt))
        }
        do {
            for try await event in stream {
                switch event {
                case .reasoning(let r):
                    reasonChars += r.count
                    try await send(conn, chunk("{\"reasoning_content\":\(jsonString(r))}", finish: "null"))
                    note()
                case .text(let t):
                    textChars += t.count
                    tail = String((tail + t).suffix(200))
                    try await send(conn, chunk("{\"content\":\(jsonString(t))}", finish: "null"))
                    note()
                case .toolCall(let calls):
                    finish = "tool_calls"
                    toolCallCount = calls.count
                    for c in calls { onLog("← tool call: \(c.name)(\(preview(c.argumentsJSON, max: 100)))\n") }
                    try await send(conn, chunk("{\"tool_calls\":\(toolCallsJSON(calls))}", finish: "null"))
                case .progress(let p):
                    // Il PREFILL può durare minuti (transcript intero al primo
                    // giro): senza queste righe il log tace e sembra un hang.
                    // "resa:" (attribuzione motore/sampler/resto a fine turno)
                    // salta il throttle: è la riga che diagnostica la lentezza.
                    if !p.isEmpty, p.hasPrefix("resa:") || Date().timeIntervalSince(lastNote) >= 2 {
                        lastNote = Date()
                        onLog("· \(p)\n")
                    }
                case .toolStream:
                    break
                }
            }
        } catch is CancellationError {
            finish = "stop"
            onLog("← client disconnesso a metà stream (generazione fermata)\n")
        }
        try await send(conn, chunk("{}", finish: "\"\(finish)\""))
        try await send(conn, Data("data: [DONE]\n\n".utf8))
        let dt = Date().timeIntervalSince(t0)
        onLog(String(format: "← SSE chiuso: finish=%@ · %d char testo (+%d reasoning, %d tool call) · %.1fs\n",
                     finish, textChars, reasonChars, toolCallCount, dt))
        if !tail.isEmpty { onLog("← coda risposta: \"…\(preview(tail, max: 200))\"\n") }
    }

    /// Non-streaming: collect the full reply into one chat.completion body.
    func bufferChat(_ conn: NWConnection, stream: AsyncThrowingStream<GenEvent, Error>,
                            id: String, model: String, created: Int) async throws {
        var content = "", reasoning = "", finish = "stop"
        var calls: [ToolCall] = []
        let t0 = Date()
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
        let dt = Date().timeIntervalSince(t0)
        onLog(String(format: "← risposta (non-stream): finish=%@ · %d char testo (+%d reasoning, %d tool call) · %.1fs\n",
                     finish, content.count, reasoning.count, calls.count, dt))
        for c in calls { onLog("← tool call: \(c.name)(\(preview(c.argumentsJSON, max: 100)))\n") }
        if !content.isEmpty { onLog("← anteprima: \"\(preview(content, max: 200))\"\n") }

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
}

