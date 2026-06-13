import Foundation
import DS4Core
import DS4Engine

/// Maps an OpenAI `/v1/chat/completions` JSON body into the engine's types.
/// Tolerant: unknown fields are ignored, missing fields take sane defaults.
enum ChatRequestParser {
    struct Parsed {
        var turns: [ChatTurn]
        var tools: [ToolSpec]
        var sampling: SamplingParams
        var think: DS4ThinkMode
        var maxTokens: Int
        var stream: Bool
        var model: String?
    }

    static func parse(_ obj: [String: Any], defaultTokens: Int) -> Parsed {
        let turns = parseMessages(obj["messages"] as? [[String: Any]] ?? [])
        let tools = parseTools(obj["tools"] as? [[String: Any]] ?? [])

        var sampling = SamplingParams()
        if let t = number(obj["temperature"]) { sampling.temperature = Float(t) }
        if let p = number(obj["top_p"]) { sampling.topP = Float(p) }
        if let mp = number(obj["min_p"]) { sampling.minP = Float(mp) }
        if let k = obj["top_k"] as? Int { sampling.topK = k }
        if let s = obj["seed"] as? Int { sampling.seed = UInt64(bitPattern: Int64(s)) }

        // reasoning_effort: high/xhigh/medium enable thinking; low/minimal/none/absent disable.
        var think: DS4ThinkMode = .none
        if let effort = (obj["reasoning_effort"] as? String)?.lowercased(),
           ["high", "xhigh", "medium"].contains(effort) {
            think = .high
        }

        let maxTokens = (obj["max_tokens"] as? Int) ?? (obj["max_completion_tokens"] as? Int) ?? defaultTokens
        let stream = (obj["stream"] as? Bool) ?? false
        let model = obj["model"] as? String

        return Parsed(turns: turns, tools: tools, sampling: sampling, think: think,
                      maxTokens: max(1, maxTokens), stream: stream, model: model)
    }

    // MARK: Anthropic /v1/messages

    static func parseAnthropic(_ obj: [String: Any], defaultTokens: Int) -> Parsed {
        var turns: [ChatTurn] = []
        let system = anthropicText(obj["system"])
        if !system.isEmpty { turns.append(.system(system)) }

        for m in obj["messages"] as? [[String: Any]] ?? [] {
            let role = m["role"] as? String ?? ""
            if let s = m["content"] as? String {
                if role == "assistant" { turns.append(.assistant(text: s, toolCalls: [])) }
                else { turns.append(.user(s)) }
                continue
            }
            let blocks = m["content"] as? [[String: Any]] ?? []
            if role == "assistant" {
                var text = ""
                var calls: [ToolCall] = []
                for (i, b) in blocks.enumerated() {
                    switch b["type"] as? String {
                    case "text": text += b["text"] as? String ?? ""
                    case "tool_use":
                        let id = b["id"] as? String ?? "call_\(i)"
                        let name = b["name"] as? String ?? ""
                        let args = jsonStringOf(b["input"]) ?? "{}"
                        if !name.isEmpty { calls.append(ToolCall(id: id, name: name, argumentsJSON: args)) }
                    default: break
                    }
                }
                turns.append(.assistant(text: text, toolCalls: calls))
            } else {
                var text = ""
                for b in blocks {
                    switch b["type"] as? String {
                    case "text": text += b["text"] as? String ?? ""
                    case "tool_result":
                        let id = b["tool_use_id"] as? String ?? ""
                        turns.append(.toolResult(callId: id, name: "", content: anthropicText(b["content"])))
                    default: break
                    }
                }
                if !text.isEmpty { turns.append(.user(text)) }
            }
        }

        var tools: [ToolSpec] = []
        for t in obj["tools"] as? [[String: Any]] ?? [] {
            guard let name = t["name"] as? String, !name.isEmpty else { continue }
            let desc = t["description"] as? String ?? ""
            let schema = jsonStringOf(t["input_schema"]) ?? #"{"type":"object","properties":{}}"#
            tools.append(ToolSpec(name: name, description: desc, parametersJSON: schema))
        }

        var sampling = SamplingParams()
        if let t = number(obj["temperature"]) { sampling.temperature = Float(t) }
        if let p = number(obj["top_p"]) { sampling.topP = Float(p) }
        if let k = obj["top_k"] as? Int { sampling.topK = k }

        var think: DS4ThinkMode = .none
        if let th = obj["thinking"] as? [String: Any], (th["type"] as? String) == "enabled" { think = .high }

        let maxTokens = (obj["max_tokens"] as? Int) ?? defaultTokens
        return Parsed(turns: turns, tools: tools, sampling: sampling, think: think,
                      maxTokens: max(1, maxTokens), stream: (obj["stream"] as? Bool) ?? false,
                      model: obj["model"] as? String)
    }

    // MARK: OpenAI /v1/completions (legacy)

    static func parseCompletion(_ obj: [String: Any], defaultTokens: Int) -> Parsed {
        var prompt = ""
        if let s = obj["prompt"] as? String { prompt = s }
        else if let arr = obj["prompt"] as? [String] { prompt = arr.joined(separator: "\n") }

        var sampling = SamplingParams()
        if let t = number(obj["temperature"]) { sampling.temperature = Float(t) }
        if let p = number(obj["top_p"]) { sampling.topP = Float(p) }
        if let k = obj["top_k"] as? Int { sampling.topK = k }

        let maxTokens = (obj["max_tokens"] as? Int) ?? defaultTokens
        return Parsed(turns: prompt.isEmpty ? [] : [.user(prompt)], tools: [], sampling: sampling,
                      think: .none, maxTokens: max(1, maxTokens),
                      stream: (obj["stream"] as? Bool) ?? false, model: obj["model"] as? String)
    }

    // MARK: OpenAI /v1/responses

    static func parseResponses(_ obj: [String: Any], defaultTokens: Int) -> Parsed {
        var turns: [ChatTurn] = []
        if let s = obj["instructions"] as? String, !s.isEmpty { turns.append(.system(s)) }

        if let s = obj["input"] as? String {
            if !s.isEmpty { turns.append(.user(s)) }
        } else if let items = obj["input"] as? [[String: Any]] {
            for (i, it) in items.enumerated() {
                switch it["type"] as? String ?? "message" {
                case "message", "":
                    let role = it["role"] as? String ?? "user"
                    let text = responsesContent(it["content"])
                    if role == "assistant" { turns.append(.assistant(text: text, toolCalls: [])) }
                    else if role == "system" || role == "developer" { turns.append(.system(text)) }
                    else { turns.append(.user(text)) }
                case "function_call":
                    let id = (it["call_id"] as? String) ?? (it["id"] as? String) ?? "call_\(i)"
                    let name = it["name"] as? String ?? ""
                    let args = it["arguments"] as? String ?? "{}"
                    if !name.isEmpty {
                        turns.append(.assistant(text: "", toolCalls: [ToolCall(id: id, name: name, argumentsJSON: args)]))
                    }
                case "function_call_output":
                    let id = it["call_id"] as? String ?? ""
                    let out = (it["output"] as? String) ?? responsesContent(it["output"])
                    turns.append(.toolResult(callId: id, name: "", content: out))
                default: break
                }
            }
        }

        var tools: [ToolSpec] = []
        for t in obj["tools"] as? [[String: Any]] ?? [] {
            guard let name = t["name"] as? String, !name.isEmpty else { continue }
            let desc = t["description"] as? String ?? ""
            let pj = jsonStringOf(t["parameters"]) ?? #"{"type":"object","properties":{}}"#
            tools.append(ToolSpec(name: name, description: desc, parametersJSON: pj))
        }

        var sampling = SamplingParams()
        if let t = number(obj["temperature"]) { sampling.temperature = Float(t) }
        if let p = number(obj["top_p"]) { sampling.topP = Float(p) }
        if let k = obj["top_k"] as? Int { sampling.topK = k }

        var think: DS4ThinkMode = .none
        if let r = obj["reasoning"] as? [String: Any],
           let e = (r["effort"] as? String)?.lowercased(), ["high", "xhigh", "medium"].contains(e) {
            think = .high
        }

        let maxTokens = (obj["max_output_tokens"] as? Int) ?? (obj["max_tokens"] as? Int) ?? defaultTokens
        return Parsed(turns: turns, tools: tools, sampling: sampling, think: think,
                      maxTokens: max(1, maxTokens), stream: (obj["stream"] as? Bool) ?? false,
                      model: obj["model"] as? String)
    }

    /// Responses `content` may be a string or an array of input_text/output_text/text parts.
    private static func responsesContent(_ value: Any?) -> String {
        if let s = value as? String { return s }
        if let parts = value as? [[String: Any]] {
            var out = ""
            for p in parts {
                switch p["type"] as? String {
                case "input_text", "output_text", "text", .none: out += p["text"] as? String ?? ""
                default: break
                }
            }
            return out
        }
        return ""
    }

    /// Anthropic `system`/`content` may be a string or an array of text blocks.
    private static func anthropicText(_ value: Any?) -> String {
        if let s = value as? String { return s }
        if let parts = value as? [[String: Any]] {
            var out = ""
            for p in parts where (p["type"] as? String) == "text" {
                out += p["text"] as? String ?? ""
            }
            return out
        }
        return ""
    }

    /// Serialize an arbitrary JSON value (tool schema / tool input) back to a string.
    private static func jsonStringOf(_ value: Any?) -> String? {
        guard let value, JSONSerialization.isValidJSONObject(value) || value is [Any],
              let d = try? JSONSerialization.data(withJSONObject: value),
              let s = String(data: d, encoding: .utf8) else { return nil }
        return s
    }

    // MARK: Messages

    private static func parseMessages(_ msgs: [[String: Any]]) -> [ChatTurn] {
        var turns: [ChatTurn] = []
        for m in msgs {
            let role = m["role"] as? String ?? ""
            let content = contentString(m["content"])
            switch role {
            case "system", "developer":
                turns.append(.system(content))
            case "user":
                turns.append(.user(content))
            case "assistant":
                turns.append(.assistant(text: content, toolCalls: parseToolCalls(m["tool_calls"])))
            case "tool", "function":
                let id = m["tool_call_id"] as? String ?? ""
                let name = m["name"] as? String ?? ""
                turns.append(.toolResult(callId: id, name: name, content: content))
            default:
                if !content.isEmpty { turns.append(.user(content)) }
            }
        }
        return turns
    }

    /// `content` may be a string or an array of content parts ({type:"text",text:…}).
    private static func contentString(_ value: Any?) -> String {
        if let s = value as? String { return s }
        if let parts = value as? [[String: Any]] {
            var out = ""
            for p in parts {
                if (p["type"] as? String) == "text", let t = p["text"] as? String { out += t }
            }
            return out
        }
        return ""
    }

    private static func parseToolCalls(_ value: Any?) -> [ToolCall] {
        guard let arr = value as? [[String: Any]] else { return [] }
        var calls: [ToolCall] = []
        for (i, c) in arr.enumerated() {
            let fn = c["function"] as? [String: Any]
            let name = fn?["name"] as? String ?? ""
            let args = fn?["arguments"] as? String ?? "{}"
            let id = c["id"] as? String ?? "call_\(i)"
            if !name.isEmpty { calls.append(ToolCall(id: id, name: name, argumentsJSON: args)) }
        }
        return calls
    }

    // MARK: Tools

    private static func parseTools(_ tools: [[String: Any]]) -> [ToolSpec] {
        var specs: [ToolSpec] = []
        for t in tools {
            guard let fn = t["function"] as? [String: Any],
                  let name = fn["name"] as? String, !name.isEmpty else { continue }
            let desc = fn["description"] as? String ?? ""
            let paramsJSON: String
            if let params = fn["parameters"],
               let d = try? JSONSerialization.data(withJSONObject: params),
               let s = String(data: d, encoding: .utf8) {
                paramsJSON = s
            } else {
                paramsJSON = #"{"type":"object","properties":{}}"#
            }
            specs.append(ToolSpec(name: name, description: desc, parametersJSON: paramsJSON))
        }
        return specs
    }

    private static func number(_ value: Any?) -> Double? {
        if let d = value as? Double { return d }
        if let i = value as? Int { return Double(i) }
        if let n = value as? NSNumber { return n.doubleValue }
        return nil
    }
}
