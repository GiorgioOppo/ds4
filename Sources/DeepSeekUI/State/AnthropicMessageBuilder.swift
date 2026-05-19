import Foundation

/// Build the JSON body for `POST /v1/messages` from this app's
/// `[StoredMessage]` transcript shape. The translation is the
/// inverse of what `AnthropicClient` does on the response side: it
/// folds tool calls / tool outputs into Anthropic's content-block
/// model.
///
/// Anthropic message format vs. OpenAI:
///   - `system` is a top-level string (or list of content blocks),
///     not a message with role "system".
///   - The assistant turn carrying tool calls uses a content-block
///     list: `[{type:"text",text:...}, {type:"tool_use", id, name,
///     input}, …]`.
///   - Tool results go in a *user* message with content blocks of
///     type `tool_result`, NOT a separate role.
///
/// `cache_control: {type:"ephemeral"}` is auto-injected on:
///   - the system block (when long enough to be cacheable — under
///     ~1024 tokens Anthropic ignores it or errors), and
///   - the last `tool_result` block of the most recent user turn
///     (Anthropic's recommended hot-cache point for agent loops).
/// Up to 4 cache breakpoints per request; we use at most 2.
enum AnthropicMessageBuilder {
    /// Anthropic's documented minimum cacheable size: 1024 tokens
    /// for Sonnet/Opus, 2048 for Haiku. We use char-count as a
    /// rough proxy (avg ~3.5 chars/token across English+code), so
    /// 3500 chars ≈ 1000 tokens. Below this threshold we omit the
    /// `cache_control` marker.
    private static let minCacheableChars = 3500

    /// Build the full request body. `model` and `maxTokens` are
    /// required; `tools` is optional (already shaped as Anthropic
    /// `{name, description, input_schema}` entries). `cacheControl`
    /// toggles the auto-injection — pass `false` to opt out (e.g.
    /// during testing).
    static func buildBody(model: String,
                           maxTokens: Int,
                           history: [StoredMessage],
                           agentSystem: String?,
                           tools: [[String: Any]]?,
                           temperature: Float? = nil,
                           topP: Float? = nil,
                           cacheControl: Bool = true) -> [String: Any]
    {
        var body: [String: Any] = [
            "model": model,
            "max_tokens": maxTokens,
        ]

        // System: agent prompt prepended to any transcript-side
        // system messages. Anthropic accepts either a plain string or
        // a list of content blocks; we use the block form so we can
        // attach cache_control.
        let systemString = composeSystem(
            history: history, agentSystem: agentSystem)
        if !systemString.isEmpty {
            if cacheControl && systemString.count >= minCacheableChars {
                body["system"] = [[
                    "type": "text",
                    "text": systemString,
                    "cache_control": ["type": "ephemeral"],
                ]]
            } else {
                body["system"] = systemString
            }
        }

        // Messages array.
        body["messages"] = composeMessages(
            history: history, cacheControl: cacheControl)

        if let tools, !tools.isEmpty {
            body["tools"] = tools
        }
        if let t = temperature { body["temperature"] = t }
        if let p = topP { body["top_p"] = p }
        return body
    }

    /// Translate OpenAI-style `tools` (the shape `composeOpenAITools`
    /// produces) into Anthropic-style. The schemas are identical
    /// (JSON Schema), only the wrapper keys differ:
    ///   OpenAI:   {type:"function", function:{name, description, parameters}}
    ///   Anthropic:{name, description, input_schema}
    static func translateTools(_ openai: [[String: Any]]?) -> [[String: Any]]? {
        guard let openai, !openai.isEmpty else { return nil }
        var out: [[String: Any]] = []
        out.reserveCapacity(openai.count)
        for t in openai {
            guard let fn = t["function"] as? [String: Any],
                  let name = fn["name"] as? String
            else { continue }
            var row: [String: Any] = ["name": name]
            if let desc = fn["description"] as? String {
                row["description"] = desc
            }
            if let params = fn["parameters"] {
                row["input_schema"] = params
            }
            out.append(row)
        }
        return out.isEmpty ? nil : out
    }

    // MARK: - System block

    private static func composeSystem(history: [StoredMessage],
                                       agentSystem: String?) -> String
    {
        var pieces: [String] = []
        if let s = agentSystem?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !s.isEmpty
        {
            pieces.append(s)
        }
        for m in history where m.role == .system {
            let trimmed = m.content
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { pieces.append(trimmed) }
        }
        return pieces.joined(separator: "\n\n")
    }

    // MARK: - Messages array

    private static func composeMessages(history: [StoredMessage],
                                         cacheControl: Bool) -> [[String: Any]]
    {
        var out: [[String: Any]] = []

        // Pre-scan: find the index of the last user-turn that will
        // carry tool_result blocks. That's where we drop the last
        // cache_control breakpoint (per Anthropic's agent-loop
        // recipe).
        var lastToolResultUserMessageIdx: Int? = nil

        // Two-pass: first build the messages without cache markers,
        // tracking which output index ends up containing tool_result
        // blocks, then re-emit the chosen entry with the marker.
        //
        // We synthesize one user message per assistant turn that
        // has toolOutputs (with one tool_result block per output).

        for msg in history where msg.role != .system {
            switch msg.role {
            case .user:
                out.append([
                    "role": "user",
                    "content": msg.content,
                ])
            case .assistant:
                let blocks = assistantContentBlocks(msg)
                out.append([
                    "role": "assistant",
                    "content": blocks,
                ])
                // Emit a synthesised user message carrying tool_result
                // blocks if the assistant turn produced any outputs.
                if let outputs = msg.toolOutputs, !outputs.isEmpty {
                    let resultBlocks = toolResultBlocks(
                        from: msg, outputs: outputs)
                    if !resultBlocks.isEmpty {
                        out.append([
                            "role": "user",
                            "content": resultBlocks,
                        ])
                        lastToolResultUserMessageIdx = out.count - 1
                    }
                }
            case .system:
                continue
            }
        }

        // Drop the cache marker on the last tool_result block, if any.
        if cacheControl,
           let idx = lastToolResultUserMessageIdx,
           var entry = out[idx] as [String: Any]?,
           var content = entry["content"] as? [[String: Any]],
           !content.isEmpty
        {
            // Mark the LAST block in the list — that's the
            // "everything up to here" cache point.
            var lastBlock = content[content.count - 1]
            lastBlock["cache_control"] = ["type": "ephemeral"]
            content[content.count - 1] = lastBlock
            entry["content"] = content
            out[idx] = entry
        }

        return out
    }

    private static func assistantContentBlocks(_ msg: StoredMessage) -> [[String: Any]]
    {
        var blocks: [[String: Any]] = []
        let trimmed = msg.content
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            blocks.append(["type": "text", "text": msg.content])
        }
        for (i, tc) in msg.toolCalls.enumerated() {
            // Anthropic requires the same id on the request as the
            // one it emitted in a prior response. If the call was
            // originally minted by Anthropic it'll look like
            // `toolu_xxx`; if we synthesized one (mixed-provider
            // history) the API just treats it as an opaque string.
            let id = tc.id ?? "call_\(msg.id.uuidString)_\(i)"
            // `arguments` is a JSON string in the OpenAI shape;
            // Anthropic wants the parsed object under `input`.
            let input: Any = parseJSONObjectOrEmpty(tc.args)
            blocks.append([
                "type": "tool_use",
                "id": id,
                "name": tc.name,
                "input": input,
            ])
        }
        return blocks
    }

    private static func toolResultBlocks(from msg: StoredMessage,
                                          outputs: [String]) -> [[String: Any]]
    {
        var blocks: [[String: Any]] = []
        for (i, output) in outputs.enumerated() where i < msg.toolCalls.count {
            let tc = msg.toolCalls[i]
            let id = tc.id ?? "call_\(msg.id.uuidString)_\(i)"
            blocks.append([
                "type": "tool_result",
                "tool_use_id": id,
                "content": output,
            ])
        }
        return blocks
    }

    /// Parse `argsJSON` as a JSON object; fall back to an empty
    /// object on failure (Anthropic rejects `tool_use.input` if it
    /// isn't a JSON object, even when the tool nominally takes no
    /// args). Mirrors `MCPClientPool.invokeQualified`'s leniency.
    private static func parseJSONObjectOrEmpty(_ s: String) -> Any {
        let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let data = trimmed.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data),
              obj is [String: Any]
        else {
            return [String: Any]()
        }
        return obj
    }
}
