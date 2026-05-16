import Foundation

/// Port of `Reference/encoding/encoding_dsv4.py`. Covers the realistic
/// chat-with-tools workflow plus the three thinking-effort modes; the
/// less-used surface (task tokens, response_format injection,
/// latest_reminder) is left for the caller to prepend manually since it
/// boils down to literal string concatenation.
public enum EncodingDSV4 {
    public static let bosToken = "<｜begin▁of▁sentence｜>"
    public static let eosToken = "<｜end▁of▁sentence｜>"
    public static let userToken = "<｜User｜>"
    public static let assistantToken = "<｜Assistant｜>"
    public static let thinkOpen = "<think>"
    public static let thinkClose = "</think>"
    public static let dsmlToken = "｜DSML｜"

    // Project / repo context delimiters. Real added_tokens in the V4
    // vocab (128815-820 in the reference checkpoint); the BPE
    // pre-splits on them so each emits exactly one id regardless of
    // surrounding bytes. Used by the "project attached to chat" flow
    // to inject a codebase as native structured context — see
    // `InferenceService.tokenizeProjectContext`.
    public static let beginOfRepoName = "<｜begin▁of▁repo▁name｜>"
    public static let endOfRepoName   = "<｜end▁of▁repo▁name｜>"
    public static let beginOfFileName = "<｜begin▁of▁file▁name｜>"
    public static let endOfFileName   = "<｜end▁of▁file▁name｜>"
    public static let beginOfFile     = "<｜begin▁of▁file｜>"
    public static let endOfFile       = "<｜end▁of▁file｜>"

    // Native tool-output delimiters (V4 added_tokens 128810-814).
    // Used to splice the host-executed tool results back into the
    // prompt right after the assistant turn that emitted the calls.
    public static let beginToolOutputs = "<｜tool▁outputs▁begin｜>"
    public static let endToolOutputs   = "<｜tool▁outputs▁end｜>"
    public static let beginToolOutput  = "<｜tool▁output▁begin｜>"
    public static let endToolOutput    = "<｜tool▁output▁end｜>"
    public static let toolSep          = "<｜tool▁sep｜>"

    /// Name of the synthetic "delegate to another agent" tool the
    /// host injects when more than one Agent is registered. Lives
    /// in the same namespace as the MCP qualified names
    /// ("server__tool"), with double leading underscores so it
    /// can't collide with a real MCP server called "delegate".
    public static let delegateToolName = "__delegate_to_agent"

    /// Reasoning instruction prepended to the system message in `.max` mode.
    /// Mirrors REASONING_EFFORT_MAX (encoding_dsv4.py:64-67).
    public static let reasoningEffortMax: String = """
        Reasoning Effort: Absolute maximum with no shortcuts permitted.
        You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.
        Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.


        """

    /// Tools section block, prepended to the system message when the request
    /// includes tool schemas. Mirrors TOOLS_TEMPLATE (encoding_dsv4.py:70-95).
    public static func toolsBlock(toolSchemasJSON: String) -> String {
        let dt = dsmlToken
        return """
            ## Tools

            You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<\(dt)tool_calls>" block like the following:

            <\(dt)tool_calls>
            <\(dt)invoke name="$TOOL_NAME">
            <\(dt)parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</\(dt)parameter>
            ...
            </\(dt)invoke>
            </\(dt)tool_calls>

            String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

            If thinking_mode is enabled (triggered by \(thinkOpen)), you MUST output your complete reasoning inside \(thinkOpen)...\(thinkClose) BEFORE any tool calls or final response.

            Otherwise, output directly after \(thinkClose) with tool calls or final response.

            ### Available Tool Schemas

            \(toolSchemasJSON)

            You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.

            """
    }

    // MARK: - Encode

    /// Encode a list of messages into the prompt string the model expects.
    /// Tool schemas, when provided, are prepended to (or merged into) the
    /// first system message. Reasoning effort `.max` prepends the
    /// REASONING_EFFORT_MAX block.
    public static func encodeMessages(_ messages: [Message],
                                       mode: ThinkingMode = .chat,
                                       toolSchemasJSON: String? = nil) -> String {
        var msgs = messages
        injectSystemAdditions(into: &msgs, mode: mode, toolSchemasJSON: toolSchemasJSON)

        var out = bosToken
        for msg in msgs {
            switch msg.role {
            case .system:
                out += msg.content
            case .user:
                out += userToken + msg.content
            case .assistant:
                out += assistantToken
                if let r = msg.reasoningContent, !r.isEmpty, mode != .chat {
                    out += thinkOpen + r + thinkClose
                }
                out += msg.content
                if !msg.toolCalls.isEmpty {
                    out += encodeToolCalls(msg.toolCalls)
                }
                out += eosToken
                if !msg.toolOutputs.isEmpty {
                    out += encodeToolOutputs(
                        callNames: msg.toolCalls.map(\.name),
                        outputs: msg.toolOutputs)
                }
            }
        }
        // Trailing assistant turn (the one the model fills in). Mirrors
        // Reference/encoding/encoding_dsv4.py:render_message lines
        // 250-258: after every user message we emit ASSISTANT_SP_TOKEN
        // and then either thinking_start_token (<think>) when the model
        // is asked to think, or thinking_end_token (</think>) for plain
        // chat (signals "no thinking, here's the response"). The earlier
        // Swift impl emitted only the assistant token, which left the
        // model in an undefined state (it expected one of the two think
        // markers) and produced incoherent first tokens.
        out += assistantToken
        switch mode {
        case .chat:
            out += thinkClose
        case .high, .max:
            out += thinkOpen
        }
        return out
    }

    private static func injectSystemAdditions(into msgs: inout [Message],
                                                mode: ThinkingMode,
                                                toolSchemasJSON: String?) {
        var prepend = ""
        if mode == .max {
            prepend += reasoningEffortMax
        }
        if let schemas = toolSchemasJSON, !schemas.isEmpty {
            prepend += toolsBlock(toolSchemasJSON: schemas)
        }
        guard !prepend.isEmpty else { return }
        if let firstSystemIdx = msgs.firstIndex(where: { $0.role == .system }) {
            msgs[firstSystemIdx].content = prepend + msgs[firstSystemIdx].content
        } else {
            msgs.insert(Message(role: .system, content: prepend), at: 0)
        }
    }

    /// Encode a list of host-executed tool results as the native
    /// `<｜tool▁outputs▁begin｜>…<｜tool▁outputs▁end｜>` block. Each
    /// result is wrapped in `<｜tool▁output▁begin｜>` / `<｜tool▁output▁end｜>`,
    /// optionally prefixed with `name<｜tool▁sep｜>` so the model can
    /// match each result back to the call by index. `callNames` may
    /// be shorter than `outputs` (e.g. when reconstructing from a
    /// raw log) — in that case the missing names are simply omitted.
    public static func encodeToolOutputs(callNames: [String],
                                          outputs: [String]) -> String {
        var out = beginToolOutputs
        for (idx, body) in outputs.enumerated() {
            out += beginToolOutput
            if idx < callNames.count {
                out += callNames[idx] + toolSep
            }
            out += body
            out += endToolOutput
        }
        out += endToolOutputs
        return out
    }

    /// Encode an array of tool calls as a single DSML `<｜DSML｜tool_calls>`
    /// block. Mirrors `tool_calls_template` + `tool_call_template` +
    /// `encode_arguments_to_dsml` (encoding_dsv4.py:52-58, 139-180).
    public static func encodeToolCalls(_ calls: [ToolCall]) -> String {
        let dt = dsmlToken
        var inner = ""
        for tc in calls {
            inner += "<\(dt)invoke name=\"\(escapeXMLAttr(tc.name))\">\n"
            inner += encodeArguments(tc.args)
            inner += "</\(dt)invoke>\n"
        }
        // Drop the trailing newline before the close to match the reference.
        if inner.hasSuffix("\n") { inner.removeLast() }
        return "<\(dt)tool_calls>\n\(inner)\n</\(dt)tool_calls>"
    }

    private static func encodeArguments(_ argsJSON: String) -> String {
        let dt = dsmlToken
        // Try to parse the args JSON into a dict. If it fails, fall through
        // and emit a single `arguments` parameter holding the raw string.
        guard let data = argsJSON.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return "<\(dt)parameter name=\"arguments\" string=\"true\">\(escapeXML(argsJSON))</\(dt)parameter>\n"
        }

        var out = ""
        for (k, v) in obj {
            let isString = v is String
            let value = isString
                ? escapeXML(v as! String)
                : (try? String(data: JSONSerialization.data(withJSONObject: v), encoding: .utf8)) ?? "\(v)"
            out += "<\(dt)parameter name=\"\(escapeXMLAttr(k))\" string=\"\(isString ? "true" : "false")\">\(value)</\(dt)parameter>\n"
        }
        return out
    }

    // MARK: - Parse

    /// Parse a model completion back into a Message.
    /// Strips trailing EOS, extracts an optional `<think>...</think>`
    /// block, and decodes any `<｜DSML｜tool_calls>` invocations.
    public static func parseCompletion(_ text: String,
                                        mode: ThinkingMode = .chat) -> Message {
        var work = text
        if work.hasSuffix(eosToken) {
            work = String(work.dropLast(eosToken.count))
        }

        var reasoning: String? = nil
        if let openRange = work.range(of: thinkOpen),
           let closeRange = work.range(of: thinkClose, range: openRange.upperBound..<work.endIndex) {
            reasoning = String(work[openRange.upperBound..<closeRange.lowerBound])
            work.removeSubrange(openRange.lowerBound..<closeRange.upperBound)
        }

        var toolCalls: [ToolCall] = []
        let openTag = "<\(dsmlToken)tool_calls>"
        let closeTag = "</\(dsmlToken)tool_calls>"
        if let openRange = work.range(of: openTag),
           let closeRange = work.range(of: closeTag, range: openRange.upperBound..<work.endIndex) {
            let block = String(work[openRange.upperBound..<closeRange.lowerBound])
            toolCalls = parseToolCalls(block)
            work.removeSubrange(openRange.lowerBound..<closeRange.upperBound)
        }

        return Message(role: .assistant,
                       content: work.trimmingCharacters(in: .whitespacesAndNewlines),
                       reasoningContent: reasoning,
                       toolCalls: toolCalls)
    }

    /// Parse `<｜DSML｜invoke>...</｜DSML｜invoke>` blocks inside the body of
    /// a tool_calls section. Each invoke produces one ToolCall whose
    /// `args` is a JSON dict reconstructed from the inner parameters.
    private static func parseToolCalls(_ body: String) -> [ToolCall] {
        let dt = dsmlToken
        var calls: [ToolCall] = []
        var cursor = body.startIndex
        let invokeOpenPrefix = "<\(dt)invoke name=\""
        let invokeClose = "</\(dt)invoke>"

        while let openStart = body.range(of: invokeOpenPrefix,
                                          range: cursor..<body.endIndex) {
            // Find the closing quote for the name.
            guard let nameClose = body.range(of: "\">",
                                              range: openStart.upperBound..<body.endIndex),
                  let invokeEnd = body.range(of: invokeClose,
                                              range: nameClose.upperBound..<body.endIndex)
            else { break }
            let name = String(body[openStart.upperBound..<nameClose.lowerBound])
            let inner = String(body[nameClose.upperBound..<invokeEnd.lowerBound])
            let args = parseParameters(inner)
            calls.append(ToolCall(name: name, args: args))
            cursor = invokeEnd.upperBound
        }
        return calls
    }

    private static func parseParameters(_ s: String) -> String {
        let dt = dsmlToken
        var dict: [String: Any] = [:]
        var cursor = s.startIndex
        let paramPrefix = "<\(dt)parameter name=\""
        let paramClose = "</\(dt)parameter>"

        while let nameStart = s.range(of: paramPrefix, range: cursor..<s.endIndex),
              let nameEnd = s.range(of: "\"",
                                     range: nameStart.upperBound..<s.endIndex),
              let stringAttrStart = s.range(of: "string=\"",
                                              range: nameEnd.upperBound..<s.endIndex),
              let stringAttrEnd = s.range(of: "\">",
                                           range: stringAttrStart.upperBound..<s.endIndex),
              let valueEnd = s.range(of: paramClose,
                                      range: stringAttrEnd.upperBound..<s.endIndex) {
            let key = String(s[nameStart.upperBound..<nameEnd.lowerBound])
            let isString = s[stringAttrStart.upperBound..<stringAttrEnd.lowerBound] == "true"
            let raw = String(s[stringAttrEnd.upperBound..<valueEnd.lowerBound])
            if isString {
                dict[key] = unescapeXML(raw)
            } else if let data = raw.data(using: .utf8),
                      let parsed = try? JSONSerialization.jsonObject(with: data) {
                dict[key] = parsed
            } else {
                dict[key] = raw
            }
            cursor = valueEnd.upperBound
        }
        guard let data = try? JSONSerialization.data(withJSONObject: dict),
              let str = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return str
    }

    // MARK: - XML escaping

    private static func escapeXML(_ s: String) -> String {
        var r = s
        r = r.replacingOccurrences(of: "&", with: "&amp;")
        r = r.replacingOccurrences(of: "<", with: "&lt;")
        r = r.replacingOccurrences(of: ">", with: "&gt;")
        return r
    }

    private static func escapeXMLAttr(_ s: String) -> String {
        var r = escapeXML(s)
        r = r.replacingOccurrences(of: "\"", with: "&quot;")
        return r
    }

    private static func unescapeXML(_ s: String) -> String {
        var r = s
        r = r.replacingOccurrences(of: "&quot;", with: "\"")
        r = r.replacingOccurrences(of: "&gt;", with: ">")
        r = r.replacingOccurrences(of: "&lt;", with: "<")
        r = r.replacingOccurrences(of: "&amp;", with: "&")
        return r
    }
}
