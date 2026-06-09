import Foundation

// Tool-calling (function calling) support for the DeepSeek-V4 chat protocol.
//
// Wire format is the AUTHORITATIVE one from the DeepSeek-V4 paper (Table 4): an
// XML-style scheme built on the special "｜DSML｜" token. A tool call looks like:
//
//   <｜DSML｜tool_calls>
//   <｜DSML｜invoke name="get_weather">
//   <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
//   <｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>
//   </｜DSML｜invoke>
//   </｜DSML｜tool_calls>
//
// String parameters use string="true" and the raw value; all other types
// (number/bool/array/object) use string="false" and a JSON-encoded value.
//
// This file is pure and model-independent (unit-tested without a GGUF). The only
// model token involved is ｜DSML｜ (already in the vocab as Tokenizer.dsmlId);
// ToolMarkup.discover confirms its exact spelling. NOTE: the paper specifies the
// CALL format precisely; the tool-RESULT block (renderToolResult) is a documented
// DSML-consistent extrapolation — verify it against the model's chat_template.

// MARK: - Value types

/// A tool the model may call: a name, a human description, and a JSON-Schema
/// object (as a JSON string) describing its parameters.
public struct ToolSpec: Sendable, Equatable, Identifiable {
    public var name: String
    public var description: String
    public var parametersJSON: String   // a JSON object, e.g. {"type":"object","properties":{…}}
    public var id: String { name }
    public init(name: String, description: String, parametersJSON: String = #"{"type":"object","properties":{}}"#) {
        self.name = name; self.description = description; self.parametersJSON = parametersJSON
    }
}

/// A single tool invocation. `argumentsJSON` is the call arguments as a JSON
/// object (parsed from the DSML parameters; used to execute the tool).
public struct ToolCall: Sendable, Equatable, Identifiable {
    public var id: String
    public var name: String
    public var argumentsJSON: String
    public init(id: String, name: String, argumentsJSON: String) {
        self.id = id; self.name = name; self.argumentsJSON = argumentsJSON
    }
}

/// One turn of a conversation, as the engine needs to render it.
public enum ChatTurn: Sendable, Equatable {
    case system(String)
    case user(String)
    case assistant(text: String, toolCalls: [ToolCall])
    case toolResult(callId: String, name: String, content: String)
}

// MARK: - DSML markup

/// The DeepSeek-V4 tool markup, built on the single special token `｜DSML｜`.
/// All tags are derived from it, so only that one token is model-specific.
public struct ToolMarkup: Sendable, Equatable {
    /// The DSML special token (vocab token, with fullwidth bars), e.g. "｜DSML｜".
    public var dsml: String
    public init(dsml: String) { self.dsml = dsml }

    /// DeepSeek-V4 default (the vocab token uses fullwidth U+FF5C bars; the paper
    /// writes |DSML| in ASCII for readability).
    public static let dsv4 = ToolMarkup(dsml: "｜DSML｜")

    public var callsOpen: String  { "<\(dsml)tool_calls>" }
    public var callsClose: String { "</\(dsml)tool_calls>" }
    public func invokeOpen(_ name: String) -> String { "<\(dsml)invoke name=\"\(name)\">" }
    public var invokeClose: String { "</\(dsml)invoke>" }
    public func paramOpen(_ name: String, string: Bool) -> String {
        "<\(dsml)parameter name=\"\(name)\" string=\"\(string)\">"
    }
    public var paramClose: String { "</\(dsml)parameter>" }

    /// Confirm the DSML token's exact spelling against the model vocab.
    public static func discover(in tokenizer: Tokenizer) -> ToolMarkup {
        for c in ["｜DSML｜", "|DSML|"] where tokenizer.tokenId(c) != nil { return ToolMarkup(dsml: c) }
        return dsv4
    }
}

// MARK: - Rendering

public enum ChatRenderer {
    /// The "## Tools" declaration block (DeepSeek-V4 paper, Table 4), appended to
    /// the system section when tools are present.
    public static func toolDeclarations(_ tools: [ToolSpec], markup m: ToolMarkup) -> String {
        guard !tools.isEmpty else { return "" }
        var s = """

        ## Tools

        You have access to a set of tools to help answer the user's question. You can \
        invoke tools by writing a "\(m.callsOpen)" block like the following:

        \(m.callsOpen)
        \(m.invokeOpen("$TOOL_NAME"))
        <\(m.dsml)parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</\(m.dsml)parameter>
        ...
        \(m.invokeClose)
        ...
        \(m.callsClose)

        String parameters should be specified as is and set `string="true"`. For all \
        other types (numbers, booleans, arrays, objects), pass the value in JSON format \
        and set `string="false"`.

        If thinking_mode is enabled (triggered by <think>), you MUST output your complete \
        reasoning inside <think>...</think> BEFORE any tool calls or final response.

        Otherwise, output directly after </think> with tool calls or final response.

        ### Available Tool Schemas


        """
        for t in tools {
            s += "{\"name\": \"\(t.name)\", \"description\": \(jsonString(t.description)), " +
                 "\"parameters\": \(t.parametersJSON)}\n"
        }
        s += "\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n"
        return s
    }

    /// Render the whole conversation to the rendered-chat string the tokenizer
    /// consumes. `think` controls the trailing reasoning marker on the open
    /// assistant turn. The first system turn (+ tool declarations) is emitted up
    /// front, then alternating user/assistant turns and tool results.
    public static func render(turns: [ChatTurn], tools: [ToolSpec], think: ThinkMode,
                              markup: ToolMarkup, bos: String = "<｜begin▁of▁sentence｜>",
                              eos: String = "<｜end▁of▁sentence｜>",
                              userTag: String = "<｜User｜>", assistantTag: String = "<｜Assistant｜>") -> String {
        var out = bos
        let systemText = turns.compactMap { if case let .system(s) = $0 { return s } else { return nil } }.first ?? ""
        let decls = toolDeclarations(tools, markup: markup)
        if !systemText.isEmpty || !decls.isEmpty { out += systemText + decls }

        for turn in turns {
            switch turn {
            case .system:
                continue
            case .user(let text):
                out += userTag + text
            case .assistant(let text, let calls):
                out += assistantTag + text
                if !calls.isEmpty { out += renderToolCalls(calls, markup: markup) }
                out += eos
            case .toolResult(_, let name, let content):
                out += renderToolResult(name: name, content: content, markup: markup)
            }
        }

        out += assistantTag
        out += think.enabled ? "<think>" : "</think>"
        return out
    }

    /// Render assistant-emitted tool calls back into the DSML block (for history).
    static func renderToolCalls(_ calls: [ToolCall], markup m: ToolMarkup) -> String {
        var s = "\n" + m.callsOpen + "\n"
        for c in calls {
            s += m.invokeOpen(c.name) + "\n"
            for p in jsonToParams(c.argumentsJSON) {
                s += m.paramOpen(p.name, string: p.isString) + p.value + m.paramClose + "\n"
            }
            s += m.invokeClose + "\n"
        }
        s += m.callsClose
        return s
    }

    /// Tool result block. NOTE: extrapolated (DSML-consistent) — the paper's
    /// Table 4 specifies the call format but not the result format; verify against
    /// the model's chat_template.
    static func renderToolResult(name: String, content: String, markup m: ToolMarkup) -> String {
        "\n<\(m.dsml)tool_outputs>\n<\(m.dsml)output name=\"\(name)\">\(content)</\(m.dsml)output>\n</\(m.dsml)tool_outputs>"
    }

    /// Decompose a JSON arguments object into DSML parameters (name, string-flag,
    /// rendered value). String values are emitted raw with string="true"; other
    /// types are JSON-encoded with string="false". Keys sorted for determinism.
    static func jsonToParams(_ json: String) -> [(name: String, isString: Bool, value: String)] {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [] }
        var out: [(name: String, isString: Bool, value: String)] = []
        for key in obj.keys.sorted() {
            let v = obj[key]!
            if let str = v as? String {
                out.append((key, true, str))
            } else {
                let d = (try? JSONSerialization.data(withJSONObject: v, options: [.fragmentsAllowed])) ?? Data("null".utf8)
                out.append((key, false, String(data: d, encoding: .utf8) ?? "null"))
            }
        }
        return out
    }
}

/// JSON-encode a string (with quotes/escapes), for the tool-schema block.
func jsonString(_ s: String) -> String {
    if let d = try? JSONSerialization.data(withJSONObject: s, options: [.fragmentsAllowed]),
       let str = String(data: d, encoding: .utf8) { return str }
    return "\"\(s)\""
}

// MARK: - Parsing tool calls from generated text

public enum ToolCallParser {
    /// Extract tool calls from a completed assistant message and return the visible
    /// text (the DSML tool-call block stripped). Parses the DSML XML scheme.
    public static func parse(_ text: String, markup m: ToolMarkup) -> (calls: [ToolCall], visibleText: String) {
        guard let start = text.range(of: m.callsOpen) else { return ([], text) }
        let visible = String(text[text.startIndex..<start.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        let afterOpen = start.upperBound
        let end = text.range(of: m.callsClose, range: afterOpen..<text.endIndex)?.lowerBound ?? text.endIndex
        let block = String(text[afterOpen..<end])

        var calls: [ToolCall] = []
        var idx = 0
        var search = block.startIndex
        let invokePrefix = "<\(m.dsml)invoke"
        while let io = block.range(of: invokePrefix, range: search..<block.endIndex) {
            let close = block.range(of: m.invokeClose, range: io.upperBound..<block.endIndex)?.lowerBound ?? block.endIndex
            let body = String(block[io.lowerBound..<close])
            if let call = parseInvoke(body, markup: m, index: idx) { calls.append(call); idx += 1 }
            search = close < block.endIndex ? block.index(after: close) : block.endIndex
            if search >= block.endIndex { break }
        }
        return (calls, visible)
    }

    /// Parse one `<DSML|invoke name="…"> …params… ` body into a ToolCall.
    static func parseInvoke(_ body: String, markup m: ToolMarkup, index: Int) -> ToolCall? {
        guard let name = attributeValue("name", in: body) else { return nil }
        var params: [(name: String, isString: Bool, value: String)] = []
        var search = body.startIndex
        let paramPrefix = "<\(m.dsml)parameter"
        while let po = body.range(of: paramPrefix, range: search..<body.endIndex) {
            guard let tagEnd = body.range(of: ">", range: po.upperBound..<body.endIndex) else { break }
            let tag = String(body[po.lowerBound..<tagEnd.upperBound])
            let pname = attributeValue("name", in: tag) ?? ""
            let isString = (attributeValue("string", in: tag) ?? "true") == "true"
            let valStart = tagEnd.upperBound
            let pclose = body.range(of: m.paramClose, range: valStart..<body.endIndex)?.lowerBound ?? body.endIndex
            let value = String(body[valStart..<pclose]).trimmingCharacters(in: .whitespacesAndNewlines)
            params.append((pname, isString, value))
            search = pclose < body.endIndex ? body.index(after: pclose) : body.endIndex
            if search >= body.endIndex { break }
        }
        return ToolCall(id: "call_\(index)", name: name, argumentsJSON: paramsToJSON(params))
    }

    /// Build a JSON arguments object from parsed DSML parameters.
    static func paramsToJSON(_ params: [(name: String, isString: Bool, value: String)]) -> String {
        var parts: [String] = []
        for p in params {
            let key = jsonString(p.name)
            let val: String
            if p.isString {
                val = jsonString(p.value)
            } else if let d = p.value.data(using: .utf8),
                      (try? JSONSerialization.jsonObject(with: d, options: [.fragmentsAllowed])) != nil {
                val = p.value                    // already valid JSON
            } else {
                val = jsonString(p.value)        // fallback: treat as a string
            }
            parts.append("\(key):\(val)")
        }
        return "{" + parts.joined(separator: ",") + "}"
    }

    /// Extract `attr="value"` from an XML-ish tag/body.
    static func attributeValue(_ attr: String, in s: String) -> String? {
        guard let r = s.range(of: "\(attr)=\"") else { return nil }
        guard let close = s.range(of: "\"", range: r.upperBound..<s.endIndex) else { return nil }
        return String(s[r.upperBound..<close.lowerBound])
    }
}
