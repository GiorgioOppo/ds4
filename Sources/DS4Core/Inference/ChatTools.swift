import Foundation

// Tool-calling (function calling) for the DeepSeek-V4 chat protocol.
//
// The renderer mirrors the model's actual `tokenizer.chat_template` (verified
// against the GGUF): an XML-style scheme on the ｜DSML｜ token. Key structure:
//   • tools are declared in a "## Tools" system block (schemas via JSON);
//   • a tool call is  <｜DSML｜tool_calls> <｜DSML｜invoke name="…"> <｜DSML｜parameter
//     name="…" string="true|false">VALUE</｜DSML｜parameter> … </｜DSML｜invoke> … </｜DSML｜tool_calls>;
//   • a tool RESULT is rendered inside a user turn as  <｜User｜><tool_result>…</tool_result>
//     (consecutive results don't repeat <｜User｜>);
//   • every assistant turn opens <｜Assistant｜> then </think> (or <think>… for a
//     reasoning turn), and closes with <｜end▁of▁sentence｜>.
// Pure and model-independent (unit-tested without a GGUF).

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
public struct ToolMarkup: Sendable, Equatable {
    /// The DSML special token (vocab token, with fullwidth bars), e.g. "｜DSML｜".
    public var dsml: String
    public init(dsml: String) { self.dsml = dsml }

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

// MARK: - Rendering (mirrors tokenizer.chat_template)

public enum ChatRenderer {
    /// The "## Tools" header text (verbatim from the model template), ending right
    /// before "### Available Tool Schemas\n\n".
    static func toolsHeader(_ m: ToolMarkup) -> String {
        let d = m.dsml
        return """
        ## Tools

        You have access to a set of tools to help answer the user question. You can invoke tools by writing a "<\(d)tool_calls>" block like the following:

        <\(d)tool_calls>
        <\(d)invoke name="$TOOL_NAME">
        <\(d)parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</\(d)parameter>
        ...
        </\(d)invoke>
        <\(d)invoke name="$TOOL_NAME2">
        ...
        </\(d)invoke>
        </\(d)tool_calls>

        String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

        If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

        Otherwise, output directly after </think> with tool calls or final response.

        ### Available Tool Schemas
        """
    }

    /// Compact tool declaration for local inference: just `name(params)` per tool
    /// plus a one-line reminder of the call format. Far fewer prefill tokens than
    /// the full block — at some risk to call reliability (it deviates from the
    /// trained "## Tools" text). The format line is kept because without ANY
    /// format hint the model won't emit the DSML markup we can parse.
    static func compactToolsDeclaration(_ tools: [ToolSpec], markup m: ToolMarkup) -> String {
        let d = m.dsml
        var s = "## Tools\n"
        for t in tools {
            s += "- \(t.name)(\(paramNames(t.parametersJSON).joined(separator: ", ")))\n"
        }
        // Single, prose-free format line (full open/close nesting once) so the model
        // still emits parsable DSML. This is the irreducible floor for working calls.
        s += "Call: <\(d)tool_calls><\(d)invoke name=\"NAME\">" +
             "<\(d)parameter name=\"P\" string=\"true|false\">V</\(d)parameter>" +
             "</\(d)invoke></\(d)tool_calls>\n"
        return s
    }

    /// Parameter names from a JSON-Schema object's "properties".
    static func paramNames(_ json: String) -> [String] {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let props = obj["properties"] as? [String: Any] else { return [] }
        return props.keys.sorted()
    }

    /// Build the system block: collected system prompts + (if tools) the tools
    /// declaration. `compact` selects the minimal name-list form. Mirrors the template.
    static func systemBlock(turns: [ChatTurn], tools: [ToolSpec], markup m: ToolMarkup,
                            compact: Bool) -> String {
        var system = ""
        var first = true
        for case let .system(s) in turns {
            system += first ? s : "\n\n" + s
            first = false
        }
        guard !tools.isEmpty else { return system }
        let toolsDecl: String
        if compact {
            toolsDecl = compactToolsDeclaration(tools, markup: m)
        } else {
            var schemas = ""
            for t in tools { schemas += functionJSON(t) + "\n" }
            toolsDecl = toolsHeader(m) + "\n\n" + schemas +
                "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls."
        }
        return system.isEmpty ? toolsDecl : system + "\n\n" + toolsDecl
    }

    /// Render the whole conversation to the rendered-chat string the tokenizer
    /// consumes. `think` controls the trailing reasoning marker; `compactTools`
    /// uses the minimal tool declaration; `addGenerationPrompt` opens an assistant turn.
    public static func render(turns: [ChatTurn], tools: [ToolSpec], think: ThinkMode,
                              markup: ToolMarkup, compactTools: Bool = false,
                              addGenerationPrompt: Bool = true,
                              bos: String = "<｜begin▁of▁sentence｜>", eos: String = "<｜end▁of▁sentence｜>",
                              userTag: String = "<｜User｜>", assistantTag: String = "<｜Assistant｜>",
                              thinkOpen: String = "<think>", thinkClose: String = "</think>") -> String {
        var out = bos + systemBlock(turns: turns, tools: tools, markup: markup, compact: compactTools)

        var pendingAssistant = false
        var pendingToolResult = false
        for turn in turns {
            switch turn {
            case .system:
                continue
            case .user(let text):
                out += userTag + text
                pendingAssistant = true; pendingToolResult = false
            case .toolResult(_, _, let content):
                if !pendingToolResult { out += userTag }
                out += "<tool_result>" + content + "</tool_result>"
                pendingAssistant = true; pendingToolResult = true
            case .assistant(let text, let calls):
                if pendingAssistant {
                    out += assistantTag + thinkClose   // past turns: reasoning discarded
                }
                out += text
                if !calls.isEmpty { out += renderToolCalls(calls, markup: markup) }
                out += eos
                pendingAssistant = false; pendingToolResult = false
            }
        }

        if addGenerationPrompt && pendingAssistant {
            out += assistantTag + (think.enabled ? thinkOpen : thinkClose)
        }
        return out
    }

    /// Render assistant-emitted tool calls into the DSML block (for history).
    static func renderToolCalls(_ calls: [ToolCall], markup m: ToolMarkup) -> String {
        var s = "\n\n" + m.callsOpen + "\n"
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

    /// The function schema (JSON, sorted keys ≈ Jinja `tojson`) for one tool.
    static func functionJSON(_ t: ToolSpec) -> String {
        let params = (try? JSONSerialization.jsonObject(with: Data(t.parametersJSON.utf8))) ?? [String: Any]()
        let fn: [String: Any] = ["name": t.name, "description": t.description, "parameters": params]
        if let d = try? JSONSerialization.data(withJSONObject: fn, options: [.sortedKeys, .withoutEscapingSlashes]),
           let s = String(data: d, encoding: .utf8) { return s }
        return "{\"name\":\(jsonString(t.name))}"
    }

    /// Decompose a JSON arguments object into DSML parameters (name, string-flag,
    /// rendered value). String values are raw with string="true"; other types are
    /// JSON-encoded with string="false". Keys sorted for determinism.
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

/// JSON-encode a string (with quotes/escapes).
func jsonString(_ s: String) -> String {
    if let d = try? JSONSerialization.data(withJSONObject: s, options: [.fragmentsAllowed]),
       let str = String(data: d, encoding: .utf8) { return str }
    return "\"\(s)\""
}

// MARK: - Parsing tool calls from generated text

public enum ToolCallParser {
    /// Extract tool calls from a completed assistant message and return the visible
    /// text (the DSML tool-call block stripped).
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

    /// Parse one `<DSML|invoke name="…"> …params…` body into a ToolCall.
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
                val = p.value
            } else {
                val = jsonString(p.value)
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
