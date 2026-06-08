import Foundation

// Tool-calling (function calling) support for the DeepSeek-V4 chat protocol.
//
// Scope of this file (pure, model-independent logic — unit-tested without a GGUF):
//   • value types for tools, tool calls, and conversation turns;
//   • `ToolMarkup`: the model-specific special-token strings used to wrap tool
//     calls / outputs, DISCOVERED from the model vocab at runtime (honoring the
//     "extract from the GGUF" choice) with a documented DeepSeek-family default;
//   • `ChatRenderer`: renders a multi-turn conversation (+ declared tools) to the
//     rendered-chat string the tokenizer consumes (`tokenizeRenderedChat`);
//   • `ToolCallParser`: extracts tool calls from generated assistant text.
//
// NOTE ON FIDELITY: the authoritative wire format lives in the GGUF's
// `tokenizer.chat_template` (Jinja). We do not execute Jinja; instead we extract
// the *actual* markup tokens the model defines (ToolMarkup.discover) and render
// with the DeepSeek-family structure. If a given DS4 build diverges, the markup
// strings and the two functions below are the single place to adjust. Verify the
// rendered prompt against `model.string("tokenizer.chat_template")` on a machine
// with the real model before relying on tool calls.

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

/// A single tool invocation emitted by the model.
public struct ToolCall: Sendable, Equatable, Identifiable {
    public var id: String
    public var name: String
    public var argumentsJSON: String   // the raw JSON arguments object the model produced
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

// MARK: - Model-specific markup tokens

/// The special-token strings that wrap tool declarations, calls, and outputs.
/// The defaults are the DeepSeek-family tokens; `discover` replaces any that the
/// model's own vocab actually defines (so we use the real tokens, not guesses).
public struct ToolMarkup: Sendable, Equatable {
    public var callsBegin: String
    public var callsEnd: String
    public var callBegin: String
    public var callEnd: String
    public var sep: String
    public var outputsBegin: String
    public var outputsEnd: String
    public var outputBegin: String
    public var outputEnd: String

    public static let deepSeekDefault = ToolMarkup(
        callsBegin: "<｜tool▁calls▁begin｜>", callsEnd: "<｜tool▁calls▁end｜>",
        callBegin: "<｜tool▁call▁begin｜>", callEnd: "<｜tool▁call▁end｜>",
        sep: "<｜tool▁sep｜>",
        outputsBegin: "<｜tool▁outputs▁begin｜>", outputsEnd: "<｜tool▁outputs▁end｜>",
        outputBegin: "<｜tool▁output▁begin｜>", outputEnd: "<｜tool▁output▁end｜>")

    public init(callsBegin: String, callsEnd: String, callBegin: String, callEnd: String,
                sep: String, outputsBegin: String, outputsEnd: String,
                outputBegin: String, outputEnd: String) {
        self.callsBegin = callsBegin; self.callsEnd = callsEnd
        self.callBegin = callBegin; self.callEnd = callEnd; self.sep = sep
        self.outputsBegin = outputsBegin; self.outputsEnd = outputsEnd
        self.outputBegin = outputBegin; self.outputEnd = outputEnd
    }

    /// Keep only the markup tokens the model's vocab actually defines; fall back
    /// to the default string for any that are absent. This is the "extract from
    /// the GGUF" step: the wire tokens come from the model, not a hardcoded guess.
    public static func discover(in tokenizer: Tokenizer) -> ToolMarkup {
        var m = deepSeekDefault
        // Try a few known alternative spellings for each role; use whichever the
        // vocab defines. (All DeepSeek variants seen use the ▁-joined names.)
        func first(_ candidates: [String], _ fallback: String) -> String {
            for c in candidates where tokenizer.tokenId(c) != nil { return c }
            return fallback
        }
        m.callsBegin = first(["<｜tool▁calls▁begin｜>", "<｜tool_calls_begin｜>"], m.callsBegin)
        m.callsEnd   = first(["<｜tool▁calls▁end｜>", "<｜tool_calls_end｜>"], m.callsEnd)
        m.callBegin  = first(["<｜tool▁call▁begin｜>", "<｜tool_call_begin｜>"], m.callBegin)
        m.callEnd    = first(["<｜tool▁call▁end｜>", "<｜tool_call_end｜>"], m.callEnd)
        m.sep        = first(["<｜tool▁sep｜>", "<｜tool_sep｜>"], m.sep)
        m.outputsBegin = first(["<｜tool▁outputs▁begin｜>", "<｜tool_outputs_begin｜>"], m.outputsBegin)
        m.outputsEnd   = first(["<｜tool▁outputs▁end｜>", "<｜tool_outputs_end｜>"], m.outputsEnd)
        m.outputBegin  = first(["<｜tool▁output▁begin｜>", "<｜tool_output_begin｜>"], m.outputBegin)
        m.outputEnd    = first(["<｜tool▁output▁end｜>", "<｜tool_output_end｜>"], m.outputEnd)
        return m
    }
}

// MARK: - Rendering

/// Renders a conversation (+ optional tools) to the rendered-chat string the
/// tokenizer consumes. Mirrors the DeepSeek chat structure: a single system
/// section (with the tool declarations appended when tools are present), then
/// alternating user / assistant turns, with tool calls and tool outputs wrapped
/// in the model markup. Ends with the assistant marker (+ <think>/</think>).
public enum ChatRenderer {
    /// Build the tool-declaration block appended to the system message.
    public static func toolDeclarations(_ tools: [ToolSpec]) -> String {
        guard !tools.isEmpty else { return "" }
        var s = "\n\n## Tools\nYou have access to the following tools. " +
                "When you decide to call a tool, emit the tool-call markup; otherwise answer normally.\n"
        for t in tools {
            s += "\n### \(t.name)\n\(t.description)\nParameters (JSON Schema): \(t.parametersJSON)\n"
        }
        return s
    }

    /// Render the whole conversation to a string. `think` controls the trailing
    /// reasoning marker on the open assistant turn; `markup` carries the model's
    /// tool tokens. The returned string is fed to `tokenizeRenderedChat`.
    public static func render(turns: [ChatTurn], tools: [ToolSpec], think: ThinkMode,
                              markup: ToolMarkup, bos: String = "<｜begin▁of▁sentence｜>",
                              eos: String = "<｜end▁of▁sentence｜>",
                              userTag: String = "<｜User｜>", assistantTag: String = "<｜Assistant｜>") -> String {
        var out = bos

        // Collect the system text (only the first system turn is used, like the
        // C template) and append tool declarations once, up front.
        let systemText = turns.compactMap { if case let .system(s) = $0 { return s } else { return nil } }.first ?? ""
        let decls = toolDeclarations(tools)
        if !systemText.isEmpty || !decls.isEmpty { out += systemText + decls }

        for turn in turns {
            switch turn {
            case .system:
                continue   // already emitted above
            case .user(let text):
                out += userTag + text
            case .assistant(let text, let calls):
                out += assistantTag + text
                if !calls.isEmpty { out += renderToolCalls(calls, markup: markup) }
                out += eos
            case .toolResult(_, let name, let content):
                out += markup.outputsBegin + markup.outputBegin + name + "\n" + content + markup.outputEnd + markup.outputsEnd
            }
        }

        // Open the assistant turn for generation.
        out += assistantTag
        out += think.enabled ? "<think>" : "</think>"
        return out
    }

    static func renderToolCalls(_ calls: [ToolCall], markup: ToolMarkup) -> String {
        var s = markup.callsBegin
        for c in calls {
            s += markup.callBegin + "function" + markup.sep + c.name + "\n```json\n" + c.argumentsJSON + "\n```" + markup.callEnd
        }
        s += markup.callsEnd
        return s
    }
}

// MARK: - Parsing tool calls from generated text

public enum ToolCallParser {
    /// Extract tool calls from a completed assistant message. Returns the calls
    /// and the visible text with the tool-call block stripped. Tolerant of the
    /// optional ```json fences DeepSeek emits around the arguments.
    public static func parse(_ text: String, markup: ToolMarkup) -> (calls: [ToolCall], visibleText: String) {
        guard let blockStart = text.range(of: markup.callsBegin) else { return ([], text) }
        let visible = String(text[text.startIndex..<blockStart.lowerBound])
        // The block runs to callsEnd if present, else to the end of the string.
        let afterBegin = blockStart.upperBound
        let blockEnd = text.range(of: markup.callsEnd, range: afterBegin..<text.endIndex)?.lowerBound ?? text.endIndex
        let block = String(text[afterBegin..<blockEnd])

        var calls: [ToolCall] = []
        var idx = 0
        var search = block.startIndex
        while let cb = block.range(of: markup.callBegin, range: search..<block.endIndex) {
            let ce = block.range(of: markup.callEnd, range: cb.upperBound..<block.endIndex)?.lowerBound ?? block.endIndex
            let body = String(block[cb.upperBound..<ce])
            if let call = parseOneCall(body, markup: markup, index: idx) { calls.append(call); idx += 1 }
            search = (ce < block.endIndex) ? block.index(after: ce) : block.endIndex
            if search >= block.endIndex { break }
        }
        return (calls, visible.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// One call body looks like: `function<sep>name\n```json\n{…}\n```` (the
    /// leading "function" type and the fences are optional / tolerated).
    static func parseOneCall(_ body: String, markup: ToolMarkup, index: Int) -> ToolCall? {
        var rest = body
        if let sep = rest.range(of: markup.sep) { rest = String(rest[sep.upperBound...]) }
        rest = rest.trimmingCharacters(in: .whitespacesAndNewlines)
        // name = up to the first newline (or the start of a ``` fence / { ).
        let nameEnd = rest.firstIndex(where: { $0 == "\n" || $0 == "{" }) ?? rest.endIndex
        let name = String(rest[rest.startIndex..<nameEnd]).trimmingCharacters(in: .whitespacesAndNewlines)
        if name.isEmpty { return nil }
        var args = String(rest[nameEnd...])
        args = stripJSONFence(args)
        if args.isEmpty { args = "{}" }
        return ToolCall(id: "call_\(index)", name: name, argumentsJSON: args)
    }

    /// Strip an optional ```json … ``` fence (or plain ``` …  ```), returning the
    /// inner JSON, else the trimmed input.
    static func stripJSONFence(_ s: String) -> String {
        var t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if let open = t.range(of: "```") {
            // drop everything up to and including the opening fence line
            var inner = String(t[open.upperBound...])
            if let nl = inner.firstIndex(of: "\n") { inner = String(inner[inner.index(after: nl)...]) }
            if let close = inner.range(of: "```") { inner = String(inner[inner.startIndex..<close.lowerBound]) }
            t = inner.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return t
    }
}
