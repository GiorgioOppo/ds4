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

