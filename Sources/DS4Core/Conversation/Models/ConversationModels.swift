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

    /// Stable identity for loop/duplicate detection. JSON object keys are sorted
    /// so formatting or key order alone cannot bypass the repeated-call guard.
    public var fingerprint: String {
        let canonical: String
        if let data = argumentsJSON.data(using: .utf8),
           let value = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]),
           JSONSerialization.isValidJSONObject(value),
           let normalized = try? JSONSerialization.data(withJSONObject: value,
                                                        options: [.sortedKeys, .withoutEscapingSlashes]),
           let text = String(data: normalized, encoding: .utf8) {
            canonical = text
        } else {
            canonical = argumentsJSON.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return name + "\u{0}" + canonical
    }
}

/// One turn of a conversation, as the engine needs to render it.
public enum ChatTurn: Sendable, Equatable {
    case system(String)
    case user(String)
    case assistant(text: String, toolCalls: [ToolCall])
    case toolResult(callId: String, name: String, content: String)
}

/// Structural validation for client-supplied tool histories.  The server APIs
/// accept uncapped message arrays, so validation must stay O(number of turns +
/// calls): build the set of preceding call ids while scanning forward instead
/// of searching the whole prefix for every tool result.
public enum ToolHistoryValidator {
    public enum ValidationError: Error, Equatable, CustomStringConvertible {
        case duplicateCallID(String)
        case unknownResultID(String)
        case duplicateResultID(String)

        public var description: String {
            switch self {
            case .duplicateCallID(let id):
                return "duplicate tool call id \(id)"
            case .unknownResultID(let id):
                return "tool result \(id) has no preceding assistant call"
            case .duplicateResultID(let id):
                return "tool call \(id) has more than one result"
            }
        }
    }

    /// Empty ids are retained for legacy `role=function` compatibility: those
    /// messages are name-addressed and cannot participate in id validation.
    public static func validate(_ turns: [ChatTurn]) throws {
        var declared = Set<String>()
        var resolved = Set<String>()
        declared.reserveCapacity(turns.count)
        resolved.reserveCapacity(turns.count)

        for turn in turns {
            switch turn {
            case .assistant(_, let calls):
                for call in calls where !call.id.isEmpty {
                    guard declared.insert(call.id).inserted else {
                        throw ValidationError.duplicateCallID(call.id)
                    }
                }
            case .toolResult(let callID, _, _):
                guard !callID.isEmpty else { continue }
                guard declared.contains(callID) else {
                    throw ValidationError.unknownResultID(callID)
                }
                guard resolved.insert(callID).inserted else {
                    throw ValidationError.duplicateResultID(callID)
                }
            case .system, .user:
                continue
            }
        }
    }
}
