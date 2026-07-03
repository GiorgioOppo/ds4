import Foundation
import DS4Core

// Tool registry for the GUI: built-in, auto-executable tools plus the plumbing to
// run a model-emitted ToolCall. Each tool lives in its OWN file under Builtins/
// (an extension of ToolRegistry); this file holds only the registry surface and
// the argument-parsing helpers shared by the tool files.

/// A tool the app can execute itself. `run` receives the model's raw JSON
/// arguments and returns a JSON (or plain text) result string.
public struct BuiltinTool: Sendable {
    public let spec: ToolSpec
    public let run: @Sendable (_ argumentsJSON: String) -> String
    public init(spec: ToolSpec, run: @escaping @Sendable (_ argumentsJSON: String) -> String) {
        self.spec = spec; self.run = run
    }
}

/// The result of a tool, ready to feed back into the conversation.
public struct ToolOutput: Sendable, Equatable {
    public var callId: String
    public var name: String
    public var content: String
    public init(callId: String, name: String, content: String) {
        self.callId = callId; self.name = name; self.content = content
    }
}

public enum ToolRegistry {
    /// The built-in tools (each defined in its own file under Builtins/).
    /// project_write/project_edit and file_write/file_add/file_modify have side
    /// effects (they modify files INSIDE the active project root only); else pure.
    public static let builtins: [BuiltinTool] = [clock, calculator, add, subtract, multiply,
                                                 webSearch, webFetch,
                                                 projectTree, projectList, projectFind,
                                                 projectRead, projectSearch,
                                                 projectWrite, projectEdit,
                                                 fileRead, fileLines, fileWrite, fileAdd, fileModify,
                                                 fileDelete, git,
                                                 agentsList, subagentSearch, subagentRun]

    /// Tools that require an imported project (a root to operate in). The sub-agent
    /// resolver drops these when no project is loaded.
    public static let projectScoped: Set<String> = [
        "project_tree", "project_list", "project_find",
        "project_read", "project_search", "project_edit", "project_write",
        "file_read", "file_lines", "file_write", "file_add", "file_modify", "file_delete", "git",
    ]

    public static func builtin(named name: String) -> BuiltinTool? {
        builtins.first { $0.spec.name == name }
    }

    /// Specs for the named subset (used to declare tools to the model).
    public static func specs(enabled names: Set<String>) -> [ToolSpec] {
        builtins.filter { names.contains($0.spec.name) }.map(\.spec)
    }

    /// Tools a sub-agent may be granted: every built-in EXCEPT the orchestration
    /// tools (no nested sub-agents; `agents_list` is for the orchestrator, not for
    /// doing work). The main agent passes a minimal subset of these to
    /// `subagent_run`; names outside this set are ignored.
    public static var subAgentGrantable: Set<String> {
        Set(builtins.map(\.spec.name)).subtracting(["subagent_run", "subagent_search", "agents_list"])
    }

    /// Run a model-emitted call against the built-ins; nil if it's not a built-in
    /// (the UI must then supply the result manually).
    public static func execute(_ call: ToolCall) -> ToolOutput? {
        guard let tool = builtin(named: call.name) else { return nil }
        return ToolOutput(callId: call.id, name: call.name, content: tool.run(call.argumentsJSON))
    }

    // MARK: - Shared argument-parsing helpers (used by the tool files in Builtins/)

    static func stringArg(_ json: String, _ key: String) -> String? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return obj[key] as? String
    }

    static func intArg(_ json: String, _ key: String) -> Int? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        if let n = obj[key] as? NSNumber { return n.intValue }
        if let s = obj[key] as? String { return Int(s) }
        return nil
    }

    /// Build a tool taking two numeric arguments `a` and `b` and returning `op(a,b)`.
    static func binaryTool(name: String, verb: String, symbol: String,
                           _ op: @escaping @Sendable (Double, Double) -> Double) -> BuiltinTool {
        let schema = #"{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}"#
        return BuiltinTool(
            spec: ToolSpec(name: name, description: "\(verb) two numbers: a \(symbol) b.", parametersJSON: schema),
            run: { argsJSON in
                guard let (a, b) = parseTwoNumbers(argsJSON) else {
                    return #"{"error":"expected numeric arguments 'a' and 'b'"}"#
                }
                return formatNumberResult(op(a, b))
            })
    }

    /// Parse `a` and `b` from a JSON arguments object. Accepts JSON numbers or
    /// numeric strings (some models quote their arguments).
    static func parseTwoNumbers(_ argsJSON: String) -> (Double, Double)? {
        guard let data = argsJSON.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let a = number(obj["a"]), let b = number(obj["b"]) else { return nil }
        return (a, b)
    }

    private static func number(_ any: Any?) -> Double? {
        if let n = any as? NSNumber { return n.doubleValue }
        if let s = any as? String { return Double(s.trimmingCharacters(in: .whitespaces)) }
        return nil
    }

    /// Render a numeric result as JSON, printing whole numbers without a ".0".
    static func formatNumberResult(_ value: Double) -> String {
        guard value.isFinite else { return #"{"error":"non-finite result"}"# }
        let s = (value.rounded() == value && abs(value) < 1e15) ? String(Int64(value)) : String(value)
        return #"{"result":\#(s)}"#
    }

    /// Safe arithmetic evaluation via a small recursive-descent parser (no
    /// NSExpression, so a malformed input returns an error instead of throwing an
    /// uncatchable ObjC exception). Supports + - * / % ^, unary minus,
    /// parentheses, the constants pi/e, and one-argument functions (sqrt, sin…).
    static func evaluateArithmetic(_ expr: String) -> String {
        guard let value = ArithmeticEvaluator.evaluate(expr) else {
            return #"{"error":"could not evaluate expression"}"#
        }
        return formatNumberResult(value)
    }
}

/// Minimal, crash-free arithmetic evaluator:
/// `expr := term (('+'|'-') term)*`,
/// `term := power (('*'|'/'|'%') power)*`,
/// `power := factor ('^' power)?`  (right-associative),
/// `factor := number | constant | function '(' expr ')' | '(' expr ')' | ('-'|'+') factor`.
/// Constants: pi, e. Functions: sqrt, cbrt, abs, exp, ln, log/log10, log2,
/// sin, cos, tan, asin, acos, atan, floor, ceil, round.
/// Returns nil on any malformed input (a non-finite result, e.g. sqrt(-1),
/// is rejected downstream by `formatNumberResult`). Pure and side-effect free.
enum ArithmeticEvaluator {
    static func evaluate(_ s: String) -> Double? {
        var p = Parser(Array(s))
        guard let v = p.parseExpression(), p.atEnd else { return nil }
        return v
    }

    static func constant(_ name: String) -> Double? {
        switch name {
        case "pi": return .pi
        case "e": return M_E
        default: return nil
        }
    }

    static func function(_ name: String, _ x: Double) -> Double? {
        switch name {
        case "sqrt": return x.squareRoot()
        case "cbrt": return cbrt(x)
        case "abs": return abs(x)
        case "exp": return exp(x)
        case "ln": return log(x)
        case "log", "log10": return log10(x)
        case "log2": return log2(x)
        case "sin": return sin(x)
        case "cos": return cos(x)
        case "tan": return tan(x)
        case "asin": return asin(x)
        case "acos": return acos(x)
        case "atan": return atan(x)
        case "floor": return floor(x)
        case "ceil": return ceil(x)
        case "round": return x.rounded()
        default: return nil
        }
    }

    private struct Parser {
        let c: [Character]
        var i = 0
        init(_ c: [Character]) { self.c = c }

        var atEnd: Bool { mutating get { skipSpaces(); return i >= c.count } }
        mutating func skipSpaces() { while i < c.count, c[i] == " " || c[i] == "\t" { i += 1 } }
        mutating func peek() -> Character? { skipSpaces(); return i < c.count ? c[i] : nil }

        mutating func parseExpression() -> Double? {
            guard var acc = parseTerm() else { return nil }
            while let op = peek(), op == "+" || op == "-" {
                i += 1
                guard let rhs = parseTerm() else { return nil }
                acc = (op == "+") ? acc + rhs : acc - rhs
            }
            return acc
        }

        mutating func parseTerm() -> Double? {
            guard var acc = parsePower() else { return nil }
            while let op = peek(), op == "*" || op == "/" || op == "%" {
                i += 1
                guard let rhs = parsePower() else { return nil }
                switch op {
                case "/": if rhs == 0 { return nil }; acc /= rhs
                case "%": if rhs == 0 { return nil }; acc = acc.truncatingRemainder(dividingBy: rhs)
                default: acc *= rhs
                }
            }
            return acc
        }

        /// Right-associative exponentiation: 2^3^2 = 2^(3^2) = 512.
        mutating func parsePower() -> Double? {
            guard let base = parseFactor() else { return nil }
            guard peek() == "^" else { return base }
            i += 1
            guard let exp = parsePower() else { return nil }
            return pow(base, exp)
        }

        mutating func parseFactor() -> Double? {
            guard let ch = peek() else { return nil }
            // Unary sign binds looser than '^' (so -2^2 = -(2^2) = -4, the
            // standard convention) but tighter than * and /.
            if ch == "-" { i += 1; guard let f = parsePower() else { return nil }; return -f }
            if ch == "+" { i += 1; return parsePower() }
            if ch == "(" {
                i += 1
                guard let v = parseExpression(), peek() == ")" else { return nil }
                i += 1
                return v
            }
            if ch.isLetter { return parseNameExpression() }
            return parseNumber()
        }

        /// A named constant (pi, e) or a one-argument function call f(expr).
        mutating func parseNameExpression() -> Double? {
            skipSpaces()
            let start = i
            while i < c.count, c[i].isLetter || c[i].isNumber { i += 1 }
            let name = String(c[start..<i]).lowercased()
            if let k = ArithmeticEvaluator.constant(name) { return k }
            guard peek() == "(" else { return nil }
            i += 1
            guard let v = parseExpression(), peek() == ")" else { return nil }
            i += 1
            return ArithmeticEvaluator.function(name, v)
        }

        mutating func parseNumber() -> Double? {
            skipSpaces()
            let start = i
            while i < c.count, c[i].isNumber || c[i] == "." { i += 1 }
            guard i > start else { return nil }
            return Double(String(c[start..<i]))
        }
    }
}
