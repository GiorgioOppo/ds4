import Foundation
import DS4Core

// Tool registry for the GUI: built-in, auto-executable demo tools plus the
// plumbing to run a model-emitted ToolCall. Tools the registry doesn't know are
// left to the UI to answer manually.

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
    /// The demo tools shipped with the app. All are pure and side-effect free.
    public static let builtins: [BuiltinTool] = [clock, calculator]

    public static func builtin(named name: String) -> BuiltinTool? {
        builtins.first { $0.spec.name == name }
    }

    /// Specs for the named subset (used to declare tools to the model).
    public static func specs(enabled names: Set<String>) -> [ToolSpec] {
        builtins.filter { names.contains($0.spec.name) }.map(\.spec)
    }

    /// Run a model-emitted call against the built-ins; nil if it's not a built-in
    /// (the UI must then supply the result manually).
    public static func execute(_ call: ToolCall) -> ToolOutput? {
        guard let tool = builtin(named: call.name) else { return nil }
        return ToolOutput(callId: call.id, name: call.name, content: tool.run(call.argumentsJSON))
    }

    // MARK: - Built-ins

    /// Current date/time in ISO-8601 (no parameters).
    static let clock = BuiltinTool(
        spec: ToolSpec(name: "now",
                       description: "Return the current local date and time in ISO-8601 format.",
                       parametersJSON: #"{"type":"object","properties":{}}"#),
        run: { _ in
            let f = ISO8601DateFormatter()
            return #"{"datetime":"\#(f.string(from: Date()))"}"#
        })

    /// Evaluate a basic arithmetic expression (+ - * / parentheses).
    static let calculator = BuiltinTool(
        spec: ToolSpec(name: "calculator",
                       description: "Evaluate a basic arithmetic expression with + - * / and parentheses.",
                       parametersJSON: #"{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}"#),
        run: { argsJSON in
            guard let data = argsJSON.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let expr = obj["expression"] as? String else {
                return #"{"error":"missing 'expression' argument"}"#
            }
            return evaluateArithmetic(expr)
        })

    /// Safe arithmetic evaluation via a small recursive-descent parser (no
    /// NSExpression, so a malformed input returns an error instead of throwing an
    /// uncatchable ObjC exception). Supports + - * / , unary minus, parentheses.
    static func evaluateArithmetic(_ expr: String) -> String {
        guard let value = ArithmeticEvaluator.evaluate(expr) else {
            return #"{"error":"could not evaluate expression"}"#
        }
        // Print integers without a trailing .0; keep finite results only.
        guard value.isFinite else { return #"{"error":"non-finite result"}"# }
        let s = (value.rounded() == value && abs(value) < 1e15)
            ? String(Int64(value)) : String(value)
        return #"{"result":\#(s)}"#
    }
}

/// Minimal, crash-free arithmetic evaluator: `expr := term (('+'|'-') term)*`,
/// `term := factor (('*'|'/') factor)*`, `factor := number | '(' expr ')' | '-' factor`.
/// Returns nil on any malformed input. Pure and side-effect free.
enum ArithmeticEvaluator {
    static func evaluate(_ s: String) -> Double? {
        var p = Parser(Array(s))
        guard let v = p.parseExpression(), p.atEnd else { return nil }
        return v
    }

    private struct Parser {
        let c: [Character]
        var i = 0
        init(_ c: [Character]) { self.c = c }

        var atEnd: Bool { skipSpaces(); return i >= c.count }
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
            guard var acc = parseFactor() else { return nil }
            while let op = peek(), op == "*" || op == "/" {
                i += 1
                guard let rhs = parseFactor() else { return nil }
                if op == "/" { if rhs == 0 { return nil }; acc /= rhs } else { acc *= rhs }
            }
            return acc
        }

        mutating func parseFactor() -> Double? {
            guard let ch = peek() else { return nil }
            if ch == "-" { i += 1; guard let f = parseFactor() else { return nil }; return -f }
            if ch == "+" { i += 1; return parseFactor() }
            if ch == "(" {
                i += 1
                guard let v = parseExpression(), peek() == ")" else { return nil }
                i += 1
                return v
            }
            return parseNumber()
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
