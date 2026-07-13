import Foundation
import DS4Core

extension ToolRegistry {
    /// Evaluate an arithmetic expression (+ - * / % ^, parentheses, functions).
    static let calculator = BuiltinTool(
        spec: ToolSpec(name: "calculator",
                       description: "Evaluate an arithmetic expression with + - * / % ^ and parentheses, the constants pi and e, and the functions sqrt, cbrt, abs, exp, ln, log, log2, sin, cos, tan, asin, acos, atan, floor, ceil, round.",
                       parametersJSON: #"{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}"#),
        run: { argsJSON in
            guard let data = argsJSON.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let expr = obj["expression"] as? String else {
                return #"{"error":"missing 'expression' argument"}"#
            }
            return evaluateArithmetic(expr)
        })
}
